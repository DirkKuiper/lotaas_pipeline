#!/usr/bin/env python3
import os
import json
import sys
import tqdm
import yaml
import time
import re
import shutil

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import convolve2d
from scipy.signal import detrend
from scipy.ndimage import zoom
from scipy.ndimage import median_filter
from numpy.polynomial import Polynomial

from lotaas_reprocessing import filterbank
from lotaas_reprocessing import plotting

from pathlib import Path
import argparse
from lotaas_reprocessing.dedispersion import backend, iter_dedispersed
from lotaas_reprocessing.dm_plan import dm_values, dm_label
from lotaas_reprocessing.trials import trial_specs, link_trial, write_manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOTAAS dedispersion stage")
    parser.add_argument("input_fil_file")
    parser.add_argument("output_directory")
    parser.add_argument("--settings", default=str(Path(__file__).resolve().parents[1] / "settings.yaml"))
    parser.add_argument("--backend", choices=["cpu", "gpu", "auto"], default="gpu")
    parser.add_argument("--pilot", action="store_true", help="Mark incomplete-beam validation runs")
    parser.add_argument("--max-samples", type=int, help="Pilot only: read a prefix of the beam")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    started = time.monotonic()
    xp = backend(args.backend)
    use_gpu = xp is not np
    from lotaas_reprocessing.numpy_utils import compute_rfi_mask
    fname = args.input_fil_file
    base_fname = os.path.basename(fname).replace(".fil", "")
    output_dir = args.output_directory
    print("Dedispersion backend:", "gpu" if use_gpu else "cpu")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Read settings
    with open(args.settings, "r") as fp:
        settings = yaml.load(fp, Loader=yaml.FullLoader)
    
    # Read filterbank
    fil = filterbank.FilterbankFile(fname, "read")
    data = np.flipud(fil.get_spectra(0, min(fil.nspec, args.max_samples) if args.max_samples else fil.nspec).T)
    fil.close()

    # Axes
    nchan, nsamp = data.shape
    tsamp = fil.tsamp
    print(tsamp)
    t = np.arange(nsamp) * tsamp
    nu = np.flipud(fil.frequencies)
    print(f"Read {nsamp} samples, {nchan} channels of {fname}")

    # RFI mask timeseries length
    block_size = settings["rfi_block_size"]

    # Compute RFI mask with timing
    print(f"Computing RFI mask with {block_size} samples")
    start_time = time.time()  # Start timing
    mask = compute_rfi_mask(data, block_size)
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"RFI mask computation time: {elapsed_time:.4f} seconds")

    # Calculate masked fraction
    masked_frac = np.sum(mask) / np.prod(mask.shape)
    print(f"Masked data fraction: {masked_frac * 100:.2f} %")

    # Normalize and offset to zero
    median = np.median(data)
    if not np.isfinite(median) or median == 0:
        raise ValueError("Invalid filterbank normalization")
    data = data / median - 1

   
    # Channels always masked, from settings rather than a literal, so a
    # change of band or station set does not need a code edit.
    bad_channels = settings.get("bad_channels") or []
    masked_channels = []
    for bad_channel in bad_channels:
        if not 0 <= bad_channel < nchan:
            raise ValueError(f"bad_channels entry {bad_channel} is outside 0..{nchan-1}")
        mask[bad_channel, :] = True
        masked_channels.append(bad_channel)
    print(f"Channels always masked: {masked_channels}")
    
    # # Apply mask
    masked_data = data.copy()
    masked_data[mask] = np.nan


    # Get dynamic observation information from filterbank header
    rfi_averaging_factor = settings["rfi_averaging_factor"]
    observation_info = plotting.get_observation_info(fil, block_size, masked_frac, rfi_averaging_factor)

   # Save RFI diagnostic plot in the main output directory
    save_path = os.path.join(output_dir, f"{base_fname}_rfi_diagnostic_plot.png")
    plotting.rfi_diagnostic_plot(
        masked_data, 
        data, 
        mask, 
        t, 
        nu, 
        rfi_averaging_factor, 
        filename=fname,
        save_path=save_path,
        observation_info=observation_info
    )

   # Masking and replacing data with random noise
    print("Replacing masked data with random noise...")
    rng = np.random.default_rng(args.seed)
    if not np.isfinite(masked_data).any():
        raise ValueError("All data were masked")
    masked_data[mask] = rng.normal(np.nanmean(masked_data), np.nanstd(masked_data), int(mask.sum()))

    print(f"nu shape: {nu.shape}, range: {nu.min()} - {nu.max()}")

    # Apply detrending along the time axis
    print("Detrending data using numpy.polynomial.Polynomial...")

    # Preallocate the detrended data array
    detrended_data = np.zeros_like(masked_data)

    # Time axis (e.g., time samples as integers)
    t = np.arange(masked_data.shape[1])

    # Fit and subtract a degree-2 polynomial for each frequency channel
    for freq_idx in range(masked_data.shape[0]):
        # Fit polynomial to the time series of the current frequency channel
        p = Polynomial.fit(t, masked_data[freq_idx, :], deg=2)
        
        # Subtract the polynomial fit to detrend
        detrended_data[freq_idx, :] = masked_data[freq_idx, :] - p(t)

    masked_data = detrended_data

    # Create FITS header
    hdr = fits.Header()
    #    hdr["RA"] = fits_hdr["RA"]
    #    hdr["DEC"] = fits_hdr["DEC"]
    hdr["MJD-OBS"] = fil.header["tstart"]
    #    hdr["SRC_NAME"] = fits_hdr["SRC_NAME"]
    hdr["CRPIX1"] = 0
    hdr["CRVAL1"] = 0
    hdr["CDELT1"] = fil.header["tsamp"]
    hdr["CRPIX2"] = 0
    hdr["CRVAL2"] = fil.header["foff"] * fil.header["nchans"] + fil.header["fch1"]
    hdr["CDELT2"] = np.abs(fil.header["foff"])

    # Extract the dedispersion plan from settings
    dedispersion_plan = settings["dedispersion_plan"]
    periodicity = settings.get("periodicity", {}) or {}
    periodicity_enabled = bool(periodicity.get("enabled", False))
    periodicity_plan = periodicity.get("dm_plan") or dedispersion_plan

    # Validate both grids before writing any products.
    trial_metadata = {"filename": fname, "samples_processed": nsamp}
    trial_specs(trial_metadata, dedispersion_plan)
    if periodicity_enabled:
        from lotaas_reprocessing.periodicity import resolved_config
        periodicity = resolved_config(periodicity)
        trial_specs(trial_metadata, periodicity_plan)

    # Refuse a beam that cannot fit rather than filling the disk part way
    # through. Trials are only removed when the CPU stage succeeds, so failed
    # beams accumulate about 3.6 GB each, and a disk filled mid-write fails
    # every beam running beside this one with a truncated trial.
    plans_to_write = [(dedispersion_plan, "DM_trials")]
    if periodicity_enabled:
        plans_to_write.append((periodicity_plan, "Periodic_DM_trials"))
    unique_trials = {(dm, int(entry["downsample"]))
                     for plan, _ in plans_to_write for entry in plan
                     for dm in dm_values(entry)}
    planned_bytes = 4 * sum(nsamp // downsample for _, downsample in unique_trials)
    free_bytes = shutil.disk_usage(output_dir).free
    if free_bytes < planned_bytes * 1.1:
        raise RuntimeError(
            f"DM trials for this beam need {planned_bytes/1e9:.1f} GB but only "
            f"{free_bytes/1e9:.1f} GB is free in {output_dir}. Reclaim trials from "
            f"failed beams with python3 -m euroflash.reclaim before retrying.")
    print(f"DM trials will occupy {planned_bytes/1e9:.1f} GB; {free_bytes/1e9:.1f} GB free")

    trial_cache = {}

    def write_trial(dm, trial_cpu, directory_name, entry):
        dm_trials_dir = os.path.join(output_dir, directory_name)
        os.makedirs(dm_trials_dir, exist_ok=True)
        dm_filename = os.path.join(dm_trials_dir, f"{base_fname}_DM{dm_label(dm)}")
        trial_cpu.tofile(f"{dm_filename}.dat.partial")
        os.replace(f"{dm_filename}.dat.partial", f"{dm_filename}.dat")
        with open(f"{dm_filename}.inf", "w") as inf_file:
            inf_file.write(f" Data file name without suffix          =  {os.path.basename(dm_filename)}\n")
            inf_file.write(f" Telescope used                         =  {fil.header.get('telescope', 'LOFAR')}\n")
            inf_file.write(f" Instrument used                        =  {fil.header.get('instrument', 'Unknown')}\n")
            inf_file.write(f" Object being observed                  =  {fil.header.get('source_name', 'Unknown')}\n")
            inf_file.write(f" J2000 Right Ascension (hh:mm:ss.ssss)  =  {fil.header.get('src_raj', '00:00:00.0000')}\n")
            inf_file.write(f" J2000 Declination     (dd:mm:ss.ssss)  =  {fil.header.get('src_dej', '+00:00:00.0000')}\n")
            inf_file.write(f" Epoch of observation (MJD)             =  {fil.header.get('tstart', 0.0)}\n")
            inf_file.write(f" Dispersion measure (cm-3 pc)           =  {dm:.2f}\n")
            inf_file.write(f" Number of bins in the time series      =  {trial_cpu.size}\n")
            inf_file.write(f" Width of each time series bin (sec)    =  {hdr['CDELT1'] * entry['downsample']:.6f}\n")
            inf_file.write(f" Total bandwidth (MHz)                  =  {np.abs(fil.header['foff']) * fil.header['nchans']:.6f}\n")
            inf_file.write(f" Number of channels                     =  {fil.header['nchans']}\n")
            inf_file.write(f" Channel bandwidth (MHz)                =  {np.abs(fil.header['foff']):.6f}\n")
        return dm_filename + ".dat"

    def write_plan(plan, directory_name):
      plan_started = time.monotonic()
      computed_trials = 0
      linked_trials = 0
      gpu_memory_peak = 0
      # The same channel FFT is recomputed once per range, but all DM trials in
      # a range share it.  This keeps periodicity cheaper than a second input
      # FFT/preprocessing pipeline and avoids host rereads of filterbanks.
      for entry in plan:
        low_dm = entry["low_dm"]
        high_dm = entry["high_dm"]
        ddm = entry["ddm"]
        downsample = entry["downsample"]

        # Generate the DM values for this range
        dms = dm_values(entry)

        # Dedisperse for this range
        print(f"Dedispersing DM range {low_dm} to {high_dm} (step {ddm}, downsample {downsample})")
        dm_trials_dir = os.path.join(output_dir, directory_name)
        os.makedirs(dm_trials_dir, exist_ok=True)
        missing = [dm for dm in dms if (dm, downsample) not in trial_cache]
        for dm, trial in (iter_dedispersed(masked_data, tsamp, nu, missing, downsample, xp) if missing else ()):
            trial_cpu = xp.asnumpy(trial) if use_gpu else trial
            trial_cache[(dm, downsample)] = write_trial(dm, trial_cpu, directory_name, entry)
            computed_trials += 1
            if use_gpu:
                gpu_memory_peak = max(gpu_memory_peak, xp.get_default_memory_pool().used_bytes())
        # Exact (DM, downsample) repeats between the transient and periodic
        # plans are hard-linked instead of recomputed or copied through the
        # host.  This is the principal reuse path for the periodic search.
        for dm in dms:
            key = (dm, downsample)
            target = os.path.join(output_dir, directory_name, f"{base_fname}_DM{dm_label(dm)}.dat")
            if trial_cache[key] != target:
                link_trial(trial_cache[key], target)
                linked_trials += 1
      return {"elapsed_seconds": time.monotonic() - plan_started,
              "computed_trials": computed_trials, "linked_trials": linked_trials,
              "gpu_memory_pool_peak_bytes": gpu_memory_peak}

    dedispersion_timing = write_plan(dedispersion_plan, "DM_trials")
    periodicity_timing = None
    if periodicity_enabled:
        periodicity_timing = write_plan(periodicity_plan, "Periodic_DM_trials")

    # Save metadata to a YAML file
    metadata_file = os.path.join(output_dir, "metadata.yaml")
    metadata = {"tsamp": tsamp, "observation_info": observation_info, "dedispersion_plan": dedispersion_plan, "periodicity": periodicity, "periodicity_enabled": periodicity_enabled, "periodicity_dm_plan": periodicity_plan if periodicity_enabled else [], "stage_timings": {"dedispersion": dedispersion_timing, "periodicity_dedispersion": periodicity_timing}, "filename": str(Path(fname).resolve()), "backend": "gpu" if use_gpu else "cpu", "samples_processed": nsamp, "pilot": args.pilot or args.max_samples is not None, "seed": args.seed,
                   # Band limits, so the search can size the tail that circular
                   # dedispersion pollutes at each DM.
                   "nu_min": float(np.min(nu)), "nu_max": float(np.max(nu)),
                   "tstart_mjd": float(fil.header["tstart"]), "bad_channels": [nchan - 1 - channel for channel in bad_channels],
                   "elapsed_seconds": time.monotonic() - started}

    # JSON keeps the host orchestrator free of container-only dependencies.
    metadata = json.loads(json.dumps(metadata, default=lambda value: value.item() if hasattr(value, "item") else str(value)))
    with open(metadata_file, "w") as fp:
        yaml.safe_dump(metadata, fp)
    from lotaas_reprocessing.periodicity import atomic_json
    atomic_json(Path(output_dir)/"metadata.json", metadata)
    write_manifest(output_dir)
    print(f"Saved metadata to {metadata_file}")
    
    print("Dedispersion complete.")
