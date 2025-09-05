#!/usr/bin/env python3
"""
GPU-accelerated full processing pipeline for FRB search from filterbank files.
Includes:
  - RFI masking
  - Polynomial detrending
  - Dedispersion over DM trials
  - Matched filtering
  - Candidate clustering and classification

Falls back to CPU-based routines if GPU is unavailable.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # For finding modules
import yaml
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from numpy.polynomial import Polynomial

from lotaas_reprocessing import (
    filterbank, plotting, cluster
)
from graveyard import classify_testing as classify

# Attempt to import GPU-accelerated modules
use_gpu = False
try:
    from lotaas_reprocessing.cupy_utils import fourier_domain_dedispersion, compute_rfi_mask
    from lotaas_reprocessing import matched_filter_gpu as matched_filter
    print("Using CuPy and GPU-accelerated matched filtering")
    use_gpu = True
except (ModuleNotFoundError, ImportError) as e:
    print(f"GPU acceleration unavailable ({e}). Falling back to CPU.")
    from lotaas_reprocessing.numpy_utils import fourier_domain_dedispersion, compute_rfi_mask
    from lotaas_reprocessing import matched_filter

def main():
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python3 pipeline.py <input_fil_file> [output_directory]")
        sys.exit(1)

    # Input filename and base name for outputs
    fname = sys.argv[1]
    base_fname = os.path.basename(fname).replace(".fil", "")
    output_dir = sys.argv[2] if len(sys.argv) > 2 else f"{base_fname}_output"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load configuration from YAML
        with open("settings.yaml", "r") as fp:
            settings = yaml.load(fp, Loader=yaml.FullLoader)

        # Load filterbank data
        fil = filterbank.FilterbankFile(fname, "read")
        data = np.flipud(fil.get_spectra(0, fil.nspec).T)
        nu = np.flipud(fil.frequencies)
        tsamp = fil.tsamp
        t = np.arange(data.shape[1]) * tsamp
        fil.close()

        print(f"Read {data.shape[1]} samples, {data.shape[0]} channels")

        # RFI masking
        block_size = settings["rfi_block_size"]
        print(f"Computing RFI mask with {block_size} samples")
        start_time = time.time()
        mask = compute_rfi_mask(data, block_size)
        print(f"RFI mask time: {time.time() - start_time:.2f}s")

        masked_frac = np.sum(mask) / mask.size
        print(f"Masked fraction: {masked_frac:.2%}")

        # Normalize data and mask bad channels
        data = data / np.median(data) - 1
        mask[138, :] = True  # Mask known bad channel
        masked_data = data.copy()
        masked_data[mask] = np.nan

        # Extract and display observation info
        rfi_averaging_factor = settings["rfi_averaging_factor"]
        observation_info = plotting.get_observation_info(
            fil, block_size, masked_frac, rfi_averaging_factor
        )

        # Save RFI diagnostic plot
        rfi_plot_path = os.path.join(output_dir, f"{base_fname}_rfi_diagnostic_plot.png")
        plotting.rfi_diagnostic_plot(
            masked_data, data, mask, t, nu, rfi_averaging_factor,
            filename=fname, save_path=rfi_plot_path, observation_info=observation_info
        )

        # Replace masked values with Gaussian noise
        print("Replacing masked data with noise...")
        noise = np.random.normal(np.nanmean(masked_data), np.nanstd(masked_data), masked_data.shape)
        masked_data[mask] = noise[mask]

        # Detrend each channel with 2nd degree polynomial
        print("Detrending...")
        detrended_data = np.zeros_like(masked_data)
        time_axis = np.arange(masked_data.shape[1])
        for i in range(masked_data.shape[0]):
            p = Polynomial.fit(time_axis, masked_data[i, :], deg=2)
            detrended_data[i, :] = masked_data[i, :] - p(time_axis)
        masked_data = detrended_data

        # Generate minimal FITS-like header for metadata
        hdr = fits.Header()
        hdr["MJD-OBS"] = fil.header["tstart"]
        hdr["CRPIX1"] = 0
        hdr["CRVAL1"] = 0
        hdr["CDELT1"] = tsamp
        hdr["CRPIX2"] = 0
        hdr["CRVAL2"] = fil.header["foff"] * fil.header["nchans"] + fil.header["fch1"]
        hdr["CDELT2"] = np.abs(fil.header["foff"])

        # Loop over dedispersion plan
        dedispersion_plan = settings["dedispersion_plan"]
        dm_trials_dir = os.path.join(output_dir, "DM_trials")
        # os.makedirs(dm_trials_dir, exist_ok=True)

        # for entry in dedispersion_plan:
        #     dms = np.arange(entry["low_dm"], entry["high_dm"], entry["ddm"])
        #     print(f"Dedispersing DMs {entry['low_dm']} to {entry['high_dm']} (step {entry['ddm']})")

        #     I_f_dm = fourier_domain_dedispersion(
        #         masked_data, hdr["CDELT1"] * entry["downsample"], nu, dms
        #     )
        #     I_f_dm = I_f_dm[:, ::entry["downsample"]]
        #     I_t_dm = np.real(np.fft.irfft(I_f_dm, axis=1)).astype("float32")

        #     for i, dm in enumerate(dms):
        #         dm_filename = os.path.join(dm_trials_dir, f"{base_fname}_DM{dm:.1f}")
        #         I_t_dm[i].tofile(f"{dm_filename}.dat")
        #         with open(f"{dm_filename}.inf", "w") as inf:
        #             inf.write(f" Data file name without suffix          =  {os.path.basename(dm_filename)}\n")
        #             inf.write(f" Telescope used                         =  {fil.header.get('telescope', 'LOFAR')}\n")
        #             inf.write(f" Instrument used                        =  {fil.header.get('instrument', 'Unknown')}\n")
        #             inf.write(f" Object being observed                  =  {fil.header.get('source_name', 'Unknown')}\n")
        #             inf.write(f" J2000 Right Ascension (hh:mm:ss.ssss)  =  {fil.header.get('src_raj', '00:00:00.0000')}\n")
        #             inf.write(f" J2000 Declination     (dd:mm:ss.ssss)  =  {fil.header.get('src_dej', '+00:00:00.0000')}\n")
        #             inf.write(f" Epoch of observation (MJD)             =  {fil.header.get('tstart', 0.0)}\n")
        #             inf.write(f" Dispersion measure (cm-3 pc)           =  {dm:.2f}\n")
        #             inf.write(f" Number of bins in the time series      =  {I_t_dm.shape[1]}\n")
        #             inf.write(f" Width of each time series bin (sec)    =  {hdr['CDELT1'] * entry['downsample']:.6f}\n")
        #             inf.write(f" Total bandwidth (MHz)                  =  {np.abs(fil.header['foff']) * fil.header['nchans']:.6f}\n")
        #             inf.write(f" Number of channels                     =  {fil.header['nchans']}\n")
        #             inf.write(f" Channel bandwidth (MHz)                =  {np.abs(fil.header['foff']):.6f}\n")

        # Run matched filtering
        # matched_filter.run_all_matched_filtering(
        #     dm_trials_dir, tsamp, output_dir, observation_info, dedispersion_plan
        # )

        # Cluster and classify detected candidates
        # all_candidates = os.path.join(output_dir, "all_detected_candidates.cands")
        clustered_output = os.path.join(output_dir, "clustered_candidates.txt")
        # print("Clustering candidates...")
        # cluster.cluster_candidates(all_candidates, clustered_output)

        print("Classifying candidates...")
        plot_dir = os.path.join(output_dir, "candidate_plots")
        os.makedirs(plot_dir, exist_ok=True)
        classify.classify_candidates(fname, clustered_output, plot_dir, observation_info)

        # Clean up temporary DM trial files
        print("Cleaning up temporary files...")
        shutil.rmtree(dm_trials_dir, ignore_errors=True)
        print("Pipeline completed successfully.")

    except Exception as e:
        print(f"Pipeline error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()