#!/usr/bin/env python3
import os
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

# Try importing GPU utilities; fallback to CPU on failure
use_gpu = False
try:
    from lotaas_reprocessing.cupy_utils import fourier_domain_dedispersion, compute_rfi_mask
    from lotaas_reprocessing import matched_filter_gpu as matched_filter
    print("Using CuPy and GPU-accelerated dedispersion.")
    use_gpu = True
except (ModuleNotFoundError, ImportError) as e:
    print(f"GPU acceleration unavailable ({e}). Falling back to CPU.")
    from lotaas_reprocessing.numpy_utils import fourier_domain_dedispersion, compute_rfi_mask
    from lotaas_reprocessing import matched_filter  # CPU-based filtering

if __name__ == "__main__":
    # Check for input arguments
    if len(sys.argv) < 2:
        print("Usage: python3 pipeline.py <input_fil_file> [output_directory]")
        sys.exit(1)

    # Input file
    fname = sys.argv[1]
    base_fname = os.path.basename(fname).replace(".fil", "")

    # Output directory
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Read settings
    with open("settings.yaml", "r") as fp:
        settings = yaml.load(fp, Loader=yaml.FullLoader)
    
    # Read filterbank
    fil = filterbank.FilterbankFile(fname, "read")
    data = np.flipud(fil.get_spectra(0, fil.nspec).T)
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
    data = data / np.median(data) - 1

   
    # Define the bad channel index
    bad_channel = 138

    # Add the bad channel to the mask
    mask[bad_channel, :] = True  # Mask all time samples for channel 138
    print(f"Channel {bad_channel} has been added to the mask.")
    
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
    random_data = np.random.normal(np.nanmean(masked_data), np.nanstd(masked_data), masked_data.shape)
    masked_data[mask] = random_data[mask]

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

    # Loop over the dedispersion plan
    for entry in dedispersion_plan:
        low_dm = entry["low_dm"]
        high_dm = entry["high_dm"]
        ddm = entry["ddm"]
        downsample = entry["downsample"]

        # Generate the DM values for this range
        dms = np.arange(low_dm, high_dm, ddm)

        # Dedisperse for this range
        print(f"Dedispersing DM range {low_dm} to {high_dm} (step {ddm}, downsample {downsample})")
        I_f_dm = fourier_domain_dedispersion(masked_data, hdr["CDELT1"] * downsample, nu, dms)

        # Apply downsampling
        I_f_dm = I_f_dm[:, ::downsample]

        # Store downsample factor per DM trial for matched filtering
        downsampling_map = {dm: downsample for dm in dms}

        # iFFT to time domain
        I_t_dm = np.real(np.fft.irfft(I_f_dm, axis=1)).astype("float32")

       # Folder to save all DM trials
        dm_trials_dir = os.path.join(output_dir, "DM_trials")
        os.makedirs(dm_trials_dir, exist_ok=True)

        # Save each DM trial
        for i, dm in enumerate(dms):
            dm_filename = os.path.join(dm_trials_dir, f"{base_fname}_DM{dm:.1f}")
            I_t_dm[i].tofile(f"{dm_filename}.dat")

            # Write .inf files
            with open(f"{dm_filename}.inf", "w") as inf_file:
                inf_file.write(f" Data file name without suffix          =  {os.path.basename(dm_filename)}\n")
                inf_file.write(f" Telescope used                         =  {fil.header.get('telescope', 'LOFAR')}\n")
                inf_file.write(f" Instrument used                        =  {fil.header.get('instrument', 'Unknown')}\n")
                inf_file.write(f" Object being observed                  =  {fil.header.get('source_name', 'Unknown')}\n")
                inf_file.write(f" J2000 Right Ascension (hh:mm:ss.ssss)  =  {fil.header.get('src_raj', '00:00:00.0000')}\n")
                inf_file.write(f" J2000 Declination     (dd:mm:ss.ssss)  =  {fil.header.get('src_dej', '+00:00:00.0000')}\n")
                inf_file.write(f" Epoch of observation (MJD)             =  {fil.header.get('tstart', 0.0)}\n")
                inf_file.write(f" Dispersion measure (cm-3 pc)           =  {dm:.2f}\n")
                inf_file.write(f" Number of bins in the time series      =  {I_t_dm.shape[1]}\n")
                inf_file.write(f" Width of each time series bin (sec)    =  {hdr['CDELT1'] * downsample:.6f}\n")
                inf_file.write(f" Total bandwidth (MHz)                  =  {np.abs(fil.header['foff']) * fil.header['nchans']:.6f}\n")
                inf_file.write(f" Number of channels                     =  {fil.header['nchans']}\n")
                inf_file.write(f" Channel bandwidth (MHz)                =  {np.abs(fil.header['foff']):.6f}\n")

    # Save metadata to a YAML file
    metadata_file = os.path.join(output_dir, "metadata.yaml")
    with open(metadata_file, "w") as fp:
        yaml.dump({"tsamp": tsamp, "observation_info": observation_info, "dedispersion_plan": dedispersion_plan, "filename": fname}, fp)

    print(f"Saved metadata to {metadata_file}")
    
    print("GPU processing complete.")