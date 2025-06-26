#!/usr/bin/env python3
import os
import sys
import tqdm
import argparse

import numpy as np
import matplotlib.pyplot as plt

from astropy.time import Time
from astropy.io import fits
from lotaas_reprocessing import filterbank

fil_header_keys = [
    "telescope_id",
    "machine_id",
    "data_type",
    "rawdatafile",
    "source_name",
    "barycentric",
    "pulsarcentric",
    "az_start",
    "za_start",
    "src_raj",
    "src_dej",
    "tstart",
    "tsamp",
    "nbits",
    "fch1",
    "foff",
    "nchans",
    "nifs"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PSRFITS table to FITS image")
    parser.add_argument("-f", "--fscrunch", help="Average in frequency [int, default: 16]",
                        type=int, default=4)
    parser.add_argument("-t", "--tscrunch", help="Average in time [int, default: 16]",
                        type=int, default=16)
    parser.add_argument("-n", "--nsub", help="Number of subints to process [int, default: all]",
                        type=int, default=None)
    parser.add_argument("-d", "--dc", help="Mask DC channel of each subint [bool, default: False]",
                        action="store_true")
    parser.add_argument("--detect-time", type=float, default=None,
                    help="Detection time in seconds (center of the chunk)")
    parser.add_argument("--window", type=float, default=10.0,
                    help="Window size in seconds around detection time [default: 10s]")
    parser.add_argument("-o", "--output", help="output source file [FITS]", metavar="FILE")
    parser.add_argument("input", help="Input source file [PSRFITS]", metavar="FILE")
    args = parser.parse_args()

    # Scrunch factors
    mchan = args.fscrunch
    msamp = args.tscrunch

    # Get SAP/BEAM IDs
    parts = os.path.basename(args.input.replace(".fits", "")).split("_")
    obsid = parts[0]
    sap = int(parts[1][3:])
    beam = int(parts[2][4:])
    
    # Get output filename
    if args.output == None:
        outfname = f"{obsid}_SAP{sap:03d}_B{beam:03d}.fits"
    else:
        outfname = args.output

    # Open input PSRFITS file
    hdu = fits.open(args.input, mmemap=True)

    # Read header information
    fits_hdr = hdu[0].header
    subint_hdr = hdu['SUBINT'].header
    nchan = subint_hdr['NCHAN']
    nsamp = subint_hdr['NSBLK']
    nsub = subint_hdr['NAXIS2']
    nbit = subint_hdr['NBITS']
    chanbw = subint_hdr['CHAN_BW']
    tsamp = subint_hdr["TBIN"]
    obsfreq = fits_hdr['OBSFREQ']

    # Fix frequency accuracy
    subbandbw = 0.1953125
    chanbw = subbandbw / np.round(subbandbw / np.abs(chanbw))

    # Fix time accuracy
    rawtsamp = 5.12e-6
    
    # Output header
    hdr = fits.Header()
    hdr["OBSID"] = obsid
    hdr["SAP"] = sap
    hdr["BEAM"] = beam
    hdr["RA"] = fits_hdr["RA"]
    hdr["DEC"] = fits_hdr["DEC"]
    hdr["DATE-OBS"] = fits_hdr["DATE-OBS"]
    hdr["SRC_NAME"] = fits_hdr["SRC_NAME"]
    hdr["CRPIX1"] = 0
    hdr["CRVAL1"] = 0
    hdr["CDELT1"] = np.round(tsamp * msamp / rawtsamp) * rawtsamp
    hdr["CRPIX2"] = 0
    hdr["CRVAL2"] = obsfreq - np.abs(chanbw) * nchan / 2 #- np.abs(chanbw) / 2 # Unsure if the last part is needed
    hdr["CDELT2"] = chanbw * mchan
    hdr["DCMASK"] = args.dc

    if args.detect_time is not None:
        subint_duration = nsamp * tsamp
        center_subint = int(args.detect_time / subint_duration)
        half_window_subints = int(args.window / subint_duration)
        start_subint = max(0, center_subint - half_window_subints)
        end_subint = min(subint_hdr['NAXIS2'], center_subint + half_window_subints + 1)
        subint_indices = range(start_subint, end_subint)
        print(f"Processing subints {start_subint} to {end_subint} (around {args.detect_time}s)")
    else:
        # Default: all subints
        nsub = args.nsub if args.nsub is not None else subint_hdr['NAXIS2']
        subint_indices = range(nsub)
    
    # Output array
    nsub_used = len(subint_indices)
    outdata = np.zeros((nchan // mchan, nsamp * nsub_used // msamp), dtype="float32")

    # Loop over subints
    for i, isub in enumerate(tqdm.tqdm(subint_indices)):
        # Get data, offsets, scales and weights
        data = hdu['SUBINT'].data[isub]['DATA']
        offsets = hdu['SUBINT'].data[isub]['DAT_OFFS']
        scales = hdu['SUBINT'].data[isub]['DAT_SCL']
        weights = hdu['SUBINT'].data[isub]['DAT_WTS']

        # Unpack 2bit to 8bit
        if nbit == 2:
            spec = np.packbits(np.unpackbits(np.squeeze(data)).reshape(-1, 2), axis=1).reshape(-1, nchan)
        elif nbit == 8:
            spec = data.reshape(-1, nchan)

        # Apply scales and offsets
        data = (spec * scales + offsets) * weights

        # Mask DC channels
        if args.dc:
            data[:, 0::16] = np.nan
        
        # Average in frequency and time
        if (msamp > 1) or (mchan > 1):
            data = np.nanmean(data.reshape(nsamp // msamp, msamp, nchan // mchan, mchan), axis=(1, 3))
        
        # Add to array
        imin = i * nsamp // msamp
        imax = imin + data.shape[0]         
        outdata[:, imin:imax] = data.T.astype("float32")

    # Save as FITS image
    #fits.PrimaryHDU(data=outdata, header=hdr).writeto(outfname, overwrite=True)

    nchan, nsamp = outdata.shape
    t = hdr["CRVAL1"] + hdr["CDELT1"] * np.arange(nsamp)
    freq = hdr["CRVAL2"] + hdr["CDELT2"] * np.arange(nchan)
    nbits = 32

    print(hdr["DATE-OBS"])

    # Create header
    fil_header = dict.fromkeys(fil_header_keys, None)
    fil_header["telescope_id"] = 11
    fil_header["machine_id"] = -1
    fil_header["data_type"] = 1 
    fil_header["rawdatafile"] = ""
    fil_header["source_name"] = hdr["SRC_NAME"]
    fil_header["barycentric"] = 0 
    fil_header["pulsarcentric"] = 0 
    fil_header["az_start"] = 0
    fil_header["za_start"] = 0
    fil_header["src_raj"] = float(hdr['RA'].replace(':',''))
    fil_header["src_dej"] = float(hdr['DEC'].replace(':',''))
    start_sample = subint_indices[0] * nsamp
    start_seconds = start_sample * tsamp
    fil_header["tstart"] = Time(hdr["DATE-OBS"], format="isot", scale="utc").mjd # + start_seconds / 86400.0
    fil_header["tsamp"] = hdr["CDELT1"]
    fil_header["foff"] = -hdr["CDELT2"]
    fil_header["fch1"] = hdr["CRVAL2"] + hdr["CDELT2"] * nchan
    fil_header["nchans"] = nchan
    fil_header["nifs"] = 1
    fil_header["nbits"] = nbits
    

    # Write file
    outfname = outfname.replace(".fits", ".fil")
    # print(f"Writing filterbank to: {outfname}")
    # outfil = filterbank.create_filterbank_file(outfname, fil_header, nbits=nbits)
    # outfil.append_spectra(np.flipud(outdata).T)
    # outfil.close()

    print(f"Writing filterbank to: {outfname}")
    outfil = filterbank.create_filterbank_file(outfname, fil_header, nbits=nbits)

    chunk_size = 1024  # adjust if needed
    n_spectra = outdata.shape[1]

    for i in tqdm.tqdm(range(0, n_spectra, chunk_size), desc="Writing spectra"):
        block = outdata[:, i:i+chunk_size].T  # shape (time, freq)
        outfil.append_spectra(np.flipud(block))

    outfil.close()
    print("File written successfully.")
