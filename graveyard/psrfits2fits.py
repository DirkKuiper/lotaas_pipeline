import os
import sys
import tqdm
import argparse

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PSRFITS table to FITS image")
    parser.add_argument("-f", "--fscrunch", help="Average in frequency [int, default: 16]",
                        type=int, default=4)
    parser.add_argument("-t", "--tscrunch", help="Average in time [int, default: 16]",
                        type=int, default=16)
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

    nsub = 8

    # Output array
    outdata = np.zeros((nchan // mchan, nsamp * nsub // msamp), dtype="float32")

    
    # Loop over subints
    for isub in tqdm.tqdm(range(nsub)):
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
        data[:, 0::16] = np.nan
        
        # Average in frequency and time
        data = np.nanmean(data.reshape(nsamp, -1, mchan), axis=2)
        data = np.nanmean(data.reshape(-1, msamp, nchan // mchan), axis=1)
        
        # Add to array
        imin = isub * nsamp // msamp
        imax = imin + data.shape[0]
        outdata[:, imin:imax] = data.T.astype("float32")

    # Save as FITS image
    fits.PrimaryHDU(data=outdata, header=hdr).writeto(outfname, overwrite=True)