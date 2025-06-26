import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Check if running inside a SLURM job
in_slurm = "SLURM_JOB_ID" in os.environ

if not in_slurm:
    from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Requantize and downsample PSRFITS data to 32-bit float")
    parser.add_argument("-f", "--fscrunch", help="Average in frequency [int, default: 4]", type=int, default=4)
    parser.add_argument("-t", "--tscrunch", help="Average in time [int, default: 16]", type=int, default=16)
    parser.add_argument("-o", "--output", help="Output PSRFITS file", metavar="FILE")
    parser.add_argument("-d", "--dc", help="Mask DC channel of each subint [bool, default: False]", action="store_true")
    parser.add_argument("input", help="Input PSRFITS file", metavar="FILE")
    args = parser.parse_args()

    # Scrunch factors
    mchan = args.fscrunch
    msamp = args.tscrunch

    # Get SAP/BEAM IDs
    parts = os.path.basename(args.input.replace(".fits", "")).split("_")
    obsid = parts[0]
    sap = int(parts[1][3:])
    beam = int(parts[2][4:])

    # Output filename
    if args.output is None:
        outfname = f"{obsid}_SAP{sap:03d}_B{beam:03d}_downsampled_32bit.fits"
    else:
        outfname = args.output

    # Open input PSRFITS file
    hdu = fits.open(args.input, mode="readonly", memmap=True)

    # Read headers
    fits_hdr = hdu[0].header
    subint_hdr = hdu['SUBINT'].header

    nchan = subint_hdr['NCHAN']
    nsamp = subint_hdr['NSBLK']
    nsub = subint_hdr['NAXIS2']
    nbit = subint_hdr['NBITS']
    chanbw = subint_hdr['CHAN_BW']
    tsamp = subint_hdr["TBIN"]
    obsfreq = fits_hdr['OBSFREQ']

    # Fix frequency and time accuracy
    subbandbw = 0.1953125
    chanbw = subbandbw / np.round(subbandbw / np.abs(chanbw))
    rawtsamp = 5.12e-6

    # Update header values for downsampling
    subint_hdr["NCHAN"] = nchan // mchan  # New number of channels
    subint_hdr["NSBLK"] = nsamp // msamp  # New number of time samples per subint
    subint_hdr["TBIN"] = np.round(tsamp * msamp / rawtsamp) * rawtsamp
    subint_hdr["CHAN_BW"] = chanbw * mchan  # New channel bandwidth
    subint_hdr["NBITS"] = -32  # Set NBITS to -32 to indicate 32-bit float

    # Create new output PSRFITS HDU list
    new_hdulist = fits.HDUList([fits.PrimaryHDU(header=fits_hdr)])

    # Define new data structure for SUBINT table with 32-bit float data
    new_cols = []
    for col in hdu['SUBINT'].columns:
        if col.name == 'DATA':
            new_cols.append(fits.Column(name='DATA', format=f'{(nchan // mchan) * (nsamp // msamp)}E',
                                        unit=col.unit, dim=f'({nchan // mchan},{nsamp // msamp})'))
        elif col.name == 'DAT_FREQ':  # Update frequency array
            new_cols.append(fits.Column(name='DAT_FREQ', format=f'{nchan // mchan}E'))
        elif col.name in ['DAT_SCL', 'DAT_OFFS', 'DAT_WTS']:  # Adjust scales, offsets, and weights
            new_cols.append(fits.Column(name=col.name, format=f'{nchan // mchan}E'))
        else:
            new_cols.append(col)

    new_coldefs = fits.ColDefs(new_cols)
    new_subint_hdu = fits.BinTableHDU.from_columns(new_coldefs, header=subint_hdr)

    # Output array for processed data
    outdata = np.zeros((nchan // mchan, nsamp * nsub // msamp), dtype="float32")

    # Extract and downsample frequency array
    orig_freq = hdu['SUBINT'].data['DAT_FREQ'][0]  # Take from first subint
    new_freq = np.mean(orig_freq.reshape(-1, mchan), axis=1)

    # Process data for each subint
    subint_iter = range(nsub)
    if not in_slurm:
        subint_iter = tqdm(subint_iter)

    for isub in subint_iter:
        # Extract data, offsets, scales, and weights
        data = hdu['SUBINT'].data[isub]['DATA']
        offsets = hdu['SUBINT'].data[isub]['DAT_OFFS']
        scales = hdu['SUBINT'].data[isub]['DAT_SCL']
        weights = hdu['SUBINT'].data[isub]['DAT_WTS']

        # Unpack 2-bit to 8-bit if necessary
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
        data = np.nanmean(data.reshape(nsamp, -1, mchan), axis=2)
        data = np.nanmean(data.reshape(-1, msamp, nchan // mchan), axis=1)
        
        # Store in output array
        imin = isub * nsamp // msamp
        imax = imin + data.shape[0]
        outdata[:, imin:imax] = data.T.astype("float32")  # Keep as float32

        # Downsample weights, scales, and offsets
        new_weights = np.ones(nchan // mchan, dtype="float32")  # Set all weights to 1
        new_scales = np.ones_like(new_weights, dtype="float32")  # Set to 1 for unscaled float data
        new_offsets = np.zeros_like(new_weights, dtype="float32")  # Set to 0 for unscaled float data

        # Store downsampled values in SUBINT table
        new_subint_hdu.data[isub]['DATA'] = outdata[:, imin:imax].T.reshape(nsamp // msamp, nchan // mchan)
        new_subint_hdu.data[isub]['DAT_WTS'] = new_weights
        new_subint_hdu.data[isub]['DAT_SCL'] = new_scales
        new_subint_hdu.data[isub]['DAT_OFFS'] = new_offsets
        new_subint_hdu.data[isub]['DAT_FREQ'] = new_freq  # Update downsampled frequency array

    # Append new SUBINT table to HDU list
    new_hdulist.append(new_subint_hdu)

    # Save new PSRFITS file with 32-bit floating point data
    new_hdulist.writeto(outfname, overwrite=True)

    print(f"Saved downsampled PSRFITS file to {outfname} with 32-bit float data.")