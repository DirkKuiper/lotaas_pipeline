#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
from astropy.io import fits
from astropy.time import Time

# Optional tqdm for local testing
in_slurm = "SLURM_JOB_ID" in os.environ
if not in_slurm:
    from tqdm import tqdm

from lotaas_reprocessing import filterbank 

def parse_ids(filename):
    """Extract OBSID, SAP, and BEAM from filename."""
    parts = os.path.basename(filename.replace(".fits", "")).split("_")
    obsid = parts[0]
    sap = int(parts[1][3:])
    beam = int(parts[2][4:])
    return obsid, sap, beam

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample PSRFITS to 32-bit filterbank")
    parser.add_argument("-f", "--fscrunch", type=int, default=4, help="Average in frequency [default: 4]")
    parser.add_argument("-t", "--tscrunch", type=int, default=16, help="Average in time [default: 16]")
    parser.add_argument("-o", "--output", help="Output filterbank file name")
    parser.add_argument("-d", "--dc", action="store_true", help="Mask DC channel [default: False]")
    parser.add_argument("input", help="Input PSRFITS file")
    args = parser.parse_args()

    mchan, msamp = args.fscrunch, args.tscrunch
    obsid, sap, beam = parse_ids(args.input)

    # Default output
    if args.output is None:
        outfname = f"{obsid}_SAP{sap:03d}_B{beam:03d}_32bit.fil"
    else:
        outfname = args.output

    # Open file and extract headers
    hdu = fits.open(args.input, mode="readonly", memmap=True)
    hdr = hdu[0].header
    subhdr = hdu["SUBINT"].header

    nchan, nsamp, nsub = subhdr["NCHAN"], subhdr["NSBLK"], subhdr["NAXIS2"]
    tsamp, nbit = subhdr["TBIN"], subhdr["NBITS"]
    obsfreq, chanbw = hdr["OBSFREQ"], subhdr["CHAN_BW"]

    # Fix bandwidths
    subbandbw = 0.1953125
    chanbw = subbandbw / np.round(subbandbw / np.abs(chanbw))
    rawtsamp = 5.12e-6

    # New frequency/time info
    new_tsamp = np.round(tsamp * msamp / rawtsamp) * rawtsamp
    new_nchan = nchan // mchan
    new_freq = np.mean(hdu["SUBINT"].data["DAT_FREQ"][0].reshape(-1, mchan), axis=1)

    # Allocate array
    outdata = np.zeros((new_nchan, nsamp * nsub // msamp), dtype="float32")

    subint_iter = range(nsub)
    if not in_slurm:
        subint_iter = tqdm(subint_iter)

    for isub in subint_iter:
        row = hdu["SUBINT"].data[isub]
        data = row["DATA"]
        offsets = row["DAT_OFFS"]
        scales = row["DAT_SCL"]
        weights = row["DAT_WTS"]

        # Unpack
        if nbit == 2:
            spec = np.packbits(np.unpackbits(np.squeeze(data)).reshape(-1, 2), axis=1).reshape(-1, nchan)
        elif nbit == 8:
            spec = data.reshape(-1, nchan)
        else:
            raise ValueError(f"Unsupported bit depth: {nbit}")

        # Scale
        data = (spec * scales + offsets) * weights

        # Mask DC
        if args.dc:
            data[:, 0::16] = np.nan

        # Downsample
        data = np.nanmean(data.reshape(nsamp, -1, mchan), axis=2)
        data = np.nanmean(data.reshape(-1, msamp, new_nchan), axis=1)

        # Store
        imin = isub * nsamp // msamp
        outdata[:, imin:imin + data.shape[0]] = data.T.astype("float32")

    # Build filterbank header
    fil_header = {
        "telescope_id": 11,
        "machine_id": -1,
        "data_type": 1,
        "source_name": hdr["SRC_NAME"],
        "barycentric": 0,
        "pulsarcentric": 0,
        "src_raj": float(hdr["RA"].replace(":", "")),
        "src_dej": float(hdr["DEC"].replace(":", "")),
        "tstart": Time(hdr["DATE-OBS"], format="isot", scale="utc").mjd,
        "tsamp": new_tsamp,
        "foff": -chanbw * mchan,
        "fch1": obsfreq - np.abs(chanbw) * nchan / 2 + chanbw * nchan,
        "nchans": new_nchan,
        "nifs": 1,
        "nbits": 32
    }

    # Write out filterbank
    print(f"Writing filterbank to {outfname}")
    outfil = filterbank.create_filterbank_file(outfname, fil_header, nbits=32)
    outfil.append_spectra(np.flipud(outdata).T)  # Match format
    outfil.close()

    print(f"Done. Output: {outfname}")