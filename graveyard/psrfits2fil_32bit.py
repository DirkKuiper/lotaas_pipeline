#!/usr/bin/env python3
import os
import argparse
import numpy as np
import tqdm
from astropy.time import Time
from astropy.io import fits
from lotaas_reprocessing import filterbank  # Ensure this library is available

# List of required filterbank headers
fil_header_keys = [
    "telescope_id", "machine_id", "data_type", "rawdatafile", "source_name",
    "barycentric", "pulsarcentric", "az_start", "za_start", "src_raj",
    "src_dej", "tstart", "tsamp", "nbits", "fch1", "foff", "nchans", "nifs"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert 32-bit PSRFITS to 32-bit SIGPROC Filterbank")
    parser.add_argument("-o", "--output", help="Output filterbank file [FIL]", metavar="FILE")
    parser.add_argument("input", help="Input PSRFITS file", metavar="FILE")
    args = parser.parse_args()

    # Get output filename
    outfname = args.output if args.output else args.input.replace(".fits", ".fil")

    # Open PSRFITS file
    hdu = fits.open(args.input, mmap=True)
    fits_hdr = hdu[0].header
    subint_hdr = hdu["SUBINT"].header

    # Extract essential parameters
    nchan = subint_hdr["NCHAN"]
    nsamp = subint_hdr["NSBLK"]
    nsub = subint_hdr["NAXIS2"]
    tsamp = subint_hdr["TBIN"]
    obsfreq = fits_hdr["OBSFREQ"]
    chanbw = subint_hdr['CHAN_BW']
    # Fix frequency accuracy
    subbandbw = 0.1953125
    chanbw = subbandbw / np.round(subbandbw / np.abs(chanbw))

    # Allocate output array for 32-bit filterbank
    outdata = np.zeros((nchan, nsamp * nsub), dtype="float32")

    # Process and convert data
    for isub in tqdm.tqdm(range(nsub)):
        # Read PSRFITS data as float32 (direct, no scaling)
        data = hdu["SUBINT"].data[isub]["DATA"].astype("float32")

        # Reshape to (nsamp, nchan)
        data = data.reshape(-1, nchan)  

        # Store converted data in output array
        imin = isub * nsamp
        imax = imin + nsamp
        outdata[:, imin:imax] = data.T  # Transpose to match filterbank format

    # Construct filterbank header
    fil_header = dict.fromkeys(fil_header_keys, None)
    fil_header["telescope_id"] = 11  # Change as needed
    fil_header["machine_id"] = -1
    fil_header["data_type"] = 1
    fil_header["rawdatafile"] = ""
    fil_header["source_name"] = fits_hdr["SRC_NAME"]
    fil_header["barycentric"] = 0
    fil_header["pulsarcentric"] = 0
    fil_header["az_start"] = 0
    fil_header["za_start"] = 0
    fil_header["src_raj"] = float(fits_hdr["RA"].replace(":", ""))
    fil_header["src_dej"] = float(fits_hdr["DEC"].replace(":", ""))
    fil_header["tstart"] = Time(fits_hdr["DATE-OBS"], format="isot", scale="utc").mjd
    fil_header["tsamp"] = tsamp
    fil_header["foff"] = -chanbw
    fil_header["fch1"] = obsfreq - np.abs(chanbw) * nchan / 2 + chanbw * nchan
    fil_header["nchans"] = nchan
    fil_header["nifs"] = 1
    fil_header["nbits"] = 32  # Ensuring 32-bit output

    # Write 32-bit filterbank file
    outfil = filterbank.create_filterbank_file(outfname, fil_header, nbits=32)
    outfil.append_spectra(np.flipud(outdata).T)  # Flip to match expected filterbank format
    outfil.close()

    print(f"Successfully converted {args.input} to 32-bit {outfname}")