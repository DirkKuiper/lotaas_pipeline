#!/usr/bin/env python3
import os
import sys
import glob
import numpy as np
from astropy.io import fits
from astropy.time import Time
from lotaas_reprocessing import filterbank
import re

def extract_beam_id(filename):
    """Extracts beam number from a PSRFITS filename."""
    match = re.search(r"_BEAM(\d{1,3})|_B(\d{1,3})", filename)
    if match:
        return int(match.group(1) or match.group(2))  # Return the first matched number
    else:
        raise ValueError(f"Could not extract beam number from filename: {filename}")

def read_psrfits_data(psrfits_file):
    """Read data from a 32-bit PSRFITS file and return it in filterbank format."""
    print(f"Reading {psrfits_file}")

    hdu = fits.open(psrfits_file, mmap=True)
    fits_hdr = hdu[0].header
    subint_hdr = hdu["SUBINT"].header

    # Extract parameters
    nchan = subint_hdr["NCHAN"]
    nsamp = subint_hdr["NSBLK"]
    nsub = subint_hdr["NAXIS2"]
    tsamp = subint_hdr["TBIN"]
    obsfreq = fits_hdr["OBSFREQ"]
    chanbw = subint_hdr["CHAN_BW"]

    # Ensure frequency accuracy
    subbandbw = 0.1953125
    chanbw = subbandbw / np.round(subbandbw / np.abs(chanbw))

    # Allocate output array
    outdata = np.zeros((nchan, nsamp * nsub), dtype="float32")

    # Convert data in-memory
    for isub in range(nsub):
        data = hdu["SUBINT"].data[isub]["DATA"].astype("float32")
        data = data.reshape(-1, nchan)
        imin, imax = isub * nsamp, (isub + 1) * nsamp
        outdata[:, imin:imax] = data.T  # Transpose to match filterbank format

    # Construct filterbank header (for writing the final flatfielded file)
    fil_header = {
        "telescope_id": 11, "machine_id": -1, "data_type": 1, "rawdatafile": "",
        "source_name": fits_hdr["SRC_NAME"], "barycentric": 0, "pulsarcentric": 0,
        "az_start": 0, "za_start": 0, "src_raj": float(fits_hdr["RA"].replace(":", "")),
        "src_dej": float(fits_hdr["DEC"].replace(":", "")),
        "tstart": Time(fits_hdr["DATE-OBS"], format="isot", scale="utc").mjd,
        "tsamp": tsamp, "foff": -chanbw,
        "fch1": obsfreq - np.abs(chanbw) * nchan / 2 + chanbw * nchan,
        "nchans": nchan, "nifs": 1, "nbits": 32
    }

    return outdata, fil_header

def compute_flatfield(obsid, psrfits_files):
    """Compute the mean beam from central beams, used for flatfielding."""
    print("Computing flatfield...")

    central_beams = [i for i in range(13, 74)]
    path = os.path.dirname(psrfits_files[0])

    mean_data = None
    count = 0

    for psrfits_file in psrfits_files:
        beam_id = extract_beam_id(psrfits_file)
        if beam_id in central_beams:
            print(f"Processing beam {beam_id} for flatfield computation...")
            data, _ = read_psrfits_data(psrfits_file)

            if mean_data is None:
                mean_data = np.zeros_like(data)

            mean_data += data
            count += 1

    if count == 0:
        raise ValueError("No central beams found for flatfield computation!")

    return mean_data / count

def apply_flatfield(obsid, psrfits_files, mean_data):
    """Apply flatfielding to all beams and save only the final flatfielded filterbanks."""
    print("Applying flatfielding...")

    for psrfits_file in psrfits_files:
        beam_id = extract_beam_id(psrfits_file)
        print(f"Flatfielding beam {beam_id}")

        # Read PSRFITS data (on the fly)
        data, fil_header = read_psrfits_data(psrfits_file)

        # Apply flatfield correction
        data /= mean_data

        # Construct output filename
        outfname = psrfits_file.replace(".fits", "_ff.fil")

        print(f"Shape of mean_data: {mean_data.shape}")  
        print(f"Shape of data (before applying flatfield): {data.shape}")
        print(f"Filterbank header channels: {fil_header['nchans']}")

        # Save only the flatfielded filterbank file
        outfil = filterbank.create_filterbank_file(outfname, fil_header, nbits=32)
        outfil.append_spectra(np.flipud(data).T)
        outfil.close()

        print(f"Saved flatfielded file: {outfname}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_psrfits.py <SAP directory>")
        sys.exit(1)

    sap_dir = sys.argv[1]
    obsid = os.path.basename(sap_dir)  # Extract obsid from the path

    # Find all 32-bit PSRFITS files in the SAP directory
    psrfits_files = sorted(glob.glob(os.path.join(sap_dir, "B*", "*_32bit.fits")))

    if not psrfits_files:
        print("No 32-bit PSRFITS files found.")
        sys.exit(1)

    # Compute flatfield mean beam
    mean_data = compute_flatfield(obsid, psrfits_files)

    # Apply flatfielding and save only final filterbanks
    apply_flatfield(obsid, psrfits_files, mean_data)