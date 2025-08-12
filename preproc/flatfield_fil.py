#!/usr/bin/env python3
import os
import sys
import glob
import numpy as np
import re
from lotaas_reprocessing import filterbank

def extract_beam_id(filename):
    """Extract beam number from a filterbank filename."""
    match = re.search(r"_BEAM(\d{1,3})|_B(\d{1,3})", filename)
    if match:
        return int(match.group(1) or match.group(2))
    else:
        raise ValueError(f"Could not extract beam number from filename: {filename}")

def read_filterbank_data(fil_file):
    """Read a .fil file and return data and header."""
    print(f"Reading filterbank: {fil_file}")
    fb = filterbank.FilterbankFile(fil_file, mode="read")
    data = np.flipud(fb.get_spectra(0, fb.nspec).T)
    hdr = fb.header
    fb.close()
    return data, hdr

def compute_flatfield(fil_files):
    """Compute the mean beam from central beams (with padding)."""
    central_beams = list(range(13, 74))
    beam_data_list = []
    nspecs = []

    # First, read all valid central beam data
    for fil_file in fil_files:
        beam_id = extract_beam_id(fil_file)
        if beam_id in central_beams:
            print(f"Including beam {beam_id} in mean...")
            data, _ = read_filterbank_data(fil_file)
            beam_data_list.append(data)
            nspecs.append(data.shape[1])

    if not beam_data_list:
        raise ValueError("No central beams found for flatfielding!")

    # Find the max number of time samples
    max_nspec = max(nspecs)
    print(f"Padding all beams to max nspec: {max_nspec}")

    # Pad and sum
    mean_data = np.zeros((beam_data_list[0].shape[0], max_nspec))
    for data in beam_data_list:
        if data.shape[1] < max_nspec:
            pad_width = max_nspec - data.shape[1]
            print(f"Padding beam from {data.shape[1]} to {max_nspec}")
            # Pad using the mean along the time axis
            pad_value = np.mean(data, axis=1, keepdims=True)
            padded = np.hstack([data, np.tile(pad_value, (1, pad_width))])
        else:
            padded = data

        mean_data += padded

    return mean_data / len(beam_data_list)

def apply_flatfield(fil_files, mean_data):
    """Apply flatfielding and save new `_ff.fil` files."""
    nchan, target_nspec = mean_data.shape

    for fil_file in fil_files:
        beam_id = extract_beam_id(fil_file)
        print(f"Flatfielding beam {beam_id}")

        data, header = read_filterbank_data(fil_file)
        if data.shape[1] < target_nspec:
            pad_width = target_nspec - data.shape[1]
            print(f"Padding beam from {data.shape[1]} to {target_nspec}")
            pad_value = np.mean(data, axis=1, keepdims=True)
            data = np.hstack([data, np.tile(pad_value, (1, pad_width))])
        elif data.shape[1] > target_nspec:
            print(f"Truncating beam from {data.shape[1]} to {target_nspec}")
            data = data[:, :target_nspec]

        data /= mean_data  # Apply flatfield

        outfname = fil_file.replace(".fil", "_ff.fil")
        print(f"Saving flatfielded file to: {outfname}")

        fb_out = filterbank.create_filterbank_file(outfname, header, nbits=32)
        fb_out.append_spectra(np.flipud(data).T)  # Transpose back for writing
        fb_out.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python flatfield_fil.py <SAP directory>")
        sys.exit(1)

    sap_dir = sys.argv[1]
    fil_files = sorted(glob.glob(os.path.join(sap_dir, "B*", "*_32bit.fil")))

    if not fil_files:
        print("No .fil files found.")
        sys.exit(1)

    mean_data = compute_flatfield(fil_files)
    apply_flatfield(fil_files, mean_data)