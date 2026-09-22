#!/usr/bin/env python3
"""Flatfield beams with bounded memory and explicit central-beam completeness."""
import argparse
import glob
import os
import re
import numpy as np
from lotaas_reprocessing import filterbank


def extract_beam_id(filename):
    match = re.search(r'_(?:BEAM|B)(\d{1,3})(?:_|\.)', os.path.basename(filename))
    if not match:
        raise ValueError(f'Cannot extract beam number: {filename}')
    return int(match.group(1))


def read_filterbank_data(filename):
    fb = filterbank.FilterbankFile(filename)
    try:
        return np.flipud(fb.get_spectra(0, fb.nspec).T), fb.header
    finally:
        fb.close()


def compute_flatfield(files):
    central = [f for f in files if 13 <= extract_beam_id(f) <= 73]
    if not central:
        raise ValueError('No central beams found for flatfielding')
    metadata = []
    reference = None
    for path in central:
        fb = filterbank.FilterbankFile(path)
        signature = (fb.nchans, fb.tsamp, fb.tstart, fb.fch1, fb.foff)
        if reference is not None and signature != reference:
            raise ValueError('Central beam channel/time grids differ')
        reference = signature
        metadata.append(fb.nspec)
        fb.close()
    mean = np.zeros((reference[0], max(metadata)), dtype=np.float64)
    # One beam at a time, instead of retaining all 61 full beam arrays.
    for path in central:
        data, _ = read_filterbank_data(path)
        mean[:, :data.shape[1]] += data
        if data.shape[1] < mean.shape[1]:
            mean[:, data.shape[1]:] += data.mean(axis=1, keepdims=True)
    mean /= len(central)
    if not np.isfinite(mean).all() or np.any(mean == 0):
        raise ValueError('Flatfield contains zero/nonfinite values')
    return mean


def apply_flatfield(files, mean):
    for path in files:
        data, header = read_filterbank_data(path)
        # Preserve the observed duration; do not fabricate padded science samples.
        if data.shape[1] > mean.shape[1]:
            raise ValueError('Beam extends beyond flatfield time grid')
        data /= mean[:, :data.shape[1]]
        output = path[:-4] + '_ff.fil'
        fb = filterbank.create_filterbank_file(output + '.partial', header, nbits=32)
        try:
            fb.append_spectra(np.flipud(data).T)
        finally:
            fb.close()
        os.replace(output + '.partial', output)
        print('Flatfielded:', output, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sap_directory')
    p.add_argument('--allow-partial', action='store_true', help='Pilot only: permit fewer than 61 central beams')
    a = p.parse_args()
    files = sorted(glob.glob(os.path.join(a.sap_directory, 'B*', '*_32bit.fil')))
    found = {extract_beam_id(f) for f in files if 13 <= extract_beam_id(f) <= 73}
    missing = set(range(13, 74)) - found
    if missing and not a.allow_partial:
        raise ValueError(f'Missing {len(missing)} central beams. Use --allow-partial only for a pilot.')
    if missing:
        print(f'PILOT: flatfield uses only {len(found)} of 61 central beams', flush=True)
    apply_flatfield(files, compute_flatfield(files))


if __name__ == '__main__':
    main()
