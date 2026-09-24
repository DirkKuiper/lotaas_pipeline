#!/usr/bin/env python3
"""Flatfield beams with bounded memory and explicit central-beam completeness.

With --level-rows, each flatfielded channel is also levelled per subintegration
of the 2-bit data the beam was made from: early-cycle (SPIDER) beams were
unpacked from psrfits_subband's 2-bit samples 64x too large with the offsets
added once, so each row's level is set by the requantiser (row mean + 78.75
sigma_row) instead of the sky, and steps at every row edge (euroflash.spider).
'auto' finds the row length from the central beams' mean, where every beam's
steps coincide.
"""
import argparse
import glob
import json
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


ROW_LENGTHS = (32, 512)   # samples of 7.864 ms per 2-bit row: 2012-14 and 2014-15 early-cycle data


def edge_excess(series, row):
    """Median jump into each row's first sample over the median jump anywhere."""
    jumps = np.abs(np.diff(np.asarray(series, dtype=np.float64)))
    n = (jumps.size // row) * row
    typical = np.median(jumps)
    if n < 4 * row or typical <= 0:
        return 1.0
    return float(np.median(jumps[:n].reshape(-1, row)[:, row - 1]) / typical)


def detect_row_length(mean):
    """(row length or 0, {row: excess}) from the central-beam mean (channels x samples).

    Each beam's steps come from its own requantiser statistics, so the mean of
    61 beams dilutes them: measured 49x at 512 for 2015 rows of 512, but only
    2.0x at both 32 and 512 for 2013 rows of 32. Rows of 32 step at every
    512th sample too; rows of 512 leave the 32-sample grid flat. Without steps
    the excess is 1.00 +- 0.02 (a median over thousands of rows).
    """
    series = mean.sum(axis=0)
    excess = {row: round(edge_excess(series, row), 2) for row in ROW_LENGTHS}
    if excess[512] >= 2.0 and excess[512] >= 3 * excess[32]:
        return 512, excess
    if excess[32] >= 1.5:
        return 32, excess
    return 0, excess


def level_rows(data, row):
    """Replace each channel's mean in every row of `row` samples by its median row mean (in place)."""
    n = (data.shape[1] // row) * row
    if row <= 0 or n == 0:
        return data
    blocks = data[:, :n].reshape(data.shape[0], -1, row)
    means = blocks.mean(axis=2, dtype=np.float64)
    level = np.median(means, axis=1, keepdims=True)
    blocks -= (means - level)[:, :, None].astype(data.dtype)
    if n < data.shape[1]:
        tail = data[:, n:]
        tail -= (tail.mean(axis=1, dtype=np.float64, keepdims=True) - level).astype(data.dtype)
    return data


def apply_flatfield(files, mean, row=0):
    for path in files:
        data, header = read_filterbank_data(path)
        # Preserve the observed duration; do not fabricate padded science samples.
        if data.shape[1] > mean.shape[1]:
            raise ValueError('Beam extends beyond flatfield time grid')
        data /= mean[:, :data.shape[1]]
        if row:
            level_rows(data, row)
        output = path[:-4] + '_ff.fil'
        fb = filterbank.create_filterbank_file(output + '.partial', header, nbits=32)
        try:
            fb.append_spectra(np.flipud(data).T)
        finally:
            fb.close()
        os.replace(output + '.partial', output)
        print('Flatfielded:', output, flush=True)


def save_mean(mean, path):
    """Keep a SAP's flatfield for its beams that arrive after the rest were searched."""
    partial = path + '.partial.npy'
    np.save(partial, mean.astype(np.float32))
    os.replace(partial, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sap_directory')
    p.add_argument('--allow-partial', action='store_true',
                   help='Permit fewer than 61 central beams: pilots, SAPs whose archive lacks a few, and SAPs '
                        'searched before their last beams arrived')
    p.add_argument('--save-mean', help='Also write the flatfield (the central beams\' mean, float32 .npy) here')
    p.add_argument('--mean', help='Flatfield the beams present with this saved mean instead of their own: the '
                                  'beams of a SAP that arrived after the rest were flatfielded and removed')
    p.add_argument('--level-rows', default='0',
                   help="Level each channel per 2-bit row of this many samples after flatfielding; 'auto' detects "
                        "32 or 512 (or none) from the central beams and records it in row-levelling.json")
    a = p.parse_args()
    files = sorted(glob.glob(os.path.join(a.sap_directory, 'B*', '*_32bit.fil')))
    record = os.path.join(a.sap_directory, 'row-levelling.json')
    if a.mean:
        if not files:
            raise ValueError(f'No unflattened beams in {a.sap_directory}')
        row = 0
        if a.level_rows == 'auto':
            # Late beams are levelled as the rest of their SAP was.
            row = json.load(open(record))['row'] if os.path.exists(record) else 0
        elif a.level_rows:
            row = int(a.level_rows)
        apply_flatfield(files, np.load(a.mean, mmap_mode='r'), row)
        return
    found = {extract_beam_id(f) for f in files if 13 <= extract_beam_id(f) <= 73}
    missing = set(range(13, 74)) - found
    if missing and not a.allow_partial:
        raise ValueError(f'Missing {len(missing)} central beams; --allow-partial flatfields without them')
    if missing:
        print(f'PARTIAL: flatfield uses {len(found)} of 61 central beams', flush=True)
    mean = compute_flatfield(files)
    if a.save_mean:
        save_mean(mean, a.save_mean)
    row = 0
    if a.level_rows == 'auto':
        row, excess = detect_row_length(mean)
        with open(record + '.partial', 'w') as f:
            json.dump({'row': row, 'edge_excess': {str(k): v for k, v in excess.items()}}, f)
        os.replace(record + '.partial', record)
        print(f'Row levelling: {row or "none"} (edge-jump excess {excess})', flush=True)
    elif a.level_rows:
        row = int(a.level_rows)
    apply_flatfield(files, mean, row)


if __name__ == '__main__':
    main()
