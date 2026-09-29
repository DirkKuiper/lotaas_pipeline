#!/usr/bin/env python3
"""Flatfield beams with bounded memory and explicit central-beam completeness.

With --level-rows, each flatfielded channel is also levelled per subintegration
of the 2-bit data the beam was made from: early-cycle (SPIDER) beams were
unpacked from psrfits_subband's 2-bit samples 64x too large with the offsets
added once, so each row's level is set by the requantiser (row mean + 78.75
sigma_row) instead of the sky, and steps at every row edge (euroflash.spider).
'auto' finds the row length from the central beams' mean, where every beam's
steps coincide.

Cells that hold no sky are left out of the central beams' mean and, once a beam
is flatfielded (and levelled), set to their channel's level: cells exactly zero,
and every cell of a sample where most channels read below DROP_LEVEL of their
level (7 sigma below it in a raw beam). They are the recording's gaps (every
beam zero for 75 s in L168048 and L169695, which stopped their flatfield, and
blocks of single channels) and the early-cycle beams' dropped 2-bit rows (whole
rows at 7% of the level in 0.3-2.3% of L167144 SAP002's rows, beam by beam),
which the search saw as broadband dips and, dedispersed, as curved seams at high
DM. Scattered low cells in single channels (RFI-damaged 2-bit rows) are kept.
"""
import argparse
import glob
import json
import os
import re
import warnings

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


DROP_LEVEL = 0.2       # of a channel's level: below it a cell holds no sky
DROP_SAMPLE = 0.5      # of a sample's channels below DROP_LEVEL: then the whole sample holds none


def lost_cells(data):
    """(channel, time) mask of the cells that hold no sky (see the module's docstring)."""
    level = np.median(data[:, ::64], axis=1, keepdims=True)
    lost = data == 0
    dropped = ((data < DROP_LEVEL * level) | lost).mean(axis=0) > DROP_SAMPLE
    lost[:, dropped] = True
    return lost


def fill_lost(data, lost):
    """Set the lost cells to their channel's level: the median of its kept cells (in place)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        level = np.nanmedian(np.where(lost[:, ::16], np.nan, data[:, ::16]), axis=1)
        fallback = np.nanmedian(level)
    level = np.where(np.isfinite(level), level, fallback if np.isfinite(fallback) else 1.0)
    np.copyto(data, level[:, None].astype(data.dtype), where=lost)
    return data


def compute_flatfield(files, series=None):
    """The central beams' mean over their kept cells; with `series`, each central beam's sum over channels is
    appended to it. A cell lost in every central beam takes its channel's median."""
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
    count = None                                     # beams kept per cell, once any cell is lost
    # One beam at a time, instead of retaining all 61 full beam arrays.
    for number, path in enumerate(central):
        data, _ = read_filterbank_data(path)
        if series is not None:
            series.append(data.sum(axis=0, dtype=np.float64))
        lost = lost_cells(data)
        if lost.any():
            if count is None:
                count = np.full(mean.shape, number, dtype=np.uint8)
            data[lost] = 0
        n = data.shape[1]
        mean[:, :n] += data
        if count is not None:
            count[:, :n] += ~lost
        if n < mean.shape[1]:
            kept = (~lost).sum(axis=1, keepdims=True)
            mean[:, n:] += data.sum(axis=1, keepdims=True) / np.maximum(kept, 1) if lost.any() else \
                data.mean(axis=1, keepdims=True)
            if count is not None:
                count[:, n:] += (kept > 0).astype(np.uint8)
    if count is None:
        mean /= len(central)
    else:
        with np.errstate(all='ignore'):
            mean /= count
        empty = count == 0
        if empty.any():
            mean[empty] = np.nan
            fill_lost(mean, empty)
            print(f'Flatfield: {int(empty.all(axis=0).sum())} samples lost in every central beam', flush=True)
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


def detect_row_length(series):
    """(row length or 0, {row: excess}) from the central beams' sums over channels.

    Each beam steps at its own requantiser's levels, so the steps are judged
    beam by beam (the median over beams), not in their mean, where 61 beams
    dilute them (2013 rows of 32: 2.0x in the mean, 2.5-4x in single beams).
    Without steps the excess is 1.00 +- 0.02 (a median over thousands of rows).
    """
    excess = {row: round(float(np.median([edge_excess(s, row) for s in series])), 2) for row in ROW_LENGTHS}
    # Rows of 512 put a real edge at only 1 in 16 of the 32-sample grid points,
    # which leaves that median at 1.0-1.15; rows of 32 raise both. Measured on
    # eight SAPs, Dec 2012 - Apr 2015: rows of 32 at 1.97-3.5, rows of 512 at
    # 1.64-15.4 (a weakly stepped June 2014 SAP at 1.64), no steps at 1.00.
    if excess[32] >= 1.3:
        return 32, excess
    if excess[512] >= 1.3:
        return 512, excess
    return 0, excess


def level_rows(data, row, lost=None):
    """Replace each channel's mean in every row of `row` samples by its median row mean (in place); with
    `lost`, over the kept cells only (a row with none is left for fill_lost)."""
    n = (data.shape[1] // row) * row
    if row <= 0 or n == 0:
        return data
    blocks = data[:, :n].reshape(data.shape[0], -1, row)
    if lost is None or not lost.any():
        means = blocks.mean(axis=2, dtype=np.float64)
        level = np.median(means, axis=1, keepdims=True)
        blocks -= (means - level)[:, :, None].astype(data.dtype)
        if n < data.shape[1]:
            tail = data[:, n:]
            tail -= (tail.mean(axis=1, dtype=np.float64, keepdims=True) - level).astype(data.dtype)
        return data
    kept = ~lost[:, :n].reshape(blocks.shape)
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        means = np.where(kept, blocks, 0).sum(axis=2, dtype=np.float64) / kept.sum(axis=2)
        level = np.nanmedian(means, axis=1, keepdims=True)
    blocks -= np.nan_to_num(means - level)[:, :, None].astype(data.dtype)
    if n < data.shape[1]:
        tail, keep = data[:, n:], ~lost[:, n:]
        with np.errstate(all='ignore'):
            tail_mean = np.where(keep, tail, 0).sum(axis=1, dtype=np.float64, keepdims=True) / keep.sum(axis=1, keepdims=True)
        tail -= np.nan_to_num(tail_mean - level).astype(data.dtype)
    return data


def apply_flatfield(files, mean, row=0):
    for path in files:
        data, header = read_filterbank_data(path)
        # Preserve the observed duration; do not fabricate padded science samples.
        if data.shape[1] > mean.shape[1]:
            raise ValueError('Beam extends beyond flatfield time grid')
        lost = lost_cells(data)
        data /= mean[:, :data.shape[1]]
        if row:
            level_rows(data, row, lost)
        if lost.any():
            fill_lost(data, lost)
            print(f'Filled {int(lost.sum())} lost cells ({100 * lost.mean():.2f}%) in {os.path.basename(path)}', flush=True)
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
    series = [] if a.level_rows == 'auto' else None
    mean = compute_flatfield(files, series)
    if a.save_mean:
        save_mean(mean, a.save_mean)
    row = 0
    if a.level_rows == 'auto':
        row, excess = detect_row_length(series)
        with open(record + '.partial', 'w') as f:
            json.dump({'row': row, 'edge_excess': {str(k): v for k, v in excess.items()}}, f)
        os.replace(record + '.partial', record)
        print(f'Row levelling: {row or "none"} (edge-jump excess {excess})', flush=True)
    elif a.level_rows:
        row = int(a.level_rows)
    apply_flatfield(files, mean, row)


if __name__ == '__main__':
    main()
