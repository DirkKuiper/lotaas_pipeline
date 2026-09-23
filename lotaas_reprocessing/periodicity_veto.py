"""A period found in many beams of one observation is RFI, not a pulsar.

A pulsar lies in one tied-array beam or a few neighbours, at one DM. Periodic
RFI reaches most beams of an observation, in every SAP, at whatever trial DM
happened to hold it, because it is narrow in frequency and dedispersion barely
moves it. In L559955 SAP000 the same 5.93, 11.66, 4.70 and 3.38 s peaks filled
the fold shortlist of 40-64 of its 73 beams, so a real signal in those beams
would never have been folded.

This runs after every beam of a batch has been searched and sifted, and before
any is folded. For each sifted-best peak it counts the other beams of the same
observation with a sifted-best peak within `veto_bins` Fourier bins. A peak is
vetoed when that count is both large (at least `veto_beams` beams, or two SAPs)
and far above what the local density of peaks gives by chance: red noise
crowds the lowest frequencies of a bad observation with peaks, and those must
not veto a long-period pulsar by coincidence. A set of matches that all sit at
one DM above 2 is a bright pulsar seen in neighbouring beams and is kept.

Standard library only: the runner calls this on the host, outside the search
container.
"""
import bisect
import json
import math
import os
from pathlib import Path
import re

ITEM = re.compile(r'(L\d+)_SAP(\d+)_BEAM(\d+)')
# Peaks within this many windows either side set the chance expectation.
NEIGHBOURHOOD = 20


def best_peaks(directory):
    """(index in periodicity_candidates.jsonl, row) of the sifted-best peaks of one beam."""
    rows = []
    path = Path(directory) / 'periodicity_candidates.jsonl'
    for index, line in enumerate(path.read_text().splitlines()):
        if line.strip():
            row = json.loads(line)
            if row.get('is_sifted_best'):
                rows.append((index, row))
    return rows


def dm_consistent(dms):
    low, high = min(dms), max(dms)
    return low >= 2 and high - low <= max(2.0, 0.1 * high)


def decide(peaks, veto_bins=1.1, veto_beams=4):
    """Veto decisions for the peaks of one observation.

    `peaks` holds dicts with frequency_hz, frequency_resolution_hz, dm, sap,
    beam and key. Returns {key: evidence} for the vetoed ones.
    """
    peaks = sorted(peaks, key=lambda p: p['frequency_hz'])
    frequencies = [p['frequency_hz'] for p in peaks]
    vetoed = {}
    for peak in peaks:
        f = peak['frequency_hz']
        tolerance = veto_bins * peak['frequency_resolution_hz']
        lo, hi = bisect.bisect_left(frequencies, f - tolerance), bisect.bisect_right(frequencies, f + tolerance)
        home = (peak['sap'], peak['beam'])
        matches = [p for p in peaks[lo:hi] if (p['sap'], p['beam']) != home]
        others = {(p['sap'], p['beam']) for p in matches}
        if not others:
            continue
        saps = {peak['sap']} | {p['sap'] for p in matches}
        # Chance: other beams' peaks per unit frequency near this one, outside
        # the window, times the window's width. Peaks, not distinct beams: a
        # beam count saturates at the number of beams and would veto noise.
        reach = NEIGHBOURHOOD * tolerance
        wide_lo = bisect.bisect_left(frequencies, f - reach)
        wide_hi = bisect.bisect_right(frequencies, f + reach)
        around = sum(1 for p in peaks[wide_lo:lo] + peaks[hi:wide_hi] if (p['sap'], p['beam']) != home)
        spanned = frequencies[wide_hi - 1] - frequencies[wide_lo] if wide_hi - wide_lo > 1 else 0.0
        width = max(min(2 * reach, spanned) - 2 * tolerance, 2 * tolerance)
        expected = around * (2 * tolerance) / width
        floor = 3 * expected + 1
        many_beams = len(others) + 1 >= veto_beams and len(others) >= floor
        two_saps = len(saps) >= 2 and len(others) >= max(1, floor)
        if not (many_beams or two_saps):
            continue
        dms = [peak['dm']] + [p['dm'] for p in matches]
        if dm_consistent(dms):
            continue
        vetoed[peak['key']] = {'frequency_hz': f, 'dm': peak['dm'], 'beams': len(others) + 1,
                               'saps': len(saps), 'expected_beams': round(expected, 3),
                               'dm_min': min(dms), 'dm_max': max(dms)}
    return vetoed


def atomic_json(path, value):
    path = Path(path)
    partial = path.with_name(path.name + '.partial')
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(partial, path)


def apply(directories, veto_bins=1.1, veto_beams=4, write=None):
    """Decide the veto for a batch and write periodicity_veto.json beside each beam's peaks.

    `directories` are beam output directories whose periodicity search is
    complete; beams are grouped by observation from their names. `write`
    limits which directories get a file (those not yet folded), while every
    directory still counts as evidence. Returns {directory: vetoed count}.
    """
    groups = {}
    for directory in map(Path, directories):
        match = ITEM.search(directory.parent.name) or ITEM.search(str(directory))
        if not match:
            continue
        groups.setdefault(match[1], []).append((directory, int(match[2]), int(match[3])))
    written = {}
    targets = set(map(Path, write)) if write is not None else None
    for observation, members in groups.items():
        peaks = []
        for directory, sap, beam in members:
            for index, row in best_peaks(directory):
                peaks.append({'frequency_hz': row['frequency_hz'],
                              'frequency_resolution_hz': row['frequency_resolution_hz'],
                              'dm': row['dm'], 'sap': sap, 'beam': beam, 'key': (str(directory), index)})
        vetoed = decide(peaks, veto_bins, veto_beams)
        rule = {'veto_bins': veto_bins, 'veto_beams': veto_beams, 'neighbourhood_windows': NEIGHBOURHOOD,
                'chance_floor': '3 x expected + 1 other beams', 'dm_consistent': 'min >= 2 and span <= max(2, 10%)'}
        for directory, sap, beam in members:
            if targets is not None and directory not in targets:
                continue
            mine = sorted(((key[1], evidence) for key, evidence in vetoed.items() if key[0] == str(directory)))
            atomic_json(directory / 'periodicity_veto.json', {
                'schema': 1, 'observation': observation, 'compared_beams': len(members),
                'compared_peaks': len(peaks), 'rule': rule,
                'vetoed_indices': [index for index, _ in mine],
                'vetoed': [dict(evidence, index=index) for index, evidence in mine]})
            written[str(directory)] = len(mine)
    return written


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--veto-bins', type=float, default=1.1)
    parser.add_argument('--veto-beams', type=int, default=4)
    arguments = parser.parse_args()
    print(json.dumps(apply(arguments.directories, arguments.veto_bins, arguments.veto_beams), indent=2))
