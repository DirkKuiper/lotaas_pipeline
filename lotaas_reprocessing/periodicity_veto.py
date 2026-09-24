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
not veto a long-period pulsar by coincidence.

A bright pulsar is seen in neighbouring beams too, so two things keep it:
half its family and the judged peak sit at one DM above 2, and its best peak
stands far above the same period in the other beams. Sidelobe detections
scatter by a few units of DM and far sidelobes fold it at unrelated DMs
(B2016+28 at DM 14.2 had members up to DM 995), so the old test, every member
within max(2, 10%), vetoed J0323+3944 in its three best beams (DM 23.6-28.6
across 51 beams).

Standard library only: the runner calls this on the host, outside the search
container.
"""
import bisect
import json
import os
from pathlib import Path
import re
import statistics

ITEM = re.compile(r'(L\d+)_SAP(\d+)_BEAM(\d+)')
# Peaks within this many windows either side set the chance expectation.
NEIGHBOURHOOD = 20
# A family at one DM: this share of its members, and at least two, within
# max(1.5, 5%) of one DM. A periodic detection fixes DM to about its pulse
# width over the 0.11 s sweep per unit DM across the band, well under one unit;
# sidelobes of J0323+3944 scattered 2.5 either way. A wider window at low DM
# swallows interference, which scatters over DM 0-6. Half, not all: far
# sidelobes are narrowband and fold a bright pulsar at any DM, and 40 of the
# 113 beams holding B2016+28 in L606924 SAP002 had it at DM 214-995.
DM_AGREEMENT = 1 / 2
DM_TOLERANCE = (1.5, 0.05)
# Interference peaks at DM ~0 in most beams. A family with this share below
# DM 1 is not a pulsar, even when the peak judged sits just above DM 2.
NEAR_ZERO = (1.0, 1 / 4)
# A peak this many times the median statistic of the same period in the other
# beams is where the signal is, not interference spread over the array.
DOMINANCE = 5.0


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


def dm_consistent(dms, home=None):
    """Most of a family at one DM above 2, and the judged peak at it: a pulsar in neighbouring beams.

    `dms` holds the DMs of the whole family, the judged peak's included; `home`
    is that peak's own DM. The centre is the member DM with the most of the
    family within tolerance of it, so strays folded far away (a third of
    B2016+28's family) move it no more than they move the pulsar, a sidelobe
    at the edge of the spread is judged like the rest, and a family spread
    evenly over DM never gathers half of itself anywhere.
    """
    if not dms or sum(1 for dm in dms if dm < NEAR_ZERO[0]) >= NEAR_ZERO[1] * len(dms):
        return False
    ordered = sorted(dms)

    def gathered(centre):
        tolerance = max(DM_TOLERANCE[0], DM_TOLERANCE[1] * centre)
        count = bisect.bisect_right(ordered, centre + tolerance) - bisect.bisect_left(ordered, centre - tolerance)
        return count, tolerance

    middle = statistics.median(ordered)
    centre = max(ordered, key=lambda c: (gathered(c)[0], -abs(c - middle)))
    count, tolerance = gathered(centre)
    if centre < 2 or (home is not None and abs(home - centre) > tolerance):
        return False
    return count >= max(2, DM_AGREEMENT * len(dms))


def dominant(statistic, others):
    """The peak holds the signal: its statistic far above the median of the same period elsewhere."""
    others = [s for s in others if s is not None]
    if statistic is None or not others:
        return False
    return statistic >= DOMINANCE * statistics.median(others)


def chance_count(peaks, frequencies, f, tolerance, home):
    """Other beams' peaks a window of this width holds by chance near f.

    The median over the neighbouring windows, not their mean. Red noise crowds
    every window near the lowest frequencies alike and still raises it, but a
    comb of RFI lines a few bins apart does not: in L603686 the 3.35 s line ten
    bins from the 3.38 s one lifted the mean to 15 peaks per window and the
    floor to 46 beams, so the 3.38 s line in 42 beams was never vetoed. Only
    windows inside the span that holds peaks count; below the lowest peak every
    window is empty by construction.
    """
    width = 2 * tolerance
    low, high = frequencies[0], frequencies[-1]
    counts = []
    for k in range(1, 4 * NEIGHBOURHOOD + 1):
        for centre in (f - k * width, f + k * width):
            if centre - tolerance < low or centre + tolerance > high:
                continue
            lo = bisect.bisect_left(frequencies, centre - tolerance)
            hi = bisect.bisect_right(frequencies, centre + tolerance)
            counts.append(sum(1 for p in peaks[lo:hi] if (p['sap'], p['beam']) != home))
        if len(counts) >= 2 * NEIGHBOURHOOD:
            break
    if len(counts) < 8:
        return None
    return statistics.median(counts)


def decide(peaks, veto_bins=1.1, veto_beams=4):
    """Veto decisions for the peaks of one observation.

    `peaks` holds dicts with frequency_hz, frequency_resolution_hz, dm, sap,
    beam, key and optionally statistic. Returns {key: evidence} for the vetoed ones.
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
        expected = chance_count(peaks, frequencies, f, tolerance, home)
        if expected is None:
            # Too few windows around (an edge of the populated band): the old
            # estimate, other beams' peaks per unit frequency near this one.
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
        if dm_consistent(dms, home=peak['dm']):
            continue
        if peak['dm'] >= 2 and dominant(peak.get('statistic'), [p.get('statistic') for p in matches]):
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
                              'dm': row['dm'], 'sap': sap, 'beam': beam, 'key': (str(directory), index),
                              'statistic': row.get('statistic')})
        vetoed = decide(peaks, veto_bins, veto_beams)
        rule = {'veto_bins': veto_bins, 'veto_beams': veto_beams, 'neighbourhood_windows': NEIGHBOURHOOD,
                'chance_floor': '3 x median other-beam peaks per neighbouring window + 1',
                'dm_consistent': f'the peak and {DM_AGREEMENT:.2f} of the family within max({DM_TOLERANCE[0]:g}, '
                                 f'{DM_TOLERANCE[1]:.0%}) of one DM >= 2, under '
                                 f'{NEAR_ZERO[1]:.2f} of it below DM {NEAR_ZERO[0]:g}',
                'dominant': f'peak DM >= 2 and statistic >= {DOMINANCE:g} x median of the other beams'}
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
