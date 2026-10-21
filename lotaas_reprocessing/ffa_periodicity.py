"""The FFA search for long periods, on the slow-pulsar pass's trials (lotaas_reprocessing.ffa, slow_periodicity).

The slow pass's FFT found 4 of 15 slow pulsars injected at 60-200 s (benchmarks/slow-pass-pilot-2026-10-21); the
FFA recovers 75-90% of a pulse train's ideal S/N at 7-230 s (tests/test_ffa.py). It searches the slow pass's
trials (no zero-DM filter, DM 0-100 at 31 ms) before they are pruned, every trial on DM_STEPS' coarser grid:
at these periods the pulses are wide and a DM step smears little. Peaks are sifted across DM and harmonics as
the FFT search's are, vetoed across the batch's beams among themselves, and folded, as ffa_periodicity_*
products: the review page judges the folds with the others.

Settings (ffa, top level; needs slow_periodicity on): enabled, period_min_seconds, period_max_seconds,
bins_min, threshold (boxcar S/N), dm_steps ([low, high, step] in DM), and fold settings over the slow pass's.
"""
import json
import os
import time
from copy import deepcopy
from pathlib import Path

PREFIX = 'ffa_'
DEFAULTS = {
    'enabled': False,
    'period_min_seconds': 4.0,
    'period_max_seconds': 300.0,
    'bins_min': 128,
    'threshold': 7.0,
    # A DM step d smears 0.109 s x d / 2 across 119-151 MHz at most: 27 ms at 0.5, 0.11 s at 2.
    'dm_steps': [[0.0, 10.0, 0.5], [10.0, 40.0, 1.0], [40.0, 100.0, 2.0]],
    # The injected pulsars ranked up to 15th among an interference-heavy beam's peaks.
    'max_folds': 16,
    # 225 s intervals: a fold up to that period covers every phase in each (the review page needs 16).
    'fold_subintegrations': 16,
}
# The requantiser's 24.16 s blocks and 4.03 s rows: their residual level steps fold at simple fractions or
# multiples of them, as the review page knows (web.periodic_quality.on_steps). Such peaks are kept but flagged
# rfi_like, so they fold last.
BLOCK_SECONDS = 3072 * 0.007864319719374176
ROW_SECONDS = 512 * 0.007864319719374176
# Simple fractions of a block (64.4 s = 8/3 of one folded across 30 DM trials of L455526 SAP000 B006), whole
# rows: sparser than the review page's rescue rule, so fewer real periods fold last.
BLOCK_RATIOS = sorted({m / n for n in range(1, 5) for m in range(1, 13)})
ROW_RATIOS = [float(m) for m in range(1, 80)]
STEP_TOLERANCE = 1e-3
# A pulse w seconds wide loses S/N within about w / 0.109 s DM units of its DM (its sweep across 119-151 MHz);
# a period kept at >= UNDISPERSED_FRACTION of its S/N over more than UNDISPERSED_FACTOR times that (and
# UNDISPERSED_MIN_DM) is not dispersed: flagged rfi_like, folded last.
SWEEP_PER_DM = 0.109
UNDISPERSED_FRACTION = 0.6
UNDISPERSED_FACTOR = 3.0
UNDISPERSED_MIN_DM = 5.0


def settings_of(settings):
    result = deepcopy(DEFAULTS)
    result.update((settings or {}).get('ffa') or {})
    return result


def on_steps(period):
    return (any(abs(period / BLOCK_SECONDS / r - 1) <= STEP_TOLERANCE for r in BLOCK_RATIOS)
            or any(abs(period / ROW_SECONDS / r - 1) <= STEP_TOLERANCE for r in ROW_RATIOS))


def undispersed(row, rows, periods=None):
    """Whether a peak's period holds its S/N over a wider span of DM than its pulse width allows.

    periods: the sorted periods of rows (rows in that order), to look only at the peaks near this one."""
    import bisect
    p = row['period_seconds']
    if periods is None:
        rows = sorted(rows, key=lambda q: q['period_seconds'])
        periods = [q['period_seconds'] for q in rows]
    near = rows[bisect.bisect_left(periods, p * (1 - 2e-3)):bisect.bisect_right(periods, p * (1 + 2e-3))]
    same = [q for q in near if q['statistic'] >= UNDISPERSED_FRACTION * row['statistic']]
    span = max((abs(q['dm'] - row['dm']) for q in same), default=0.0)
    width = row['width_bins'] * row['period_seconds'] / row['ffa_bins']
    return span > max(UNDISPERSED_MIN_DM, UNDISPERSED_FACTOR * width / SWEEP_PER_DM)


def selected(specs, steps):
    """Names of the trials on the FFA's DM grid: for each [low, high, step], the trial nearest each grid DM."""
    by_dm = sorted((spec['dm'], name) for name, spec in specs.items())
    chosen = []
    for low, high, step in steps:
        dm = low
        while dm < high - 1e-9:
            best = min(by_dm, key=lambda t: abs(t[0] - dm))
            if abs(best[0] - dm) <= step / 2 and best[1] not in chosen:
                chosen.append(best[1])
            dm += step
    return chosen


def config_of(metadata):
    """The fold configuration of the FFA's folds: the slow pass's, with the FFA's fold settings over it."""
    from .periodicity import resolved_config
    ffa = metadata['ffa']
    config = dict(metadata['slow_periodicity_config'])
    config.update({k: ffa[k] for k in ('max_folds', 'fold_subintegrations') if k in ffa})
    return resolved_config(config)


def search_ffa(trial_dir, output_dir, metadata, config=None):
    """Search the selected trials, sift the peaks; folding (fold_periodicity with PREFIX) waits for the veto.

    metadata: the slow pass's view (slow_periodicity.metadata_view) of the beam's metadata."""
    import numpy as np
    from . import ffa
    from .periodicity import _write_jsonl, atomic_json, sift_bounded, valid_trial
    from .trials import trial_specs
    settings = metadata['ffa']
    config = config or config_of(metadata)
    trial_dir, output_dir = Path(trial_dir), Path(output_dir)
    for name in ('periodicity_search_summary.json', 'periodicity_summary.json'):
        (output_dir / (PREFIX + name)).unlink(missing_ok=True)
    specs = trial_specs(metadata, metadata['periodicity_dm_plan'])
    names = selected(specs, settings['dm_steps'])
    steps = {name: next((s for lo, hi, s in settings['dm_steps'] if lo <= specs[name]['dm'] < hi), 0.0) for name in names}
    started, cpu_started = time.perf_counter(), time.process_time()
    raw_path = output_dir / (PREFIX + 'periodicity_raw_candidates.jsonl')
    partial = raw_path.with_name(raw_path.name + '.partial')
    candidates, coverage = [], []
    with partial.open('w') as raw:
        for name in names:
            spec = specs[name]
            data, polluted = valid_trial(trial_dir / name, metadata, spec['dm'], spec['downsample'])
            dt = float(metadata['tsamp']) * spec['downsample']
            seconds = len(data) * dt
            peaks = ffa.search(np.asarray(data, dtype=np.float64), dt, settings['period_min_seconds'],
                               min(settings['period_max_seconds'], seconds / 3), settings['bins_min'],
                               settings['threshold'])
            rows = []
            for p in peaks:
                rows.append({'frequency_hz': 1.0 / p['period_seconds'], 'period_seconds': p['period_seconds'],
                             'frequency_resolution_hz': 1.0 / seconds, 'statistic': p['snr'], 'floor_excess': p['excess'],
                             'statistic_kind': 'ffa_boxcar_snr', 'harmonic_count': 1, 'width_bins': p['width_bins'],
                             'ffa_bins': p['bins'], 'ffa_downsample': p['downsample'], 'turns': p['turns'],
                             'observation_seconds': seconds, 'effective_sampling_seconds': dt,
                             'rfi_like': on_steps(p['period_seconds']), 'rfi_harmonics': [],
                             'dm': spec['dm'], 'dm_step': steps[name], 'trial_file': name, 'trimmed_samples': polluted})
            for row in rows:
                raw.write(json.dumps(row, sort_keys=True) + '\n')
            candidates += rows
            coverage.append({'dm': spec['dm'], 'valid_samples': len(data), 'peaks': len(rows),
                             'effective_sampling_seconds': dt})
    os.replace(partial, raw_path)
    search_seconds = time.perf_counter() - started
    ordered = sorted(candidates, key=lambda q: q['period_seconds'])
    periods = [q['period_seconds'] for q in ordered]
    for row in candidates:
        row['undispersed'] = undispersed(row, ordered, periods)
        row['rfi_like'] = row['rfi_like'] or row['undispersed']
    sifted, sift_limit = sift_bounded(candidates, config)
    _write_jsonl(output_dir / (PREFIX + 'periodicity_candidates.jsonl'), sifted)
    atomic_json(output_dir / (PREFIX + 'periodicity_coverage.json'), coverage)
    files = [raw_path, output_dir / (PREFIX + 'periodicity_candidates.jsonl'), output_dir / (PREFIX + 'periodicity_coverage.json')]
    summary = {'schema': 'lotaas.ffa.search.v1', 'complete': True, 'algorithm': 'fast_folding_boxcar',
               'configuration': dict(settings, fold=config), 'trials_searched': len(names),
               'raw_candidates': len(candidates), 'sifted_candidates': len(sifted),
               'best_candidates': sum(r['is_sifted_best'] for r in sifted), 'sift_input_limit': sift_limit,
               'rfi_like_candidates': sum(r['rfi_like'] for r in candidates),
               'elapsed_seconds': time.perf_counter() - started, 'cpu_seconds': time.process_time() - cpu_started,
               'stage_seconds': {'search': search_seconds},
               'outputs': {str(p.relative_to(output_dir)): p.stat().st_size for p in files}}
    atomic_json(output_dir / (PREFIX + 'periodicity_search_summary.json'), summary)
    return summary
