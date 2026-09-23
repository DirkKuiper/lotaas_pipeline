"""Running-baseline whitening and per-event merging in the single-pulse search."""
import json

import numpy as np
import pytest

from lotaas_reprocessing.matched_filter import (baseline_window, merge_events, run_matched_filtering,
                                                running_baseline)

DT = 0.007864319719374176


def red_noise(n, seed, slow=1.0, tau_samples=400):
    """White noise plus an AR(1) wander holding `slow` times the white variance."""
    rng = np.random.default_rng(seed)
    white = rng.normal(size=n)
    a = np.exp(-1.0 / tau_samples)
    drive = rng.normal(size=n) * np.sqrt(slow * (1 - a * a))
    wander = np.empty(n)
    wander[0] = 0.0
    for i in range(1, n):
        wander[i] = a * wander[i - 1] + drive[i]
    return (white + wander).astype(np.float32)


def test_baseline_follows_a_slow_drift_and_ignores_a_short_pulse():
    t = np.arange(40000)
    x = (3 * np.sin(2 * np.pi * t / 5000.0)).astype(np.float32)
    x[20000:20010] += 50
    base = running_baseline(x, 512)
    residual = x - base
    assert np.std(residual[1000:19000]) < 0.05  # drift removed
    assert residual[20000:20010].min() > 49  # the pulse keeps its height


def test_a_local_stretch_gets_the_whole_series_values():
    x = red_noise(50000, 3)
    whole = running_baseline(x, 600)
    for lo, hi in ((0, 700), (12345, 16000), (49000, 50000)):
        assert np.allclose(running_baseline(x, 600, lo, hi), whole[lo:hi])


def test_window_scales_with_width_and_has_a_floor():
    assert baseline_window(1, DT, 1, 2.0) == round(2.0 / DT)
    assert baseline_window(100, DT, 1, 2.0) == 6400
    assert baseline_window(3, DT, 32, 2.0) == 192


def test_merging_keeps_one_strongest_crossing_per_event():
    centres = np.array([100, 101, 102, 101.5, 300, 301])
    strengths = np.array([5.1, 6.0, 5.2, 7.5, 5.0, 5.5])
    widths = np.array([1, 1, 1, 4, 1, 1])
    keep = merge_events(centres, strengths, widths)
    assert keep.tolist() == [3, 5]


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_whitening_raises_a_pulse_in_red_noise(tmp_path, seed):
    x = red_noise(200000, seed)
    x[100000:100003] += 5.0  # white-noise S/N ~8.7
    path = tmp_path / 'red.dat'
    x.tofile(path)
    plain = run_matched_filtering(path, DT, 50., max_duration=1.)
    white = run_matched_filtering(path, DT, 50., max_duration=1., baseline_seconds=2.)

    def at_pulse(found):
        times, _, snr, _ = found
        near = np.abs(times - 100001 * DT) < 0.05
        return snr[near].max() if near.any() else 0.0
    assert at_pulse(white) > 1.2 * at_pulse(plain)


def test_whitening_leaves_white_noise_significance_alone(tmp_path):
    x = np.random.default_rng(9).normal(size=300000).astype(np.float32)
    x[150000:150002] += 4.0
    path = tmp_path / 'white.dat'
    x.tofile(path)
    a = run_matched_filtering(path, DT, 50., max_duration=1.)
    b = run_matched_filtering(path, DT, 50., max_duration=1., baseline_seconds=2.)
    near = lambda f: f[2][np.abs(f[0] - 150000.5 * DT) < 0.05].max()
    assert abs(near(b) / near(a) - 1) < 0.05


def test_merged_crossings_give_the_same_clusters(tmp_path):
    from lotaas_reprocessing import cluster
    rng = np.random.default_rng(5)
    rows = {False: [], True: []}
    for dm in np.arange(40, 60, 0.5):
        x = rng.normal(size=60000).astype(np.float32)
        x[30000:30004] += 4.0 * np.exp(-((dm - 50) / 3) ** 2)
        x[10000:10002] += 6.0 * (abs(dm - 44) < 1)
        p = tmp_path / f'DM{dm}.dat'
        x.tofile(p)
        for merge in (False, True):
            t, d, s, w = run_matched_filtering(p, DT, dm, max_duration=1., merge=merge)
            rows[merge].append(np.column_stack([d, s, t, np.rint(t / DT), w]))
    out = {}
    for merge in (False, True):
        cands = tmp_path / f'm{merge}.cands'
        with open(cands, 'w') as f:
            f.write('# DM S/N Time Sample Width\n')
            np.savetxt(f, np.concatenate(rows[merge]), fmt=['%.3f', '%.3f', '%.6f', '%d', '%d'])
        plan = [{'low_dm': 40.0, 'high_dm': 60.0, 'ddm': 0.5, 'downsample': 1}]
        cluster.cluster_candidates(str(cands), str(tmp_path / f'c{merge}.txt'), plan=plan)
        out[merge] = np.loadtxt(tmp_path / f'c{merge}.txt', skiprows=1, ndmin=2, usecols=range(5))
    assert len(np.concatenate(rows[True])) < len(np.concatenate(rows[False])) / 3
    assert np.array_equal(out[True], out[False])


def test_whitened_evidence_uses_the_searched_series(tmp_path):
    from lotaas_reprocessing.single_pulse_quality import measure_clusters
    trials = tmp_path / 'DM_trials'
    trials.mkdir()
    x = red_noise(120000, 11, slow=2.0)
    x[60000:60003] += 3.5
    x.tofile(trials / 'beam_DM50.0.dat')
    (tmp_path / 'clustered_candidates.txt').write_text(
        'DM S/N Time Sample Filter_Width\n' + f'50.0 8.0 {60001 * DT:.6f} 60001 3\n')
    base = {'filename': 'beam.fil', 'tsamp': DT, 'nu_min': None, 'nu_max': None,
            'dedispersion_plan': [{'low_dm': 0.0, 'high_dm': 100.0, 'ddm': 1.0, 'downsample': 1}]}
    raw = measure_clusters(tmp_path, dict(base, single_pulse={}))
    white = measure_clusters(tmp_path, dict(base, single_pulse={'baseline_seconds': 2.0}))
    (key,) = raw
    assert white[key]['local_snr'] > raw[key]['local_snr']


def test_whitened_evidence_keeps_the_raw_statistic_and_routes_on_both():
    from lotaas_reprocessing.single_pulse_quality import review_route
    limits = {'min_local_snr': 5.0, 'min_raw_local_snr': 4.0}
    assert review_route(0.1, {'local_snr': 7.0, 'raw_local_snr': 4.5}, limits, 0) is None
    assert review_route(0.1, {'local_snr': 7.0, 'raw_local_snr': 3.0}, limits, 0) == 'unconfirmed'
    assert review_route(0.1, {'local_snr': 4.0, 'raw_local_snr': 9.0}, limits, 0) == 'unconfirmed'
    # An unwhitened search records no raw statistic, so the raw floor never applies.
    assert review_route(0.1, {'local_snr': 7.0}, limits, 0) is None


def test_whitened_evidence_records_both_statistics(tmp_path):
    from lotaas_reprocessing.single_pulse_quality import measure_clusters
    trials = tmp_path / 'DM_trials'
    trials.mkdir()
    x = red_noise(120000, 12, slow=2.0)
    x[60000:60003] += 3.5
    x.tofile(trials / 'beam_DM50.0.dat')
    (tmp_path / 'clustered_candidates.txt').write_text(
        'DM S/N Time Sample Filter_Width\n' + f'50.0 8.0 {60001 * DT:.6f} 60001 3\n')
    meta = {'filename': 'beam.fil', 'tsamp': DT, 'nu_min': None, 'nu_max': None,
            'dedispersion_plan': [{'low_dm': 0.0, 'high_dm': 100.0, 'ddm': 1.0, 'downsample': 1}],
            'single_pulse': {'baseline_seconds': 2.0}}
    (record,) = measure_clusters(tmp_path, meta).values()
    raw = measure_clusters(tmp_path, dict(meta, single_pulse={}))
    assert record['raw_local_snr'] == next(iter(raw.values()))['local_snr']
    assert record['local_snr'] > record['raw_local_snr']
