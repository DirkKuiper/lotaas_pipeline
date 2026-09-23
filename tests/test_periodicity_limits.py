"""A bright source or an instrumental comb must not end or swamp a periodic search."""
import json

import numpy as np
import pytest

from lotaas_reprocessing.periodicity import (fast_length, resolved_config, run_periodicity_search,
                                             search_trial, sift_bounded,
                                             sift_candidates, thin_candidates)


def config(**overrides):
    values = {"period_min_seconds": 0.05, "period_max_seconds": 100.0, "harmonics": [1, 2, 4, 8, 16],
              "threshold": 12.0, "red_noise_window_bins": 31, "rfi_frequencies_hz": [], "rfi_tolerance_bins": 2,
              "sift_dm_tolerance": 2.0, "sift_period_fraction": 0.001, "max_folds": 0, "catalogue_match": False}
    values.update(overrides)
    return resolved_config(values)


def bright_pulse_train(dt=0.01, period=3.0318, n=2 ** 17, amplitude=40.0, seed=9):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    data = rng.normal(size=n)
    data[(t % period) < 2 * dt] += amplitude          # a very narrow, very bright pulse
    return data.astype('float32')


def test_a_bright_pulsar_trial_is_thinned_not_aborted():
    stats = {}
    rows = search_trial(bright_pulse_train(), 0.01, config(max_candidates_per_trial=40), stats=stats)
    assert stats['peaks_found'] > 40, 'the test needs a crowded trial'
    assert len(rows) <= 40 and stats['peaks_dropped'] == stats['peaks_found'] - len(rows)
    best = max(rows, key=lambda r: r['statistic'])
    assert abs(best['frequency_hz'] * 3.0318 - round(best['frequency_hz'] * 3.0318)) < 0.01
    # One row per native Fourier bin at most.
    bins = sorted(round(r['frequency_hz'] / r['frequency_resolution_hz']) for r in rows)
    assert all(b - a > 1 for a, b in zip(bins, bins[1:]))


def test_thinning_keeps_the_strongest_distinct_frequencies():
    rows = [{'frequency_hz': f, 'frequency_resolution_hz': 1.0, 'statistic': s}
            for f, s in [(10, 5), (10.4, 9), (11, 3), (50, 20), (80, 1), (120, 7)]]
    kept, dropped = thin_candidates(rows, 3)
    assert [r['frequency_hz'] for r in kept] == [10.4, 50, 120] and dropped == 3
    assert thin_candidates(rows, 10) == (rows, 0)


def test_sifting_is_bounded_by_halving_instead_of_failing():
    rows = [{'period_seconds': 1.0 + 1e-6 * i, 'dm': 10.0 + 0.1 * (i % 40), 'dm_step': 0.1,
             'rfi_like': False, 'statistic': float(i)} for i in range(400)]
    with pytest.raises(RuntimeError, match='work limit'):
        sift_candidates(rows, config(max_sift_comparisons=2000))
    sifted, limit = sift_bounded(rows, config(max_sift_comparisons=2000))
    assert 1 <= limit < 400 and len(sifted) == 400
    left = [r for r in sifted if r['sift_status'] == 'not_sifted_beam_limit']
    assert len(left) == 400 - limit
    # The strongest are the ones sifted.
    assert min(r['statistic'] for r in sifted if r['sift_status'] != 'not_sifted_beam_limit') > \
        max(r['statistic'] for r in left)


def test_a_beam_over_its_candidate_limit_completes_and_records_it(tmp_path):
    trials = tmp_path / 'Periodic_DM_trials'
    trials.mkdir()
    plan = [{'low_dm': 10, 'high_dm': 14, 'ddm': 1, 'downsample': 1}]
    for dm in (10.0, 11.0, 12.0, 13.0):
        bright_pulse_train(n=2 ** 16, seed=int(dm)).tofile(trials / f'beam_DM{dm}.dat')
    metadata = {'tsamp': .01, 'filename': 'beam.fil', 'samples_processed': 2 ** 16, 'nu_min': 120.,
                'nu_max': 160., 'dedispersion_plan': plan, 'periodicity_dm_plan': plan}
    summary = run_periodicity_search(trials, tmp_path, metadata,
                                     config(max_candidates_per_trial=30, max_candidates_per_beam=50))
    assert summary['complete'] is True and summary['trials_searched'] == 4
    assert summary['crowded_trials'] == 4 and summary['peaks_dropped_in_crowded_trials'] > 0
    assert summary['sift_input_limit'] == 50 and summary['candidates_not_sifted'] == summary['raw_candidates'] - 50
    coverage = json.loads((tmp_path / 'periodicity_coverage.json').read_text())
    assert all(c['peaks_found'] >= c['candidates'] for c in coverage)


def test_fast_lengths_are_smooth_close_and_never_shorter():
    for n in (457035, 457728, 229000, 13005, 16, 17, 1000003):
        m = fast_length(n)
        assert n <= m <= n + max(1, n // 40)
        value = 2 * m
        for prime in (2, 3, 5, 7):
            while value % prime == 0:
                value //= prime
        assert value == 1


def test_the_top_of_the_band_is_whitened_like_the_rest():
    """A one-bin final noise block (the real-only Nyquist term) whitened the band edge ~100-fold."""
    from lotaas_reprocessing.periodicity import noise_baseline
    rng = np.random.default_rng(21)
    for n in range(4096, 4096 + 64):
        native = rng.exponential(size=n)
        native[-1] = 0.01            # a real-only Nyquist term far below the exponential mean
        baseline = noise_baseline(native, 257, real_only_last=True)
        assert 0.6 < baseline[-2] < 1.6 and 0.6 < baseline[-1] < 1.6   # was ~0.01 before the fix


def old_final_block(native_length, width=257):
    start, block = 1, 31
    while True:
        stop = min(start + block, native_length)
        if stop == native_length:
            return stop - start
        start, block = stop, min(width, max(31, int(block * 1.5)))


def test_white_noise_gives_no_peaks_at_the_trial_nyquist_period():
    """Lengths whose last noise block was a single bin produced 'detections' at P = 2 dt."""
    rng = np.random.default_rng(22)
    dt = 0.0078643
    lengths = [n for n in range(30000, 32000, 2) if old_final_block(n // 2 + 1) == 1][:4]
    assert lengths, 'the test needs lengths that used to leave a one-bin final block'
    for n in lengths:
        x = rng.normal(size=n)
        # The real-only Nyquist term is chi-squared with one degree of freedom,
        # so occasionally near zero; remove it entirely for the worst case.
        alternating = (-1.) ** np.arange(n)
        x -= alternating * (x @ alternating) / n
        rows = search_trial(x, dt, config(period_min_seconds=0.016, fft_fast_lengths=False))
        assert not rows, (n, [(r['period_seconds'], r['statistic']) for r in rows])


def test_padding_to_a_fast_length_keeps_every_sample_and_the_detection():
    dt, period = 0.01, 2.56
    data = np.random.default_rng(12).normal(size=100003).astype('float32')
    data[(np.arange(100003) * dt % period) < 0.03] += 1.5
    padded = search_trial(data, dt, config(fft_fast_lengths=True))
    plain = search_trial(data, dt, config(fft_fast_lengths=False))
    best_padded = max(padded, key=lambda r: r['statistic'])
    best_plain = max(plain, key=lambda r: r['statistic'])
    assert best_padded['fft_samples'] == fast_length(100003) > 100003
    assert best_padded['observation_seconds'] == best_plain['observation_seconds'] == 100003 * dt
    assert abs(best_padded['period_seconds'] - period) < 2e-3 and abs(best_plain['period_seconds'] - period) < 2e-3
    assert best_padded['statistic'] > 0.8 * best_plain['statistic']
