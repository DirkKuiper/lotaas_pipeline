"""The sift joins a period's run over DM only; harmonics are judged at the fold shortlist, after the veto.

Until 27 October 2026 harmonics were joined transitively: J2029+3744 (1.2168 s) sat in a group of 1,820
peaks led by 24.336 s = 20 P, which the cross-beam veto removed as a requantiser step period.
"""
import numpy as np

from lotaas_reprocessing.periodicity import resolved_config, same_source, sift_bounded, sift_candidates
from lotaas_reprocessing.periodicity_folding import shortlist


def config(**overrides):
    values = {"period_min_seconds": 0.016, "period_max_seconds": 300.0, "harmonics": [1, 2, 4, 8, 16, 32],
              "threshold": 12.0, "red_noise_window_bins": 31, "rfi_frequencies_hz": [], "rfi_tolerance_bins": 2,
              "sift_dm_tolerance": 2.0, "sift_period_fraction": 0.001, "max_folds": 16, "catalogue_match": False}
    values.update(overrides)
    return resolved_config(values)


def run(period, dms, peak, step=0.2, at=None):
    """A signal's peaks over DM: strongest at `at` (default the middle), falling off either side."""
    at = np.median(dms) if at is None else at
    return [dict(period_seconds=period, frequency_hz=1 / period, frequency_resolution_hz=1 / 3600, dm=float(dm),
                 dm_step=step, statistic=peak - 0.5 * abs(dm - at), rfi_like=False) for dm in dms]


def j2029_like():
    P = 1.2168136652342303
    dms = np.round(np.arange(184.0, 198.8, 0.2), 1)
    rows = run(P, dms, 82.8, at=190.6)
    for k in range(2, 33):
        rows += run(k * P, dms, 82.0 - 1.2 * k, at=190.6)
    # The step period, over a wide DM range, lifted above the pulsar where the pulsar's 20th harmonic adds.
    rows += run(20 * P * 1.00002, np.round(np.arange(136.6, 371.2, 0.2), 1), 88.7, at=192.8)
    return P, rows


def test_a_pulsar_is_not_carried_off_by_a_stronger_step_period():
    P, rows = j2029_like()
    sifted = sift_candidates(rows, config())
    best = [r for r in sifted if r['is_sifted_best']]
    # One signal per period: P, 2-32 P, the step period joined with the pulsar's own 20 P run.
    assert len(best) == 32
    step = max(best, key=lambda r: r['statistic'])
    assert abs(step['period_seconds'] / P - 20) < 0.01
    # The veto removes the step period: the pulsar takes the first fold slot, and none of its own multiples follow.
    for r in sifted:
        r['multibeam_rfi'] = r['is_sifted_best'] and r is step
    chosen = shortlist(sifted, config())
    assert len(chosen) == 1 and abs(chosen[0]['period_seconds'] - P) < 1e-9 and chosen[0]['dm'] == 190.6
    # Not vetoed, the step period folds first, and still leaves the pulsar its own slot (P is 1/20 of it).
    for r in sifted:
        r['multibeam_rfi'] = False
    chosen = shortlist(sifted, config())
    assert [round(r['period_seconds'] / P) for r in chosen] == [20, 1]


def test_a_crowded_beam_stays_many_signals():
    # 300 periods 1.6% apart over 1-100 s, many of them near whole ratios of others.
    rng = np.random.default_rng(3)
    periods = np.geomspace(1.0, 100.0, 300)
    rows = []
    for p in periods:
        rows += run(float(p), np.arange(10.0, 14.0, 0.5), rng.uniform(13, 40), step=0.5)
    sifted, limit = sift_bounded(rows, config(max_sifted_signals=1000))
    assert limit == len(rows)
    assert sum(r['is_sifted_best'] for r in sifted) == 300


def test_same_source_multiples_to_the_highest_harmonic_fractions_to_16():
    c = config()
    chosen = {'period_seconds': 2.0, 'dm': 50.0, 'signal_dm_min': 48.0, 'signal_dm_max': 52.0}
    def row(p, dm=50.0):
        return {'period_seconds': p, 'dm': dm, 'signal_dm_min': dm - 1, 'signal_dm_max': dm + 1}
    assert same_source(row(2.0), chosen, c)
    assert same_source(row(4.0), chosen, c) and same_source(row(2.0 * 31), chosen, c)
    assert not same_source(row(2.0 * 33), chosen, c)
    assert same_source(row(2.0 / 16), chosen, c)
    assert not same_source(row(2.0 / 20), chosen, c)
    assert same_source(row(2.0 * 1.5), chosen, c) and same_source(row(2.0 * 15 / 2), chosen, c), 'm/n aliases, n <= 4'
    assert not same_source(row(2.0 * 7 / 5), chosen, c) and not same_source(row(2.0 * 1.37), chosen, c)
    assert not same_source(row(4.0, dm=60.0), chosen, c), 'a harmonic at another DM is another source'
    # Only the search's own harmonic sums lift a k x P peak: with 16 at most, 20 P is not P again.
    assert not same_source(row(2.0 * 20), chosen, config(harmonics=[1, 2, 4, 8, 16]))


def test_the_beam_limit_no_longer_cuts_a_full_search():
    # Every trial's 100 candidates of a 3,802-trial beam are sifted.
    assert resolved_config({})['max_candidates_per_beam'] >= 3802 * 100


def test_a_beam_keeps_its_strongest_signals_and_marks_the_rest():
    periods = np.geomspace(1.0, 100.0, 300)
    rows = []
    for k, p in enumerate(periods):
        rows += run(float(p), np.arange(10.0, 12.0, 0.5), 13.0 + k * 0.1, step=0.5)
    for r in rows:  # a flagged signal ranks after every clean one, whatever its strength
        if r['period_seconds'] == periods[-1]:
            r['rfi_like'] = True
    sifted = sift_candidates(rows, config(max_sifted_signals=50))
    best = [r for r in sifted if r['is_sifted_best']]
    assert len(best) == 50
    assert min(r['statistic'] for r in best) > max(r['statistic'] for r in sifted if r['sift_status'] == 'signal_limit')
    assert not any(r['period_seconds'] == periods[-1] for r in best), 'the strongest, flagged, signal goes'
    # 250 signals go; the flagged one keeps its rfi_like status.
    assert len({r['sift_group'] for r in sifted if r['sift_status'] == 'signal_limit'}) == 249


def peak(period, dm, statistic, vetoed=False, span=1.0, **extra):
    return dict(period_seconds=period, frequency_hz=1 / period, frequency_resolution_hz=1 / 3600, dm=dm,
                signal_dm_min=dm - span, signal_dm_max=dm + span, statistic=statistic, rfi_like=False,
                is_sifted_best=True, multibeam_rfi=vetoed, **extra)


def test_an_alias_of_interference_takes_no_fold_slot():
    rows = [peak(2.0, 10.0, 500.0, vetoed=True, span=20.0),   # a 2 s comb, vetoed across the beams
            peak(62.0, 1.2, 250.0), peak(40.0, 16.2, 240.0),  # its 31st and 20th multiples, not vetoed
            peak(1.2168, 15.0, 30.0)]                          # a pulsar beside it
    assert [r['period_seconds'] for r in shortlist(rows, config())] == [1.2168]


def test_red_noise_within_a_fourier_bin_folds_once():
    # 0.15-0.8 bins apart at 230-290 s, one DM.
    rows = [peak(p, 0.2, 19.0 - k * 0.1) for k, p in enumerate((267.24, 263.58, 284.36, 271.0))]
    assert [r['period_seconds'] for r in shortlist(rows, config())] == [267.24]


def test_below_fold_low_dm_last_folds_after_the_rest():
    rows = [peak(48.3, 0.3, 156.0), peak(0.9311, 41.6, 12.9)]
    assert [r['dm'] for r in shortlist(rows, config(max_folds=1))] == [0.3]
    assert [r['dm'] for r in shortlist(rows, config(max_folds=1, fold_low_dm_last=1.0))] == [41.6]
