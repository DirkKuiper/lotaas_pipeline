import json

import numpy as np

from web.indexer import Indexer
from web.periodic_quality import assess, measure
from web.tests.conftest import ITEM
from web.tests.test_app import client_for


def diagnostic(seed=7, amplitude=4.0, intermittent=False, drifting=False):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(32, 64))
    for row in range(32):
        if intermittent and row >= 4:
            continue
        phase = (18 + row * 5) % 64 if drifting else 18
        for offset in (0, 1):
            values[row, (phase + offset) % 64] += amplitude
    counts = np.full(values.shape, 100)
    bands = rng.normal(size=(16, 64))
    bands[:, 18:20] += amplitude
    return {'subintegration_sums': values * counts, 'subintegration_counts': counts,
            'profile': values.mean(axis=0), 'errors': np.ones(64),
            'frequencies': np.linspace(0.33 - 1e-5, 0.33 + 1e-5, 33), 'fold_chi2': np.ones(33),
            'dm_curve': np.array([[26.0, 14], [26.1, 18], [26.2, 24], [26.3, 18], [26.4, 14]]),
            'subbands': bands}


def fold_record(**changes):
    return dict({'dm': 26.2, 'dm_step': 0.1, 'period_seconds': 3.03179,
                 'refined_period_seconds': 3.03179, 'effective_sampling_seconds': 0.007864,
                 'statistic': 24.0, 'harmonic_count': 16, 'fold_chi2': 630.0, 'fold_bins': 64,
                 'catalogue_matches': [], 'rfi_like': False}, **changes)


def test_repeated_pulse_passes_without_needing_a_catalogue_match():
    quality = assess(fold_record(), diagnostic())
    assert quality['strong'] and quality['repeatability'] >= 5 and quality['persistence'] >= 0.75
    assert quality['band_support'] >= 0.6 and quality['dm_trials'] == 5


def test_noise_impulses_and_drifting_structure_do_not_pass():
    for seed in range(30):
        assert not assess(fold_record(), diagnostic(seed=seed, amplitude=0))['strong']
    assert not assess(fold_record(), diagnostic(amplitude=50, intermittent=True))['strong']
    assert not assess(fold_record(), diagnostic(amplitude=50, drifting=True))['strong']


def test_dm_band_sampling_and_invalid_data_defer_without_deletion(tmp_path):
    data = diagnostic()
    assert not assess(fold_record(dm=0), data)['strong']
    data['dm_curve'] = np.array([[26.2, 24]])
    assert not assess(fold_record(), data)['strong']
    data['dm_curve'] = np.array([[0, 24], [26.1, 23], [26.2, 24], [26.3, 23]])
    assert not assess(fold_record(), data)['strong']
    data = diagnostic()
    data['subbands'][:, 18:20] -= 8
    assert not assess(fold_record(), data)['strong']
    assert not assess(fold_record(effective_sampling_seconds=1), diagnostic())['strong']
    assert not measure(fold_record(), tmp_path / 'missing.npz')['strong']
    np.savez(tmp_path / 'invalid.npz', **dict(diagnostic(), subintegration_counts=np.zeros((32, 64))))
    assert not measure(fold_record(), tmp_path / 'invalid.npz')['strong']


def save_fold(directory, name, period=3.03179, amplitude=4.0, catalogue=False):
    folder = directory / 'periodicity_plots'
    folder.mkdir(exist_ok=True)
    row = fold_record(period_seconds=period, refined_period_seconds=period,
                      plot=f'periodicity_plots/{name}.png', fold_data=f'periodicity_plots/{name}.npz',
                      catalogue_matches=[{'name': 'Known source'}] if catalogue else [])
    with (directory / 'periodicity_folded_candidates.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    np.savez(folder / f'{name}.npz', **diagnostic(amplitude=amplitude))


def save_low_dm_fold(directory, name, period, statistic=20.0, folds='periodicity_folded_candidates.jsonl'):
    folder = directory / 'periodicity_plots'
    folder.mkdir(exist_ok=True)
    row = fold_record(dm=1.4, period_seconds=period, refined_period_seconds=period, statistic=statistic,
                      plot=f'periodicity_plots/{name}.png', fold_data=f'periodicity_plots/{name}.npz')
    with (directory / folds).open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    data = diagnostic()
    data['dm_curve'] = np.array([[1.2, 14], [1.3, 18], [1.4, 24], [1.5, 18], [1.6, 14]])
    np.savez(folder / f'{name}.npz', **data)


def other_beam(directory, beam, declination):
    other = directory.parents[1] / ITEM.replace('BEAM025', f'BEAM{beam:03d}') / directory.name
    other.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((directory / 'metadata.json').read_text())
    metadata['observation_info']['DEC (J2000)'] = declination
    (other / 'metadata.json').write_text(json.dumps(metadata))
    return other


def test_low_dm_period_in_a_beam_and_its_neighbours_reaches_the_queue(cfg, campaign):
    # As PSR J0311+1402 at DM 1.4 in L606916 SAP001 beam 53 and three neighbours: a source on the sky, while
    # the same period in four beams 2 deg apart stays interference.
    directory = campaign['beam_dir']
    save_low_dm_fold(directory, 'local', 13.6358, statistic=85.0)
    save_low_dm_fold(directory, 'spread', 7.77123, statistic=85.0)
    for beam, declination in ((26, '+68:39:23.00'), (27, '+68:10:35.00'), (28, '+68:39:23.00')):
        save_low_dm_fold(other_beam(directory, beam, declination), 'local', 13.6358)
    for beam, declination in ((40, '+70:24:59.00'), (41, '+66:24:59.00'), (42, '+70:24:59.00')):
        save_low_dm_fold(other_beam(directory, beam, declination), 'spread', 7.77123)
    indexer = Indexer(cfg)
    indexer.run_pass()
    families = dict(indexer.db.execute('SELECT c.period, MAX(f.radius) FROM candidates c '
                                       'JOIN periodic_families f USING(key) GROUP BY c.period'))
    assert abs(families[13.6358] - 0.24) < 0.01 and families[7.77123] > 1.9
    states = {}
    for period, status, reasons in indexer.db.execute('SELECT c.period, t.status, t.reasons FROM candidates c '
                                                      'JOIN periodic_triage t USING(key)'):
        states.setdefault(period, []).append((status, json.loads(reasons)))
    assert sorted(s for s, _ in states[13.6358]) == ['related', 'related', 'related', 'strong']
    assert sorted(s for s, _ in states[7.77123]) == ['deferred', 'related', 'related', 'related']
    assert all('Period is flagged as interference' in r and 'Best DM is close to zero' in r
               for _, r in states[7.77123])
    assert '13.635800' in client_for(cfg).get('/periodic').text
    indexer.db.close()


def test_a_slow_pass_fold_joins_the_beams_folds_and_its_family(cfg, campaign):
    # The slow-pulsar pass (lotaas_reprocessing.slow_periodicity) writes slow_periodicity_folded_candidates.jsonl:
    # J0311+1402's fundamental there, in its beam and three neighbours, is one source with them.
    directory = campaign['beam_dir']
    slow = 'slow_periodicity_folded_candidates.jsonl'
    save_low_dm_fold(directory, 'slow_periodic_001', 40.9073, statistic=325.0, folds=slow)
    for beam, declination in ((26, '+68:39:23.00'), (27, '+68:10:35.00'), (28, '+68:39:23.00')):
        save_low_dm_fold(other_beam(directory, beam, declination), 'slow_periodic_001', 40.9073, folds=slow)
    indexer = Indexer(cfg)
    indexer.run_pass()
    rows = indexer.db.execute("SELECT c.period, t.status FROM candidates c JOIN periodic_triage t USING(key) "
                              "WHERE c.kind='periodic'").fetchall()
    assert sorted(status for _, status in rows) == ['related', 'related', 'related', 'strong']
    assert '40.907300' in client_for(cfg).get('/periodic').text
    indexer.db.close()


def test_default_queue_groups_strong_folds_and_keeps_deferred_accessible(cfg, campaign):
    directory = campaign['beam_dir']
    save_fold(directory, 'real', catalogue=True)
    save_fold(directory, 'harmonic', period=3.03179 / 2)
    save_fold(directory, 'noise', period=7.12345, amplitude=0)
    save_fold(directory, 'second_real', period=2.56789)
    # A stronger incoherent-beam diagnostic must not hide the coherent source.
    incoherent = directory.parents[1] / ITEM.replace('BEAM025', 'BEAM012') / directory.name
    incoherent.mkdir(parents=True)
    (incoherent / 'metadata.json').write_bytes((directory / 'metadata.json').read_bytes())
    save_fold(incoherent, 'incoherent', amplitude=10)
    indexer = Indexer(cfg)
    indexer.run_pass()
    states = dict(indexer.db.execute('SELECT c.period,t.status FROM candidates c JOIN periodic_triage t USING(key) '
                                   'WHERE c.item=?', (ITEM,)))
    assert states == {3.03179: 'strong', 3.03179 / 2: 'related', 7.12345: 'deferred', 2.56789: 'strong'}
    client = client_for(cfg)
    page = client.get('/periodic').text
    assert '3.031790' in page and '2.567890' in page and '7.123450' not in page and '1.515895' not in page
    assert 'B012' not in page
    assert '7.123450' in client.get('/periodic?triage=all').text
    assert '1.515895' in client.get('/periodic?triage=related').text
    deferred_id = indexer.db.execute('SELECT id FROM candidates WHERE period=7.12345').fetchone()[0]
    assert 'Deferred for follow-up' in client.get(f'/verify/{deferred_id}').text
    queue = client.get('/verify?kind=periodic', follow_redirects=False)
    assert queue.status_code == 303 and '1 of 2' in client.get(queue.headers['location']).text
    with indexer.db:
        indexer.derive()
    assert indexer.db.execute('SELECT COUNT(*) FROM periodic_triage').fetchone()[0] == 5
    assert indexer.db.execute("SELECT COUNT(*) FROM periodic_triage WHERE status='strong'").fetchone()[0] == 3
    indexer.db.close()


def test_a_slow_folds_intervals_are_merged_to_hold_enough_turns():
    from web.periodic_quality import merged_intervals, merges
    data = diagnostic()
    sums, counts = np.asarray(data['subintegration_sums'], float), np.asarray(data['subintegration_counts'], float)
    slow = fold_record(period_seconds=40.9073, refined_period_seconds=40.9073, observation_seconds=3600.0)
    merged_sums, merged_counts = merged_intervals(slow, sums, counts)
    assert merged_sums.shape == (11, 64) and np.allclose(merged_sums.sum(axis=0), sums.sum(axis=0))
    assert merges(slow) and not merges(fold_record())
    fast_sums, _ = merged_intervals(fold_record(), sums, counts)
    assert fast_sums is sums
    assert assess(fold_record(), data) == assess(fold_record(observation_seconds=3600.0), data)


def test_a_fold_longer_than_its_intervals_is_assessed_once_they_are_merged():
    # At 200 s each 112 s interval covers half the phase bins: 'Incomplete phase coverage' made such folds
    # unassessable (17,754 of 20,000 deferred archive folds at P >= 5 s).
    data = diagnostic(amplitude=6.0)
    counts = np.full((32, 64), 100.0)
    for row in range(32):
        empty = slice(32, 64) if row % 2 else slice(0, 32)
        counts[row, empty] = 0
    sums = np.asarray(data['subintegration_sums'], float) * (counts > 0)
    long = fold_record(period_seconds=200.0, refined_period_seconds=200.0, observation_seconds=3600.0)
    quality = assess(long, dict(data, subintegration_sums=sums, subintegration_counts=counts))
    assert 'Fold diagnostics unavailable or invalid' not in quality['reasons'] and quality['repeatability'] > 0


def pulses_at(directory, period, phase, dm=26.2, turns=(100, 420, 730, 1010)):
    """Single-pulse clusters at this DM and rotational phase in these turns, beside the fixture's own."""
    with (directory / 'clustered_candidates.txt').open('a') as stream:
        for i, turn in enumerate(turns):
            t = (turn + phase) * period
            stream.write(f'{dm}\t8.0\t{t:.4f}\t{int(t / 0.00786)}\t3\t1\t{40 + i}\n')


def triage_of(cfg):
    indexer = Indexer(cfg)
    indexer.run_pass()
    rows = indexer.db.execute('SELECT t.status, t.reasons, t.evidence FROM periodic_triage t').fetchall()
    indexer.db.close()
    return [(status, json.loads(reasons), json.loads(evidence)) for status, reasons, evidence in rows]


def test_pulses_on_its_phase_show_a_fold_the_time_splits_defer(cfg, campaign):
    # A nulling or flickering pulsar fails the time splits however bright; its own single pulses arriving on the
    # fold's phase in several turns are independent evidence (J0311+1402: 18 of 19 on phase).
    from web.periodic_quality import pulse_phase
    directory = campaign['beam_dir']
    save_fold(directory, 'flicker', period=20.74, amplitude=50)
    data = diagnostic(amplitude=50, intermittent=True)
    np.savez(directory / 'periodicity_plots' / 'flicker.npz', **data)
    phase, _ = pulse_phase(np.asarray(data['subintegration_sums'], float), np.asarray(data['subintegration_counts'], float))
    pulses_at(directory, 20.74, phase, turns=(10, 61, 102, 150))
    [(status, reasons, evidence)] = triage_of(cfg)
    assert status == 'strong' and not reasons
    assert evidence['pulses']['on'] == 4 and evidence['pulses']['log10p'] <= -4
    assert set(evidence['reasons']) & {'Pulse window does not repeat strongly in every time split',
                                       'Pulse phase is not supported in enough time intervals'}


def test_pulses_off_its_phase_leave_the_fold_deferred(cfg, campaign):
    from web.periodic_quality import pulse_phase
    directory = campaign['beam_dir']
    save_fold(directory, 'flicker', period=20.74, amplitude=50)
    data = diagnostic(amplitude=50, intermittent=True)
    np.savez(directory / 'periodicity_plots' / 'flicker.npz', **data)
    phase, _ = pulse_phase(np.asarray(data['subintegration_sums'], float), np.asarray(data['subintegration_counts'], float))
    pulses_at(directory, 20.74, (phase + 0.5) % 1, turns=(10, 61, 102, 150))
    [(status, reasons, evidence)] = triage_of(cfg)
    assert status == 'deferred' and evidence['pulses']['on'] == 0
