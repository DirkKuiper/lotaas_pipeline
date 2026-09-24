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
