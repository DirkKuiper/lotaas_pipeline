import json
import os

import numpy as np

from web import sigproc
from web.indexer import Indexer
from web.snippets import Snippets, cut, downsample_for
from web.tests.conftest import ITEM, TSAMP, synthetic_filterbank

PLAN = [{'low_dm': 0.0, 'high_dm': 150.6, 'downsample': 1}, {'low_dm': 150.6, 'high_dm': 300.0, 'downsample': 2}]


def detection(**extra):
    return dict({'key': f'candidate|{ITEM}|DM30.000|W3|SN12.000', 'type': 'candidate', 'item': ITEM, 'dm': 30.0,
                 'snr': 12.0, 'width_samples': 3, 'time_seconds': 15.0}, **extra)


def test_downsample_follows_the_plan():
    assert downsample_for(30, PLAN) == 1 and downsample_for(200, PLAN) == 2 and downsample_for(900, PLAN) == 2


def test_cut_keeps_the_sweep_and_its_time_reference(tmp_path):
    source = synthetic_filterbank(tmp_path / f'{ITEM}.fil')
    path = cut(source, detection(), tmp_path / 'out', PLAN, bad_channels=[5])
    header, data = sigproc.open_data(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['t0_relative'] < -(meta['sweep_seconds'] + 4.9)
    assert data.shape[0] * TSAMP > 2 * meta['sweep_seconds'] + 9.9
    assert np.isclose(header['tstart'], 57713.157638888886 + meta['start_sample'] * TSAMP / 86400)
    assert meta['bad_channels'] == [5]


def test_cut_decimates_to_the_search_resolution(tmp_path):
    source = synthetic_filterbank(tmp_path / f'{ITEM}.fil', dm=200.0, t_pulse=10.0, nsamp=12000)
    path = cut(source, detection(dm=200.0, time_seconds=10.0), tmp_path / 'out', PLAN)
    header, _ = sigproc.open_data(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['downsample'] == 2 and np.isclose(header['tsamp'], 2 * TSAMP)
    assert meta['start_sample'] % 2 == 0


def test_cut_pads_at_the_start_of_the_file(tmp_path):
    source = synthetic_filterbank(tmp_path / f'{ITEM}.fil', t_pulse=1.0)
    path = cut(source, detection(time_seconds=1.0), tmp_path / 'out', PLAN)
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['start_sample'] < 0


def test_hold_cut_release(cfg, campaign):
    prepared = campaign['prepared'] / 'B025'
    prepared.mkdir(parents=True)
    original = synthetic_filterbank(prepared / f'{ITEM}.fil')
    with campaign['state'].db() as db:
        db.execute("UPDATE saps SET state='dispatched' WHERE key='L1163405_SAP000'")
    Indexer(cfg).run_pass()
    snippets = Snippets(cfg)
    snippets.sources_scanned = 1e12        # no search of the source roots here
    first = snippets.run_pass()
    held = cfg.held / 'L1163405_SAP000' / original.name
    assert first['linked'] == 1 and held.is_file() and os.stat(held).st_nlink == 2
    # The campaign deletes its copy after the search and marks the SAP searched.
    original.unlink()
    with campaign['state'].db() as db:
        db.execute("UPDATE saps SET state='searched' WHERE key='L1163405_SAP000'")
    second = snippets.run_pass()
    assert second['cut'] == 1 and second['released'] == 1 and not held.exists()
    made = list(cfg.snippets.glob('*.fil'))
    assert len(made) == 1 and 'DM30.000' in made[0].name
    meta = json.loads(made[0].with_suffix('.json').read_text())
    assert meta['how'] == 'held' and meta['bad_channels'] == [5, 7]


def test_backfill_finds_a_surviving_copy(cfg, campaign):
    synthetic_filterbank(cfg.source_roots[0] / f'{ITEM}.fil')
    Indexer(cfg).run_pass()
    result = Snippets(cfg).run_pass()
    assert result['backfilled'] == 1
    meta = json.loads(next(cfg.snippets.glob('*.json')).read_text())
    assert meta['how'] == 'found on disk'
