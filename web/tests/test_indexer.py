from pathlib import Path

from web.indexer import Indexer, fp16_of, run_of, sexagesimal
from web.keys import sp_key
from web.tests.conftest import ITEM


def test_path_helpers():
    path = f'/home/dkuiper/lotaas-runs/run-a/work/processed/{ITEM}/abc123/candidate_plots'
    assert run_of(path) == 'run-a' and fp16_of(path) == 'abc123'
    assert abs(sexagesimal('08:47:08.00', hours=True) - 131.7833) < 1e-3
    assert abs(sexagesimal('-01:30:00') + 1.5) < 1e-9


def test_index_mirrors_and_derives(cfg, campaign):
    indexer = Indexer(cfg)
    indexer.run_pass()
    db = indexer.db
    assert db.execute('SELECT COUNT(*) FROM saps').fetchone()[0] == 2
    assert db.execute('SELECT COUNT(*) FROM files').fetchone()[0] == 3
    assert db.execute("SELECT key FROM obs_sap WHERE observation='L559289' AND sap=0").fetchone()[0] == 'L1163405_SAP000'
    candidate = db.execute("SELECT * FROM candidates WHERE type='candidate'").fetchone()
    # The key existing verdicts were saved under.
    assert candidate['key'] == sp_key('candidate', ITEM, 30.0, 3, 12.0) == f'candidate|{ITEM}|DM30.000|W3|SN12.000'
    assert candidate['sap_key'] == 'L1163405_SAP000'
    assert candidate['plot_id'] is not None and candidate['run_name'] == 'run-a'
    info = db.execute("SELECT * FROM sap_info WHERE key='L1163405_SAP000'").fetchone()
    assert info['beams_searched'] == 1 and info['pointing'] == 'P1254B' and abs(info['dec_deg'] - 68.416) < 1e-2


def test_state_changes_become_transitions(cfg, campaign):
    indexer = Indexer(cfg)
    indexer.run_pass()
    with campaign['state'].db() as db:
        db.execute("UPDATE files SET state='requested', updated=200.0 WHERE sap_key='L1163405_SAP001'")
    indexer.run_pass()
    rows = indexer.db.execute("SELECT * FROM transitions").fetchall()
    assert [(r['old'], r['new'], r['time']) for r in rows] == [('pending', 'requested', 200.0)]


def test_new_ledger_rows_are_picked_up(cfg, campaign):
    indexer = Indexer(cfg)
    indexer.run_pass()
    with campaign['ledger'].connect() as db:
        db.execute("INSERT INTO attempts(item,stage,fingerprint,status,started,host) "
                   "VALUES ('x','dedisperse','f','running',1e12,'h')")
    indexer.run_pass()
    with campaign['ledger'].connect() as db:
        db.execute("UPDATE attempts SET status='success' WHERE item='x'")
    indexer.run_pass()
    assert indexer.db.execute("SELECT status FROM attempts WHERE item='x'").fetchone()[0] == 'success'


def test_slack_columns_are_dropped_and_verdicts_kept(cfg):
    import sqlite3
    from web import store
    cfg.prepare()
    with sqlite3.connect(cfg.reviews_db) as db:
        db.execute('CREATE TABLE reviews (id INTEGER PRIMARY KEY, key TEXT NOT NULL, reviewer TEXT NOT NULL, '
                   'label TEXT NOT NULL, note TEXT, dm REAL, created REAL NOT NULL, slack_ts TEXT)')
        db.execute("INSERT INTO reviews(key,reviewer,label,created) VALUES ('k', 'dk', 'noise', 1.0)")
    reviews = store.reviews(cfg)
    assert 'slack_ts' not in {row[1] for row in reviews.execute('PRAGMA table_info(reviews)')}
    assert [tuple(r) for r in reviews.execute('SELECT key, label FROM reviews')] == [('k', 'noise')]


def test_results_of_benchmarks_and_the_injection_lane_are_pilot_runs(cfg, campaign):
    import json
    import shutil
    indexer = Indexer(cfg)
    indexer.run_pass()
    row = indexer.db.execute('SELECT dir, pilot FROM beams').fetchone()
    assert not row['pilot']
    fold = {'plot': 'periodicity_plots/periodic_001.png', 'dm': 450.0, 'period_seconds': 23.2, 'statistic': 18.5}
    (Path(row['dir']) / 'periodicity_folded_candidates.jsonl').write_text(json.dumps(fold) + '\n')
    indexer.run_pass(full=True)
    # An injection twin is a copy of the beam: its metadata says "pilot": false, and it finds the production beam's
    # candidates and folds under the same keys, besides its own (the injected bursts').
    node = Path(row['dir']).parents[2]                         # .../<run>/<node>/processed/<item>/<fp16>
    copy = cfg.result_roots[0] / 'benchmarks' / 'trial' / node.name
    shutil.copytree(node, copy)
    twin_dir = copy / Path(row['dir']).relative_to(node)
    own = dict(fold, plot='periodicity_plots/periodic_002.png', period_seconds=11.0)
    (twin_dir / 'periodicity_folded_candidates.jsonl').write_text(json.dumps(fold) + '\n' + json.dumps(own) + '\n')
    indexer.run_pass(full=True)
    db = indexer.db
    flags = {('/benchmarks/' in r['dir']): r['pilot'] for r in db.execute('SELECT dir, pilot FROM beams')}
    assert flags == {True: 1, False: 0}
    # The production fold keeps its place; the twin's own fold is a pilot's.
    folds = db.execute("SELECT dir, pilot, period FROM candidates WHERE kind='periodic' ORDER BY period").fetchall()
    assert [('/benchmarks/' in r['dir'], r['pilot'], r['period']) for r in folds] == [(True, 1, 11.0), (False, 0, 23.2)]
    # The production beam's single-pulse candidates stay the campaign's, not the newer twin's.
    sp = db.execute("SELECT dir, pilot FROM candidates WHERE kind='sp'").fetchall()
    assert sp and all('/benchmarks/' not in r['dir'] and not r['pilot'] for r in sp)


def test_folds_a_twin_took_over_are_restored_once(cfg, campaign):
    import json
    import shutil
    from web.indexer import meta_set
    indexer = Indexer(cfg)
    indexer.run_pass()
    row = indexer.db.execute('SELECT dir FROM beams').fetchone()
    fold = {'plot': 'periodicity_plots/periodic_001.png', 'dm': 450.0, 'period_seconds': 23.2, 'statistic': 18.5}
    (Path(row['dir']) / 'periodicity_folded_candidates.jsonl').write_text(json.dumps(fold) + '\n')
    node = Path(row['dir']).parents[2]
    copy = cfg.result_roots[0] / 'injections' / 'results' / 'lane' / node.name
    shutil.copytree(node, copy)
    indexer.run_pass(full=True)
    db = indexer.db
    # As the index stood before 5 October 2026: the twin's row holds the production fold's key.
    twin_dir = str(copy / Path(row['dir']).relative_to(node))
    with db:
        db.execute('UPDATE periodic SET dir=?, pilot=1', (twin_dir,))
        meta_set(db, 'twin_folds_restored', 0)
    indexer.run_pass()
    assert [tuple(r) for r in db.execute('SELECT dir, pilot FROM periodic')] == [(row['dir'], 0)]
