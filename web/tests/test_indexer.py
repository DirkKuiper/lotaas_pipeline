from postproc.notify_candidates import key_for
from web.indexer import Indexer, fp16_of, run_of, sexagesimal
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
    # The same key the Slack notifier gives its post.
    assert candidate['key'] == key_for({'item': ITEM, 'dm': 30.0, 'width': 3, 'snr': 12.0})
    assert candidate['slack_sent'] == 95.0 and candidate['sap_key'] == 'L1163405_SAP000'
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
