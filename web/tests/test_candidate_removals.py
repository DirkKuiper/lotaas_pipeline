import json

from web import store
from web.indexer import Indexer
from web.tests.test_app import client_for, fold


def test_removal_survives_refresh_and_rebuild_without_losing_evidence(cfg, campaign):
    beam_dir = campaign['beam_dir']
    fold(beam_dir, 0.0, 5.9323, 20.0, 'artifact')
    fold(beam_dir, 26.3, 3.0317926, 560.3, 'keep')
    source = (beam_dir / 'periodicity_folded_candidates.jsonl').read_bytes()
    indexer = Indexer(cfg)
    indexer.run_pass()
    removed = dict(indexer.db.execute("SELECT key,id FROM candidates WHERE period=5.9323").fetchone())
    kept = indexer.db.execute("SELECT id FROM candidates WHERE period=3.0317926").fetchone()[0]
    sp_before = {r[0] for r in indexer.db.execute("SELECT key FROM candidates WHERE kind='sp'")}
    reviews = store.reviews(cfg)
    with reviews:
        reviews.execute('INSERT INTO candidate_removals VALUES (?,?,?,?,?)',
                        (removed['key'], 'Confirmed artifact', json.dumps({'audit': 'test'}), 'test', 1.0))
    reviews.close()

    # A normal pass and a completely rebuilt index both respect the removal.
    for rebuild in (False, True):
        if rebuild:
            indexer.db.close()
            cfg.index_db.unlink()
            indexer = Indexer(cfg)
        indexer.run_pass()
        assert not indexer.db.execute('SELECT 1 FROM candidates WHERE key=?', (removed['key'],)).fetchone()
        assert indexer.db.execute('SELECT 1 FROM periodic WHERE key=?', (removed['key'],)).fetchone()
        assert indexer.db.execute('SELECT 1 FROM periodic_families WHERE key=?', (removed['key'],)).fetchone()
        assert {r[0] for r in indexer.db.execute("SELECT key FROM candidates WHERE kind='sp'")} == sp_before
        assert (beam_dir / 'periodicity_folded_candidates.jsonl').read_bytes() == source
        client = client_for(cfg)
        assert '5.932300' not in client.get('/periodic?type=all&multibeam=include').text
        assert '3.031793' in client.get('/periodic?triage=all').text
        assert client.get(f"/verify/{removed['id']}").status_code == 404
        assert client.get(f'/verify/{kept}').status_code == 200
    indexer.db.close()
