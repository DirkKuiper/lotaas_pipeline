"""Beams a /findings record cites keep their data (euroflash.findings.reviewed_items)."""
import json
import sqlite3
from types import SimpleNamespace

from euroflash.findings import cited_keys, periodic_key, reviewed_items


def reviews_db(path, reviews, findings):
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE reviews (key TEXT, label TEXT, created REAL)')
    db.execute('CREATE TABLE findings (source TEXT, names TEXT, reviewer TEXT, status TEXT, claims TEXT, '
               'summary TEXT, evidence TEXT, checked TEXT, created REAL)')
    db.executemany('INSERT INTO reviews VALUES (?,?,?)', reviews)
    db.executemany('INSERT INTO findings VALUES (?,?,?,?,?,?,?,?,?)',
                   [(s, '[]', 'Claude', st, '[]', '', json.dumps(ev), '', t) for s, st, ev, t in findings])
    db.commit()
    db.close()


def test_a_cited_fold_keeps_its_beam_and_a_settled_source_does_not(tmp_path):
    beams = {f'downsampled_L1_SAP000_BEAM00{i}_32bit_ff': {'item': f'downsampled_L1_SAP000_BEAM00{i}_32bit_ff',
                                                          'fp': 'abc', 'folds': [{'plot': f'p{i}.png'}]}
             for i in range(3)}
    index = SimpleNamespace(beams=beams)
    keys = [periodic_key(beams[item], beams[item]['folds'][0]) for item in sorted(beams)]
    path = tmp_path / 'reviews.sqlite'
    reviews_db(path, [(k, 'known', 1.0) for k in keys],
               [('J1848+0647', 'finding', [keys[0]], 2.0),
                ('J0000+00', 'follow_up', [keys[1]], 1.0), ('J0000+00', 'not_new', [keys[1]], 3.0),
                ('J1111+11', 'not_real', [keys[2]], 2.0)])
    assert cited_keys(path) == {keys[0]}
    assert reviewed_items(path, index) == {sorted(beams)[0]}
    assert cited_keys(tmp_path / 'missing.sqlite') == frozenset()
