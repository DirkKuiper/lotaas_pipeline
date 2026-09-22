"""Inspect campaign coverage, stage attempts and failures."""
import argparse
import json
from pathlib import Path
from euroflash.ledger import Ledger
from euroflash.provenance import coverage


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('ledger',type=Path)
    a=p.parse_args()
    ledger=Ledger(a.ledger)
    result=ledger.summary()
    result['archive_beam_history']=coverage(ledger)
    with ledger.connect() as db:
        result['analyzed_beams_by_scope']=[dict(r) for r in db.execute('''
            SELECT CASE r.pilot WHEN 0 THEN 'production' WHEN 1 THEN 'pilot' ELSE 'unclassified_legacy' END AS scope,
            COUNT(DISTINCT a.item) AS beams FROM attempts a LEFT JOIN runs r USING(fingerprint)
            WHERE a.stage='classify' AND a.status='success' GROUP BY scope''')]
    with ledger.connect() as db:
        result['latest_stages']=[dict(r) for r in db.execute('''
            SELECT stage,status,COUNT(*) AS count FROM attempts
            WHERE id IN (SELECT MAX(id) FROM attempts GROUP BY item,stage,fingerprint)
            GROUP BY stage,status''')]
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        result['detections']=[dict(r) for r in db.execute('SELECT detection_type,COUNT(*) AS count FROM detections GROUP BY detection_type')] if 'detections' in tables else []
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
