"""Reconcile extracted archive receipts with the campaign's beam identities."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit
from euroflash.ledger import Ledger


def reconcile(directory, ledger):
    by_name = {}
    with ledger.connect() as db:
        for row in db.execute('SELECT uri FROM inputs'):
            by_name.setdefault(Path(urlsplit(row[0]).path).name, []).append(row[0])
    linked = 0
    for marker in directory.rglob('*.extracted.json'):
        value = json.loads(marker.read_text())
        receipt = value['archive']
        matches = by_name.get(Path(urlsplit(receipt['url']).path).name, [])
        if len(matches) != 1:
            raise ValueError(f'Expected one known archive URI for {marker}, found {len(matches)}')
        # FITS deleted after conversion are recorded in the marker; only an
        # unexplained absence is an error.
        if not value.get('raw_deleted') and any(not Path(path).is_file() for path in value['fits']):
            raise ValueError(f'Extraction receipt references missing FITS: {marker}')
        ledger.record_archive(matches[0], value['request_id'], receipt, value['fits'], marker)
        linked += 1
    return linked


def coverage(ledger, exclude_beams=None):
    """Search history of retrieved beams; excluded (incoherent) beams are not counted."""
    from euroflash.beams import INCOHERENT_BEAMS
    beams = tuple(INCOHERENT_BEAMS if exclude_beams is None else exclude_beams) or (-1,)
    with ledger.connect() as db:
        rows = db.execute('''WITH beam_history AS (
            SELECT b.uri,b.item,MAX(CASE WHEN a.status='success' AND r.pilot=0 THEN 1 ELSE 0 END) AS searched
            FROM archive_beams b LEFT JOIN attempts a ON a.item=b.item AND a.stage='classify'
            LEFT JOIN runs r USING(fingerprint) WHERE b.beam NOT IN (%s) GROUP BY b.uri,b.item''' % ','.join('?' * len(beams)) + '''
        ), archive_history AS (
            SELECT uri,COUNT(*) AS beams,SUM(searched) AS searched FROM beam_history GROUP BY uri
        ) SELECT COUNT(*) AS mapped_archives,COALESCE(SUM(beams),0) AS mapped_beam_identities,
            COALESCE(SUM(CASE WHEN beams=searched THEN 1 ELSE 0 END),0) AS archives_with_all_beam_ids_searched,
            COALESCE(SUM(CASE WHEN beams>searched THEN 1 ELSE 0 END),0) AS archives_with_beam_ids_pending
            FROM archive_history''', beams).fetchone()
        result = dict(rows)
        result['retrieved_archives_without_beam_mapping'] = db.execute('''SELECT COUNT(*) FROM inputs i
            WHERE availability='retrieved' AND NOT EXISTS (SELECT 1 FROM archive_beams b WHERE b.uri=i.uri)''').fetchone()[0]
        result['excluded_beams'] = [b for b in beams if b >= 0]
        result['scope'] = 'Beam identity history across recorded production runs, excluding the incoherent beam. Does not assert byte equivalence of archive replicas or scientific validation of every run.'
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    args = parser.parse_args()
    ledger = Ledger(args.ledger)
    print('Linked archive receipts:', reconcile(args.input, ledger))
    print(json.dumps(coverage(ledger), indent=2))


if __name__ == '__main__':
    main()
