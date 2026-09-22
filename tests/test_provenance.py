import json
from pathlib import Path
import tempfile
import unittest

from euroflash.ledger import Ledger
from euroflash.provenance import coverage, reconcile


class ArchiveProvenanceTests(unittest.TestCase):
    def test_archive_id_mapping_and_pilot_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = Ledger(root/'campaign.sqlite')
            uri = 'srm://example.test/lt5_004/1263256/L1263256_SAP001_B005_P000_bf_93e6078f.tar'
            ledger.discover([uri], 'lt5_004', 'test')
            raw = root/'L603682_SAP1_BEAM5_2bit.fits'
            raw.touch()
            marker = root/'archive.extracted.json'
            marker.write_text(json.dumps({'request_id': 1, 'archive': {
                'url': uri.replace('srm:', 'https:'), 'sha256': '0'*64, 'bytes': 1}, 'fits':[str(raw)]}))
            self.assertEqual(reconcile(root, ledger), 1)
            self.assertEqual(reconcile(root, ledger), 1)
            result = coverage(ledger)
            self.assertEqual(result['mapped_archives'], 1)
            self.assertEqual(result['archives_with_beam_ids_pending'], 1)
            with ledger.connect() as db:
                item = db.execute('SELECT item FROM archive_beams').fetchone()[0]
            self.assertEqual(item, 'downsampled_L603682_SAP001_BEAM005_32bit_ff')
            ledger.register_run('pilot', {'pilot': True})
            attempt = ledger.start(item, 'classify', 'pilot', root/'log', [])
            ledger.finish(attempt, [raw])
            self.assertEqual(coverage(ledger)['archives_with_beam_ids_pending'], 1)
            ledger.register_run('production', {'pilot': False})
            attempt = ledger.start(item, 'classify', 'production', root/'log', [])
            ledger.finish(attempt, [raw])
            self.assertEqual(coverage(ledger)['archives_with_all_beam_ids_searched'], 1)
            other = uri.replace('example.test', 'replica.test')
            ledger.discover([other], 'lt5_004', 'test')
            with self.assertRaisesRegex(ValueError, 'Expected one known archive URI'):
                reconcile(root, ledger)


if __name__ == '__main__':
    unittest.main()
