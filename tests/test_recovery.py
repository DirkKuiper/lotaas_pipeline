import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from euroflash.recovery import report


class RecoveryReportTests(unittest.TestCase):
    def test_requires_matching_accepted_pulse_and_completed_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            beam = 'downsampled_L603682_SAP001_BEAM005_32bit_ff'
            fingerprint = 'a'*64
            product = root/'processed'/beam/fingerprint[:16]
            product.mkdir(parents=True)
            (root/'run.json').write_text(json.dumps({'fingerprint': fingerprint}))
            (product/'all_detected_candidates.cands').write_text('# header\n26.2 12.48 2465.653 313524 1\n')
            (product/'clustered_candidates.txt').write_text('DM\tS/N\tTime\tSample\tFilter_Width\n26.2\t12.48\t2465.653\t313524\t1\n')
            db = sqlite3.connect(root/'ledger-snapshot.sqlite')
            db.executescript('''CREATE TABLE attempts(id INTEGER, item TEXT, fingerprint TEXT, stage TEXT, status TEXT, seconds REAL);
                CREATE TABLE beam_runs(id INTEGER, output_dir TEXT);
                CREATE TABLE detections(beam_run_id INTEGER, candidate_dm REAL, time_seconds REAL);''')
            db.executemany('INSERT INTO attempts VALUES (?,?,?,?,?,?)',
                           [(i,beam,fingerprint,stage,'success',1.) for i,stage in enumerate(['dedisperse','classify'])])
            db.execute('INSERT INTO beam_runs VALUES (1,?)', (str(product/'candidate_plots'),))
            db.execute('INSERT INTO detections VALUES (1,26.2,2465.653)')
            db.commit()
            target = {'observation':'L603682','sap':1,'beam':5,'target_dm':26.2,'target_time_seconds':2465.653}
            self.assertTrue(report(target, root)['reference_pulse_recovered'])
            db.execute('UPDATE detections SET time_seconds=1000'); db.commit()
            result = report(target, root)
            self.assertEqual(result['threshold_crossings']['matching_rows'], 1)
            self.assertFalse(result['reference_pulse_recovered'])
            db.execute('UPDATE detections SET time_seconds=2465.653')
            db.execute("UPDATE attempts SET status='failed' WHERE stage='classify'"); db.commit()
            self.assertFalse(report(target, root)['reference_pulse_recovered'])
            db.close()


if __name__ == '__main__':
    unittest.main()
