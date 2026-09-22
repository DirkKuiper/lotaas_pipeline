"""Slack candidate notification: credentials, matching, and send-once behaviour.

No test contacts Slack. The upload test drives the real client against a stub
transport so the three-step external upload order is still checked.
"""
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from db.initialize_db import initialize_database
from euroflash.ledger import Ledger
from postproc import notify_candidates as notify
from postproc.slack_client import Slack, SlackError, load_credentials

BEAM = 'downsampled_L559289_SAP000_BEAM025_32bit_ff'


def make_plot(root, fingerprint, name, item=BEAM):
    path = root/'efc-gpu-01'/'processed'/item/fingerprint/'candidate_plots'/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + b'0'*64)
    return path


def make_ledger(root, detections=()):
    path = root/'campaign.sqlite'
    Ledger(path)
    initialize_database(str(path))
    db = sqlite3.connect(path)
    run = db.execute("INSERT INTO beam_runs(beam_id,observation_date,outcome) VALUES (?,?,?)",
                     (BEAM+'.fil', '2016-11-21 03:47:00.000', 'classified')).lastrowid
    for dm, snr, width, kind, name, probability in detections:
        db.execute("INSERT INTO detections(beam_id,candidate_dm,snr,width_samples,detection_type,"
                   "pulsar_name,classification_probability,beam_run_id,time_seconds,sample_number)"
                   " VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (BEAM+'.fil', dm, snr, width, kind, name, probability, run, 2130.971121, 270967))
    db.commit(); db.close()
    return path


class FakeSlack:
    channel = 'C0TEST'
    enabled = True

    def __init__(self):
        self.uploads = []
        self.messages = []

    def upload(self, path, title=None, comment=None):
        self.uploads.append((Path(path), comment))
        return 'F%03d' % len(self.uploads)

    def post_message(self, text):
        self.messages.append(text)
        return '1.0'


class CredentialTests(unittest.TestCase):
    def test_environment_wins_over_config_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary)/'slackrc'
            config.write_text('bot_token = xoxb-from-file\nchannel_id = C0FILE\n')
            with mock.patch.dict(os.environ, {'SLACK_BOT_TOKEN': 'xoxb-from-env',
                                              'LOTAAS_SLACK_CONFIG': str(config)}, clear=True):
                self.assertEqual(load_credentials(), ('xoxb-from-env', 'C0FILE'))
            with mock.patch.dict(os.environ, {'LOTAAS_SLACK_CONFIG': str(config)}, clear=True):
                self.assertEqual(load_credentials(), ('xoxb-from-file', 'C0FILE'))

    def test_missing_configuration_disables_rather_than_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(os.environ, {'LOTAAS_SLACK_CONFIG': str(Path(temporary)/'absent')}, clear=True):
                self.assertEqual(load_credentials(), (None, None))
                self.assertFalse(Slack().enabled)

    def test_a_non_bot_token_is_rejected(self):
        with mock.patch.dict(os.environ, {'SLACK_BOT_TOKEN': 'xoxp-user-token'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'xoxb-'):
                load_credentials()


class UploadTests(unittest.TestCase):
    def test_external_upload_reserves_sends_then_completes(self):
        calls = []

        def transport(request, timeout=None, context=None):
            url = request.full_url
            calls.append((url, request.data))
            if 'getUploadURLExternal' in url:
                body = {'ok': True, 'upload_url': 'https://files.test/put', 'file_id': 'F1'}
            elif url == 'https://files.test/put':
                return mock.MagicMock(__enter__=lambda s: mock.Mock(read=lambda: b'OK - 72'),
                                      __exit__=lambda *a: None)
            else:
                body = {'ok': True, 'files': [{'id': 'F1'}]}
            payload = json.dumps(body).encode()
            return mock.MagicMock(__enter__=lambda s: mock.Mock(read=lambda: payload),
                                  __exit__=lambda *a: None)

        with tempfile.TemporaryDirectory() as temporary:
            plot = Path(temporary)/'DM131.6_Width1_SNR8.001.png'
            plot.write_bytes(b'x'*72)
            slack = Slack(token='xoxb-test', channel='C0TEST')
            with mock.patch('urllib.request.urlopen', side_effect=transport):
                self.assertEqual(slack.upload(plot, comment='hello'), 'F1')
        self.assertIn('files.getUploadURLExternal', calls[0][0])
        self.assertEqual(calls[1][0], 'https://files.test/put')
        self.assertEqual(calls[1][1], b'x'*72)
        self.assertIn('files.completeUploadExternal', calls[2][0])
        completion = json.loads(calls[2][1])
        self.assertEqual(completion['channel_id'], 'C0TEST')
        self.assertEqual(completion['initial_comment'], 'hello')

    def test_an_empty_plot_is_never_uploaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            plot = Path(temporary)/'DM1_Width1_SNR9.png'
            plot.touch()
            with self.assertRaisesRegex(SlackError, 'empty'):
                Slack(token='xoxb-test', channel='C0TEST').upload(plot)


class NotificationTests(unittest.TestCase):
    def test_recorded_candidate_is_posted_once_with_its_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_plot(root, '3739f1e04141461d', 'DM131.6_Width1_SNR8.001.png')
            db = notify.connect(make_ledger(root, [(131.6, 8.001, 1, 'candidate', None, 0.9936)]))
            slack = FakeSlack()
            first = notify.run_once(slack, db, [root], 25, False, False, False)
            self.assertEqual(first['posted'], 1)
            comment = slack.uploads[0][1]
            self.assertIn('DM = 131.60 pc/cm', comment)
            self.assertIn('S/N = 8.00', comment)
            self.assertIn('FETCH max probability = 0.99', comment)
            self.assertIn('2130.971 s (sample 270967)', comment)
            # A second pass, and a re-search under a new fingerprint, are the
            # same candidate and must not be announced again.
            self.assertEqual(notify.run_once(slack, db, [root], 25, False, False, False)['posted'], 0)
            make_plot(root, 'bc31bf0fdf2e07a6', 'DM131.6_Width1_SNR8.001.png')
            self.assertEqual(notify.run_once(slack, db, [root], 25, False, False, False)['posted'], 0)
            self.assertEqual(len(slack.uploads), 1)
            db.close()

    def test_unrecorded_plots_are_held_back_then_labelled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_plot(root, '0cc9ca24a70a055b', 'DM83.2_Width20_SNR37.817.png',
                      item='INJECTED_L559289_SAP000_BEAM013_ff')
            db = notify.connect(make_ledger(root))
            slack = FakeSlack()
            held = notify.run_once(slack, db, [root], 25, False, False, False)
            self.assertEqual((held['posted'], held['skipped_unrecorded']), (0, 1))
            self.assertFalse(slack.uploads)
            allowed = notify.run_once(slack, db, [root], 25, False, True, False)
            self.assertEqual(allowed['posted'], 1)
            self.assertIn('not a recorded campaign candidate', slack.uploads[0][1])
            db.close()

    def test_a_pass_never_exceeds_its_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            detections = []
            for index in range(5):
                make_plot(root, 'aaaa', f'DM{100+index}.0_Width1_SNR9.0.png')
                detections.append((100.0+index, 9.0, 1, 'candidate', None, 0.9))
            db = notify.connect(make_ledger(root, detections))
            slack = FakeSlack()
            first = notify.run_once(slack, db, [root], 2, False, False, False)
            self.assertEqual((first['posted'], first['held_by_limit']), (2, 3))
            self.assertEqual(notify.run_once(slack, db, [root], 2, False, False, False)['posted'], 2)
            self.assertEqual(notify.run_once(slack, db, [root], 2, False, False, False)['posted'], 1)
            self.assertEqual(len(slack.uploads), 5)
            db.close()

    def test_dry_run_sends_nothing_and_stays_repeatable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_plot(root, 'aaaa', 'DM131.6_Width1_SNR8.001.png')
            db = notify.connect(make_ledger(root, [(131.6, 8.001, 1, 'candidate', None, 0.99)]))
            slack = FakeSlack()
            self.assertEqual(notify.run_once(slack, db, [root], 25, True, False, False)['posted'], 1)
            self.assertFalse(slack.uploads)
            self.assertEqual(notify.run_once(slack, db, [root], 25, True, False, False)['posted'], 1)
            db.close()

    def test_best_redetection_per_pulsar_is_announced_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            db = notify.connect(make_ledger(root, [
                (26.2, 11.0, 2, 'known_pulsar', 'J1713+0747', None),
                (26.2, 19.5, 2, 'known_pulsar', 'J1713+0747', None)]))
            slack = FakeSlack()
            self.assertEqual(notify.run_once(slack, db, [root], 25, False, False, True)['redetections'], 1)
            self.assertIn('highest S/N=19.50', slack.messages[0])
            self.assertEqual(notify.run_once(slack, db, [root], 25, False, False, True)['redetections'], 0)
            db.close()

    def test_beam_name_comes_from_the_processed_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            plot = make_plot(Path(temporary), 'aaaa', 'DM1.0_Width1_SNR9.0.png')
            self.assertEqual(notify.beam_of(plot), BEAM)

    def test_an_ambiguous_detection_match_is_not_guessed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_plot(root, 'aaaa', 'DM131.6_Width1_SNR8.001.png')
            # Two ledger rows inside the matching tolerance: refuse to pick one.
            db = notify.connect(make_ledger(root, [(131.6, 8.001, 1, 'candidate', None, 0.99),
                                                   (131.61, 8.002, 1, 'candidate', None, 0.88)]))
            result = notify.run_once(FakeSlack(), db, [root], 25, False, False, False)
            self.assertEqual((result['posted'], result['skipped_unrecorded']), (0, 1))
            db.close()


if __name__ == '__main__':
    unittest.main()
