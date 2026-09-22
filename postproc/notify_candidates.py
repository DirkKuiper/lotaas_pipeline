"""Post candidate plots produced by the search to Slack, once each.

This runs on the head node, beside the campaign ledger, and never inside the
search itself. Keeping it here has three consequences that matter:

* the Slack token stays on the head node and is never copied into the source
  snapshot or the image sent to a compute node;
* ``lotaas_reprocessing``, ``pipeline`` and ``euroflash`` are unchanged, so the
  run fingerprint that identifies every finished beam is not disturbed by a
  notification change;
* a notification can be replayed for work that has already finished, because
  the plots and the ledger rows are the durable record.

By default only candidates with a recorded ``candidate`` detection in the
ledger are posted. Injection tests and other synthetic plots are deliberately
kept out of the campaign database, so that rule keeps artificial signals out of
the channel unless they are asked for explicitly.
"""
import argparse
import json
import logging
from pathlib import Path
import re
import sqlite3
import time

from postproc.slack_client import Slack, SlackError

logger = logging.getLogger(__name__)

# classify.py writes f"DM{dm}_Width{width}_SNR{snr}.png"
PLOT = re.compile(r"^DM(?P<dm>[-\d.]+)_Width(?P<width>\d+)_SNR(?P<snr>[-\d.]+)\.png$")
# .../processed/<item>/<fingerprint>/candidate_plots/<plot>.png
DM_TOLERANCE = 0.05
SNR_TOLERANCE = 0.05

SCHEMA = '''
CREATE TABLE IF NOT EXISTS slack_notifications (
    key TEXT PRIMARY KEY, kind TEXT NOT NULL, beam_id TEXT,
    plot_path TEXT, slack_file_id TEXT, channel TEXT NOT NULL, sent REAL NOT NULL);
'''


def connect(ledger):
    db = sqlite3.connect(ledger, timeout=120)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


def beam_of(plot):
    """Recover the beam item name from a candidate plot's location."""
    parts = plot.resolve().parts
    if 'processed' in parts:
        index = len(parts) - 1 - parts[::-1].index('processed')
        if index + 1 < len(parts):
            return parts[index + 1]
    # candidate_plots/<plot>.png directly under an output directory
    return plot.resolve().parent.parent.name


def discover(roots):
    """Return every candidate plot below the given directories, newest last."""
    found = {}
    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError(f'Not a directory: {root}')
        for plot in root.rglob('candidate_plots/*.png'):
            match = PLOT.match(plot.name)
            if not match:
                logger.warning('Skipping unrecognised plot name: %s', plot)
                continue
            item = beam_of(plot)
            record = {'path': plot, 'item': item,
                      'dm': float(match['dm']), 'width': int(match['width']),
                      'snr': float(match['snr']), 'mtime': plot.stat().st_mtime}
            # The same candidate re-searched under a new fingerprint lands in a
            # different directory; it is still one candidate.
            found.setdefault(key_for(record), record)
    return sorted(found.values(), key=lambda r: (r['mtime'], str(r['path'])))


def key_for(record):
    """A stable identity for a candidate, independent of the run fingerprint."""
    return 'candidate|{item}|DM{dm:.3f}|W{width}|SN{snr:.3f}'.format(**record)


def detection_for(db, record):
    """Find the ledger detection row a plot was made from, if it was recorded."""
    rows = db.execute(
        "SELECT d.*, b.observation_date, b.output_dir FROM detections d"
        " LEFT JOIN beam_runs b ON b.id = d.beam_run_id"
        " WHERE d.detection_type = 'candidate' AND d.beam_id LIKE ?"
        "   AND abs(d.candidate_dm - ?) <= ? AND abs(d.snr - ?) <= ? AND d.width_samples = ?",
        (record['item'] + '%', record['dm'], DM_TOLERANCE,
         record['snr'], SNR_TOLERANCE, record['width'])).fetchall()
    return dict(rows[0]) if len(rows) == 1 else None


def candidate_message(record, detection):
    lines = ['*New candidate detected!*', record['item']]
    if detection and detection.get('observation_date') not in (None, 'Unknown'):
        lines.append(f"Observed: {detection['observation_date']}")
    lines.append(f"DM = {record['dm']:.2f} pc/cm³")
    lines.append(f"S/N = {record['snr']:.2f}")
    lines.append(f"Width = {record['width']} samples")
    if detection and detection.get('time_seconds') is not None:
        lines.append(f"Time = {detection['time_seconds']:.3f} s (sample {detection['sample_number']})")
    if detection and detection.get('classification_probability') is not None:
        lines.append(f"FETCH max probability = {detection['classification_probability']:.2f}")
    if detection is None:
        lines.append('_No matching detection in the campaign ledger; not a recorded campaign candidate._')
    return '\n'.join(lines)


def redetections(db):
    """Best known-pulsar redetection per (beam, pulsar), as the classifier reports."""
    return [dict(r) for r in db.execute(
        "SELECT beam_id, pulsar_name, MAX(snr) AS snr, candidate_dm, width_samples"
        " FROM detections WHERE detection_type = 'known_pulsar' AND pulsar_name IS NOT NULL"
        " GROUP BY beam_id, pulsar_name")]


def sent_keys(db):
    return {r['key'] for r in db.execute('SELECT key FROM slack_notifications')}


def record_sent(db, key, kind, beam_id, plot_path, file_id, channel):
    with db:
        db.execute('INSERT OR REPLACE INTO slack_notifications VALUES (?,?,?,?,?,?,?)',
                   (key, kind, beam_id, plot_path, file_id, channel, time.time()))


def run_once(slack, db, roots, limit, dry_run, unrecorded, include_redetections):
    already = sent_keys(db)
    posted, skipped, pending = [], 0, 0
    for record in discover(roots):
        key = key_for(record)
        if key in already:
            continue
        detection = detection_for(db, record)
        if detection is None and not unrecorded:
            logger.info('No ledger detection for %s; use --unrecorded to post it anyway', record['path'])
            skipped += 1
            continue
        if len(posted) >= limit:
            pending += 1
            continue
        message = candidate_message(record, detection)
        if dry_run:
            print('--- would post', record['path'], '---')
            print(message)
            posted.append(key)
            continue
        file_id = slack.upload(record['path'], title=record['path'].name, comment=message)
        record_sent(db, key, 'candidate', record['item'], str(record['path']), file_id, slack.channel)
        logger.info('Posted %s', record['path'])
        posted.append(key)

    announced = 0
    if include_redetections:
        for row in redetections(db):
            key = 'redetection|{beam_id}|{pulsar_name}|SN{snr:.3f}'.format(**row)
            if key in already:
                continue
            text = (f"*Redetected:* {row['pulsar_name']}  DM={row['candidate_dm']:.2f}"
                    f"  highest S/N={row['snr']:.2f}  Width={row['width_samples']}"
                    f"  ({row['beam_id']})")
            if dry_run:
                print('--- would post ---')
                print(text)
            else:
                slack.post_message(text)
                record_sent(db, key, 'redetection', row['beam_id'], None, None, slack.channel)
            announced += 1
    return {'posted': len(posted), 'redetections': announced,
            'skipped_unrecorded': skipped, 'held_by_limit': pending}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('roots', type=Path, nargs='+', help='Result directories to scan for candidate_plots/')
    p.add_argument('--ledger', type=Path, required=True, help='Campaign SQLite database')
    p.add_argument('--channel', help='Override the configured channel ID')
    p.add_argument('--limit', type=int, default=25, help='Maximum plots to post per pass (default: 25)')
    p.add_argument('--dry-run', action='store_true', help='Print what would be posted without sending')
    p.add_argument('--unrecorded', action='store_true',
                   help='Also post plots with no matching campaign detection, labelled as such')
    p.add_argument('--redetections', action='store_true', help='Also announce known-pulsar redetections')
    p.add_argument('--watch', action='store_true', help='Keep scanning while a search runs')
    p.add_argument('--interval', type=int, default=60, help='Seconds between passes with --watch')
    p.add_argument('--no-verify-tls', action='store_true',
                   help='Only for hosts with a broken certificate store')
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    if a.limit < 1:
        p.error('--limit must be positive')
    if a.interval < 5:
        p.error('--interval must be at least 5 seconds')

    slack = Slack(channel=a.channel, verify=not a.no_verify_tls)
    if not slack.enabled:
        p.error('No Slack token configured; set SLACK_BOT_TOKEN or write ~/.config/lotaas/slackrc')
    if not a.dry_run:
        print(json.dumps(slack.check()), flush=True)
    db = connect(a.ledger)
    try:
        while True:
            try:
                result = run_once(slack, db, a.roots, a.limit, a.dry_run, a.unrecorded, a.redetections)
                print(json.dumps(result), flush=True)
            except SlackError as error:
                # A Slack outage must not end a watch that is tracking a live run.
                logger.error('%s', error)
                if not a.watch:
                    raise
            if not a.watch:
                break
            time.sleep(a.interval)
    finally:
        db.close()


if __name__ == '__main__':
    main()
