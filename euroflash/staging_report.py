"""What each staging window delivered: bytes off tape, files, complete SAPs, throttling.

The campaign driver changes its window of SAP requests in flight at the times
a schedule gives (--staging-schedule). This reads the campaign state and the
ledger and reports every phase of that schedule. Each phase is reported whole
and without its first hours, in which requests submitted under the previous
window are still arriving.

    python3 -m euroflash.staging_report --root CAMPAIGN --ledger LEDGER.sqlite \\
        --schedule SCHEDULE.json [--ramp-hours 6] [--json OUT.json]
"""
import argparse
import calendar
import json
from pathlib import Path
import sqlite3
import statistics
import time


def epoch(value):
    return value if isinstance(value, (int, float)) else calendar.timegm(time.strptime(value, '%Y-%m-%dT%H:%M:%SZ'))


def phases(schedule, now=None):
    now = time.time() if now is None else now
    entries = sorted(json.loads(Path(schedule).read_text())['phases'], key=lambda p: epoch(p['start']))
    rows = []
    for index, phase in enumerate(entries):
        start = epoch(phase['start'])
        end = epoch(entries[index + 1]['start']) if index + 1 < len(entries) else epoch(phase.get('end', now))
        if start < now:
            rows.append(dict(phase, start_unix=start, end_unix=min(end, now)))
    return rows


def measure(state_db, ledger_db, start, end):
    """Staging and retrieval between two times."""
    ledger = sqlite3.connect(f'file:{ledger_db}?mode=ro', uri=True)
    state = sqlite3.connect(f'file:{state_db}?mode=ro', uri=True)
    try:
        # The retrieve attempt is named after the archive tar, as is the receipt's URI.
        sizes = {Path(uri).name.removesuffix('.tar'): size
                 for uri, size in ledger.execute('SELECT uri, bytes FROM archive_receipts')}
        items = [item for (item,) in ledger.execute('''SELECT item FROM attempts WHERE stage='retrieve'
            AND status='success' AND finished >= ? AND finished < ?''', (start, end))]
        retrieved = (len(items), sum(sizes.get(item, 0) for item in items))
        seconds = [r[0] for r in ledger.execute('''SELECT seconds FROM attempts WHERE stage='retrieve'
            AND status='success' AND finished >= ? AND finished < ? AND seconds IS NOT NULL''', (start, end))]
        converted = ledger.execute('''SELECT COUNT(*) FROM attempts WHERE stage='downsample' AND status='success'
            AND finished >= ? AND finished < ?''', (start, end)).fetchone()[0]
        events = dict(state.execute('''SELECT kind, COUNT(*) FROM events WHERE time >= ? AND time < ?
            GROUP BY kind''', (start, end)).fetchall())
    finally:
        ledger.close()
        state.close()
    days = max(end - start, 1) / 86400
    return {'hours': round((end - start) / 3600, 2),
            'files_retrieved': retrieved[0], 'bytes': retrieved[1],
            'tb_per_day': round(retrieved[1] / 1e12 / days, 3),
            'files_per_day': round(retrieved[0] / days, 1),
            'files_converted': converted,
            'saps_complete': events.get('prepared', 0),
            'saps_per_day': round(events.get('prepared', 0) / days, 2),
            'saps_searched': events.get('searched', 0),
            'requests_submitted': events.get('submitted', 0),
            'throttled_429_503': events.get('throttled', 0),
            'restaged': events.get('restage', 0),
            'refused': events.get('refused', 0),
            'files_failed': events.get('file_failed', 0),
            'retrieve_seconds_median': round(statistics.median(seconds), 1) if seconds else None}


def report(root, ledger, schedule, ramp_hours=6.0, now=None):
    root = Path(root)
    out = []
    for phase in phases(schedule, now):
        start, end = phase['start_unix'], phase['end_unix']
        row = {'label': phase.get('label'), 'max_staging_saps': phase['max_staging_saps'],
               'start': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(start)),
               'end': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(end)),
               'whole': measure(root/'campaign-state.sqlite', ledger, start, end)}
        if end - start > ramp_hours * 3600:
            row['after_ramp'] = measure(root/'campaign-state.sqlite', ledger, start + ramp_hours * 3600, end)
        out.append(row)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--ledger', type=Path, required=True)
    p.add_argument('--schedule', type=Path, required=True)
    p.add_argument('--ramp-hours', type=float, default=6.0)
    p.add_argument('--json', type=Path, help='Also write the report here')
    a = p.parse_args()
    rows = report(a.root, a.ledger, a.schedule, a.ramp_hours)
    for row in rows:
        w = row['whole']
        print(f"{row['label'] or '-':12s} window {row['max_staging_saps']:3d}  {row['start']} +{w['hours']:6.1f} h  "
              f"{w['tb_per_day']:6.2f} TB/day  {w['files_per_day']:7.1f} files/day  {w['saps_per_day']:5.2f} SAPs/day  "
              f"429/503 {w['throttled_429_503']}  restaged {w['restaged']}")
        if 'after_ramp' in row:
            r = row['after_ramp']
            print(f"{'':12s} after {a.ramp_hours:.0f} h ramp: {r['tb_per_day']:6.2f} TB/day  "
                  f"{r['files_per_day']:7.1f} files/day  {r['saps_per_day']:5.2f} SAPs/day")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
