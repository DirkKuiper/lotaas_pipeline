"""Reclaim DM trial directories that can no longer be retried.

The CPU stage removes its trials once it succeeds, so a beam that fails keeps
roughly 4.88 GB of unique payload with periodicity enabled. That is deliberate: it lets the search be
repeated without redoing dedispersion. It was not bounded, so a burst of
failures fills a node disk and every beam running beside them then fails on
a truncated trial.

A trial directory is dead once its beam has been classified successfully
under the same fingerprint, or once its last attempt failed longer ago than
the retention window and no retry is coming. Nothing here reads or writes
science results; it removes only intermediate trials whose successor is
already recorded in the ledger.
"""
import argparse
from pathlib import Path
import shutil
import sqlite3
import time


def directory_bytes(path, seen=None):
    seen = set() if seen is None else seen
    total = 0
    for entry in path.rglob('*'):
        if not entry.is_file():
            continue
        stat = entry.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity not in seen:
            total += stat.st_size
            seen.add(identity)
    return total


def classify(db, item, fingerprint, retention_seconds, now):
    """Why this trial directory may go, or None if it must stay."""
    # A retry now runs independently checkpointed branches before finalization.
    # An old aggregate failure must not permit deleting their live inputs.
    active = db.execute(
        "SELECT 1 FROM attempts WHERE item=? AND fingerprint=? AND status='running' "
        "AND id IN (SELECT MAX(id) FROM attempts WHERE item=? AND fingerprint=? GROUP BY stage)",
        (item, fingerprint, item, fingerprint)).fetchone()
    if active:
        return None
    row = db.execute(
        'SELECT status,finished FROM attempts WHERE item=? AND stage=? AND fingerprint=?'
        ' ORDER BY id DESC LIMIT 1', (item, 'classify', fingerprint)).fetchone()
    if row is None:
        return None  # No attempt recorded: the CPU stage has not run yet.
    status, finished = row
    if status == 'success':
        return 'classified'
    if status == 'failed' and finished and now - finished > retention_seconds:
        age = (now - finished) / 86400
        return f'failed {age:.1f} days ago'
    return None


def survey(work, ledger, retention_seconds, now=None):
    now = time.time() if now is None else now
    found = []
    seen_inodes = set()
    with sqlite3.connect(f'file:{ledger}?mode=ro', uri=True) as db:
        # A node writes work/processed/...; the head collects it under a
        # directory named for the node it came from.
        candidates = set(Path(work).glob('processed/*/*/DM_trials'))
        candidates |= set(Path(work).glob('*/processed/*/*/DM_trials'))
        candidates |= set(Path(work).glob('processed/*/*/Periodic_DM_trials'))
        candidates |= set(Path(work).glob('*/processed/*/*/Periodic_DM_trials'))
        for trials in sorted(candidates):
            if not trials.is_dir():
                continue
            item, fingerprint = trials.parent.parent.name, trials.parent.name
            # Attempts record the full fingerprint; the directory holds its head.
            row = db.execute(
                'SELECT DISTINCT fingerprint FROM attempts WHERE item=? AND fingerprint LIKE ?',
                (item, fingerprint + '%')).fetchone()
            if row is None:
                continue
            reason = classify(db, item, row[0], retention_seconds, now)
            if reason:
                found.append((trials, directory_bytes(trials, seen_inodes), reason))
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True, help='Node or collected work directory')
    p.add_argument('--ledger', type=Path, required=True)
    p.add_argument('--retention-days', type=float, default=7.,
                   help="Keep a failed beam's trials this long, so it can be retried")
    p.add_argument('--apply', action='store_true', help='Delete; otherwise only report')
    a = p.parse_args()
    found = survey(a.work, a.ledger, a.retention_days * 86400)
    total = 0
    for trials, size, reason in found:
        total += size
        print(f'{"removing" if a.apply else "reclaimable"}  {size/1e9:7.2f} GB  {reason:24s}  {trials}')
        if a.apply:
            shutil.rmtree(trials, ignore_errors=True)
    verb = 'Reclaimed' if a.apply else 'Reclaimable'
    beams = len({trials.parent for trials, _, _ in found})
    print(f'{verb}: {total/1e9:.2f} GB across {beams} beams')
    if found and not a.apply:
        print('Re-run with --apply to delete.')


if __name__ == '__main__':
    main()
