"""Durable per-stage attempts and outcomes; SQLite rollback journal supports NFS."""
from contextlib import contextmanager
import json
from pathlib import Path
import re
import socket
import sqlite3
import time


class Ledger:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS runs (
                    fingerprint TEXT PRIMARY KEY, pilot INTEGER NOT NULL,
                    metadata TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS inputs (
                    uri TEXT PRIMARY KEY, project TEXT NOT NULL, discovered REAL NOT NULL,
                    source TEXT NOT NULL, availability TEXT NOT NULL DEFAULT 'unknown');
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY, item TEXT NOT NULL, stage TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, status TEXT NOT NULL,
                    started REAL NOT NULL, finished REAL, seconds REAL,
                    host TEXT NOT NULL, device TEXT, log TEXT, error TEXT,
                    outputs TEXT, command TEXT);
                CREATE INDEX IF NOT EXISTS attempts_lookup ON attempts(item,stage,fingerprint);
                CREATE TABLE IF NOT EXISTS archive_receipts (
                    uri TEXT PRIMARY KEY REFERENCES inputs(uri), request_id INTEGER,
                    webdav_url TEXT NOT NULL, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL,
                    receipt_path TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS archive_beams (
                    uri TEXT NOT NULL REFERENCES archive_receipts(uri), raw_path TEXT NOT NULL,
                    item TEXT NOT NULL, observation TEXT NOT NULL, sap INTEGER NOT NULL,
                    beam INTEGER NOT NULL, PRIMARY KEY(uri,raw_path));
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=120)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def register_run(self, fingerprint, metadata):
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO runs VALUES (?,?,?,?)',
                       (fingerprint, int(metadata.get('pilot', True)), json.dumps(metadata, sort_keys=True), time.time()))

    def discover(self, urls, project, source):
        with self.connect() as db:
            db.executemany('INSERT OR IGNORE INTO inputs(uri,project,discovered,source) VALUES (?,?,?,?)',
                           [(u, project, time.time(), source) for u in urls])

    def record_archive(self, uri, request_id, receipt, fits, receipt_path):
        """Link an archive receipt to beam identities read from extracted names.

        Beam identity links support coverage reconciliation; they do not assert
        that two archive replicas have identical bytes or processing history.
        """
        beams = []
        for path in fits:
            match = re.search(r'(L\d+)_SAP(\d+)_(?:BEAM|B)(\d+)', Path(path).name)
            if match:
                obs, sap, beam = match[1], int(match[2]), int(match[3])
                item = f'downsampled_{obs}_SAP{sap:03d}_BEAM{beam:03d}_32bit_ff'
                beams.append((uri, str(Path(path).resolve()), item, obs, sap, beam))
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO archive_receipts VALUES (?,?,?,?,?,?)',
                       (uri, request_id, receipt['url'], receipt['sha256'], receipt['bytes'], str(receipt_path)))
            db.execute('DELETE FROM archive_beams WHERE uri=?', (uri,))
            db.executemany('INSERT INTO archive_beams VALUES (?,?,?,?,?,?)', beams)
            db.execute("UPDATE inputs SET availability='retrieved' WHERE uri=?", (uri,))

    def completed(self, item, stage, fingerprint):
        with self.connect() as db:
            row = db.execute('SELECT outputs FROM attempts WHERE item=? AND stage=? AND fingerprint=? AND status=? ORDER BY id DESC LIMIT 1',
                             (item, stage, fingerprint, 'success')).fetchone()
        if not row:
            return False
        outputs = json.loads(row['outputs'])
        return bool(outputs) and all(Path(p).is_file() and Path(p).stat().st_size == size for p, size in outputs.items())

    def start(self, item, stage, fingerprint, log, command, device=None):
        with self.connect() as db:
            cur = db.execute('INSERT INTO attempts(item,stage,fingerprint,status,started,host,device,log,command) VALUES (?,?,?,?,?,?,?,?,?)',
                             (item, stage, fingerprint, 'running', time.time(), socket.gethostname(), device, str(log), json.dumps(command)))
            return cur.lastrowid

    def finish(self, attempt, outputs=(), error=None):
        files = {str(p): Path(p).stat().st_size for p in outputs} if error is None else {}
        now = time.time()
        with self.connect() as db:
            db.execute('UPDATE attempts SET status=?,finished=?,seconds=?-started,error=?,outputs=? WHERE id=?',
                       ('failed' if error else 'success', now, now, error, json.dumps(files), attempt))

    def summary(self):
        with self.connect() as db:
            return {'inputs': [dict(r) for r in db.execute('SELECT project,availability,COUNT(*) AS count FROM inputs GROUP BY project,availability')],
                    'attempts': [dict(r) for r in db.execute('SELECT stage,status,COUNT(*) AS count,SUM(seconds) AS seconds FROM attempts GROUP BY stage,status')],
                    'errors': [dict(r) for r in db.execute("SELECT item,stage,error,log FROM attempts WHERE status='failed' ORDER BY id DESC LIMIT 20")]}
