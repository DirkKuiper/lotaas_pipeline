"""Keep a whole campaign moving: stage, retrieve, convert, flatfield, dispatch.

Staging one request at a time and preparing it by hand left the tape system,
the network and the cluster idle for most of each day. This driver keeps a
rolling window of SAP-sized StageIT requests in flight and processes every
file the moment dCache holds it on disk:

    pending -> staging -> prepared -> dispatched -> searched     (per SAP)
                                   +-> partial -> prepared -> ...  (searched a few beams short)
    pending -> requested -> online -> working -> converted -> searched | kept   (per file)

Each file is downloaded, extracted, converted to a 32-bit filterbank and its
tar and PSRFITS deleted before the next is taken; the download receipt and
extraction marker keep its provenance. A SAP is flatfielded once all its
beams are converted, or once it has waited `--partial-after-hours` a few beams
short: then its flatfield is saved, the beams present are searched, and the
late ones are flatfielded with the saved flatfield and searched as they come
(state 'partial'). Its unflattened filterbanks are removed, and it is optionally
dispatched to the GPU nodes with `euroflash.cluster`. Once searched, a beam's
flatfielded filterbank is removed unless something was found in it: a
candidate FETCH accepted, or a periodic fold that the rest of its observation
and of the survey does not explain (euroflash.findings). Those are kept for
review, and released again when a later SAP of the observation explains them.

StageIT's "online" is only a hint: request 991619 was reported online while
dCache held every file on tape. A file is fetched only when dCache itself
reports it on disk. Everything is bounded: requests in flight, prepared SAPs
awaiting a search, and free disk. State lives in SQLite beside the data, so a
restart resumes where it stopped. Create a file named STOP in the campaign
root, or send SIGTERM, to stop after the current step.
"""
import argparse
import calendar
import functools
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
CENTRAL = range(13, 74)
# A SAP searched before its last beams arrived keeps its flatfield here for them.
MEAN = 'flatfield-mean.npy'
OUTSTANDING = ('pending', 'requested', 'online', 'working')
FINAL = {'success', 'failed', 'aborted', 'partial success'}
NAME = re.compile(r'^L(\d+)_SAP(\d+)_B(\d+)_')


def basename(url):
    return Path(urlsplit(url).path).name


def parse_inventory(path):
    """SAP-grouped archive files from a list of SRM URLs, in a stable order."""
    rows = []
    for line in Path(path).read_text().splitlines():
        surl = line.strip()
        if not surl or surl.startswith('#'):
            continue
        match = NAME.match(basename(surl))
        if not match:
            continue
        rows.append({'surl': surl, 'name': basename(surl), 'archive_obs': match[1],
                     'sap': int(match[2]), 'beam': int(match[3])})
    rows.sort(key=lambda r: (r['archive_obs'], r['sap'], r['beam']))
    return rows


def sap_key(archive_obs, sap):
    return f'L{archive_obs}_SAP{int(sap):03d}'


class State:
    """Durable campaign state. One writer at a time; every call is a short transaction."""

    def __init__(self, path):
        self.path = str(path)
        self.lock = threading.Lock()
        with self.db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS saps (
                    key TEXT PRIMARY KEY, position INTEGER NOT NULL, files INTEGER NOT NULL,
                    state TEXT NOT NULL, detail TEXT, sap_dir TEXT, run_name TEXT, updated REAL);
                CREATE TABLE IF NOT EXISTS files (
                    surl TEXT PRIMARY KEY, name TEXT NOT NULL, sap_key TEXT NOT NULL,
                    beam INTEGER NOT NULL, state TEXT NOT NULL, request_id INTEGER,
                    submissions INTEGER NOT NULL DEFAULT 0, failures INTEGER NOT NULL DEFAULT 0,
                    locality TEXT, checked REAL, detail TEXT, fil TEXT, updated REAL);
                CREATE INDEX IF NOT EXISTS files_by_state ON files(state, sap_key);
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY, sap_key TEXT, submitted REAL, files INTEGER,
                    status TEXT, checked REAL, final_seen REAL);
                CREATE TABLE IF NOT EXISTS events (
                    time REAL NOT NULL, kind TEXT NOT NULL, subject TEXT, detail TEXT);
            ''')

    @contextmanager
    def db(self):
        with self.lock:
            db = sqlite3.connect(self.path, timeout=120)
            db.row_factory = sqlite3.Row
            try:
                with db:
                    yield db
            finally:
                db.close()

    def event(self, kind, subject, detail=''):
        with self.db() as db:
            db.execute('INSERT INTO events VALUES (?,?,?,?)', (time.time(), kind, subject, str(detail)[:2000]))

    def load(self, rows, missing_central, exclude_beams=(), allowed_missing=0):
        """Queue the inventory. A SAP whose archive lacks more than allowed_missing
        central beams is 'incomplete' and never staged; one that lacks fewer is
        searched without them."""
        now = time.time()
        with self.db() as db:
            saps = {}
            for row in rows:
                saps.setdefault(sap_key(row['archive_obs'], row['sap']), []).append(row)
            for position, (key, files) in enumerate(sorted(saps.items())):
                missing = missing_central(files)
                incomplete = len(missing) > allowed_missing
                db.execute('INSERT OR IGNORE INTO saps(key,position,files,state,detail,updated) VALUES (?,?,?,?,?,?)',
                           (key, position, len(files), 'incomplete' if incomplete else 'pending',
                            f'missing central beams {missing}' if missing else None, now))
                if missing and not incomplete:
                    # Queued as incomplete under a stricter limit: stage it now.
                    db.execute("UPDATE saps SET state='pending',updated=? WHERE key=? AND state='incomplete'", (now, key))
                db.executemany('INSERT OR IGNORE INTO files(surl,name,sap_key,beam,state,updated) VALUES (?,?,?,?,?,?)',
                               [(f['surl'], f['name'], key, f['beam'],
                                 'excluded' if f['beam'] in exclude_beams else 'pending', now) for f in files])
            if exclude_beams:
                # Files queued before the exclusion are no longer staged or fetched.
                db.execute("UPDATE files SET state='excluded',detail='incoherent beam',updated=? WHERE beam IN (%s) "
                           "AND state IN ('pending','requested','online')" % ','.join('?' * len(exclude_beams)),
                           (now, *exclude_beams))

    def set_file(self, surl, **values):
        values['updated'] = time.time()
        with self.db() as db:
            db.execute('UPDATE files SET ' + ','.join(f'{k}=?' for k in values) + ' WHERE surl=?',
                       [*values.values(), surl])

    def set_sap(self, key, **values):
        values['updated'] = time.time()
        with self.db() as db:
            db.execute('UPDATE saps SET ' + ','.join(f'{k}=?' for k in values) + ' WHERE key=?',
                       [*values.values(), key])

    def rows(self, sql, *args):
        with self.db() as db:
            return [dict(r) for r in db.execute(sql, args)]

    def counts(self):
        with self.db() as db:
            return {'saps': dict(db.execute('SELECT state,COUNT(*) FROM saps GROUP BY state').fetchall()),
                    'files': dict(db.execute('SELECT state,COUNT(*) FROM files GROUP BY state').fetchall())}


class AdoptedProcess:
    """poll()/wait() for a cluster run started by an earlier driver process."""

    def __init__(self, pid, run_name, node=None):
        self.pid, self.run_name, self.node, self.returncode = pid, run_name, node, None

    def poll(self):
        try:
            command = Path(f'/proc/{self.pid}/cmdline').read_bytes().split(b'\0')
            if b'euroflash.cluster' in command and self.run_name.encode() in command:
                return None
        except OSError:
            pass
        self.returncode = -1          # unknown: the results decide, as for any run
        return self.returncode

    def wait(self):
        while self.poll() is None:
            time.sleep(10)
        return self.returncode


def running_dispatch(run_name):
    """The live euroflash.cluster process for this run, if any."""
    if not run_name:
        return None
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc/'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if b'euroflash.cluster' in argv and b'--run-name' in argv:
            i = argv.index(b'--run-name')
            if i + 1 < len(argv) and argv[i + 1].decode(errors='replace') == run_name:
                node = argv[argv.index(b'--nodes') + 1].decode() if b'--nodes' in argv else None
                return AdoptedProcess(int(proc.name), run_name, node)
    return None


def staging_window(schedule, default, now=None):
    """SAP requests allowed in flight now, from a schedule of timed phases.

    The schedule is JSON: {"phases": [{"start": "2026-09-24T16:00:00Z",
    "max_staging_saps": 16, "label": "window-16"}, ...]}. The latest phase
    that has started applies; before the first, the command-line default.
    """
    if not schedule:
        return default, None
    try:
        phases = json.loads(Path(schedule).read_text())['phases']
    except (OSError, ValueError, KeyError):
        return default, None
    now = time.time() if now is None else now
    current = None
    for phase in phases:
        start = phase['start']
        start = start if isinstance(start, (int, float)) else calendar.timegm(
            time.strptime(start, '%Y-%m-%dT%H:%M:%SZ'))
        if start <= now and (current is None or start >= current[0]):
            current = (start, phase)
    if current is None:
        return default, None
    return int(current[1]['max_staging_saps']), current[1].get('label')


def flattened(path):
    """The flatfielded filterbank beside a converted one."""
    path = Path(path)
    return path.with_name(path.stem + '_ff.fil')


def missing_central_beams(files):
    present = {f['beam'] for f in files}
    return [b for b in CENTRAL if b not in present]


class Campaign:
    def __init__(self, options, api=None, locate=None, runner=None):
        self.o = options
        self.root = Path(options.root).resolve()
        for name in ('downloads', 'extracted', 'prepared', 'logs', 'dispatch', 'results'):
            (self.root/name).mkdir(parents=True, exist_ok=True)
        self.state = State(self.root/'campaign-state.sqlite')
        from euroflash.ledger import Ledger
        self.ledger = Ledger(options.ledger)
        if api is None:
            from staging.client import StageIT
            api = StageIT()
        if locate is None:
            from staging.dcache import locality as locate
        self.api, self.locate = api, locate
        self.runner = runner or self._runner()
        self.manifests = {}
        self.manifest_lock = threading.Lock()
        self.flatfield_pool = ThreadPoolExecutor(max_workers=max(1, getattr(options, 'flatfield_workers', 2)))
        self.flatfield_jobs = {}
        self.throttle_until = 0.
        # Cluster runs in flight by run name. Each holds one GPU node until the
        # node's own stages end (<run>/<node>.gpu-done); a CPU node may still be
        # classifying and folding the batch after that.
        self.dispatches = {}
        self.dispatch_failures = {}
        self.dispatch_retry_after = {}
        self.window = (getattr(options, 'max_staging_saps', 8), None)
        self.stopping = False

    def _runner(self):
        from euroflash.run import Runner, fingerprint
        args = SimpleNamespace(work=self.root/'prepared', ledger=Path(self.o.ledger).resolve(),
                               settings=Path(self.o.settings), image=Path(self.o.image),
                               input=self.root/'extracted', pilot=False, preprocess_workers=1,
                               convert_only=False, delete_raw=True)
        runner = Runner(args)
        runner.fp = fingerprint(runner.settings, runner.image,
                                {'pilot': False, 'max_samples': None, 'backend': 'gpu', 'prepared': False})
        runner.ledger.register_run(runner.fp, {'pilot': False, 'campaign': str(self.root),
                                               'settings': str(runner.settings), 'image': str(runner.image)})
        return runner

    # ------------------------------------------------------------ staging
    def staging_limit(self):
        window = staging_window(getattr(self.o, 'staging_schedule', None), self.o.max_staging_saps)
        if window != self.window:
            self.state.event('staging_window', window[1] or 'default', f'{self.window[0]} -> {window[0]} SAPs in flight')
            self.window = window
        return window[0]

    def admit(self):
        """Start staging further SAPs while every bound allows it."""
        free_tb = shutil.disk_usage(self.root).free / 1e12
        staging = self.state.rows("SELECT COUNT(*) AS n FROM saps WHERE state='staging'")[0]['n']
        waiting = self.state.rows("SELECT COUNT(*) AS n FROM saps WHERE state IN ('prepared','dispatched')")[0]['n']
        limit = self.staging_limit()
        target = getattr(self.o, 'staging_files_target', 0)
        outstanding = self.state.rows("""SELECT COUNT(*) AS n FROM files f JOIN saps s ON s.key=f.sap_key
            WHERE s.state IN ('staging','partial') AND f.state IN ('pending','requested','online','working')""")[0]['n']
        admitted = []
        while (staging < limit and waiting < self.o.max_prepared_saps
               and free_tb > self.o.min_free_tb and not self.stopping
               and (not target or outstanding < target)):
            candidates = self.state.rows("SELECT key FROM saps WHERE state='pending' ORDER BY position LIMIT 1")
            if self.o.only_sap:
                candidates = self.state.rows(
                    "SELECT key FROM saps WHERE state='pending' AND key IN (%s) ORDER BY position LIMIT 1"
                    % ','.join('?' * len(self.o.only_sap)), *self.o.only_sap)
            if not candidates:
                break
            key = candidates[0]['key']
            self.state.set_sap(key, state='staging', detail=None)
            admitted.append(key)
            staging += 1
            outstanding += self.state.rows("SELECT COUNT(*) AS n FROM files WHERE sap_key=? AND state='pending'", key)[0]['n']
        return admitted

    def request(self):
        """Submit one StageIT request per staging SAP for its unrequested files."""
        submitted = []
        for sap in self.state.rows("SELECT key FROM saps WHERE state IN ('staging','partial') ORDER BY position"):
            files = self.state.rows("SELECT surl,submissions FROM files WHERE sap_key=? AND state='pending'", sap['key'])
            if not files:
                continue
            try:
                request_id = int(self.api.submit([f['surl'] for f in files]))
            except Exception as error:
                self.state.event('submit_failed', sap['key'], error)
                continue
            now = time.time()
            with self.state.db() as db:
                db.execute('INSERT OR REPLACE INTO requests(id,sap_key,submitted,files,status) VALUES (?,?,?,?,?)',
                           (request_id, sap['key'], now, len(files), 'submitted'))
                db.executemany("UPDATE files SET state='requested',request_id=?,submissions=submissions+1,updated=? WHERE surl=?",
                               [(request_id, now, f['surl']) for f in files])
            self.state.event('submitted', sap['key'], f'request {request_id}: {len(files)} files')
            submitted.append(request_id)
        return submitted

    def manifest(self, request_id, refresh=False):
        with self.manifest_lock:
            cached = self.manifests.get(request_id)
            if cached and not refresh and time.time() - cached[0] < 6 * 3600:
                return cached[1]
        manifest = self.api.downloads(request_id)
        with self.manifest_lock:
            self.manifests[request_id] = (time.time(), manifest)
        return manifest

    def _token(self, manifest, surl):
        from staging.client import tokens_for_url
        urls = {basename(u): u for u in manifest['urls']}
        url = urls.get(basename(surl))
        if url is None:
            raise KeyError(f'{basename(surl)} is not in the request manifest')
        return url, tokens_for_url(manifest, url)

    def refresh(self):
        """Poll requests; a file becomes 'online' only when dCache says so."""
        now = time.time()
        for request in self.state.rows('''SELECT DISTINCT r.* FROM requests r JOIN files f ON f.request_id=r.id
                                          WHERE f.state='requested' ORDER BY r.submitted'''):
            try:
                status = self.api.status(request['id'])
            except Exception as error:
                self.state.event('status_failed', request['sap_key'], error)
                continue
            current = str(status.get('currentStatus', '')).lower()
            response = json.loads(status.get('response') or '{}')
            online = {basename(u) for u in response.get('online') or []}
            errors = response.get('errors') or {}
            error_names = {basename(k) for k in (errors if isinstance(errors, (dict, list)) else [])
                           if isinstance(k, str)}
            final_seen = request['final_seen'] or (now if current in FINAL else None)
            with self.state.db() as db:
                db.execute('UPDATE requests SET status=?,checked=?,final_seen=? WHERE id=?',
                           (current, now, final_seen, request['id']))
            files = self.state.rows("SELECT * FROM files WHERE request_id=? AND state='requested'", request['id'])
            for f in files:
                if f['name'] in error_names:
                    self.retry(f, f'StageIT reported an error for this file (request {request["id"]})')
            files = [f for f in files if f['name'] not in error_names]
            hinted = [f for f in files if f['name'] in online or current in FINAL]
            due = [f for f in hinted if not f['checked'] or now - f['checked'] >= self.o.locality_interval]
            if due:
                try:
                    manifest = self.manifest(request['id'])
                except Exception as error:
                    self.state.event('manifest_failed', request['sap_key'], error)
                    continue
                for f in due:
                    try:
                        _, tokens = self._token(manifest, f['surl'])
                    except (KeyError, ValueError) as error:
                        self.retry(f, error)
                        continue
                    where = self.locate(f['surl'], tokens[0])
                    if where in ('ONLINE', 'ONLINE_AND_NEARLINE'):
                        self.state.set_file(f['surl'], state='online', locality=where, checked=now)
                    else:
                        self.state.set_file(f['surl'], locality=where, checked=now)
            # Reported done but still on tape, or never finishing: ask again.
            stale = ((final_seen and now - final_seen > self.o.restage_after_hours * 3600)
                     or now - request['submitted'] > self.o.request_timeout_hours * 3600)
            if stale:
                for f in self.state.rows("SELECT * FROM files WHERE request_id=? AND state='requested'", request['id']):
                    self.retry(f, f'not on disk {self.o.restage_after_hours} h after request {request["id"]} '
                                  f'reported {current or "no status"} (dCache: {f["locality"]})')

    def retry(self, f, reason):
        """Back to 'pending' for another request, or 'failed' after too many."""
        if f['submissions'] >= self.o.max_submissions:
            self.state.set_file(f['surl'], state='failed', detail=str(reason))
            self.state.event('file_failed', f['name'], reason)
        else:
            self.state.set_file(f['surl'], state='pending', detail=str(reason), checked=None)
            self.state.event('restage', f['name'], reason)

    # ---------------------------------------------------------- retrieval
    def retrieve(self):
        """Download, extract, convert and delete raw data for files on disk."""
        if time.time() < self.throttle_until:
            return 0
        files = self.state.rows('''SELECT f.* FROM files f JOIN saps s ON s.key=f.sap_key
                                   WHERE f.state='online' ORDER BY s.position,f.beam LIMIT ?''',
                                self.o.download_workers * self.o.files_per_worker)
        if not files:
            return 0
        for f in files:
            self.state.set_file(f['surl'], state='working')
        with ThreadPoolExecutor(max_workers=self.o.download_workers) as pool:
            done = sum(pool.map(self.process_file, files))
        return done

    def process_file(self, f):
        from euroflash.download import download, extract
        from euroflash.rawdata import delete_archive
        target = self.root/'downloads'/f['name']
        item = target.stem
        # Beside its extraction folder, where raw-data deletion records itself.
        marker = self.root/'extracted'/(item + '.extracted.json')
        attempt = None
        try:
            previous = json.loads(marker.read_text()) if marker.is_file() else None
            if previous and previous['fits'] and all(Path(p).is_file() for p in previous['fits']):
                # A conversion retry: the extracted FITS are still here.
                fits = [Path(p) for p in previous['fits']]
            else:
                manifest = self.manifest(f['request_id'])
                url, tokens = self._token(manifest, f['surl'])
                attempt = self.ledger.start(item, 'retrieve', 'http-tar-v1', target.with_suffix('.receipt.json'),
                                            ['StageIT', str(f['request_id']), url])
                receipt = download(url, tokens, target, 64 * 2**30)
                fits = extract(target, self.root/'extracted'/item)
                marker.write_text(json.dumps({'request_id': f['request_id'], 'archive': receipt,
                                              'fits': [str(p) for p in fits]}, indent=2))
                self.ledger.record_archive(f['surl'], f['request_id'], receipt, fits, marker)
                self.ledger.finish(attempt, [marker, *fits])
                attempt = None
                delete_archive(target)
            from euroflash.beams import excluded
            if any(excluded(raw, self.o.exclude_beams) for raw in fits):
                # The archive says incoherentstokes whatever the beam number.
                from euroflash.rawdata import delete_converted
                for raw in fits:
                    delete_converted(raw)
                self.state.set_file(f['surl'], state='excluded', detail='incoherent beam (archive layout)')
                self.state.event('excluded', f['name'], 'incoherentstokes in the archive; not converted')
                return 0
            fils = []
            for raw in fits:
                sap_dir, fil = self.runner.convert(raw, delete_raw=True)
                fils.append(str(fil))
            self.state.set_file(f['surl'], state='converted', fil=json.dumps(fils), detail=None)
            return 1
        except PermissionError as error:
            # Every macaroon refused: the file fell back to tape, or the token expired.
            if attempt is not None:
                self.ledger.finish(attempt, error=str(error))
            self.state.set_file(f['surl'], state='requested', checked=None, detail=str(error))
            with self.manifest_lock:
                self.manifests.pop(f['request_id'], None)
            self.state.event('refused', f['name'], error)
            return 0
        except Exception as error:
            if attempt is not None:
                self.ledger.finish(attempt, error=str(error))
            if any(code in str(error) for code in ('HTTP Error 429', 'HTTP Error 503')):
                # SURF is throttling us, not refusing the file: back off, no strike.
                self.throttle_until = time.time() + self.o.throttle_seconds
                self.state.set_file(f['surl'], state='online', detail=str(error))
                self.state.event('throttled', f['name'], error)
                return 0
            failures = f['failures'] + 1
            self.state.set_file(f['surl'], state='failed' if failures >= self.o.max_failures else 'online',
                                failures=failures, detail=str(error)[:2000])
            self.state.event('retrieve_failed', f['name'], error)
            return 0

    # ---------------------------------------------------------- flatfield
    def prepare(self):
        """Flatfield SAPs that are complete or given up on, those stalled a few beams short, and late beams."""
        prepared = []
        now = time.time()
        for sap in self.state.rows("SELECT * FROM saps WHERE state IN ('staging','partial') ORDER BY position"):
            if sap['key'] in self.flatfield_jobs:
                continue
            files = self.state.rows('SELECT * FROM files WHERE sap_key=?', sap['key'])
            if sap['state'] == 'partial':
                self.prepare_late(sap, files)
                continue
            outstanding = [f for f in files if f['state'] in OUTSTANDING]
            if outstanding and not self.stalled(files, outstanding, now):
                continue
            lost = sorted(f['beam'] for f in files if f['state'] == 'failed' and f['beam'] in CENTRAL)
            converted = [Path(p) for f in files if f['state'] == 'converted' for p in json.loads(f['fil'])]
            if len(lost) > getattr(self.o, 'partial_missing', 0) or not converted:
                self.state.set_sap(sap['key'], state='attention',
                                   detail=f'central beams not retrieved: {lost}' if lost else 'no beam converted')
                self.state.event('attention', sap['key'], 'cannot flatfield')
                continue
            directories = {p.parent.parent for p in converted}
            if len(directories) != 1:
                self.state.set_sap(sap['key'], state='attention', detail=f'beams span {sorted(map(str, directories))}')
                continue
            sap_dir = directories.pop()
            flattened = [p.with_name(p.stem + '_ff.fil') for p in converted]
            if any(p.is_file() for p in converted) or not all(p.is_file() for p in flattened):
                # A SAP short of central beams (lost, absent from the archive or
                # still to come) is flatfielded with those present; one still
                # waiting for beams keeps its flatfield for them.
                present = {f['beam'] for f in files if f['state'] == 'converted'}
                short = any(b not in present for b in CENTRAL)
                job = functools.partial(self.runner.flatfield, sap_dir, allow_partial=short,
                                        save_mean=str(sap_dir/MEAN) if outstanding else None)
                # Flatfielding reads the whole SAP (~4 min); keep downloading meanwhile.
                self.flatfield_jobs[sap['key']] = self.flatfield_pool.submit(job)
                self.state.set_sap(sap['key'], state='flatfielding', sap_dir=str(sap_dir))
                if outstanding:
                    self.state.event('partial', sap['key'], f'searched without {len(outstanding)} beams still '
                                     f'being retrieved: {sorted(f["beam"] for f in outstanding)}')
            else:
                self.finish_flatfield(sap['key'], sap_dir, None)
                prepared.append(sap['key'])
        for key, job in list(self.flatfield_jobs.items()):
            if job.done():
                del self.flatfield_jobs[key]
                sap_dir = Path(self.state.rows('SELECT sap_dir FROM saps WHERE key=?', key)[0]['sap_dir'])
                if self.finish_flatfield(key, sap_dir, job.exception()):
                    prepared.append(key)
        return prepared

    def stalled(self, files, outstanding, now):
        """A few beams short, with no beam of the SAP converted for --partial-after-hours."""
        hours = getattr(self.o, 'partial_after_hours', 0)
        if not hours or len(outstanding) > getattr(self.o, 'partial_missing', 0):
            return False
        done = [f['updated'] or 0 for f in files if f['state'] == 'converted']
        return bool(done) and now - max(done) >= hours * 3600

    def prepare_late(self, sap, files):
        """Late beams of a SAP searched without them: flatfield each set with the SAP's saved flatfield."""
        sap_dir = Path(sap['sap_dir'])
        late = [Path(p) for f in files if f['state'] == 'converted' for p in json.loads(f['fil']) if Path(p).is_file()]
        if late:
            job = functools.partial(self.runner.flatfield, sap_dir, mean=str(sap_dir/MEAN))
            self.flatfield_jobs[sap['key']] = self.flatfield_pool.submit(job)
            self.state.set_sap(sap['key'], state='flatfielding')
        elif not any(f['state'] in OUTSTANDING for f in files) and not self.unsearched(sap['key']):
            lost = sorted(f['beam'] for f in files if f['state'] == 'failed')
            (sap_dir/MEAN).unlink(missing_ok=True)
            self.state.set_sap(sap['key'], state='searched',
                               detail=f'searched without beams never retrieved: {lost}' if lost else None)
            self.state.event('searched', sap['key'], 'the last late beams are in' if not lost else f'gave up on {lost}')

    def finish_flatfield(self, key, sap_dir, error):
        if error is not None:
            self.state.set_sap(key, state='attention', detail=f'flatfield failed: {error}'[:2000])
            self.state.event('flatfield_failed', key, error)
            return False
        if not self.o.keep_unflattened:
            for path in sap_dir.glob('B*/*_32bit.fil'):
                path.unlink()
        self.state.set_sap(key, state='prepared', sap_dir=str(sap_dir), detail=None)
        self.state.event('prepared', key, str(sap_dir))
        return True

    # ----------------------------------------------------------- dispatch
    def gpu_busy(self, node):
        """A GPU node is busy while a run on it has not finished its GPU stages."""
        for process in self.dispatches.values():
            if process.node == node and not (self.root/'results'/process.run_name/f'{node}.gpu-done').exists():
                return True
        return False

    def dispatch(self):
        """Search prepared SAPs: a batch per free GPU node, several runs in flight."""
        # Running (or adopted) runs are followed even with dispatch turned off.
        for run_name, process in list(self.dispatches.items()):
            if process.poll() is not None:
                self.finish_dispatch(process)
        started = []
        for node in (self.o.dispatch_nodes or []) if not self.stopping else []:
            on_node = sum(1 for p in self.dispatches.values() if p.node == node)
            if (self.gpu_busy(node) or on_node >= getattr(self.o, 'batches_per_node', 2)
                    or time.time() < self.dispatch_retry_after.get(node, 0)):
                continue
            ready = self.state.rows("SELECT * FROM saps WHERE state='prepared' ORDER BY position LIMIT ?",
                                    self.o.dispatch_saps)
            if not ready:
                break
            started.append(self.start_dispatch(node, ready))
        return started or ('running' if self.dispatches else None)

    def start_dispatch(self, node, ready):
        base = time.strftime('campaign-%Y%m%d-%H%M%S-') + node.removeprefix('efc-').replace('-', '')
        run_name, n = base, 1
        while (self.root/'dispatch'/run_name).exists() or (self.root/'results'/run_name).exists():
            n += 1
            run_name = f'{base}-{n}'
        batch = self.root/'dispatch'/run_name
        batch.mkdir(parents=True)
        from euroflash.beams import excluded
        for sap in ready:
            for path in self.unsearched(sap['key']):
                if not excluded(path, self.o.exclude_beams):
                    os.link(path, batch/path.name)   # hard link: no copy, originals survive cleanup
        command = [sys.executable, '-m', 'euroflash.cluster', '--input', str(batch),
                   '--work', str(self.root/'results'/run_name), '--ledger', str(Path(self.o.ledger).resolve()),
                   '--nodes', node, '--run-name', run_name, '--gpus', self.o.gpus,
                   '--workers-per-gpu', str(self.o.workers_per_gpu), '--cpu-workers', str(self.o.cpu_workers),
                   '--image', str(self.o.image), '--settings', str(self.o.settings),
                   '--exclude-beams', *map(str, self.o.exclude_beams), '--skip-trials', '--cleanup-remote']
        if getattr(self.o, 'remote_root', None):
            command += ['--remote-root', self.o.remote_root]
        if getattr(self.o, 'cpu_nodes', None):
            command += ['--cpu-nodes', *self.o.cpu_nodes, '--cpu-tier-workers', str(getattr(self.o, 'cpu_tier_workers', 64)),
                        '--cpu-lock-dir', str(self.root/'.cpu-slots')]
        if self.o.control_dir:
            command += ['--control-dir', str(self.o.control_dir)]
        for timeout in getattr(self.o, 'stage_timeout', []):
            command += ['--stage-timeout', timeout]
        log = (self.root/'logs'/f'{run_name}.log').open('a')
        process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        process.run_name, process.node = run_name, node
        self.dispatches[run_name] = process
        for sap in ready:
            self.state.set_sap(sap['key'], state='dispatched', run_name=run_name)
        self.state.event('dispatched', run_name, f'{node}: ' + ','.join(s['key'] for s in ready))
        return run_name

    def searched_items(self, run_name):
        succeeded, seen = set(), False
        for snapshot in (self.root/'results'/run_name).glob('*/ledger-snapshot.sqlite'):
            seen = True
            db = sqlite3.connect(f'file:{snapshot}?mode=ro', uri=True)
            try:
                latest = db.execute('''SELECT item,status FROM attempts WHERE id IN
                    (SELECT MAX(id) FROM attempts WHERE stage='classify' GROUP BY item)''').fetchall()
            finally:
                db.close()
            succeeded |= {item for item, status in latest if status == 'success'}
        return succeeded, seen

    def finish_dispatch(self, process):
        self.dispatches.pop(process.run_name, None)
        run_name, node = process.run_name, getattr(process, 'node', None)
        succeeded, collected = self.searched_items(run_name)
        saps = self.state.rows("SELECT * FROM saps WHERE state='dispatched' AND run_name=?", run_name)
        if not collected:
            # Nothing ran (SSH, node health): put the SAPs back and back off that node.
            self.dispatch_failures[node] = self.dispatch_failures.get(node, 0) + 1
            self.dispatch_retry_after[node] = time.time() + min(4 * 3600, 600 * 2 ** self.dispatch_failures[node])
            for sap in saps:
                self.state.set_sap(sap['key'], state='prepared', run_name=None,
                                   detail=f'dispatch {run_name} exited {process.returncode} before searching')
            self.state.event('dispatch_failed', run_name, f'exit {process.returncode}; see logs/{run_name}.log')
        else:
            self.dispatch_failures[node] = 0
            found = self.findings(run_name)
            for sap in saps:
                done = kept = failed = 0
                for f in self.state.rows("SELECT * FROM files WHERE sap_key=? AND state='converted'", sap['key']):
                    beams = [flattened(p) for p in json.loads(f['fil'] or '[]')]
                    if not beams or any(b.stem not in succeeded for b in beams):
                        failed += 1
                        continue
                    done += 1
                    if self.o.keep_prepared or any(b.stem in found for b in beams):
                        kept += 1
                        why = next((found[b.stem] for b in beams if b.stem in found), 'every beam is kept')
                        self.state.set_file(f['surl'], state='kept',
                                            detail=f'searched in {run_name}; kept for review: {why}'[:2000])
                    else:
                        for path in beams:
                            path.unlink(missing_ok=True)
                        self.state.set_file(f['surl'], state='searched', detail=f'searched in {run_name}')
                outstanding = self.state.rows(f"""SELECT COUNT(*) AS n FROM files WHERE sap_key=?
                    AND state IN ({','.join('?' * len(OUTSTANDING))})""", sap['key'], *OUTSTANDING)[0]['n']
                if failed and not self.retried(sap['key']):
                    # One more attempt before a person is asked: the 25 M-crossing
                    # guard and a node lost mid-run both left beams unsearched.
                    self.state.set_sap(sap['key'], state='prepared', run_name=None,
                                       detail=f'automatic retry: {failed} beams not searched in {run_name}')
                    self.state.event('retry', sap['key'], f'automatic: {failed} beams not searched in {run_name}')
                elif failed:
                    self.state.set_sap(sap['key'], state='attention', detail=f'{failed} beams not searched in {run_name}')
                elif outstanding:
                    self.state.set_sap(sap['key'], state='partial', detail=f'waiting for {outstanding} late beams')
                else:
                    if sap['sap_dir']:
                        (Path(sap['sap_dir'])/MEAN).unlink(missing_ok=True)
                    self.state.set_sap(sap['key'], state='searched', detail=None)
                self.state.event('searched', sap['key'], f'{done}/{done + failed} beams in {run_name}; '
                                                         f'{kept} kept for review')
            if not self.o.keep_prepared:
                observations = {m[1] for sap in saps for m in [re.search(r'(L\d+)/SAP', sap['sap_dir'] or '')] if m}
                self.release_kept(observations, run_name)
        shutil.rmtree(self.root/'dispatch'/run_name, ignore_errors=True)

    def judge(self):
        """The periodic index, brought up to date with every run, and a judge over it."""
        from euroflash.findings import Index, Judge
        if getattr(self, 'periodic_index', None) is None:
            self.periodic_index = Index(self.root)
        self.periodic_index.backfill(self.root/'results')
        return Judge(self.periodic_index)

    def findings(self, run_name):
        """{beam stem: why} for the beams of one run worth keeping (euroflash.findings)."""
        from euroflash.findings import run_candidates
        judge = self.judge()
        found = {}
        for item, beam in judge.index.beams.items():
            if item and beam['run'] == run_name:
                keep, why = judge.keep(beam)
                if keep:
                    found[item] = why
        # A beam whose periodic search never ran still keeps a FETCH candidate.
        for item, n in run_candidates(self.root/'results'/run_name).items():
            found.setdefault(item, f'{n} FETCH candidate(s)')
        return found

    def release_kept(self, observations=None, run_name=None, apply=True):
        """Remove kept filterbanks that nothing holds any more; returns [(name, why)].

        A beam kept for a fold loses it when a later SAP of its observation
        shows the fold's frequency across beams at scattered DMs, or when the
        rule itself changed (`rekeep`). A FETCH candidate, or a reviewer's
        astro or unsure verdict on any candidate of the beam, always holds it.
        """
        from euroflash.findings import reviewed_items
        judge = self.judge()
        reviews = getattr(self.o, 'reviews', None) or self.root.parent/'web'/'reviews.sqlite'
        protected = reviewed_items(reviews, judge.index)
        released = []
        for f in self.state.rows("SELECT * FROM files WHERE state='kept'"):
            beams = [flattened(p) for p in json.loads(f['fil'] or '[]')]
            entries = [judge.index.beams.get(b.stem) for b in beams]
            if not beams or any(e is None for e in entries):
                continue
            if observations is not None and not any(e['observation'] in observations for e in entries):
                continue
            if any(b.stem in protected for b in beams):
                continue
            verdicts = [judge.keep(e) for e in entries]
            if any(keep for keep, _ in verdicts):
                continue
            why = '; '.join(w for _, w in verdicts)
            released.append((f['name'], why))
            if apply:
                for path in beams:
                    path.unlink(missing_ok=True)
                self.state.set_file(f['surl'], state='searched',
                                    detail=f'released{" after " + run_name if run_name else ""}: {why}'[:2000])
                self.state.event('released', f['name'], why)
        return released

    def retried(self, key):
        return bool(self.state.rows("SELECT 1 FROM events WHERE kind='retry' AND subject=? LIMIT 1", key))

    def unsearched(self, key):
        """Flatfielded beams of a SAP that still need a search."""
        paths = []
        for f in self.state.rows("SELECT fil FROM files WHERE sap_key=? AND state='converted'", key):
            paths += [p for p in map(flattened, json.loads(f['fil'] or '[]')) if p.is_file()]
        return sorted(paths)

    # --------------------------------------------------------------- loop
    def status(self):
        counts = self.state.counts()
        day = time.time() - 86400
        converted = self.state.rows("SELECT COUNT(*) AS n FROM files WHERE state='converted' AND updated>?", day)[0]['n']
        report = {'time_unix': time.time(), 'root': str(self.root), **counts,
                  'files_converted_last_24h': converted,
                  'free_tb': round(shutil.disk_usage(self.root).free / 1e12, 2),
                  'active_requests': self.state.rows('''SELECT r.id,r.sap_key,r.status,r.files,
                        SUM(f.state='requested') AS waiting FROM requests r JOIN files f ON f.request_id=r.id
                        GROUP BY r.id HAVING waiting>0 ORDER BY r.submitted'''),
                  'attention': self.state.rows("SELECT key,detail FROM saps WHERE state='attention'"),
                  'kept_for_review': self.kept_summary(),
                  'dispatch_running': sorted(self.dispatches),
                  'dispatch_nodes': {p.run_name: p.node for p in self.dispatches.values()},
                  'staging_window': {'max_staging_saps': self.window[0], 'phase': self.window[1]},
                  'staging_files_target': getattr(self.o, 'staging_files_target', 0),
                  'recent_events': self.state.rows('SELECT * FROM events ORDER BY time DESC LIMIT 20')}
        partial = self.root/'status.json.partial'
        partial.write_text(json.dumps(report, indent=2, default=str))
        os.replace(partial, self.root/'status.json')
        return report

    def recover(self):
        """Undo what an interrupted driver left half-done."""
        with self.state.db() as db:
            db.execute("UPDATE files SET state='online' WHERE state='working'")
            db.execute("UPDATE saps SET state='staging' WHERE state='flatfielding'")
        for sap in self.state.rows("SELECT * FROM saps WHERE state='staging' AND sap_dir IS NOT NULL"):
            if (Path(sap['sap_dir'])/MEAN).is_file() and self.state.rows(
                    "SELECT 1 FROM files WHERE sap_key=? AND state IN ('searched','kept') LIMIT 1", sap['key']):
                self.state.set_sap(sap['key'], state='partial')
        # Searched before per-beam states existed: both filterbanks deleted.
        for f in self.state.rows("""SELECT f.surl,f.fil FROM files f JOIN saps s ON s.key=f.sap_key
                                    WHERE f.state='converted' AND s.state NOT IN ('staging','flatfielding')"""):
            paths = [Path(p) for p in json.loads(f['fil'] or '[]')]
            if paths and not any(p.is_file() or flattened(p).is_file() for p in paths):
                self.state.set_file(f['surl'], state='searched', detail='searched before per-beam states')
        self.apply_exclusions()
        for sap in self.state.rows("SELECT * FROM saps WHERE state='dispatched'"):
            if sap['run_name'] in self.dispatches:
                continue
            running = running_dispatch(sap['run_name'])
            if running is not None:
                # A cluster run outlived the driver that started it: follow it
                # rather than dispatch the same SAPs again beside it.
                self.dispatches[sap['run_name']] = running
                self.state.event('dispatch_adopted', sap['key'], f'{sap["run_name"]} pid {running.pid}')
                continue
            succeeded, collected = self.searched_items(sap['run_name'])
            if collected:
                self.finish_dispatch(SimpleNamespace(run_name=sap['run_name'], returncode=None, node=None))
            else:
                self.state.set_sap(sap['key'], state='prepared', run_name=None,
                                   detail=f'dispatch {sap["run_name"]} interrupted by a driver restart')
                shutil.rmtree(self.root/'dispatch'/str(sap['run_name']), ignore_errors=True)
                self.state.event('dispatch_interrupted', sap['key'], sap['run_name'])

    def apply_exclusions(self):
        """Remove excluded beams already converted, and release SAPs held only by them."""
        from euroflash.beams import excluded
        beams = tuple(self.o.exclude_beams)
        if not beams:
            return
        marks = ','.join('?' * len(beams))
        for f in self.state.rows(f"SELECT surl,fil FROM files WHERE beam IN ({marks}) AND state='converted'", *beams):
            for path in json.loads(f['fil'] or '[]'):
                Path(path).unlink(missing_ok=True)
                Path(path).with_name(Path(path).stem + '_ff.fil').unlink(missing_ok=True)
            self.state.set_file(f['surl'], state='excluded', detail='incoherent beam; converted data removed')
        for sap in self.state.rows("SELECT * FROM saps WHERE sap_dir IS NOT NULL AND state IN ('prepared','attention')"):
            sap_dir = Path(sap['sap_dir'])
            for path in sap_dir.glob('B*/*.fil'):
                if excluded(path, beams):
                    path.unlink()
            if sap['state'] == 'attention' and (sap['detail'] or '').endswith(' beams not searched in ' + str(sap['run_name'])):
                if not self.unsearched(sap['key']):
                    self.state.set_sap(sap['key'], state='searched', detail='only excluded beams were unsearched')
                    self.state.event('searched', sap['key'], 'released: only the excluded beam had failed')

    def kept_summary(self):
        paths = [flattened(p) for f in self.state.rows("SELECT fil FROM files WHERE state='kept'")
                 for p in json.loads(f['fil'] or '[]')]
        present = [p for p in paths if p.is_file()]
        return {'beams': len(present), 'tb': round(sum(p.stat().st_size for p in present) / 1e12, 3)}

    def tick(self):
        self.refresh()
        self.retrieve()
        self.prepare()
        # After preparing, so a SAP that just left the window is replaced at once.
        self.admit()
        self.request()
        self.dispatch()
        return self.status()

    def should_stop(self):
        return self.stopping or (self.root/'STOP').exists()

    def run(self):
        self.recover()
        while True:
            report = self.tick()
            print(time.strftime('%Y-%m-%d %H:%M:%S'), json.dumps({k: report[k] for k in ('saps', 'files', 'free_tb')}),
                  flush=True)
            idle = not self.state.rows("SELECT 1 FROM saps WHERE state IN "
                                       "('pending','staging','flatfielding','prepared','dispatched','partial') LIMIT 1")
            if self.o.once or self.should_stop() or (idle and not self.dispatches):
                break
            # Retrieval work is taken in bounded batches; go straight back for more.
            if self.state.rows("SELECT 1 FROM files WHERE state='online' LIMIT 1"):
                continue
            deadline = time.time() + self.o.poll_seconds
            while time.time() < deadline and not self.should_stop():
                time.sleep(min(5, max(0, deadline - time.time())))
        if self.flatfield_jobs:
            # Record flatfields still running rather than redo them next start.
            wait(list(self.flatfield_jobs.values()))
            self.prepare()
        if not self.o.once:
            for process in list(self.dispatches.values()):
                process.wait()
                self.finish_dispatch(process)
        self.status()
        self.flatfield_pool.shutdown(wait=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='Run (or resume) the campaign loop')
    run.add_argument('--root', type=Path, required=True, help='Campaign directory: state, transient data, prepared beams')
    run.add_argument('--inventory', type=Path, required=True, help='SRM URLs, one per line')
    run.add_argument('--ledger', type=Path, required=True)
    run.add_argument('--image', type=Path, default=REPO/'containers/euroflash-runtime.sif')
    run.add_argument('--settings', type=Path, default=REPO/'settings.yaml')
    run.add_argument('--max-staging-saps', type=int, default=8, help='SAP requests in flight at once')
    run.add_argument('--staging-files-target', type=int, default=0,
                     help='Replenish outstanding files up to this target, bounded by max-staging-saps; 0 disables')
    run.add_argument('--max-prepared-saps', type=int, default=20, help='Prepared SAPs allowed to wait for a search')
    run.add_argument('--min-free-tb', type=float, default=5.0, help='Stop admitting SAPs below this free space')
    run.add_argument('--download-workers', type=int, default=4)
    run.add_argument('--files-per-worker', type=int, default=4, help='Files each worker takes per loop pass')
    run.add_argument('--poll-seconds', type=float, default=300)
    run.add_argument('--locality-interval', type=float, default=900, help='Seconds between dCache checks of a file')
    run.add_argument('--restage-after-hours', type=float, default=12)
    run.add_argument('--request-timeout-hours', type=float, default=96)
    run.add_argument('--max-submissions', type=int, default=4)
    run.add_argument('--max-failures', type=int, default=3)
    run.add_argument('--flatfield-workers', type=int, default=2)
    run.add_argument('--throttle-seconds', type=float, default=120,
                     help='Pause downloads this long after SURF answers 429/503')
    run.add_argument('--only-sap', nargs='+', help='Restrict admission to these SAP keys, e.g. L1163405_SAP001')
    run.add_argument('--exclude-beams', type=int, nargs='+', default=[12],
                     help='Beams never staged, converted or searched (default: 12, the incoherent beam)')
    run.add_argument('--partial-after-hours', type=float, default=2.0,
                     help='Search a SAP without its last --partial-missing beams once none has arrived for this long; '
                          'the late ones are searched as they come (0 waits for every beam)')
    run.add_argument('--partial-missing', type=int, default=3,
                     help='Beams a SAP may still be waiting for, or have lost, and be searched without')
    run.add_argument('--max-missing-central', type=int, default=3,
                     help='Stage and search a SAP whose archive lacks up to this many central beams')
    run.add_argument('--keep-unflattened', action='store_true')
    run.add_argument('--keep-prepared', action='store_true',
                     help='Keep every flatfielded beam after its search, not only those with findings')
    run.add_argument('--staging-schedule', type=Path,
                     help='JSON phases that change --max-staging-saps at set times (the staging experiment)')
    run.add_argument('--dispatch-nodes', nargs='+', help='Search prepared SAPs on these GPU nodes')
    run.add_argument('--dispatch-saps', type=int, default=2, help='SAPs per cluster run')
    run.add_argument('--batches-per-node', type=int, default=2,
                     help='Runs per GPU node at once: one in its GPU stages, the others finishing on CPU nodes')
    run.add_argument('--cpu-nodes', nargs='+', help='Classify and search for periodicity on these CPU nodes')
    run.add_argument('--cpu-tier-workers', type=int, default=64, help='Concurrent beams on a CPU node')
    run.add_argument('--remote-root', help='Run directory on the compute nodes (euroflash.cluster default if unset)')
    run.add_argument('--gpus', default='0,1')
    run.add_argument('--workers-per-gpu', type=int, default=3)
    run.add_argument('--cpu-workers', type=int, default=24)
    run.add_argument('--stage-timeout', action='append', default=[], metavar='STAGE=SECONDS')
    run.add_argument('--control-dir', type=Path)
    run.add_argument('--reviews', type=Path,
                     help="The web layer's verdicts (default: web/reviews.sqlite beside the campaign root); "
                          'a beam with an astro or unsure verdict is never released')
    run.add_argument('--once', action='store_true', help='One pass, then exit')
    status = sub.add_parser('status', help='Print the campaign state summary')
    status.add_argument('--root', type=Path, required=True)
    rekeep = sub.add_parser('rekeep', help='Judge every kept beam again and list (--apply: release) what nothing holds')
    rekeep.add_argument('--root', type=Path, required=True)
    rekeep.add_argument('--reviews', type=Path, help='As for run')
    rekeep.add_argument('--apply', action='store_true', help='Delete their filterbanks; without it, only list them')
    retry = sub.add_parser('retry', help='Queue SAPs in attention for another search of their unsearched beams')
    retry.add_argument('--root', type=Path, required=True)
    retry.add_argument('keys', nargs='*', help='SAP keys; all SAPs in attention with prepared beams if omitted')
    return p


def retry_saps(root, keys=()):
    """Put SAPs in attention whose unsearched beams are still prepared back in the dispatch queue."""
    state = State(Path(root)/'campaign-state.sqlite')
    queued = []
    for sap in state.rows("SELECT * FROM saps WHERE state='attention' AND sap_dir IS NOT NULL"):
        if keys and sap['key'] not in keys:
            continue
        if any(flattened(p).is_file() for f in state.rows(
                "SELECT fil FROM files WHERE sap_key=? AND state='converted'", sap['key'])
               for p in json.loads(f['fil'] or '[]')):
            state.set_sap(sap['key'], state='prepared', run_name=None, detail='retry requested: ' + (sap['detail'] or ''))
            state.event('retry', sap['key'], sap['detail'])
            queued.append(sap['key'])
    return queued


def rekeep(root, reviews=None, apply=False):
    """Judge every kept beam under the current rule: what would be (or was) released."""
    root = Path(root)
    campaign = Campaign.__new__(Campaign)
    campaign.root, campaign.o = root, SimpleNamespace(reviews=reviews, keep_prepared=False)
    campaign.state = State(root/'campaign-state.sqlite')
    kept = campaign.state.rows("SELECT COUNT(*) AS n FROM files WHERE state='kept'")[0]['n']
    released = campaign.release_kept(apply=apply)
    reasons = {}
    for _, why in released:
        for part in why.split('; '):
            key = part.split(' ')[0] if not part.startswith('recurs') else 'recurs'
            reasons[key] = reasons.get(key, 0) + 1
    return {'kept': kept, 'released' if apply else 'would_release': len(released),
            'remaining': kept - len(released), 'fold_reasons': reasons}


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    if a.command == 'status':
        print((a.root/'status.json').read_text())
        return
    if a.command == 'retry':
        print('Queued for another search:', ' '.join(retry_saps(a.root, a.keys)) or 'nothing')
        return
    if a.command == 'rekeep':
        print(json.dumps(rekeep(a.root, a.reviews, a.apply), indent=2))
        return
    a.root.mkdir(parents=True, exist_ok=True)
    with (a.root/'.campaign.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error(f'Another campaign driver is running in {a.root}')
        campaign = Campaign(a)
        campaign.state.load(parse_inventory(a.inventory), missing_central_beams, tuple(a.exclude_beams),
                            a.max_missing_central)
        def stop(*_):
            campaign.stopping = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        campaign.run()


if __name__ == '__main__':
    main()
