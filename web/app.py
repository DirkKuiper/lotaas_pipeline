"""Pages and JSON for campaign progress, coverage, candidates and their review.

Pages read web.sqlite (with reviews.sqlite attached) and never the pipeline's
own databases. A background thread keeps the index and the snippets current.
The head node is shared, so every request needs a token cookie; open the URL
printed by `python -m web url` once to set it.
"""
import base64
from collections import OrderedDict
from contextlib import asynccontextmanager
import json
import logging
import math
import os
from pathlib import Path
import secrets
import threading
import time
import warnings
from urllib.parse import urlencode

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from web import store
from web.dynspec import Snippet, parse_mask, sweep_seconds
from web.keys import beam_label, parse_item
from web.store import LABELS

logger = logging.getLogger(__name__)
HERE = Path(__file__).parent
COOKIE = 'lotaas_web'
STATE_ORDER = ['pending', 'staging', 'flatfielding', 'prepared', 'dispatched', 'searched', 'attention',
               'incomplete']
FILE_ORDER = ['pending', 'requested', 'online', 'working', 'converted', 'searched', 'kept', 'failed', 'excluded']
QUEUE_TYPES = ('candidate', 'known_pulsar', 'periodic')


def token(cfg):
    """The shared secret, created private to this user on first use."""
    path = Path(cfg.token_file)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(mode=0o600)
        path.write_text(secrets.token_urlsafe(24) + '\n')
    os.chmod(path, 0o600)
    return path.read_text().strip()


def clean(value, digits=4):
    """JSON-safe copies of numpy values: NaN becomes null, floats are rounded."""
    if isinstance(value, dict):
        return {k: clean(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v, digits) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist(), digits)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if not math.isfinite(value) else round(value, digits)
    if isinstance(value, np.integer):
        return int(value)
    return value


def encode_image(image):
    """A float32 (rows, columns) array as base64 for a Float32Array in the page."""
    image = np.ascontiguousarray(image, dtype='<f4')
    return {'rows': int(image.shape[0]), 'columns': int(image.shape[1]),
            'data': base64.b64encode(image.tobytes()).decode()}


def ago(value):
    if value is None:
        return '—'
    seconds = time.time() - float(value)
    for unit, size in (('d', 86400), ('h', 3600), ('min', 60)):
        if abs(seconds) >= size:
            return f'{seconds / size:.0f} {unit} ago' if unit != 'd' else f'{seconds / size:.1f} d ago'
    return f'{seconds:.0f} s ago'


def duration(seconds):
    if seconds is None:
        return '—'
    seconds = float(seconds)
    if seconds >= 86400:
        return f'{seconds / 86400:.1f} d'
    if seconds >= 3600:
        return f'{int(seconds // 3600)} h {int(seconds % 3600 // 60):02d} m'
    if seconds >= 60:
        return f'{int(seconds // 60)} m {int(seconds % 60):02d} s'
    return f'{seconds:.1f} s'


def when(value):
    return '—' if value is None else time.strftime('%Y-%m-%d %H:%M', time.gmtime(float(value)))


def size(value):
    value = float(value or 0)
    for unit in ('B', 'kB', 'MB', 'GB', 'TB'):
        if value < 1000 or unit == 'TB':
            return f'{value:.0f} {unit}' if unit == 'B' else f'{value:.1f} {unit}'
        value /= 1000


class Cache:
    """A small thread-safe LRU for loaded snippets and their DM responses."""

    def __init__(self, capacity):
        self.capacity, self.items, self.lock = capacity, OrderedDict(), threading.Lock()

    def peek(self, key):
        with self.lock:
            return self.items.get(key)

    def get(self, key, make):
        with self.lock:
            if key in self.items:
                self.items.move_to_end(key)
                return self.items[key]
        value = make()
        with self.lock:
            self.items[key] = value
            while len(self.items) > self.capacity:
                self.items.popitem(last=False)
        return value


def kept_beams(db):
    """Beams whose flatfielded filterbank the campaign keeps for review, with their size."""
    found = []
    for row in db.execute("SELECT f.surl, f.name, f.sap_key, f.beam, f.fil, f.updated, a.item FROM files f "
                          "LEFT JOIN archive_beams a ON a.uri=f.surl WHERE f.state='kept'"):
        for path in json.loads(row['fil'] or '[]'):
            path = Path(path)
            path = path.with_name(path.stem + '_ff.fil')
            try:
                size = path.stat().st_size
            except OSError:
                size = None
            found.append(dict(row, path=str(path), bytes=size))
    return found


def view_payload(snippet, dm=None, tscrunch=1, nsub=81, window=2.0, mask=(), clip=99.0):
    view = snippet.view(dm=dm, tscrunch=tscrunch, nsub=nsub, window=window or None, mask=mask,
                        clip=min(max(clip, 50.0), 100.0))
    # Ascending frequency, so the heatmap's rows run up the axis.
    order = np.argsort(view['freqs'])
    return clean({
        'dm': view['dm'], 'tsamp': view['tsamp'], 'tscrunch': view['tscrunch'], 'nsub': view['nsub'],
        'times': view['times'], 'freqs': view['freqs'][order], 'image': encode_image(view['image'][order]),
        'zmin': view['zmin'], 'zmax': view['zmax'], 'series': view['series'], 'boxcar': view['boxcar'],
        'peak_snr': view['peak_snr'], 'width': view['width'], 'best_snr': view['best_snr'],
        'best_width': view['best_width'], 'spectrum_on': view['spectrum_on'][order],
        'spectrum_off': view['spectrum_off'][order], 'masked': view['masked'],
        'sweep': sweep_seconds(snippet.dm, view['freqs'][order])}, 5)


def dm_payload(snippet, channels=()):
    r = snippet.dm_response(mask=list(channels))
    return clean(dict(r, plane=encode_image(r['plane']), smearing=r['smearing_ms'],
                      meta={k: snippet.meta.get(k) for k in ('dm', 'snr', 'width_samples', 'tsamp',
                                                             'downsample', 'source', 'how')}), 5)


def fold_payload(directory, fold_data, row):
    path = Path(directory) / fold_data
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    counts, sums = data.get('subintegration_counts'), data.get('subintegration_sums')
    with np.errstate(invalid='ignore', divide='ignore'):
        subints = np.where(counts > 0, sums / counts, np.nan) if counts is not None else None
    fold = json.loads(row)
    # Each row about its own mean, so slow baseline changes do not hide the pulse.
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        if subints is not None:
            subints = subints - np.nanmean(subints, axis=1, keepdims=True)
        subbands = data.get('subbands')
        if subbands is not None and subbands.size:
            subbands = subbands - np.nanmean(subbands, axis=1, keepdims=True)
    # Trial frequencies differ by ~1e-7 Hz: send offsets from the search frequency, in microhertz.
    centre = fold.get('frequency_hz')
    offsets = (data['frequencies'] - centre) * 1e6 if centre and data.get('frequencies') is not None else None
    refined = (fold['refined_frequency_hz'] - centre) * 1e6 if centre and fold.get('refined_frequency_hz') else None
    try:
        beam_meta = json.loads((Path(directory) / 'metadata.json').read_text())
        band = [beam_meta.get('nu_min'), beam_meta.get('nu_max')]
    except (OSError, ValueError):
        band = [None, None]
    return clean({
        'profile': data.get('profile'), 'errors': data.get('errors'), 'subints': subints,
        'subbands': subbands if subbands is not None and subbands.size else None,
        'offsets_uhz': offsets, 'refined_offset_uhz': refined, 'fold_chi2': data.get('fold_chi2'),
        'dm_curve': data.get('dm_curve'), 'band': band, 'period': fold.get('refined_period_seconds'),
        'dm': fold.get('dm'), 'observation_seconds': fold.get('observation_seconds')}, 6)


def background(cfg, stop):
    """Index, then hold/cut snippets, then wait; errors are logged, never fatal."""
    from web.indexer import Indexer
    from web.snippets import Snippets
    indexer, snippets = Indexer(cfg), Snippets(cfg)
    while not stop.is_set():
        begun = time.time()
        try:
            indexer.run_pass()
            result = snippets.run_pass()
            if any(result.values()):
                logger.info('Snippets: %s', result)
                indexer.sync_snippets()
                indexer.derive()
        except Exception:
            logger.exception('Background pass failed')
        stop.wait(max(5.0, cfg.index_seconds - (time.time() - begun)))


def create_app(cfg, run_background=True):
    cfg.prepare()
    store.index(cfg).close()
    store.reviews(cfg).close()
    secret = token(cfg) if cfg.auth else None
    stop = threading.Event()
    @asynccontextmanager
    async def lifespan(app):
        if run_background:
            threading.Thread(target=background, args=(cfg, stop), daemon=True, name='index').start()
        yield
        stop.set()

    app = FastAPI(title='LOTAAS campaign', docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    templates = Jinja2Templates(directory=str(HERE / 'templates'))
    env = templates.env
    env.filters.update(ago=ago, duration=duration, when=when, size=size, beam=beam_label,
                       kind=lambda t: {'candidate': 'FETCH positive', 'known_pulsar': 'known pulsar',
                                       'rejected': 'FETCH reject', 'periodic': 'periodic',
                                       'periodic_rfi': 'periodic, RFI-like'}.get(t, t),
                       json=lambda v: Markup(json.dumps(clean(v)).replace('</', '<\\/')))
    env.globals.update(LABELS=LABELS, now=time.time)
    app.mount('/static', StaticFiles(directory=str(HERE / 'static')), name='static')
    snippets = Cache(8)
    responses = Cache(64)

    @app.middleware('http')
    async def authenticate(request: Request, call_next):
        if secret is None or request.url.path.startswith(('/static/', '/assets/')):
            return await call_next(request)
        if secrets.compare_digest(request.cookies.get(COOKIE, ''), secret):
            return await call_next(request)
        given = request.query_params.get('token')
        if given and secrets.compare_digest(given, secret):
            rest = {k: v for k, v in request.query_params.items() if k != 'token'}
            response = RedirectResponse(request.url.path + ('?' + urlencode(rest) if rest else ''), 303)
            response.set_cookie(COOKIE, secret, httponly=True, samesite='strict', max_age=180 * 86400)
            return response
        return HTMLResponse(templates.get_template('login.html').render(), status_code=401)

    def page(request, name, **context):
        with store.reading(cfg) as db:
            health = meta(db, 'health', {})
            last = meta(db, 'last_index', {})
            unreviewed = db.execute(f"""SELECT COUNT(*) FROM candidates c WHERE c.type IN
                ('candidate','known_pulsar') AND NOT EXISTS (SELECT 1 FROM r.reviews v WHERE v.key=c.key)""").fetchone()[0]
        return templates.TemplateResponse(request, name, dict(
            context, health=health, last_index=last, unreviewed=unreviewed, path=request.url.path))

    # ---------------------------------------------------------- overview
    @app.get('/', response_class=HTMLResponse)
    def overview(request: Request):
        now = time.time()
        with store.reading(cfg) as db:
            saps = dict(db.execute('SELECT state, COUNT(*) FROM saps GROUP BY state').fetchall())
            files = dict(db.execute('SELECT state, COUNT(*) FROM files GROUP BY state').fetchall())
            beams_searched = db.execute("""SELECT COUNT(DISTINCT item) FROM attempts
                WHERE stage='classify' AND status='success'""").fetchone()[0]
            campaign_beams = db.execute("""SELECT COUNT(DISTINCT a.item) FROM attempts a
                JOIN runs r ON r.fingerprint=a.fingerprint
                WHERE a.stage='classify' AND a.status='success' AND r.pilot=0""").fetchone()[0]
            hours = 48
            since = now - hours * 3600
            converted = hourly(db, "SELECT updated FROM files WHERE state='converted' AND updated>?", since, hours)
            searched = hourly(db, """SELECT finished FROM attempts WHERE stage='classify' AND status='success'
                AND finished>?""", since, hours)
            day = now - 86400
            beams_day = db.execute("""SELECT COUNT(DISTINCT item) FROM attempts WHERE stage='classify'
                AND status='success' AND finished>?""", (day,)).fetchone()[0]
            kept = kept_beams(db)
            files_day = db.execute("SELECT COUNT(*) FROM files WHERE state='converted' AND updated>?",
                                   (day,)).fetchone()[0]
            attention = rows(db, "SELECT key, detail, run_name, updated FROM saps WHERE state='attention' "
                                 "ORDER BY updated DESC")
            events = rows(db, 'SELECT * FROM events ORDER BY time DESC LIMIT 40')
            dispatched = rows(db, "SELECT key, run_name, updated FROM saps WHERE state='dispatched'")
            review = rows(db, """SELECT c.*, (SELECT label FROM r.reviews v WHERE v.key=c.key
                ORDER BY created DESC LIMIT 1) AS label FROM candidates c WHERE c.type='candidate'
                ORDER BY COALESCE(c.slack_sent, 0) DESC, c.found DESC LIMIT 8""")
        total = sum(saps.values())
        remaining = sum(v for k, v in saps.items() if k not in ('searched', 'incomplete'))
        # Whichever is slower, bringing files off tape or searching them, sets the pace.
        to_convert = sum(files.get(k, 0) for k in ('pending', 'requested', 'online', 'working'))
        to_search = to_convert + files.get('converted', 0)
        paces = [n / rate for n, rate in ((to_convert, files_day), (to_search, beams_day)) if rate]
        eta_days = max(paces) if len(paces) == 2 else None
        pending_files = to_convert
        return page(request, 'overview.html', saps=saps, files=files, total=total, remaining=remaining,
                    beams_searched=beams_searched, campaign_beams=campaign_beams, eta_days=eta_days,
                    beams_day=beams_day, files_day=files_day, pending_files=pending_files, to_search=to_search,
                    kept=kept, kept_bytes=sum(k['bytes'] or 0 for k in kept),
                    state_order=STATE_ORDER, file_order=FILE_ORDER, attention=attention, events=events,
                    dispatched=dispatched, review=review,
                    charts={'hours': hours, 'converted': converted, 'searched': searched, 'now': now,
                            'saps': [[k, saps.get(k, 0)] for k in STATE_ORDER],
                            'files': [[k, files.get(k, 0)] for k in FILE_ORDER]})

    # ----------------------------------------------------------- staging
    @app.get('/staging', response_class=HTMLResponse)
    def staging(request: Request):
        now = time.time()
        with store.reading(cfg) as db:
            health = meta(db, 'health', {})
            options = (health.get('driver') or {}).get('options') or {}
            restage = float(options.get('restage-after-hours', 12))
            timeout = float(options.get('request-timeout-hours', 96))
            requests = rows(db, """SELECT r.*, SUM(f.state='requested') AS waiting, SUM(f.state='online') AS online,
                    SUM(f.state='working') AS working, SUM(f.state='converted') AS converted,
                    SUM(f.state='failed') AS failed, SUM(f.state='excluded') AS excluded,
                    SUM(f.locality IN ('ONLINE','ONLINE_AND_NEARLINE')) AS on_disk, COUNT(f.surl) AS tracked
                FROM requests r LEFT JOIN files f ON f.request_id=r.id GROUP BY r.id ORDER BY r.submitted DESC""")
            troubled = rows(db, """SELECT * FROM files WHERE failures>0 OR submissions>1 OR state='failed'
                ORDER BY updated DESC LIMIT 200""")
            events = rows(db, """SELECT * FROM events WHERE kind IN ('submitted','submit_failed','status_failed',
                'manifest_failed','restage','refused','throttled','file_failed','retrieve_failed','excluded')
                ORDER BY time DESC LIMIT 100""")
            # Tape latency: request submission to dCache reporting the file on disk.
            latency = [r[0] / 3600 for r in db.execute("""SELECT t.time - q.submitted FROM transitions t
                JOIN files f ON f.surl=t.subject JOIN requests q ON q.id=f.request_id
                WHERE t.kind='file' AND t.new='online' AND t.time>q.submitted""")]
            kinds = rows(db, """SELECT kind, CAST((time - ?) / 3600 AS INTEGER) AS hour, COUNT(*) AS n FROM events
                WHERE time>? AND kind IN ('submitted','restage','refused','throttled','retrieve_failed','file_failed')
                GROUP BY kind, hour""", now - 72 * 3600, now - 72 * 3600)
        for r in requests:
            r['age_hours'] = (now - r['submitted']) / 3600 if r['submitted'] else None
            r['active'] = (r['waiting'] or 0) + (r['online'] or 0) + (r['working'] or 0) > 0
        return page(request, 'staging.html', requests=requests, troubled=troubled, events=events,
                    restage=restage, timeout=timeout,
                    charts={'latency': latency, 'kinds': kinds, 'start': now - 72 * 3600})

    # ---------------------------------------------------------- coverage
    @app.get('/coverage', response_class=HTMLResponse)
    def coverage(request: Request):
        with store.reading(cfg) as db:
            saps = rows(db, """SELECT s.key, s.position, s.files, s.state, s.detail, s.run_name, s.updated,
                    i.observation, i.sap, i.pointing, i.ra_deg, i.dec_deg, i.observed, i.beams_searched,
                    i.fingerprints, i.candidates, i.max_snr, COALESCE(x.n, 0) AS excluded
                FROM saps s LEFT JOIN sap_info i ON i.key=s.key
                LEFT JOIN (SELECT sap_key, COUNT(*) AS n FROM files WHERE state='excluded' GROUP BY sap_key) x
                    ON x.sap_key=s.key ORDER BY s.position""")
        states = {}
        for s in saps:
            states[s['state']] = states.get(s['state'], 0) + 1
        sky = [{'key': s['key'], 'ra': s['ra_deg'], 'dec': s['dec_deg'], 'state': s['state'],
                'pointing': s['pointing'], 'searched': s['beams_searched']}
               for s in saps if s['ra_deg'] is not None]
        return page(request, 'coverage.html', saps=saps, states=states, state_order=STATE_ORDER,
                    charts={'sky': sky})

    @app.get('/sap/{key}', response_class=HTMLResponse)
    def sap(request: Request, key: str):
        with store.reading(cfg) as db:
            row = db.execute('SELECT * FROM saps WHERE key=?', (key,)).fetchone()
            if row is None:
                raise HTTPException(404, f'No SAP {key}')
            info = db.execute('SELECT * FROM sap_info WHERE key=?', (key,)).fetchone()
            files = rows(db, 'SELECT * FROM files WHERE sap_key=? ORDER BY beam', key)
            requests = rows(db, 'SELECT * FROM requests WHERE sap_key=? ORDER BY submitted', key)
            events = rows(db, """SELECT * FROM events WHERE subject=? OR subject LIKE ? OR detail LIKE ?
                ORDER BY time DESC LIMIT 200""", key, key + '%', '%' + key + '%')
            beams = []
            if info:
                prefix = f'downsampled_{info["observation"]}_SAP{info["sap"]:03d}_BEAM%'
                beams = rows(db, """SELECT b.*, (SELECT status FROM attempts a WHERE a.item=b.item AND a.stage='classify'
                        ORDER BY a.id DESC LIMIT 1) AS classify,
                        (SELECT COUNT(*) FROM candidates c WHERE c.item=b.item AND c.type='candidate') AS positives
                    FROM beams b WHERE b.item LIKE ? ORDER BY b.beam, b.mtime DESC""", prefix)
                stages = rows(db, """SELECT item, stage, status, fingerprint, seconds, finished FROM attempts
                    WHERE id IN (SELECT MAX(id) FROM attempts WHERE item LIKE ? GROUP BY item, stage)""", prefix)
                # Earlier stages name the beam differently: the archive tar, and L<obs>_SAP<n>_B<beam>.
                short = f'{info["observation"]}_SAP{info["sap"]:03d}'
                names = {f['name'][:-4] if f['name'].endswith('.tar') else f['name']: f['beam'] for f in files}
                for a in rows(db, """SELECT item, stage, status, fingerprint, seconds, finished FROM attempts
                        WHERE id IN (SELECT MAX(id) FROM attempts WHERE (item LIKE ? AND stage='downsample')
                        OR (stage='retrieve' AND item LIKE ?) GROUP BY item, stage)""", short + '_B%', key + '_B%'):
                    beam_number = names.get(a['item']) if a['stage'] == 'retrieve' else int(a['item'].rsplit('_B', 1)[1])
                    if beam_number is not None:
                        stages.append(dict(a, item=f'downsampled_{short}_BEAM{beam_number:03d}_32bit_ff'))
                flatfield = db.execute("SELECT status, seconds, finished FROM attempts WHERE item=? AND stage='flatfield' "
                                       "ORDER BY id DESC LIMIT 1", (short,)).fetchone()
            else:
                stages, flatfield = [], None
        by_item = {}
        for s in stages:
            by_item.setdefault(s['item'], {})[s['stage']] = s
        layout = [{'beam': b['beam'], 'ra': b['ra_deg'], 'dec': b['dec_deg'], 'snr': b['max_cluster_snr'],
                   'clusters': b['clusters'], 'item': b['item'], 'positives': b['positives']}
                  for b in beams if b['ra_deg'] is not None]
        return page(request, 'sap.html', sap=dict(row), info=dict(info) if info else None, files=files,
                    requests=requests, events=events, beams=beams, stages=by_item,
                    stage_names=['retrieve', 'downsample', 'dedisperse', 'single_pulse', 'periodicity', 'classify'],
                    flatfield=dict(flatfield) if flatfield else None,
                    charts={'layout': layout})

    @app.get('/beam/{item}', response_class=HTMLResponse)
    def beam(request: Request, item: str):
        with store.reading(cfg) as db:
            results = rows(db, 'SELECT * FROM beams WHERE item=? ORDER BY mtime DESC', item)
            attempts = rows(db, 'SELECT * FROM attempts WHERE item=? ORDER BY id DESC', item)
            if not results and not attempts:
                raise HTTPException(404, f'No record of {item}')
            plots = rows(db, 'SELECT * FROM plots WHERE item=? ORDER BY kind, name', item)
            found = rows(db, """SELECT c.*, (SELECT label FROM r.reviews v WHERE v.key=c.key ORDER BY created DESC
                LIMIT 1) AS label FROM candidates c WHERE c.item=? ORDER BY c.kind, c.snr DESC""", item)
            archive = db.execute('SELECT uri FROM archive_beams WHERE item=? LIMIT 1', (item,)).fetchone()
            file = db.execute('SELECT * FROM files WHERE surl=?', (archive['uri'],)).fetchone() if archive else None
            runs = rows(db, 'SELECT * FROM beam_runs WHERE item=? ORDER BY id DESC', item)
            kept = next((k for k in kept_beams(db) if k['item'] == item), None)
        clusters = []
        if results:
            try:
                lines = (Path(results[0]['dir']) / 'clustered_candidates.txt').read_text().splitlines()[1:]
                for line in lines:
                    f = line.split()
                    if len(f) >= 5:
                        clusters.append([float(f[0]), float(f[1]), float(f[2]), int(float(f[4]))])
            except OSError:
                pass
        sap_key = next((c['sap_key'] for c in found if c['sap_key']), None)
        return page(request, 'beam.html', item=item, results=results, attempts=attempts, plots=plots,
                    found=found, file=dict(file) if file else None, runs=runs, sap_key=sap_key, kept=kept,
                    charts={'clusters': clusters, 'found': [
                        {'dm': c['dm'], 'time': c['time'], 'snr': c['snr'], 'type': c['type'], 'id': c['id']}
                        for c in found if c['kind'] == 'sp']})

    # -------------------------------------------------------- candidates
    def candidate_filter(params):
        clauses, args = [], []
        kind = params.get('type', 'queue')
        if kind == 'queue':
            clauses.append("c.type IN ('candidate','known_pulsar','periodic')")
        elif kind != 'all':
            clauses.append('c.type=?')
            args.append(kind)
        if params.get('pilot') != 'include':
            clauses.append('COALESCE(c.pilot, 0)=0')
        if params.get('min_snr'):
            clauses.append('c.snr>=?')
            args.append(float(params['min_snr']))
        if params.get('q'):
            clauses.append('c.item LIKE ?')
            args.append('%' + params['q'] + '%')
        review = params.get('review', 'all')
        latest = '(SELECT label FROM r.reviews v WHERE v.key=c.key ORDER BY created DESC LIMIT 1)'
        if review == 'unreviewed':
            clauses.append(f'{latest} IS NULL')
        elif review == 'reviewed':
            clauses.append(f'{latest} IS NOT NULL')
        elif review in LABELS:
            clauses.append(f'{latest}=?')
            args.append(review)
        return ' AND '.join(clauses) or '1=1', args, latest

    ORDERS = {'recent': 'COALESCE(c.slack_sent, 0) DESC, c.found DESC, c.snr DESC',
              'snr': 'c.snr DESC', 'dm': 'c.dm', 'probability': 'c.probability DESC'}

    def queue_ids(db, params):
        where, args, _ = candidate_filter(params)
        order = ORDERS.get(params.get('sort', 'recent'), ORDERS['recent'])
        return [r[0] for r in db.execute(f'SELECT c.id FROM candidates c WHERE {where} ORDER BY {order}', args)]

    @app.get('/candidates', response_class=HTMLResponse)
    def candidates(request: Request):
        params = dict(request.query_params)
        params.setdefault('type', 'queue')
        where, args, latest = candidate_filter(params)
        order = ORDERS.get(params.get('sort', 'recent'), ORDERS['recent'])
        number = max(1, int(params.get('page', 1) or 1))
        with store.reading(cfg) as db:
            count = db.execute(f'SELECT COUNT(*) FROM candidates c WHERE {where}', args).fetchone()[0]
            found = rows(db, f"""SELECT c.*, {latest} AS label,
                    (SELECT COUNT(*) FROM r.reviews v WHERE v.key=c.key) AS reviews
                FROM candidates c WHERE {where} ORDER BY {order} LIMIT 100 OFFSET ?""", *args, (number - 1) * 100)
            types = dict(db.execute('SELECT type, COUNT(*) FROM candidates GROUP BY type').fetchall())
        query = urlencode({k: v for k, v in params.items() if k != 'page'})
        return page(request, 'candidates.html', found=found, count=count, params=params, number=number,
                    pages=max(1, math.ceil(count / 100)), types=types, query=query)

    @app.get('/verify', response_class=HTMLResponse)
    def verify_queue(request: Request):
        params = dict(request.query_params)
        with store.reading(cfg) as db:
            ids = queue_ids(db, dict(params, review=params.get('review', 'unreviewed')))
        if ids:
            query = urlencode({k: v for k, v in params.items()})
            return RedirectResponse(f'/verify/{ids[0]}' + (f'?{query}' if query else ''), 303)
        return page(request, 'empty.html', message='Nothing waiting for review with these filters.')

    @app.get('/verify/{cid}', response_class=HTMLResponse)
    def verify(request: Request, cid: str):
        params = dict(request.query_params)
        with store.reading(cfg) as db:
            row = db.execute('SELECT * FROM candidates WHERE id=?', (cid,)).fetchone()
            if row is None:
                raise HTTPException(404, 'No such candidate')
            candidate = dict(row)
            queue = queue_ids(db, dict(params, review=params.get('review', 'all')))
            reviews = rows(db, 'SELECT * FROM r.reviews WHERE key=? ORDER BY created DESC', candidate['key'])
            snippet = db.execute('SELECT meta FROM snippets WHERE key=?', (candidate['key'],)).fetchone()
            plots = rows(db, 'SELECT id, kind, name FROM plots WHERE key=? OR (dir=? AND kind IN '
                             "('overview','clusters','rfi')) ORDER BY kind", candidate['key'], candidate['dir'])
            beam_run = db.execute("""SELECT observation_date, output_dir FROM beam_runs WHERE item=?
                ORDER BY id DESC LIMIT 1""", (candidate['item'],)).fetchone()
            periodic = db.execute('SELECT row, fold_data FROM periodic WHERE key=?', (candidate['key'],)).fetchone()
            periodic = dict(periodic) if periodic else None
            kept = next((k for k in kept_beams(db) if k['item'] == candidate['item']), None)
            others = rows(db, """SELECT id, type, dm, snr, time FROM candidates WHERE item=? AND id<>?
                AND kind=? ORDER BY snr DESC LIMIT 12""", candidate['item'], cid, candidate['kind'])
        initial = {}
        if snippet and candidate['kind'] == 'sp':
            loaded = load_snippet(cid)
            initial['view'] = view_payload(loaded)
            cached = responses.peek((str(loaded.path), ()))
            if cached is not None:
                initial['dm'] = cached
        if candidate['kind'] == 'periodic' and periodic and periodic['fold_data']:
            try:
                initial['fold'] = fold_payload(candidate['dir'], periodic['fold_data'], periodic['row'])
            except (OSError, ValueError, KeyError):
                pass
        position = queue.index(cid) if cid in queue else None
        neighbours = {'previous': queue[position - 1] if position else None,
                      'next': queue[position + 1] if position is not None and position + 1 < len(queue) else None,
                      'position': None if position is None else position + 1, 'total': len(queue)}
        query = urlencode(params)
        context = dict(candidate=candidate, reviews=reviews, plots=plots, neighbours=neighbours, query=query,
                       snippet=json.loads(snippet['meta']) if snippet else None, others=others,
                       observed=beam_run['observation_date'] if beam_run else None,
                       slack_threads=cfg.slack_threads, labels=LABELS, kept=kept, initial=initial)
        if candidate['kind'] == 'periodic':
            context['fold'] = json.loads(periodic['row']) if periodic else {}
            return page(request, 'verify_periodic.html', **context)
        return page(request, 'verify_sp.html', **context)

    # --------------------------------------------------------------- API
    def load_snippet(cid):
        with store.reading(cfg) as db:
            row = db.execute('SELECT path FROM snippets WHERE id=?', (cid,)).fetchone()
        if row is None or not Path(row['path']).is_file():
            raise HTTPException(404, 'No snippet for this candidate')
        return snippets.get(row['path'], lambda: Snippet(row['path']))

    @app.get('/api/sp/{cid}/view')
    def sp_view(cid: str, dm: float | None = None, tscrunch: int = 1, nsub: int = 81, window: float = 2.0,
                mask: str = '', clip: float = 99.0):
        snippet = load_snippet(cid)
        return JSONResponse(view_payload(snippet, dm, tscrunch, nsub, window,
                                         parse_mask(mask, snippet.data.shape[1]), clip))

    @app.get('/api/sp/{cid}/dm')
    def sp_dm(cid: str, mask: str = ''):
        snippet = load_snippet(cid)
        channels = tuple(parse_mask(mask, snippet.data.shape[1]))
        return JSONResponse(responses.get((str(snippet.path), channels), lambda: dm_payload(snippet, channels)))

    @app.get('/api/periodic/{cid}')
    def periodic_fold(cid: str):
        with store.reading(cfg) as db:
            row = db.execute("""SELECT p.fold_data, p.dir, p.row FROM candidates c JOIN periodic p ON p.key=c.key
                WHERE c.id=?""", (cid,)).fetchone()
        if row is None or not row['fold_data'] or not (Path(row['dir']) / row['fold_data']).is_file():
            raise HTTPException(404, 'No fold archive for this candidate')
        return JSONResponse(fold_payload(row['dir'], row['fold_data'], row['row']))

    @app.post('/api/review')
    async def review(request: Request):
        body = await request.json()
        label = body.get('label')
        reviewer = (body.get('reviewer') or '').strip()[:64]
        if label not in LABELS or not reviewer:
            raise HTTPException(400, 'A label and a reviewer name are required')
        with store.reading(cfg) as db:
            row = db.execute('SELECT key FROM candidates WHERE id=?', (body.get('id'),)).fetchone()
        if row is None:
            raise HTTPException(404, 'No such candidate')
        note = (body.get('note') or '').strip()[:4000]
        dm = body.get('dm')
        slack_ts = None
        if body.get('slack') and cfg.slack_threads:
            from web.slackthread import post_verdict
            try:
                slack_ts = post_verdict(cfg, row['key'], label, note, reviewer, dm)
            except Exception as error:
                logger.error('Slack reply failed: %s', error)
                raise HTTPException(502, f'Saved nothing: the Slack reply failed ({error})')
        db = store.reviews(cfg)
        try:
            with db:
                db.execute('INSERT INTO reviews(key,reviewer,label,note,dm,created,slack_ts) VALUES (?,?,?,?,?,?,?)',
                           (row['key'], reviewer, label, note, dm, time.time(), slack_ts))
            history = rows(db, 'SELECT * FROM reviews WHERE key=? ORDER BY created DESC', row['key'])
        finally:
            db.close()
        return JSONResponse(clean({'reviews': history}))

    @app.get('/plot/{plot_id}')
    def plot(plot_id: int):
        with store.reading(cfg) as db:
            row = db.execute('SELECT path FROM plots WHERE id=?', (plot_id,)).fetchone()
        if row is None or not row['path'].endswith('.png') or not Path(row['path']).is_file():
            raise HTTPException(404, 'Plot not found')
        return FileResponse(row['path'], media_type='image/png')

    @app.get('/snippet/{cid}.fil')
    def download(cid: str):
        with store.reading(cfg) as db:
            row = db.execute('SELECT path FROM snippets WHERE id=?', (cid,)).fetchone()
        if row is None or not Path(row['path']).is_file():
            raise HTTPException(404, 'No snippet for this candidate')
        return FileResponse(row['path'], media_type='application/octet-stream', filename=Path(row['path']).name)

    @app.get('/api/health')
    def api_health():
        with store.reading(cfg) as db:
            return JSONResponse(clean({'health': meta(db, 'health', {}), 'index': meta(db, 'last_index', {})}))

    @app.get('/assets/plotly.min.js')
    def plotly_js():
        import plotly
        return FileResponse(Path(plotly.__file__).parent / 'package_data' / 'plotly.min.js',
                            media_type='text/javascript', headers={'Cache-Control': 'max-age=86400'})

    return app


def rows(db, sql, *args):
    return [dict(r) for r in db.execute(sql, args)]


def meta(db, name, default=None):
    row = db.execute('SELECT value FROM meta WHERE name=?', (name,)).fetchone()
    return json.loads(row['value']) if row else default


def hourly(db, sql, since, hours):
    counts = [0] * hours
    for (value,) in db.execute(sql, (since,)):
        if value is not None:
            index = int((value - since) // 3600)
            if 0 <= index < hours:
                counts[index] += 1
    return counts
