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

from euroflash.beams import INCOHERENT_BEAMS
from web import store
from web.dynspec import SUBBANDS, Snippet, parse_mask, sweep_seconds
from web.keys import beam_label, parse_item
from web.store import LABELS

logger = logging.getLogger(__name__)
HERE = Path(__file__).parent
COOKIE = 'lotaas_web'
STATE_ORDER = ['not_queued', 'pending', 'staging', 'processing', 'flatfielding', 'prepared', 'dispatched', 'partial',
               'searched', 'attention', 'incomplete']
FILE_ORDER = ['not_queued', 'pending', 'requested', 'online', 'working', 'converted', 'searched', 'kept', 'failed', 'excluded']
# Single-pulse and periodic candidates are listed and reviewed apart. 'queue' is
# what waits for a person; the other types can be listed but are not queued.
KINDS = {'sp': {'types': ('candidate', 'known_pulsar', 'rejected', 'unclassified', 'unconfirmed'),
                'queue': ('candidate', 'known_pulsar'),
                'page': '/single-pulse', 'sort': 'recent'},
         'periodic': {'types': ('periodic', 'periodic_rfi'), 'queue': ('periodic',),
                      'page': '/periodic', 'sort': 'evidence'}}
CENTRE_MHZ = 135.25  # LOTAAS band centre, when a beam's own band is unknown
# A period found in several beams, or in two SAPs, of one observation is RFI,
# unless every fold of it sits at one DM above zero, as a bright pulsar seen in
# neighbouring beams would. The indexer groups the folds (periodic_families).
MULTIBEAM_BEAMS = 4


# A single-pulse event at the same (DM-aligned) moment in this many beams of its
# observation, at scattered DMs, is interference: hidden from the queue and list
# unless asked for (?coincident=include), like multi-beam periods.
COINCIDENT_BEAMS = 5


# How indexer.derive_known saw that a pulsar was in the observation.
KNOWN_ROUTES = {'fold': 'a fold at its period', 'redetection': 'the classifier redetected it',
                'rotation': 'these pulses keep its rotation'}


# The periodic search's shortest period; millisecond pulsars are out of its reach.
MIN_PERIOD_SECONDS = 0.016


def coincident_sql(alias):
    return f'({alias}.beams >= {COINCIDENT_BEAMS} AND NOT {alias}.consistent)'


def multibeam_sql(alias):
    f = alias
    return (f'(({f}.beams >= {MULTIBEAM_BEAMS} OR {f}.saps >= 2) AND NOT ({f}.dm_min >= 2 '
            f'AND {f}.dm_max - {f}.dm_min <= MAX(2, 0.1 * {f}.dm_max)))')


def typical_scattering(dm, frequency_mhz):
    """Median scatter broadening in seconds (Bhat et al. 2004, eq. 5).

    A review hint only: single lines of sight lie up to about ten times either
    side of it. A pulsar broadened by more than its period folds to a flat line.
    """
    if not dm or dm <= 0:
        return None
    x = math.log10(dm)
    return 10 ** (-6.46 + 0.154 * x + 1.07 * x * x - 3.86 * math.log10(frequency_mhz / 1000)) / 1000


def fold_summary(candidate):
    """Figures from a periodic fold record that help rank it in a list."""
    fold = json.loads(candidate.get('fold_row') or '{}')
    bins, chi2 = fold.get('fold_bins'), fold.get('fold_chi2')
    tau = typical_scattering(candidate.get('dm'), candidate.get('centre_mhz') or CENTRE_MHZ)
    period = candidate.get('period')
    return {'harmonics': fold.get('harmonic_count'), 'fold_bins': bins,
            'reduced_chi2': chi2 / (bins - 1) if chi2 is not None and bins and bins > 1 else None,
            'catalogue': ', '.join(m.get('name', '') for m in fold.get('catalogue_matches') or []),
            'tau': tau, 'smeared': bool(tau and period and tau > period)}


def number(params, name):
    """A numeric filter from the query string, or None when absent or not a number."""
    try:
        value = float(params.get(name) or 'nan')
    except ValueError:
        return None
    return value if math.isfinite(value) else None


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
    for unit in ('B', 'kB', 'MB', 'GB', 'TB', 'PB'):
        if value < 1000 or unit == 'PB':
            return f'{value:.0f} {unit}' if unit == 'B' else f'{value:.2f} {unit}' if unit == 'PB' else f'{value:.1f} {unit}'
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


def default_view(snippet):
    """The first display: about 64 subbands and a quarter of the pulse per time bin, smoothed a little."""
    from web.dynspec import detail_view
    return detail_view(snippet.width, snippet.data.shape[1])


def view_presets(snippet):
    """The displays a reviewer switches between: {name: (nsub, tscrunch, smooth)}."""
    from web.dynspec import SMOOTH_PIXELS, suggested_view
    nchans = snippet.data.shape[1]
    return {'detail': (*default_view(snippet), SMOOTH_PIXELS),
            'matched': (*suggested_view(snippet.meta.get('snr'), snippet.width, nchans), 0.0),
            'full': (nchans, 1, 0.0)}


def view_payload(snippet, dm=None, tscrunch=None, nsub=None, window=-1.0, mask=(), clip=99.0, auto_mask=True,
                 smooth=None):
    from web.dynspec import SMOOTH_PIXELS
    suggested_nsub, suggested_tscrunch = default_view(snippet)
    nsub = nsub or suggested_nsub
    tscrunch = tscrunch or suggested_tscrunch
    smooth = SMOOTH_PIXELS if smooth is None else max(0.0, min(float(smooth), 4.0))
    view = snippet.view(dm=dm, tscrunch=tscrunch, nsub=nsub, window=window or None, mask=mask,
                        clip=min(max(clip, 50.0), 100.0), auto_mask=auto_mask, smooth=smooth)
    # Ascending frequency, so the heatmap's rows run up the axis.
    order = np.argsort(view['freqs'])
    return clean({
        'dm': view['dm'], 'tsamp': view['tsamp'], 'tscrunch': view['tscrunch'], 'nsub': view['nsub'],
        'times': view['times'], 'freqs': view['freqs'][order], 'image': encode_image(view['image'][order]),
        'zmin': view['zmin'], 'zmax': view['zmax'], 'series': view['series'], 'boxcar': view['boxcar'],
        'peak_snr': view['peak_snr'], 'width': view['width'], 'best_snr': view['best_snr'],
        'best_width': view['best_width'], 'spectrum_on': view['spectrum_on'][order],
        'analysis_tsamp': view['analysis_tsamp'], 'width_seconds': view['width_seconds'],
        'reference_windows': view['reference_windows'], 'automatic_bad': view['automatic_bad'],
        'coverage_seconds': view['coverage_seconds'],
        'spectrum_off': view['spectrum_off'][order], 'masked': view['masked'],
        'unmasked_peak_snr': view['unmasked_peak_snr'], 'pixel_snr': view['pixel_snr'],
        'smooth': view['smooth'], 'smoothed_pixel_snr': view['smoothed_pixel_snr'],
        'profiles': view['profiles'], 'profile_freqs': view['profile_freqs'],
        'suggested': {'nsub': suggested_nsub, 'tscrunch': suggested_tscrunch},
        'sweep': sweep_seconds(snippet.dm, view['freqs'][order])}, 5)


def dm_payload(snippet, channels=(), auto_mask=True):
    r = snippet.dm_response(mask=list(channels), auto_mask=auto_mask)
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
                                       'unclassified': 'not sent to FETCH',
                                       'unconfirmed': 'low local significance',
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
            waiting = {}
            for kind in KINDS:
                where, args, _ = candidate_filter({'kind': kind, 'review': 'unreviewed'})
                waiting[kind] = db.execute(f'SELECT COUNT(*) FROM candidates c WHERE {where}', args).fetchone()[0]
        return templates.TemplateResponse(request, name, dict(
            context, health=health, last_index=last, waiting=waiting, path=request.url.path))

    # ---------------------------------------------------------- overview
    def catalogue_scope(request, forecast):
        project = request.query_params.get('project', '')
        if project:
            scope = next((p for p in forecast.get('projects', []) if p['project'] == project), None)
            if scope is None:
                raise HTTPException(404, 'Project not available in the catalogue')
            return scope, project
        return next((s for s in forecast.get('scopes', []) if s['key'] == 'all'), None), ''

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
            forecast = meta(db, 'forecast', {})
            rate = next((r for r in forecast.get('rates', []) if r['hours'] == 24), {})
            inventory = next((s for s in forecast.get('scopes', []) if s['key'] == 'inventory'), {})
            scope, selected_project = catalogue_scope(request, forecast)
            early = next((s for s in forecast.get('scopes', []) if s['key'] == 'ec'), None)
            if scope:
                saps, files = scope['sap_states'], scope['file_states']
            estimate = next((p for p in (scope or inventory).get('projections', []) if p['hours'] == 24), {})
            beams_day, files_day = rate.get('beams'), rate.get('files')
            kept = kept_beams(db)
            attention = rows(db, "SELECT key, detail, run_name, updated FROM saps WHERE state='attention' "
                                 "ORDER BY updated DESC")
            events = rows(db, 'SELECT * FROM events ORDER BY time DESC LIMIT 40')
            dispatched = rows(db, "SELECT key, run_name, updated FROM saps WHERE state='dispatched'")
            review = rows(db, """SELECT c.*, (SELECT label FROM r.reviews v WHERE v.key=c.key
                ORDER BY created DESC LIMIT 1) AS label FROM candidates c WHERE c.type='candidate'
                ORDER BY c.found DESC LIMIT 8""")
            _, lotaas = lotaas_summary(db)
            _, pulsars = pulsar_summary(db)
        total = sum(saps.values())
        remaining = sum(v for k, v in saps.items() if k not in ('searched', 'incomplete'))
        early_day = next((p for p in (early or {}).get('projections', []) if p['hours'] == 24), {})
        return page(request, 'overview.html', lotaas=lotaas, pulsars=pulsars, saps=saps, files=files, total=total, remaining=remaining,
                    early=early, early_day=early_day,
                    beams_searched=beams_searched, campaign_beams=campaign_beams,
                    eta_days=estimate.get('pace_days'), forecast=forecast, inventory=inventory,
                    scope=scope, selected_project=selected_project, estimate=estimate,
                    beams_day=beams_day, files_day=files_day, to_search=inventory.get('remaining_beams'),
                    kept=kept, kept_bytes=sum(k['bytes'] or 0 for k in kept),
                    state_order=STATE_ORDER, file_order=FILE_ORDER, attention=attention, events=events,
                    dispatched=dispatched, review=review,
                    charts={'hours': hours, 'retrieved': forecast.get('charts', {}).get('retrieved', [0] * hours),
                            'searched': forecast.get('charts', {}).get('searched', [0] * hours),
                            'now': forecast.get('time', now),
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
            forecast = meta(db, 'forecast', {})
            scope, selected_project = catalogue_scope(request, forecast)
            early = scope is not None and scope.get('key') == 'ec'
            if scope and not early:
                condition, args = ('c.project=?', (selected_project,)) if selected_project else ('1', ())
                catalogue_saps = rows(db, f"""SELECT c.*, i.ra_deg, i.dec_deg, i.pointing
                    FROM catalogue_saps c LEFT JOIN sap_info i ON i.key=c.queue_key
                    WHERE {condition} ORDER BY c.project, c.observation, c.sap""", *args)
                unmapped = rows(db, f"""SELECT c.* FROM catalogue_observations c
                    WHERE {condition} AND NOT EXISTS (SELECT 1 FROM catalogue_saps s
                        WHERE s.project=c.project AND s.observation=c.observation)
                    ORDER BY c.project, c.observation""", *args)
                sky = [{'key': s['queue_key'], 'ra': s['ra_deg'], 'dec': s['dec_deg'], 'state': s['state'],
                        'pointing': s['pointing'], 'searched': s['searched_files']}
                       for s in catalogue_saps if s['ra_deg'] is not None]
                return page(request, 'catalogue_coverage.html', saps=catalogue_saps, unmapped=unmapped,
                            scope=scope, forecast=forecast, selected_project=selected_project,
                            states=scope['sap_states'], state_order=STATE_ORDER, charts={'sky': sky})
            saps = rows(db, """SELECT s.key, s.position, s.files, s.state, s.detail, s.run_name, s.updated,
                    i.observation, i.sap, i.pointing, i.ra_deg, i.dec_deg, i.observed, i.beams_searched,
                    i.fingerprints, i.candidates, i.max_snr, COALESCE(x.n, 0) AS excluded, s.source
                FROM saps s LEFT JOIN sap_info i ON i.key=s.key
                LEFT JOIN (SELECT sap_key, COUNT(*) AS n FROM files WHERE state='excluded' GROUP BY sap_key) x
                    ON x.sap_key=s.key WHERE ? = 0 OR s.source='spider' ORDER BY s.position""", int(early))
        states = {}
        for s in saps:
            states[s['state']] = states.get(s['state'], 0) + 1
        sky = [{'key': s['key'], 'ra': s['ra_deg'], 'dec': s['dec_deg'], 'state': s['state'],
                'pointing': s['pointing'], 'searched': s['beams_searched']}
               for s in saps if s['ra_deg'] is not None]
        return page(request, 'coverage.html', saps=saps, states=states, state_order=STATE_ORDER,
                    early=early, charts={'sky': sky})

    @app.get('/catalogue/{project}/{observation}/{sap}', response_class=HTMLResponse)
    def catalogue_sap(request: Request, project: str, observation: str, sap: int):
        with store.reading(cfg) as db:
            row = db.execute('SELECT * FROM catalogue_saps WHERE project=? AND observation=? AND sap=?',
                             (project, observation, sap)).fetchone()
            if row is None:
                raise HTTPException(404, 'SAP not found in the catalogue')
            from web.forecast import SEARCHED_URIS, excluded_sql
            files = rows(db, f"""SELECT c.name, c.bytes, c.beam, c.part, f.sap_key,
                CASE WHEN s.uri IS NOT NULL THEN 'searched'
                     WHEN f.surl IS NULL THEN 'not_queued' ELSE f.state END AS state
                FROM catalogue_files c LEFT JOIN files f ON f.surl=c.uri
                LEFT JOIN ({SEARCHED_URIS}) s ON s.uri=c.uri
                WHERE c.project=? AND c.observation=? AND c.sap=? AND {excluded_sql(cfg, 'c.beam')}
                ORDER BY c.beam, c.part, c.name""", project, observation, sap)
        return page(request, 'catalogue_sap.html', sap=dict(row), files=files)

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
                        OR (stage='retrieve' AND (item LIKE ? OR item IN (SELECT substr(name, 1, length(name) - 4)
                            FROM files WHERE sap_key=?))) GROUP BY item, stage)""", short + '_B%', key + '_B%', key):
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
        levelling = None
        if row['sap_dir']:
            try:
                levelling = json.loads((Path(row['sap_dir']) / 'row-levelling.json').read_text())
            except (OSError, ValueError):
                pass
        return page(request, 'sap.html', sap=dict(row), info=dict(info) if info else None, files=files,
                    levelling=levelling,
                    requests=requests, events=events, beams=beams, stages=by_item,
                    stage_names=['retrieve', 'downsample', 'dedisperse', 'single_pulse', 'sp_classify', 'periodicity',
                                 'periodicity_fold', 'classify'],
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
        kind = params.get('kind') if params.get('kind') in KINDS else 'sp'
        clauses, args = ['c.kind=?'], [kind]
        chosen = params.get('type', 'queue')
        types = (KINDS[kind]['queue'] if chosen == 'queue' else KINDS[kind]['types'] if chosen == 'all'
                 else (chosen,))
        clauses.append(f"c.type IN ({','.join('?' * len(types))})")
        args.extend(types)
        if params.get('pilot') != 'include':
            clauses.append('COALESCE(c.pilot, 0)=0')
        if params.get('incoherent') != 'include':
            # Searched before the campaign left the incoherent beam out.
            for beam in INCOHERENT_BEAMS:
                clauses.append("c.item NOT LIKE ? ESCAPE '\\'")
                args.append(f'%\\_BEAM{beam:03d}\\_%')
        for name, clause in (('min_snr', 'c.snr>=?'), ('min_dm', 'c.dm>=?'), ('max_dm', 'c.dm<=?'),
                             ('min_period', 'c.period>=?'), ('max_period', 'c.period<=?')):
            value = number(params, name)
            if value is not None:
                clauses.append(clause)
                args.append(value)
        if params.get('q'):
            clauses.append('c.item LIKE ?')
            args.append('%' + params['q'] + '%')
        if kind == 'sp' and params.get('coincident') != 'include':
            clauses.append(f'NOT EXISTS (SELECT 1 FROM sp_coincidence x WHERE x.key=c.key AND {coincident_sql("x")})')
        if kind == 'sp' and params.get('known') != 'include':
            # Pulses of a catalogued pulsar seen away from its own beam (indexer.derive_known).
            clauses.append('NOT EXISTS (SELECT 1 FROM sp_known k WHERE k.key=c.key)')
        if kind == 'periodic' and params.get('multibeam') != 'include':
            clauses.append(f'NOT EXISTS (SELECT 1 FROM periodic_families pf WHERE pf.key=c.key '
                           f'AND {multibeam_sql("pf")})')
        if kind == 'periodic':
            triage = params.get('triage', 'strong' if chosen == 'queue' else 'all')
            if triage in ('strong', 'deferred', 'related'):
                clauses.append('EXISTS (SELECT 1 FROM periodic_triage t WHERE t.key=c.key AND t.status=?)')
                args.append(triage)
        review = params.get('review', 'all')
        latest = '(SELECT label FROM r.reviews v WHERE v.key=c.key ORDER BY created DESC LIMIT 1)'
        if review == 'unreviewed':
            clauses.append(f'{latest} IS NULL')
        elif review == 'reviewed':
            clauses.append(f'{latest} IS NOT NULL')
        elif review in LABELS:
            clauses.append(f'{latest}=?')
            args.append(review)
        return ' AND '.join(clauses), args, latest

    # A periodic candidate's snr column holds its search statistic.
    ORDERS = {'recent': 'c.found DESC, c.snr DESC',
              'snr': 'c.snr DESC', 'dm': 'c.dm', 'probability': 'c.probability DESC', 'period': 'c.period',
              'evidence': '(SELECT score FROM periodic_triage t WHERE t.key=c.key) DESC, c.snr DESC'}

    LATEST = '(SELECT label FROM r.reviews v WHERE v.key=c.key ORDER BY created DESC LIMIT 1)'

    def ordering(params):
        """Candidates without a verdict first, then the chosen order, in every list and queue."""
        kind = params.get('kind') if params.get('kind') in KINDS else 'sp'
        return f'({LATEST} IS NOT NULL), ' + ORDERS.get(params.get('sort'), ORDERS[KINDS[kind]['sort']])

    def queue_rows(db, params):
        """[(id, reviewed)] of a queue, in its order."""
        where, args, _ = candidate_filter(params)
        return [(r[0], r[1] is not None) for r in db.execute(
            f'SELECT c.id, {LATEST} FROM candidates c WHERE {where} ORDER BY {ordering(params)}', args)]

    def queue_ids(db, params):
        return [cid for cid, _ in queue_rows(db, params)]

    def next_unreviewed(queue, cid):
        """The first candidate without a verdict after this one in the queue, wrapping round."""
        ids = [q[0] for q in queue]
        start = ids.index(cid) + 1 if cid in ids else 0
        for other, reviewed in queue[start:] + queue[:start]:
            if not reviewed and other != cid:
                return other
        return None

    def listing(request, kind, template):
        params = dict(request.query_params, kind=kind)
        params.setdefault('type', 'queue')
        if kind == 'periodic':
            params.setdefault('triage', 'strong' if params['type'] == 'queue' else 'all')
        where, args, latest = candidate_filter(params)
        try:
            number_ = max(1, int(params.get('page') or 1))
        except ValueError:
            number_ = 1
        with store.reading(cfg) as db:
            count = db.execute(f'SELECT COUNT(*) FROM candidates c WHERE {where}', args).fetchone()[0]
            found = rows(db, f"""SELECT c.*, {latest} AS label,
                    (SELECT COUNT(*) FROM r.reviews v WHERE v.key=c.key) AS reviews, p.row AS fold_row,
                    (SELECT (b.nu_min + b.nu_max) / 2 FROM beams b WHERE b.item=c.item LIMIT 1) AS centre_mhz,
                    f.beams AS family_beams, f.saps AS family_saps, f.dm_min AS family_dm_min,
                    f.dm_max AS family_dm_max, COALESCE({multibeam_sql('f')}, 0) AS multibeam,
                    t.status AS triage_status, t.score AS repeatability, t.members AS group_members,
                    t.reasons AS triage_reasons, x.beams AS coincident_beams, x.saps AS coincident_saps,
                    x.dm_min AS coincident_dm_min, x.dm_max AS coincident_dm_max,
                    COALESCE({coincident_sql('x')}, 0) AS coincident,
                    k.name AS known_name, k.separation_deg AS known_separation, k.route AS known_route
                FROM candidates c LEFT JOIN periodic p ON p.key=c.key
                LEFT JOIN periodic_families f ON f.key=c.key
                LEFT JOIN periodic_triage t ON t.key=c.key
                LEFT JOIN sp_coincidence x ON x.key=c.key
                LEFT JOIN sp_known k ON k.key=c.key WHERE {where}
                ORDER BY {ordering(params)} LIMIT 100 OFFSET ?""", *args, (number_ - 1) * 100)
            # Counts per type under the other filters, for the type selector.
            every, every_args, _ = candidate_filter(dict(params, type='all'))
            types = dict(db.execute(f'SELECT c.type, COUNT(*) FROM candidates c WHERE {every} GROUP BY c.type',
                                    every_args).fetchall())
            triage_counts = {}
            hidden = known_hidden = 0
            if kind == 'sp' and params.get('coincident') != 'include':
                shown, shown_args, _ = candidate_filter(dict(params, coincident='include'))
                hidden = db.execute(f'SELECT COUNT(*) FROM candidates c WHERE {shown}', shown_args).fetchone()[0] - count
            if kind == 'sp' and params.get('known') != 'include':
                shown, shown_args, _ = candidate_filter(dict(params, known='include'))
                known_hidden = db.execute(f'SELECT COUNT(*) FROM candidates c WHERE {shown}',
                                          shown_args).fetchone()[0] - count
            if kind == 'periodic':
                all_where, all_args, _ = candidate_filter(dict(params, triage='all'))
                triage_counts = dict(db.execute(f'''SELECT COALESCE(t.status, 'pending'), COUNT(*)
                    FROM candidates c LEFT JOIN periodic_triage t ON t.key=c.key
                    WHERE {all_where} GROUP BY t.status''', all_args).fetchall())
        if kind == 'periodic':
            for c in found:
                c.update(fold_summary(c))
                c['triage_reasons'] = json.loads(c['triage_reasons'] or '[]')
        query = urlencode({k: v for k, v in params.items() if k not in ('page', 'kind')})
        return page(request, template, found=found, count=count, params=params, number=number_,
                    pages=max(1, math.ceil(count / 100)), types=types, query=query, kind=kind,
                    triage_counts=triage_counts, coincident_hidden=hidden, known_hidden=known_hidden)

    @app.get('/single-pulse', response_class=HTMLResponse)
    def single_pulse(request: Request):
        return listing(request, 'sp', 'single_pulse.html')

    @app.get('/periodic', response_class=HTMLResponse)
    def periodic(request: Request):
        return listing(request, 'periodic', 'periodic.html')

    @app.get('/candidates')
    def candidates(request: Request):
        # The combined list was split in two; old links land on the right half.
        params = dict(request.query_params)
        target = '/periodic' if params.get('type') in KINDS['periodic']['types'] else '/single-pulse'
        if params.get('type') == 'periodic':
            params['type'] = 'queue'
        return RedirectResponse(target + ('?' + urlencode(params) if params else ''), 301)

    @app.get('/verify', response_class=HTMLResponse)
    def verify_queue(request: Request):
        params = dict(request.query_params)
        params['kind'] = params.get('kind') if params.get('kind') in KINDS else 'sp'
        with store.reading(cfg) as db:
            ids = queue_ids(db, dict(params, review=params.get('review', 'unreviewed')))
        if ids:
            return RedirectResponse(f'/verify/{ids[0]}?{urlencode(params)}', 303)
        return page(request, 'empty.html', message='Nothing waiting for review with these filters.',
                    back=KINDS[params['kind']]['page'], section=params['kind'])

    @app.get('/verify/{cid}', response_class=HTMLResponse)
    def verify(request: Request, cid: str):
        params = dict(request.query_params)
        with store.reading(cfg) as db:
            row = db.execute('SELECT * FROM candidates WHERE id=?', (cid,)).fetchone()
            if row is None:
                raise HTTPException(404, 'No such candidate')
            candidate = dict(row)
            # Previous/next stay among candidates of the same kind.
            params['kind'] = candidate['kind'] if candidate['kind'] in KINDS else 'sp'
            rows_ = queue_rows(db, dict(params, review=params.get('review', 'all')))
            queue = [q[0] for q in rows_]
            reviews = rows(db, 'SELECT * FROM r.reviews WHERE key=? ORDER BY created DESC', candidate['key'])
            snippet = db.execute('SELECT meta FROM snippets WHERE key=?', (candidate['key'],)).fetchone()
            plots = rows(db, 'SELECT id, kind, name FROM plots WHERE key=? OR (dir=? AND kind IN '
                             "('overview','clusters','rfi')) ORDER BY kind", candidate['key'], candidate['dir'])
            beam_run = db.execute("""SELECT observation_date, output_dir FROM beam_runs WHERE item=?
                ORDER BY id DESC LIMIT 1""", (candidate['item'],)).fetchone()
            periodic = db.execute('SELECT row, fold_data FROM periodic WHERE key=?', (candidate['key'],)).fetchone()
            periodic = dict(periodic) if periodic else None
            triage = db.execute('SELECT * FROM periodic_triage WHERE key=?', (candidate['key'],)).fetchone()
            triage = dict(triage) if triage else None
            if triage:
                triage['evidence'] = json.loads(triage['evidence'])
                triage['reasons'] = json.loads(triage['reasons'])
                triage['representative_id'] = db.execute('SELECT id FROM candidates WHERE key=?',
                                                       (triage['representative'],)).fetchone()[0]
            family = db.execute(f"""SELECT f.*, {multibeam_sql('f')} AS multibeam FROM periodic_families f
                WHERE f.key=?""", (candidate['key'],)).fetchone()
            family = dict(family) if family else None
            relatives = rows(db, """SELECT c.id, c.item, c.dm, c.period, c.snr FROM periodic_families f
                JOIN candidates c ON c.key=f.key WHERE f.family=? AND f.key<>? ORDER BY c.snr DESC LIMIT 40""",
                family['family'], candidate['key']) if family else []
            kept = next((k for k in kept_beams(db) if k['item'] == candidate['item']), None)
            coincidence = db.execute(f'SELECT x.*, {coincident_sql("x")} AS rfi FROM sp_coincidence x WHERE x.key=?',
                                     (candidate['key'],)).fetchone()
            coincidence = dict(coincidence) if coincidence else None
            sweep = db.execute('SELECT * FROM sp_sweep WHERE key=?', (candidate['key'],)).fetchone()
            sweep = dict(sweep) if sweep else None
            known = db.execute('SELECT * FROM sp_known WHERE key=?', (candidate['key'],)).fetchone()
            known = dict(known) if known else None
            if known:
                known['evidence'] = ', '.join(KNOWN_ROUTES[r] for r in known['route'].split('+') if r in KNOWN_ROUTES)
            others = rows(db, """SELECT id, type, dm, snr, time FROM candidates WHERE item=? AND id<>?
                AND kind=? ORDER BY snr DESC LIMIT 12""", candidate['item'], cid, candidate['kind'])
        initial = {}
        display = None
        if snippet and candidate['kind'] == 'sp':
            loaded = load_snippet(cid)
            initial['view'] = view_payload(loaded)
            nsub, tscrunch = default_view(loaded)
            presets = view_presets(loaded)
            display = {'nsub': nsub, 'tscrunch': tscrunch, 'width': loaded.width, 'presets': presets,
                       'smooth': presets['detail'][2],
                       'scrunches': sorted({1, 2, 4, 8, 16, 32} | {p[1] for p in presets.values()}),
                       'subbands': [n for n in SUBBANDS if n <= loaded.data.shape[1] and loaded.data.shape[1] % n == 0],
                       'fch1': float(loaded.header['fch1']), 'foff': float(loaded.header['foff']),
                       'nchans': int(loaded.data.shape[1])}
            cached = responses.peek((str(loaded.path), (), True))
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
                      'position': None if position is None else position + 1, 'total': len(queue),
                      'next_unreviewed': next_unreviewed(rows_, cid),
                      'unreviewed': sum(1 for other, reviewed in rows_ if not reviewed and other != cid)}
        query = urlencode(params)
        context = dict(candidate=candidate, reviews=reviews, plots=plots, neighbours=neighbours, query=query,
                       snippet=json.loads(snippet['meta']) if snippet else None, others=others,
                       observed=beam_run['observation_date'] if beam_run else None,
                       labels=LABELS, kept=kept, initial=initial, display=display,
                       section=params['kind'], back=KINDS[params['kind']]['page'], coincidence=coincidence,
                       known=known, sweep=sweep)
        if candidate['kind'] == 'periodic':
            context['fold'] = json.loads(periodic['row']) if periodic else {}
            context.update(family=family, relatives=relatives, triage=triage)
            return page(request, 'verify_periodic.html', **context)
        return page(request, 'verify_sp.html', **context)

    # ------------------------------------------------ pulsars LOFAR has not seen
    PULSAR_SHOWS = {'unseen': 'Never reported by LOFAR', 'found': 'Found here, never reported by LOFAR',
                    'all': 'Every catalogued pulsar near a searched beam'}

    def pulsar_summary(db):
        found = rows(db, """SELECT k.*,
                (SELECT id FROM candidates c WHERE c.kind='sp' AND c.item=k.sp_item AND c.snr=k.sp_best_snr
                 ORDER BY c.type='known_pulsar' DESC LIMIT 1) AS sp_id,
                (SELECT id FROM candidates c WHERE c.kind='periodic' AND c.item=k.periodic_item
                 AND c.snr=k.periodic_best LIMIT 1) AS periodic_id
            FROM catalogue_pulsars k""")
        for k in found:
            k['unseen'] = not k['lofar']
            k['status'] = ('both' if k['sp_count'] and k['periodic_count'] else 'single pulses' if k['sp_count']
                           else 'periodic' if k['periodic_count'] else
                           'millisecond' if k['period'] < MIN_PERIOD_SECONDS else
                           'scattered' if (k['scatter_ms'] or 0) / 1000 > k['period'] else 'not found')
            k['found'] = k['status'] in ('both', 'single pulses', 'periodic')
        unseen = [k for k in found if k['unseen']]
        summary = {'near': len(found), 'found': sum(k['found'] for k in found),
                   'lotaas': sum(1 for k in found if 'LOTAAS' in (k['lofar'] or '')),
                   'lofar_other': sum(1 for k in found if k['lofar'] and 'LOTAAS' not in k['lofar']),
                   'unseen': len(unseen), 'unseen_found': sum(k['found'] for k in unseen),
                   'unseen_limits': sum(1 for k in unseen if k['lofar_limits']),
                   'unseen_open': sum(1 for k in unseen if k['status'] == 'not found'),
                   'unseen_out': sum(1 for k in unseen if k['status'] in ('millisecond', 'scattered'))}
        return found, summary

    @app.get('/pulsars', response_class=HTMLResponse)
    def pulsars(request: Request):
        from web.indexer import BEAM_RADIUS_DEG, SEARCHED_RADIUS_DEG
        show = request.query_params.get('show', 'unseen')
        show = show if show in PULSAR_SHOWS else 'unseen'
        with store.reading(cfg) as db:
            found, summary = pulsar_summary(db)
        chosen = {'unseen': lambda k: k['unseen'], 'found': lambda k: k['unseen'] and k['found'],
                  'all': lambda k: True}[show]
        # Found first: a pulsar no LOFAR paper reports, found here, is the point of the page.
        # Then the brightest of those not found, then those the search cannot see.
        order = {'both': 0, 'periodic': 1, 'single pulses': 2, 'not found': 3, 'scattered': 4, 'millisecond': 5}
        shown = sorted((k for k in found if chosen(k)),
                       key=lambda k: (order[k['status']], -(k['flux_mjy'] or 0), k['psrj']))
        return page(request, 'pulsars.html', found=shown, summary=summary, show=show, shows=PULSAR_SHOWS,
                    searched_radius=SEARCHED_RADIUS_DEG, beam_radius=BEAM_RADIUS_DEG, min_period=MIN_PERIOD_SECONDS)

    # ------------------------------------------------------ LOTAAS sources
    LOTAAS_SHOWS = {'searched': 'In fields searched so far', 'all': 'All published LOTAAS sources',
                    'discoveries': 'LOTAAS discoveries', 'single': 'Found by LOTAAS in single pulses',
                    'missed': 'Searched but not found', 'found': 'Redetected'}

    def lotaas_summary(db):
        # The links open the redetection itself, as on /pulsars: the lowest id of the beam's
        # candidates was any event there (a FETCH reject at DM 2164.8 for B0823+26).
        rows_ = rows(db, """SELECT k.*,
                (SELECT id FROM candidates c WHERE c.kind='sp' AND c.item=k.sp_item AND c.snr=k.sp_best_snr
                 ORDER BY c.type='known_pulsar' DESC LIMIT 1) AS sp_id,
                (SELECT id FROM candidates c WHERE c.kind='periodic' AND c.item=k.periodic_item
                 AND c.snr=k.periodic_best LIMIT 1) AS periodic_id
            FROM lotaas_sources k""")
        for k in rows_:
            k['searched'] = k['observation'] is not None
            k['status'] = ('not searched yet' if not k['searched'] else
                           'both' if k['sp_count'] and k['periodic_count'] else
                           'single pulses' if k['sp_count'] else 'periodic' if k['periodic_count'] else
                           'out of reach' if k['period'] < MIN_PERIOD_SECONDS else 'missed')
            k['lotaas_single'] = k['lotaas_mode'] != 'periodic'
            k['lotaas_periodic'] = k['lotaas_mode'] != 'single pulse'
        searched = [k for k in rows_ if k['searched']]
        found = [k for k in searched if k['status'] not in ('missed', 'out of reach')]
        summary = {
            'total': len(rows_), 'discoveries': sum(k['discovery'] for k in rows_),
            'single': sum(k['lotaas_single'] for k in rows_), 'searched': len(searched), 'found': len(found),
            'periodic_expected': sum(k['lotaas_periodic'] for k in searched),
            'periodic_found': sum(1 for k in searched if k['lotaas_periodic'] and k['periodic_count']),
            'single_expected': sum(k['lotaas_single'] for k in searched),
            'single_found': sum(1 for k in searched if k['lotaas_single'] and k['sp_count']),
            'sp_any': sum(1 for k in searched if k['sp_count']),
            'periodic_any': sum(1 for k in searched if k['periodic_count']),
            'missed': sum(1 for k in searched if k['status'] == 'missed'),
            'out_of_reach': sum(1 for k in searched if k['status'] == 'out of reach'),
            'missed_near': sum(1 for k in searched if k['status'] == 'missed' and (k['beams_near'] or 0) > 0)}
        return rows_, summary

    @app.get('/lotaas', response_class=HTMLResponse)
    def lotaas_sources(request: Request):
        from web.indexer import BEAM_RADIUS_DEG, SEARCHED_RADIUS_DEG
        show = request.query_params.get('show', 'searched')
        show = show if show in LOTAAS_SHOWS else 'searched'
        with store.reading(cfg) as db:
            found, summary = lotaas_summary(db)
        chosen = {'searched': lambda k: k['searched'], 'all': lambda k: True,
                  'discoveries': lambda k: k['discovery'], 'single': lambda k: k['lotaas_single'],
                  'missed': lambda k: k['status'] == 'missed',
                  'found': lambda k: k['status'] in ('both', 'periodic', 'single pulses')}[show]
        # What was redetected first, then what was missed, then what the search cannot see or has not reached.
        order = {'both': 0, 'periodic': 1, 'single pulses': 2, 'missed': 3, 'out of reach': 4, 'not searched yet': 5}
        shown = sorted((k for k in found if chosen(k)),
                       key=lambda k: (order[k['status']], k['separation_deg'] if k['searched'] else 0, k['psrj']))
        return page(request, 'lotaas.html', found=shown, summary=summary, show=show, shows=LOTAAS_SHOWS,
                    searched_radius=SEARCHED_RADIUS_DEG, beam_radius=BEAM_RADIUS_DEG)

    # --------------------------------------------------------------- API
    def load_snippet(cid):
        with store.reading(cfg) as db:
            row = db.execute('SELECT path FROM snippets WHERE id=?', (cid,)).fetchone()
        if row is None or not Path(row['path']).is_file():
            raise HTTPException(404, 'No snippet for this candidate')
        return snippets.get(row['path'], lambda: Snippet(row['path']))

    @app.get('/api/sp/{cid}/view')
    def sp_view(cid: str, dm: float | None = None, tscrunch: int | None = None, nsub: int | None = None,
                window: float = -1.0, mask: str = '', clip: float = 99.0, auto_mask: bool = True,
                smooth: float | None = None):
        snippet = load_snippet(cid)
        return JSONResponse(view_payload(snippet, dm, tscrunch, nsub, window,
                                         parse_mask(mask, snippet.data.shape[1]), clip, auto_mask, smooth))

    @app.get('/api/sp/{cid}/dm')
    def sp_dm(cid: str, mask: str = '', auto_mask: bool = True):
        snippet = load_snippet(cid)
        channels = tuple(parse_mask(mask, snippet.data.shape[1]))
        return JSONResponse(responses.get((str(snippet.path), channels, auto_mask),
                                         lambda: dm_payload(snippet, channels, auto_mask)))

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
        mask = (body.get('mask') or '').strip()[:500]
        if mask:
            # A verdict reached with channels removed by hand must say which.
            note = (note + '\n' if note else '') + f'[reviewer mask: {mask}]'
        db = store.reviews(cfg)
        try:
            with db:
                db.execute('INSERT INTO reviews(key,reviewer,label,note,dm,created) VALUES (?,?,?,?,?,?)',
                           (row['key'], reviewer, label, note, dm, time.time()))
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

    @app.get('/api/forecast')
    def api_forecast():
        with store.reading(cfg) as db:
            return JSONResponse(clean(meta(db, 'forecast', {})))

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
