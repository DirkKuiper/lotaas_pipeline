"""Build web.sqlite from the campaign's own records, reading them and nothing else.

A pass
  1. mirrors the campaign state database (SAPs, files, requests, events) and
     records every state change it sees, timed by the source's own 'updated';
  2. mirrors the ledger (runs, attempts, beam runs, detections, Slack posts,
     archive beams), incrementally where rows only append;
  3. scans the results tree for searched beams, their plots and periodic folds;
  4. derives per-SAP coverage and one candidate table across both searches;
  5. records the health of the driver, the dispatch, SSH and the disk.

Sources are attached read-only and each statement against them is short, so
the driver, which writes them in rollback-journal mode, is held up for no
more than a single query.
"""
import bisect
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from urllib.parse import quote

from web import store
from web.keys import item_of, parse_item, short_id, sp_key

logger = logging.getLogger(__name__)

ARCHIVE = re.compile(r'L(\d+)_SAP(\d+)_B(\d+)_')
PLOT = re.compile(r'^DM(?P<dm>[-\d.]+)_Width(?P<width>\d+)_SNR(?P<snr>[-\d.]+)\.png$')
POINTING = re.compile(r'(P\d+[A-Z]?)')
PRUNE = {'downloads', 'extracted', 'prepared', 'dispatch', 'data', 'logs', 'staging', 'catalogue',
         'held', 'snippets', 'containers', 'trials', '__pycache__', 'input', 'source'}
STALE_SECONDS = 2 * 86400
FAMILY_TOLERANCE = 5e-4  # relative period difference within one periodic family
# Single-pulse coincidence: events within this many seconds (or 1.5 widths), once
# aligned for their DM, are one moment. With the ~0.85 events a second of an
# RFI-rich observation, four other beams fall inside +-0.5 s by chance 1.2% of
# the time, five 0.25%.
COINCIDENCE_SECONDS = 0.5
KINDS_QUEUED = ('candidate', 'known_pulsar')
K_DM = 4148.808
COINCIDENT_BEAMS = 5      # as web.app: this many beams at scattered DMs is interference
# Catalogued pulsars this close to a searched beam are listed; within
# BEAM_RADIUS_DEG of one the search should see a bright one.
KNOWN_PULSAR_RADIUS_DEG = 2.0
BEAM_RADIUS_DEG = 0.5
# A published source counts as searched once a campaign beam lies this close.
SEARCHED_RADIUS_DEG = 1.0


def uri(path):
    return f'file:{quote(str(path))}?mode=ro'


def run_of(output_dir):
    match = re.search(r'/lotaas-runs/([^/]+)/', output_dir or '')
    return match[1] if match else None


def fp16_of(output_dir):
    parts = Path(output_dir or '').parts
    if 'processed' in parts:
        index = len(parts) - 1 - parts[::-1].index('processed')
        if index + 2 < len(parts):
            return parts[index + 2]
    return None


def observation_of(item):
    parsed = parse_item(item)
    return parsed[0] if parsed else None


def sap_of(item):
    parsed = parse_item(item)
    return parsed[1] if parsed else None


def key_of(kind, item, dm, width, snr):
    if None in (kind, item, dm, width, snr):
        return None
    return sp_key(kind, item, dm, width, snr)


def sexagesimal(text, hours=False):
    """'09:16:45.00' or '+71:18:29.00' in degrees."""
    try:
        sign = -1 if str(text).strip().startswith('-') else 1
        parts = [abs(float(p)) for p in str(text).strip().lstrip('+-').split(':')]
        value = parts[0] + parts[1] / 60 + (parts[2] if len(parts) > 2 else 0) / 3600
        return sign * value * (15 if hours else 1)
    except (ValueError, IndexError, TypeError):
        return None


def meta_get(db, name, default=None):
    row = db.execute('SELECT value FROM meta WHERE name=?', (name,)).fetchone()
    return json.loads(row['value']) if row else default


def meta_set(db, name, value):
    db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (name, json.dumps(value)))


class Indexer:
    def __init__(self, cfg):
        self.cfg = cfg.prepare()
        self.db = store.index(cfg)
        store.reviews(cfg).close()
        self.db.execute('ATTACH DATABASE ? AS review_state', (uri(cfg.reviews_db),))
        for name, arity, function in [('item_of', 1, item_of), ('run_of', 1, run_of), ('fp16_of', 1, fp16_of),
                                      ('observation_of', 1, observation_of), ('sap_of', 1, sap_of),
                                      ('archive_item', 1, lambda u: Path(u).stem),
                                      ('key_of', 5, key_of), ('short_id', 1, short_id)]:
            self.db.create_function(name, arity, function, deterministic=True)

    # -------------------------------------------------------------- state
    def sync_state(self):
        db = self.db
        if not self.cfg.state_db.is_file():
            return
        db.execute('ATTACH DATABASE ? AS st', (uri(self.cfg.state_db),))
        try:
            first = db.execute('SELECT 1 FROM files LIMIT 1').fetchone() is None
            with db:
                if not first:
                    db.execute("""INSERT INTO transitions(time,kind,subject,old,new)
                        SELECT s.updated,'file',s.surl,m.state,s.state FROM st.files s
                        JOIN files m ON m.surl=s.surl WHERE s.state IS NOT m.state""")
                    db.execute("""INSERT INTO transitions(time,kind,subject,old,new)
                        SELECT s.updated,'sap',s.key,m.state,s.state FROM st.saps s
                        JOIN saps m ON m.key=s.key WHERE s.state IS NOT m.state""")
                db.execute("""INSERT OR REPLACE INTO files
                    SELECT surl,name,sap_key,beam,state,request_id,submissions,failures,locality,checked,
                           detail,fil,updated FROM st.files s
                    WHERE NOT EXISTS (SELECT 1 FROM files m WHERE m.surl=s.surl AND m.updated IS s.updated)""")
                db.execute("""INSERT OR REPLACE INTO saps
                    SELECT key,position,files,state,detail,sap_dir,run_name,updated FROM st.saps s
                    WHERE NOT EXISTS (SELECT 1 FROM saps m WHERE m.key=s.key AND m.updated IS s.updated)""")
                db.execute('DELETE FROM requests')
                db.execute('INSERT INTO requests SELECT id,sap_key,submitted,files,status,checked,final_seen '
                           'FROM st.requests')
                last = db.execute('SELECT COALESCE(MAX(source_rowid),0) FROM events').fetchone()[0]
                db.execute('INSERT OR IGNORE INTO events SELECT rowid,time,kind,subject,detail FROM st.events '
                           'WHERE rowid>?', (last,))
        finally:
            db.execute('DETACH DATABASE st')

    # ------------------------------------------------------------- ledger
    def sync_ledger(self):
        db = self.db
        if not self.cfg.ledger.is_file():
            return
        db.execute('ATTACH DATABASE ? AS lg', (uri(self.cfg.ledger),))
        try:
            tables = {r[0] for r in db.execute("SELECT name FROM lg.sqlite_master WHERE type='table'")}
            now = time.time()
            with db:
                db.execute('INSERT OR REPLACE INTO runs SELECT fingerprint,pilot,metadata,created FROM lg.runs')
                # Rows are appended, and a running attempt is later finished in place:
                # re-read from the oldest attempt still running (ignoring abandoned ones).
                low = db.execute("SELECT MIN(id) FROM attempts WHERE status='running' AND started>?",
                                 (now - STALE_SECONDS,)).fetchone()[0]
                if low is None:
                    low = db.execute('SELECT COALESCE(MAX(id),0)+1 FROM attempts').fetchone()[0]
                db.execute('DELETE FROM attempts WHERE id>=?', (low,))
                db.execute("""INSERT INTO attempts SELECT id,item,stage,fingerprint,status,started,finished,seconds,
                    host,device,log,error FROM lg.attempts WHERE id>=?""", (low,))
                if 'beam_runs' in tables:
                    stale = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(now - STALE_SECONDS))
                    low = db.execute("SELECT MIN(id) FROM beam_runs WHERE outcome='processing' "
                                     "AND processing_timestamp>?", (stale,)).fetchone()[0]
                    if low is None:
                        low = db.execute('SELECT COALESCE(MAX(id),0)+1 FROM beam_runs').fetchone()[0]
                    db.execute('DELETE FROM beam_runs WHERE id>=?', (low,))
                    db.execute("""INSERT INTO beam_runs SELECT id,beam_id,item_of(beam_id),observation_date,
                        processing_timestamp,outcome,num_candidates,num_redetections,highest_snr,output_dir,
                        run_of(output_dir),fp16_of(output_dir),error_message,code_version
                        FROM lg.beam_runs WHERE id>=?""", (low,))
                if 'detections' in tables:
                    # Re-import of a node ledger may update recent rows in place.
                    low = max(0, db.execute('SELECT COALESCE(MAX(id),0) FROM detections').fetchone()[0] - 2000)
                    db.execute('DELETE FROM detections WHERE id>?', (low,))
                    db.execute("""INSERT INTO detections SELECT id,beam_id,item_of(beam_id),
                        key_of(detection_type,item_of(beam_id),candidate_dm,width_samples,snr),
                        candidate_dm,snr,width_samples,detection_type,pulsar_name,classification_probability,
                        beam_run_id,time_seconds,sample_number FROM lg.detections WHERE id>?""", (low,))
                if 'slack_notifications' in tables:
                    db.execute('DELETE FROM slack')
                    db.execute('INSERT INTO slack SELECT key,kind,beam_id,plot_path,slack_file_id,channel,sent '
                               'FROM lg.slack_notifications')
                if 'archive_beams' in tables:
                    last = meta_get(db, 'archive_rowid', 0)
                    db.execute('INSERT OR REPLACE INTO archive_beams SELECT uri,raw_path,item,observation,sap,beam '
                               'FROM lg.archive_beams WHERE rowid>?', (last,))
                    meta_set(db, 'archive_rowid',
                             db.execute('SELECT COALESCE(MAX(rowid),0) FROM lg.archive_beams').fetchone()[0])
                if 'archive_receipts' in tables:
                    db.execute('DELETE FROM archive_receipts')
                    db.execute('INSERT INTO archive_receipts SELECT uri,archive_item(uri),bytes '
                               'FROM lg.archive_receipts')
        finally:
            db.execute('DETACH DATABASE lg')
        self.map_saps()

    def map_saps(self):
        """Archive SAP keys (L1163405_SAP000) against observation SAPs (L559289 SAP 0)."""
        pairs = {}
        for row in self.db.execute('SELECT uri,item FROM archive_beams'):
            archive, parsed = ARCHIVE.search(Path(row['uri']).name), parse_item(row['item'])
            if archive and parsed:
                pairs[(parsed[0], parsed[1])] = f'L{archive[1]}_SAP{int(archive[2]):03d}'
        for row in self.db.execute('SELECT key,sap_dir FROM saps WHERE sap_dir IS NOT NULL'):
            match = re.search(r'/(L\d+)/SAP(\d+)$', row['sap_dir'])
            if match:
                pairs[(match[1], int(match[2]))] = row['key']
        with self.db:
            self.db.executemany('INSERT OR REPLACE INTO obs_sap VALUES (?,?,?)',
                                [(o, s, k) for (o, s), k in pairs.items()])

    # ------------------------------------------------------------ results
    def processed_dirs(self, max_depth=6):
        """Every '<node>/processed' directory under the result roots."""
        skip = {self.cfg.data.resolve()}
        for root in self.cfg.result_roots:
            stack = [(Path(root), 0)]
            while stack:
                directory, depth = stack.pop()
                try:
                    entries = list(os.scandir(directory))
                except OSError:
                    continue
                for entry in entries:
                    if not entry.is_dir(follow_symlinks=False) or entry.name.startswith('.'):
                        continue
                    if entry.name == 'processed':
                        yield Path(entry.path)
                    elif entry.name not in PRUNE and depth + 1 < max_depth \
                            and Path(entry.path).resolve() not in skip:
                        stack.append((Path(entry.path), depth + 1))

    def scan_results(self, full=False):
        db = self.db
        now = time.time()
        known = {r['path']: r['mtime'] for r in db.execute('SELECT path,mtime FROM result_dirs')}
        full = full or now - meta_get(db, 'results_full_scan', 0) > 3600
        indexed = 0
        for processed in self.processed_dirs():
            try:
                mtime = processed.stat().st_mtime
            except OSError:
                continue
            if not full and known.get(str(processed)) == mtime and now - mtime > 7200:
                continue
            indexed += self.index_processed(processed)
            with db:
                db.execute('INSERT OR REPLACE INTO result_dirs VALUES (?,?,?)', (str(processed), mtime, now))
        if full:
            with db:
                meta_set(db, 'results_full_scan', now)
        return indexed

    @staticmethod
    def run_identity(node_dir):
        try:
            run = json.loads((node_dir / 'run.json').read_text())
        except (OSError, ValueError):
            run = {}
        name = run_of(str(run.get('settings', ''))) or node_dir.parent.name
        return name, run.get('fingerprint'), run.get('pilot')

    def index_processed(self, processed):
        node_dir = processed.parent
        run_name, fingerprint, pilot = self.run_identity(node_dir)
        count = 0
        for item_dir in os.scandir(processed):
            if not item_dir.is_dir():
                continue
            for fp_dir in os.scandir(item_dir.path):
                path = Path(fp_dir.path)
                if not fp_dir.is_dir() or not (path / 'metadata.json').is_file():
                    continue
                mtime = max(path.stat().st_mtime, (path / 'metadata.json').stat().st_mtime)
                row = self.db.execute('SELECT mtime FROM beams WHERE dir=?', (str(path),)).fetchone()
                if row and row['mtime'] == mtime:
                    continue
                try:
                    self.index_beam(path, item_dir.name, run_name, node_dir.name, fingerprint, pilot, mtime)
                    count += 1
                except Exception as error:
                    logger.warning('Could not index %s: %s', path, error)
        return count

    def index_beam(self, path, item, run_name, node, fingerprint, pilot, mtime):
        meta = json.loads((path / 'metadata.json').read_text())
        info = meta.get('observation_info') or {}
        parsed = parse_item(item) or (None, None, None)
        pointing = POINTING.search(str(info.get('Object', '')))
        clusters, max_snr = 0, None
        try:
            for line in (path / 'clustered_candidates.txt').read_text().splitlines()[1:]:
                fields = line.split()
                if len(fields) > 1:
                    clusters += 1
                    max_snr = max(max_snr or float('-inf'), float(fields[1]))
        except (OSError, ValueError):
            clusters = None
        summary = {}
        for name in ('single_pulse_summary.json', 'periodicity_summary.json'):
            try:
                summary[name] = json.loads((path / name).read_text())
            except (OSError, ValueError):
                summary[name] = {}
        periodic_rows = []
        try:
            lines = (path / 'periodicity_folded_candidates.jsonl').read_text().splitlines()
        except OSError:
            lines = []
        for rank, line in enumerate(lines, 1):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            # The key postproc.notify_periodicity gives its Slack post.
            key = 'periodicity|' + hashlib.sha256((item + '|' + path.name + '|' + row['plot']).encode()).hexdigest()
            names = ', '.join(m.get('name', '') for m in row.get('catalogue_matches') or [])
            periodic_rows.append((key, str(path), item, path.name, meta.get('pilot', pilot), rank, row.get('dm'),
                                  row.get('refined_period_seconds') or row.get('period_seconds'),
                                  row.get('statistic'), row.get('harmonic_count'), row.get('fold_chi2'),
                                  int(bool(row.get('rfi_like'))), names, row.get('plot'), row.get('fold_data'),
                                  json.dumps(row)))
        plots = []
        for name, kind in ((f'{item}_rfi_diagnostic_plot.png', 'rfi'),
                           ('all_matched_filter_overview.png', 'overview'),
                           ('dm_vs_time_clusters.png', 'clusters')):
            if (path / name).is_file():
                plots.append((str(path), item, kind, name, str(path / name), None))
        for plot in sorted((path / 'candidate_plots').glob('*.png')):
            match = PLOT.match(plot.name)
            key = sp_key('candidate', item, match['dm'], match['width'], match['snr']) if match else None
            plots.append((str(path), item, 'candidate', plot.name, str(plot), key))
        by_plot = {r[13]: r[0] for r in periodic_rows}
        for plot in sorted((path / 'periodicity_plots').glob('*.png')):
            plots.append((str(path), item, 'periodic', plot.name, str(plot),
                          by_plot.get(f'periodicity_plots/{plot.name}')))
        sp = summary['single_pulse_summary.json']
        pd = summary['periodicity_summary.json']
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO beams VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', (
                str(path), item, path.name, run_name, node, fingerprint, meta.get('pilot', pilot),
                parsed[0], parsed[1], parsed[2], pointing[1] if pointing else info.get('Object'),
                sexagesimal(info.get('RA (J2000)'), hours=True), sexagesimal(info.get('DEC (J2000)')),
                info.get('Observation Date'), meta.get('tstart_mjd'), meta.get('tsamp'),
                meta.get('nu_min'), meta.get('nu_max'), meta.get('elapsed_seconds'),
                None if not sp else int(bool(sp.get('complete'))), clusters, max_snr,
                pd.get('folded_candidates'), pd.get('raw_candidates'), mtime))
            self.db.execute('DELETE FROM plots WHERE dir=?', (str(path),))
            self.db.executemany('INSERT OR REPLACE INTO plots(dir,item,kind,name,path,key) VALUES (?,?,?,?,?,?)',
                                plots)
            self.db.execute('DELETE FROM periodic WHERE dir=?', (str(path),))
            self.db.executemany('INSERT OR REPLACE INTO periodic VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                periodic_rows)

    # ------------------------------------------------------------ derived
    def sync_snippets(self):
        rows = []
        for sidecar in self.cfg.snippets.glob('*.json'):
            fil = sidecar.with_suffix('.fil')
            if not fil.is_file():
                continue
            try:
                meta = json.loads(sidecar.read_text())
            except ValueError:
                continue
            rows.append((meta['id'], meta['key'], str(fil), json.dumps(meta)))
        with self.db:
            self.db.execute('DELETE FROM snippets')
            self.db.executemany('INSERT OR REPLACE INTO snippets VALUES (?,?,?,?)', rows)

    def derive(self):
        db = self.db
        with db:
            db.execute('DELETE FROM candidates')
            db.execute("""INSERT INTO candidates(key,id,kind,type,item,dm,snr,width,time,probability,pulsar,
                    fp16,run_name,detections,found)
                SELECT d.key, short_id(d.key), 'sp', d.detection_type, d.item, d.candidate_dm, d.snr,
                       d.width_samples, d.time_seconds, d.classification_probability, d.pulsar_name,
                       b.fp16, b.run_name, g.n, f.processing_timestamp
                FROM (SELECT key, MAX(id) AS last, MIN(beam_run_id) AS first, COUNT(*) AS n
                      FROM detections WHERE key IS NOT NULL GROUP BY key) g
                JOIN detections d ON d.id=g.last
                LEFT JOIN beam_runs b ON b.id=d.beam_run_id
                LEFT JOIN beam_runs f ON f.id=g.first""")
            db.execute("""INSERT OR REPLACE INTO candidates(key,id,kind,type,item,dm,snr,period,statistic,fp16,
                    run_name,pilot,dir,detections)
                SELECT p.key, short_id(p.key), 'periodic',
                       CASE WHEN p.rfi_like THEN 'periodic_rfi' ELSE 'periodic' END,
                       p.item, p.dm, p.statistic, p.period, p.statistic, p.fp16, b.run_name, p.pilot, p.dir, 1
                FROM periodic p LEFT JOIN beams b ON b.dir=p.dir""")
            # The retained folds still provide cross-beam evidence, but explicitly
            # removed candidates must not return to the dashboard after a refresh.
            db.execute('DELETE FROM candidates WHERE key IN (SELECT key FROM review_state.candidate_removals)')
            db.execute("""UPDATE candidates SET dir=(SELECT dir FROM beams WHERE beams.item=candidates.item
                    AND beams.fp16=candidates.fp16 ORDER BY mtime DESC LIMIT 1) WHERE kind='sp'""")
            db.execute("""UPDATE candidates SET pilot=(SELECT pilot FROM beams WHERE beams.dir=candidates.dir)
                    WHERE kind='sp'""")
            db.execute("""UPDATE candidates SET plot_id=(SELECT id FROM plots WHERE plots.key=candidates.key
                    ORDER BY id DESC LIMIT 1)""")
            db.execute('UPDATE candidates SET slack_sent=(SELECT sent FROM slack WHERE slack.key=candidates.key)')
            db.execute('UPDATE candidates SET snippet=(SELECT path FROM snippets WHERE snippets.key=candidates.key)')
            db.execute("""UPDATE candidates SET sap_key=(SELECT key FROM obs_sap o
                    WHERE o.observation=observation_of(candidates.item) AND o.sap=sap_of(candidates.item))""")
            self.derive_saps()
            self.derive_families()
            self.derive_coincidence()
            self.derive_known_pulsars()
            self.derive_lotaas()
            from web.periodic_quality import sync
            sync(db)

    def derive_families(self):
        """Group the periodic folds of each observation that share a period.

        A pulsar appears in one beam or a few neighbours, at one DM. Periodic RFI
        reaches most beams of an observation, in every SAP, and at any trial DM,
        because it is narrow in frequency and dedispersion barely moves it. Folds
        are chained in period order while neighbours differ by at most
        FAMILY_TOLERANCE. Refined periods within one RFI family scatter by
        ~1e-4. Only equal periods are matched: allowing harmonics up to 16
        merged unrelated long-period folds by chance.
        """
        by_observation = {}
        for row in self.db.execute('SELECT key, item, dm, period FROM periodic WHERE period > 0'):
            parsed = parse_item(row['item'])
            if parsed:
                by_observation.setdefault(parsed[0], []).append(
                    (row['period'], row['key'], (parsed[1], parsed[2]), row['dm']))
        rows = []
        for observation, folds in by_observation.items():
            folds.sort()
            start = 0
            for end in range(1, len(folds) + 1):
                if end < len(folds) and folds[end][0] <= folds[end - 1][0] * (1 + FAMILY_TOLERANCE):
                    continue
                members = folds[start:end]
                beams = {m[2] for m in members}
                dms = [m[3] for m in members if m[3] is not None]
                family = f'{observation}|{members[0][0]:.9g}'
                rows.extend((m[1], family, len(beams), len({b[0] for b in beams}),
                             min(dms, default=None), max(dms, default=None)) for m in members)
                start = end
        with self.db:
            self.db.execute('DELETE FROM periodic_families')
            self.db.executemany('INSERT OR REPLACE INTO periodic_families VALUES (?,?,?,?,?,?)', rows)

    def derive_coincidence(self):
        """Single-pulse events at one moment in other beams of the candidate's observation.

        Interference reaches many beams at once and, being undispersed, peaks at
        whatever trial DM suits its shape: the start of L603674 stepped in level
        in every beam, and L603670 jumped 13% at 72-97 s in 32 of 35. Dedispersing
        an undispersed impulse at DM X moves its band-averaged centre earlier by
        the mean channel delay, K_DM X (1/(f_lo f_hi) - 1/f_hi^2), so events are
        compared at their time plus that delay: one impulse lines up across DMs,
        and a bright pulsar's pulse in neighbouring beams lines up at one DM.
        """
        from lotaas_reprocessing.periodicity_veto import dm_consistent
        bands = {r['item']: (r['nu_min'], r['nu_max'], r['tsamp']) for r in self.db.execute(
            'SELECT item, nu_min, nu_max, tsamp FROM beams WHERE nu_min > 0 AND nu_max > nu_min')}
        by_observation = {}
        for row in self.db.execute("SELECT key, type, item, dm, time, width FROM candidates "
                                   "WHERE kind='sp' AND time IS NOT NULL"):
            parsed = parse_item(row['item'])
            if not parsed:
                continue
            low, high, tsamp = bands.get(row['item'], (119.45, 151.04, 0.007864))
            dm = row['dm'] or 0.0
            aligned = row['time'] + K_DM * dm * (1 / (low * high) - 1 / high ** 2)
            by_observation.setdefault(parsed[0], []).append(
                (aligned, (parsed[1], parsed[2]), dm, row['key'], row['type'], (row['width'] or 1) * (tsamp or 0.007864)))
        rows = []
        for events in by_observation.values():
            events.sort()
            times = [e[0] for e in events]
            for t, beam, dm, key, kind, width in events:
                if kind not in KINDS_QUEUED:
                    continue
                tolerance = max(COINCIDENCE_SECONDS, 1.5 * width)
                near = events[bisect.bisect_left(times, t - tolerance):bisect.bisect_right(times, t + tolerance)]
                others = [e for e in near if e[1] != beam]
                beams = {e[1] for e in others} | {beam}
                if len(beams) < 2:
                    continue
                dms = [dm] + [e[2] for e in others]
                rows.append((key, len(beams), len({b[0] for b in beams}), min(dms), max(dms),
                             int(dm_consistent(dms, home=dm))))
        with self.db:
            self.db.execute('DELETE FROM sp_coincidence')
            self.db.executemany('INSERT OR REPLACE INTO sp_coincidence VALUES (?,?,?,?,?,?)', rows)

    def campaign_evidence(self):
        """What the campaign's own (non-pilot) search holds, for judging catalogued sources.

        beams: {observation: {item: (ra, dec)}}; folds: {observation: [rows]};
        singles: {(observation, pulsar): [(snr, item)]} of the classifier's
        redetections, less those a reviewer called noise or RFI and those seen
        across beams at scattered DMs (the start of L603674 was announced nine
        times as J0152+0948); known: {observation: [(dm, snr, item)]} of FETCH
        positives a reviewer called a known source.
        """
        beams = {}
        for r in self.db.execute("""SELECT item, observation, ra_deg, dec_deg FROM beams WHERE ra_deg IS NOT NULL
                AND COALESCE(pilot, 0)=0"""):
            beams.setdefault(r['observation'], {})[r['item']] = (r['ra_deg'], r['dec_deg'])
        folds = {}
        for r in self.db.execute("""SELECT item, dm, period, statistic FROM periodic
                WHERE period > 0 AND COALESCE(pilot, 0)=0"""):
            parsed = parse_item(r['item'])
            if parsed:
                folds.setdefault(parsed[0], []).append(r)
        latest = "(SELECT label FROM review_state.reviews v WHERE v.key=c.key ORDER BY created DESC LIMIT 1)"
        singles = {}
        for r in self.db.execute(f"""SELECT c.item, c.pulsar, c.snr FROM candidates c
                WHERE c.kind='sp' AND c.type='known_pulsar' AND COALESCE(c.pilot, 0)=0
                AND COALESCE({latest}, '') NOT IN ('noise', 'rfi')
                AND NOT EXISTS (SELECT 1 FROM sp_coincidence x WHERE x.key=c.key
                                AND x.beams >= {COINCIDENT_BEAMS} AND NOT x.consistent)"""):
            parsed = parse_item(r['item'])
            if parsed:
                singles.setdefault((parsed[0], r['pulsar']), []).append((r['snr'], r['item']))
        known = {}
        for r in self.db.execute(f"""SELECT c.item, c.dm, c.snr FROM candidates c WHERE c.kind='sp'
                AND c.type='candidate' AND COALESCE(c.pilot, 0)=0 AND {latest}='known'"""):
            parsed = parse_item(r['item'])
            if parsed:
                known.setdefault(parsed[0], []).append((r['dm'], r['snr'], r['item']))
        return beams, folds, singles, known

    @staticmethod
    def fields(beams):
        """{observation: (centre (ra, dec), spread in degrees)} of the searched beams."""
        from euroflash import psrcat
        out = {}
        for observation, members in beams.items():
            ras = [math.radians(ra) for ra, _ in members.values()]
            centre = (math.degrees(math.atan2(sum(map(math.sin, ras)), sum(map(math.cos, ras)))) % 360,
                      sum(dec for _, dec in members.values()) / len(members))
            out[observation] = (centre, max(psrcat.separation(*centre, ra, dec) for ra, dec in members.values()))
        return out

    def derive_known_pulsars(self):
        """Every catalogued pulsar within reach of a campaign-searched beam, and what the search found of it.

        The first night's cross-beam veto removed J0323+3944 from the periodic
        results of its own observation; only a check against the catalogue
        shows a loss like that. Single pulses count through the classifier's
        redetections (campaign_evidence), periods through harmonic matching of
        every fold of the observation (euroflash.psrcat). Pilot and validation
        runs (pilot) are left out: the campaign's own search is judged.
        """
        from euroflash import psrcat
        pulsars = psrcat.load()
        beams, folds, singles, _ = self.campaign_evidence()
        rows = []
        for observation, (centre, spread) in self.fields(beams).items() if pulsars else ():
            field = psrcat.cone(pulsars, *centre, spread + KNOWN_PULSAR_RADIUS_DEG)
            near = {}
            for item, (ra, dec) in beams[observation].items() if field else ():
                for p in psrcat.cone(field, ra, dec, KNOWN_PULSAR_RADIUS_DEG):
                    best = near.get(p['name'])
                    count = (best[3] if best else 0) + (p['separation_deg'] <= BEAM_RADIUS_DEG)
                    if best is None or p['separation_deg'] < best[1]:
                        near[p['name']] = (p, p['separation_deg'], item, count)
                    else:
                        near[p['name']] = best[:3] + (count,)
            for name, (p, separation, item, count) in near.items():
                sp = sorted(singles.get((observation, name), []), reverse=True)
                matched = [(f['statistic'] or 0, f['item'], found[1]) for f in folds.get(observation, [])
                           for found in [psrcat.match(f['period'], f['dm'] or 0.0, [p])] if found]
                best = max(matched, default=(None, None, None))
                flux, source = psrcat.flux_at(p)
                rows.append((observation, name, p.get('bname'), p['dm'], 1 / p['f0'], flux, source, item, separation,
                             count, len(sp), sp[0][0] if sp else None, sp[0][1] if sp else None,
                             len(matched), best[0], best[1], best[2]))
        with self.db:
            self.db.execute('DELETE FROM pulsar_recovery')
            self.db.executemany('INSERT OR REPLACE INTO pulsar_recovery VALUES '
                                '(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', rows)

    def derive_lotaas(self):
        """Every published LOTAAS source against what this campaign has searched and found.

        A source counts as searched once a campaign beam lies within
        SEARCHED_RADIUS_DEG of it; the evidence of every observation holding
        such a beam counts toward it. Single pulses are the classifier's
        redetections under the catalogue name, and FETCH positives a reviewer
        called a known source at its DM; periods any fold at its DM and period,
        a harmonic, a multiple or a small fraction.
        """
        from euroflash import psrcat
        from web import lotaas
        sources = lotaas.sources()
        beams, folds, singles, known = self.campaign_evidence()
        fields = self.fields(beams)
        rows = []
        for src in sources:
            best, observations, near = None, set(), 0
            for observation, (centre, spread) in fields.items():
                if psrcat.separation(*centre, src['ra'], src['dec']) > spread + SEARCHED_RADIUS_DEG:
                    continue
                for item, (ra, dec) in beams[observation].items():
                    separation = psrcat.separation(ra, dec, src['ra'], src['dec'])
                    if separation <= SEARCHED_RADIUS_DEG:
                        observations.add(observation)
                        near += separation <= BEAM_RADIUS_DEG
                    if best is None or separation < best[0]:
                        best = (separation, item, observation)
            sp, matched = [], []
            for observation in observations:
                sp += singles.get((observation, src['name']), [])
                sp += [(snr, item) for dm, snr, item in known.get(observation, [])
                       if abs(dm - src['dm']) <= max(1.0, 0.05 * src['dm'])]
                matched += [(f['statistic'] or 0, f['item'], found[1]) for f in folds.get(observation, [])
                            for found in [psrcat.match(f['period'], f['dm'] or 0.0, [src])] if found]
            sp.sort(reverse=True)
            top = max(matched, default=(None, None, None))
            flux, flux_source = psrcat.flux_at(src)
            searched = best is not None and best[0] <= SEARCHED_RADIUS_DEG
            rows.append((src['name'], src.get('bname'), src['dm'], src['period'], flux, flux_source,
                         int(src['discovery']), src['reference'], int(src['rrat']), src['lotaas_mode'],
                         src['lotaas_note'], best[2] if searched else None, best[1] if searched else None,
                         best[0] if best else None, near, len(sp), sp[0][0] if sp else None,
                         sp[0][1] if sp else None, len(matched), top[0], top[1], top[2]))
        with self.db:
            self.db.execute('DELETE FROM lotaas_sources')
            self.db.executemany('INSERT OR REPLACE INTO lotaas_sources VALUES '
                                '(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', rows)

    def derive_saps(self):
        db = self.db
        searched = {}
        for row in db.execute("""SELECT item, GROUP_CONCAT(DISTINCT substr(fingerprint,1,8)) AS fps
                FROM attempts WHERE stage='classify' AND status='success' GROUP BY item"""):
            parsed = parse_item(row['item'])
            if parsed:
                entry = searched.setdefault((parsed[0], parsed[1]), [0, set()])
                entry[0] += 1
                entry[1].update((row['fps'] or '').split(','))
        positions = {}
        for row in db.execute('SELECT observation,sap,pointing,ra_deg,dec_deg,observed FROM beams '
                              'WHERE ra_deg IS NOT NULL'):
            positions.setdefault((row['observation'], row['sap']), []).append(row)
        found = {r['sap_key']: (r['n'], r['snr']) for r in db.execute(
            "SELECT sap_key, COUNT(*) AS n, MAX(snr) AS snr FROM candidates WHERE type='candidate' GROUP BY sap_key")}
        rows = []
        for mapping in db.execute('SELECT observation,sap,key FROM obs_sap'):
            place = positions.get((mapping['observation'], mapping['sap']), [])
            ra = dec = None
            if place:
                # A circular mean, so a SAP straddling RA 0h is not placed at 12h.
                angles = [math.radians(p['ra_deg']) for p in place]
                ra = math.degrees(math.atan2(sum(map(math.sin, angles)), sum(map(math.cos, angles)))) % 360
                dec = sum(p['dec_deg'] for p in place) / len(place)
            count, fps = searched.get((mapping['observation'], mapping['sap']), [0, set()])
            n, snr = found.get(mapping['key'], (0, None))
            rows.append((mapping['key'], mapping['observation'], mapping['sap'],
                         place[0]['pointing'] if place else None, ra, dec,
                         place[0]['observed'] if place else None, count,
                         ','.join(sorted(f for f in fps if f)), n, snr))
        db.execute('DELETE FROM sap_info')
        db.executemany('INSERT OR REPLACE INTO sap_info VALUES (?,?,?,?,?,?,?,?,?,?,?)', rows)

    # ------------------------------------------------------------- health
    def health(self):
        from web import health
        report = health.check(self.cfg)
        with self.db:
            meta_set(self.db, 'health', report)
        return report

    def run_pass(self, full=False):
        started = time.time()
        timings = {}
        for name, step in (('state', self.sync_state), ('ledger', self.sync_ledger),
                           ('results', lambda: self.scan_results(full)), ('snippets', self.sync_snippets),
                           ('derive', self.derive), ('forecast', self.forecast), ('health', self.health)):
            begun = time.time()
            try:
                step()
            except Exception:
                logger.exception('Index step %s failed', name)
            timings[name] = round(time.time() - begun, 3)
        with self.db:
            meta_set(self.db, 'last_index', {'time': time.time(), 'seconds': round(time.time() - started, 3),
                                             'steps': timings})
        # Keeps the planner's statistics current as the tables grow.
        self.db.execute('PRAGMA main.optimize')
        return timings

    def forecast(self):
        from web.forecast import update
        update(self.db, self.cfg)
