"""Catalogue-backed workload and wall-clock pace, confined to the web index.

Catalogue labels are the source author's file-count heuristic, not a format
or tape-availability check. Transfer uses bytes; search uses completed beams.
First successful completions avoid counting retries or changed fingerprints
as new progress. All-project estimates extrapolate the measured campaign rate.
"""
import collections
import csv
from decimal import Decimal, InvalidOperation
import json
import logging
from pathlib import Path
import re
import sqlite3
import subprocess
import time

logger = logging.getLogger(__name__)
BEAM = re.compile(r'^L\d+_SAP(\d+)_B(\d+)(?:_S\d+)?_P(\d+)_bf(?:_|\.)')
CATALOGUE_VERSION = 2


def get(db, name, default=None):
    row = db.execute('SELECT value FROM meta WHERE name=?', (name,)).fetchone()
    return json.loads(row[0]) if row else default


def put(db, name, value):
    db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (name, json.dumps(value)))


def catalogue(db, root):
    """Atomically reload changed CSVs; unavailable input never becomes zero work."""
    root = Path(root)
    try:
        paths = [root / 'observations.csv', *sorted((root / 'csv').glob('*.csv'))]
        if len(paths) < 2:
            raise ValueError('Catalogue observation summary and project CSVs are required')
        stamp = [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in paths]
        previous = get(db, 'observation_catalogue', {})
        if (previous.get('available') and previous.get('stamp') == stamp
                and previous.get('version') == CATALOGUE_VERSION):
            return previous
        observations = {}
        by_kind = collections.Counter()
        with paths[0].open(newline='') as stream:
            for row in csv.DictReader(stream):
                key = (row['project_id'].upper(), row['obsid'])
                if key in observations:
                    raise ValueError(f'Duplicate observation summary: {key}')
                if row['obstype'] not in ('survey', 'confirmation', 'unknown'):
                    raise ValueError(f'Unknown observation type: {row["obstype"]}')
                observations[key] = row
                by_kind[row['obstype']] += 1
        counts, sizes = collections.Counter(), collections.Counter()
        with db:
            db.execute('DELETE FROM catalogue_files')
            db.execute('DELETE FROM catalogue_observations')
            for path in paths[1:]:
                project = path.stem.upper()
                batch = []
                with path.open(newline='') as stream:
                    for row in csv.DictReader(stream):
                        obs = 'L' + row['OBSERVATIONID'].lstrip('L')
                        key = (project, obs)
                        kind = observations[key]['obstype']
                        size = int(row['FILESIZE'])
                        if size < 0:
                            raise ValueError('Negative archive file size')
                        counts[key] += 1
                        sizes[key] += size
                        match = BEAM.match(row['FILENAME'])
                        if not match:
                            continue  # summary archives and legacy non-beam layouts
                        encoding = ('gzip' if row['FILENAME'].endswith('.gz') else
                                    'tar' if row['FILENAME'].endswith('.tar') else 'other')
                        batch.append((row['URI'], project, obs, kind, row['FILENAME'], size,
                                      *map(int, match.groups()), encoding))
                        if len(batch) >= 5000:
                            db.executemany('INSERT INTO catalogue_files VALUES (?,?,?,?,?,?,?,?,?,?)', batch)
                            batch.clear()
                db.executemany('INSERT INTO catalogue_files VALUES (?,?,?,?,?,?,?,?,?,?)', batch)
            for key, row in observations.items():
                if counts[key] != int(row['nfiles']):
                    raise ValueError(f'File count differs from observation summary: {key}')
                # Summary GB is rounded to three decimal places; CSV byte sizes are exact.
                if abs(Decimal(sizes[key]) - Decimal(row['totsize_gb']) * 10**9) > 500001:
                    raise ValueError(f'Byte total differs from observation summary: {key}')
                db.execute('INSERT INTO catalogue_observations VALUES (?,?,?,?,?)',
                           (*key, row['obstype'], counts[key], sizes[key]))
            try:
                commit = subprocess.run(
                    ['git', '-C', str(root), 'log', '-1', '--format=%H %cs'],
                    text=True, capture_output=True, timeout=5).stdout.strip().split()
            except (OSError, subprocess.TimeoutExpired):
                commit = []
            summary = {'available': True, 'stamp': stamp, 'version': CATALOGUE_VERSION, 'indexed': time.time(),
                       'commit': commit[0] if commit else None,
                       'date': commit[1] if len(commit) > 1 else None,
                       'observations': len(observations), 'kinds': dict(by_kind),
                       'projects': len(paths) - 1, 'all_archive_bytes': sum(sizes.values()),
                       'survey_archive_bytes': sum(sizes[k] for k, v in observations.items()
                                                   if v['obstype'] == 'survey')}
            put(db, 'observation_catalogue', summary)
        return summary
    except (OSError, ValueError, KeyError, InvalidOperation, csv.Error, sqlite3.IntegrityError) as error:
        summary = {'available': False, 'error': str(error), 'checked': time.time()}
        with db:
            put(db, 'observation_catalogue', summary)
        logger.warning('Observation catalogue unavailable: %s', error)
        return summary


# The ledger has multiple attempts and may have several fingerprints for one
# beam. Count its first successful production completion once, excluding pilots.
SEARCH = """SELECT a.item, MIN(a.finished) AS finished FROM attempts a
    JOIN runs r ON r.fingerprint=a.fingerprint
    WHERE a.stage='classify' AND a.status='success' AND r.pilot=0
    AND a.finished IS NOT NULL GROUP BY a.item"""
SEARCHED_URIS = f"""SELECT ab.uri, MIN(s.finished) AS finished
    FROM archive_beams ab LEFT JOIN ({SEARCH}) s ON s.item=ab.item GROUP BY ab.uri
    HAVING COUNT(s.item)=COUNT(*)"""


def excluded_sql(cfg, column):
    return column + ' NOT IN (' + ','.join(str(int(b)) for b in cfg.exclude_beams) + ')' if cfg.exclude_beams else '1'


def source_sql(source, table='p'):
    """SAPs of one source ('lta' or 'spider'); every SAP when source is None."""
    return f"COALESCE({table}.source,'lta')='{source}'" if source in ('lta', 'spider') else '1'


def rates(db, cfg, now, source=None):
    """Successful arrivals/completions in fixed wall-clock windows, including idle time."""
    allowed = excluded_sql(cfg, 'f.beam') + ' AND ' + source_sql(source)
    arrivals = list(db.execute(f"""SELECT r.uri, r.bytes, MIN(a.finished) AS finished
        FROM archive_receipts r JOIN files f ON f.surl=r.uri LEFT JOIN saps p ON p.key=f.sap_key
        JOIN attempts a ON a.item=r.item AND a.stage='retrieve' AND a.status='success'
        WHERE {allowed} AND f.state!='excluded' AND a.finished IS NOT NULL
        GROUP BY r.uri"""))
    searches = list(db.execute(f"""SELECT s.item, s.finished FROM ({SEARCH}) s
        WHERE EXISTS (SELECT 1 FROM archive_beams ab JOIN files f ON f.surl=ab.uri
                      LEFT JOIN saps p ON p.key=f.sap_key
                      WHERE ab.item=s.item AND {allowed} AND f.state!='excluded')"""))
    windows = []
    for hours in (6, 24):
        first = now - hours * 3600
        incoming = [r for r in arrivals if first < r['finished'] <= now]
        finished = [r for r in searches if first < r['finished'] <= now]
        total_bytes = sum(r['bytes'] for r in incoming)
        windows.append({'hours': hours, 'files': len(incoming), 'bytes': total_bytes,
                        'bytes_per_day': total_bytes * 24 / hours,
                        'beams': len(finished), 'beams_per_day': len(finished) * 24 / hours})
    history = [r['finished'] for r in arrivals] + [r['finished'] for r in searches]
    charts = {'retrieved': [0] * 48, 'searched': [0] * 48}
    for name, rows in (('retrieved', arrivals), ('searched', searches)):
        for row in rows:
            index = int((row['finished'] - (now - 48 * 3600)) // 3600)
            if 0 <= index < 48:
                charts[name][index] += 1
    return windows, charts, min(history) if history else None


def spider_scope(db, cfg, now):
    """Early-cycle LOTAAS from SPIDER, shaped like a catalogue project (overview, filter, coverage).

    Its SAPs are all in the campaign from the start, so their states are the
    driver's own. Sizes are known once fetched; the rest are taken at the
    median fetched size (the tars are all ~1.19 GB).
    """
    allowed = excluded_sql(cfg, 'f.beam') + " AND f.state!='excluded' AND " + source_sql('spider')
    row = dict(db.execute(f"""SELECT COUNT(*) AS files, COUNT(r.uri) AS retrieved_files,
        COALESCE(SUM(r.bytes),0) AS retrieved_bytes, COUNT(s.uri) AS searched_files,
        COUNT(DISTINCT substr(f.sap_key, 1, instr(f.sap_key, '_SAP') - 1)) AS observations
        FROM files f JOIN saps p ON p.key=f.sap_key LEFT JOIN archive_receipts r ON r.uri=f.surl
        LEFT JOIN ({SEARCHED_URIS}) s ON s.uri=f.surl WHERE {allowed}""").fetchone())
    if not row['files']:
        return None
    sizes = [r[0] for r in db.execute(f"""SELECT r.bytes FROM archive_receipts r JOIN files f ON f.surl=r.uri
        JOIN saps p ON p.key=f.sap_key WHERE {allowed} ORDER BY r.bytes""")]
    typical = sizes[len(sizes) // 2] if sizes else 1_190_000_000
    remaining = (row['files'] - row['retrieved_files']) * typical
    sap_states = dict(db.execute("SELECT state, COUNT(*) FROM saps p WHERE " + source_sql('spider') + " GROUP BY state"))
    file_states = dict(db.execute(f'SELECT f.state, COUNT(*) FROM files f JOIN saps p ON p.key=f.sap_key '
                                  f'WHERE {allowed} GROUP BY f.state'))
    saps = sum(sap_states.values())
    scope = dict(row, key='ec', project='EC_LOTAAS', label='Early-cycle LOTAAS from SPIDER (EC_LOTAAS)',
                 bytes=row['retrieved_bytes'] + remaining, remaining_bytes=remaining, unknown_sizes=0,
                 in_campaign=row['files'], remaining_beams=row['files'] - row['searched_files'],
                 unmapped_observations=0, kinds={'survey': row['observations']}, gzip_files=0,
                 duplicate_beam_keys=0, all_archive_bytes=row['retrieved_bytes'] + remaining,
                 sap_states=sap_states, file_states=file_states, saps=saps, queued_saps=saps,
                 searched_saps=sap_states.get('searched', 0))
    scope['projections'] = [projection(scope, rate) for rate in rates(db, cfg, now, 'spider')[0]]
    return scope


def projection(scope, rate):
    """Missing rates stay unknown unless that stage has no work remaining."""
    download = (None if scope['unknown_sizes'] else
                0 if scope['remaining_bytes'] == 0 else
                scope['remaining_bytes'] / rate['bytes_per_day'] if rate['bytes_per_day'] else None)
    search = (0 if scope['remaining_beams'] == 0 else
              scope['remaining_beams'] / rate['beams_per_day'] if rate['beams_per_day'] else None)
    return {'hours': rate['hours'], 'download_days': download, 'search_days': search,
            'pace_days': max(download, search) if download is not None and search is not None else None}


def update(db, cfg, now=None):
    now = time.time() if now is None else now
    source = catalogue(db, cfg.observation_catalogue)
    # Throughput charts count everything searched; paces are per source, since SPIDER
    # beams need no tape and would flatter the LTA projections.
    windows, charts, first = rates(db, cfg, now)
    lta_windows = rates(db, cfg, now, 'lta')[0]
    allowed = excluded_sql(cfg, 'f.beam')
    # Only use source sizes when the catalogue is available and validated.
    join_catalogue = 'c.uri=f.surl' if source['available'] else '0'
    current = dict(db.execute(f"""SELECT COUNT(*) AS files,
        COALESCE(SUM(COALESCE(c.bytes,r.bytes)),0) AS bytes,
        COALESCE(SUM(c.bytes IS NULL AND r.bytes IS NULL),0) AS unknown_sizes,
        COUNT(r.uri) AS retrieved_files,
        COALESCE(SUM(CASE WHEN r.uri IS NULL THEN c.bytes ELSE 0 END),0) AS remaining_bytes,
        COUNT(s.uri) AS searched_files,
        COALESCE(SUM(p.state='incomplete'),0) AS blocked_files,
        COUNT(DISTINCT CASE WHEN p.state='incomplete' THEN p.key END) AS incomplete_saps,
        COALESCE(SUM(p.state='attention'),0) AS attention_files
        FROM files f LEFT JOIN catalogue_files c ON {join_catalogue}
        LEFT JOIN archive_receipts r ON r.uri=f.surl
        LEFT JOIN ({SEARCHED_URIS}) s ON s.uri=f.surl
        LEFT JOIN saps p ON p.key=f.sap_key WHERE {allowed} AND f.state!='excluded'
        AND {source_sql('lta')}""").fetchone())
    current.update(key='inventory', label='Current campaign inventory',
                   remaining_beams=current['files'] - current['searched_files'])
    scopes = [current]
    projects = []
    if source['available']:
        # Every catalogue archive is present even before the dispatcher knows it.
        # URI joins avoid guessing progress from filenames. Materialize once.
        db.execute('DROP TABLE IF EXISTS temp.catalogue_progress')
        db.execute(f"""CREATE TEMP TABLE catalogue_progress AS SELECT c.*,
            r.uri IS NOT NULL AS retrieved, s.uri IS NOT NULL AS searched,
            f.surl IS NOT NULL AS queued, f.sap_key AS queue_key, p.state AS sap_state,
            CASE WHEN s.uri IS NOT NULL THEN 'searched'
                 WHEN f.surl IS NULL THEN 'not_queued' ELSE f.state END AS state
            FROM catalogue_files c LEFT JOIN archive_receipts r ON r.uri=c.uri
            LEFT JOIN ({SEARCHED_URIS}) s ON s.uri=c.uri
            LEFT JOIN files f ON f.surl=c.uri LEFT JOIN saps p ON p.key=f.sap_key
            WHERE {excluded_sql(cfg, 'c.beam')}""")
        db.execute('CREATE INDEX temp.progress_scope ON catalogue_progress(project, observation, sap)')
        db.execute('DELETE FROM catalogue_saps')
        db.execute("""INSERT INTO catalogue_saps SELECT project, observation, sap, kind,
            COUNT(*), SUM(bytes), SUM(retrieved), SUM(searched), SUM(queued),
            CASE WHEN SUM(searched)=COUNT(*) THEN 'searched'
                 WHEN SUM(queued)=0 THEN 'not_queued'
                 WHEN SUM(sap_state='attention')>0 OR SUM(state='failed')>0 THEN 'attention'
                 WHEN SUM(sap_state='incomplete')>0 THEN 'incomplete'
                 WHEN SUM(sap_state='dispatched')>0 THEN 'dispatched'
                 WHEN SUM(sap_state='flatfielding')>0 THEN 'flatfielding'
                 WHEN SUM(state='converted')>0 THEN 'prepared'
                 WHEN SUM(searched)>0 OR SUM(state='working')>0 THEN 'processing'
                 WHEN SUM(state IN ('requested','online'))>0 THEN 'staging'
                 ELSE 'pending' END, MIN(queue_key)
            FROM catalogue_progress GROUP BY project, observation, sap""")

        def summarize(key, label, condition='1', params=()):
            row = dict(db.execute(f"""SELECT COUNT(*) AS files, COALESCE(SUM(bytes),0) AS bytes,
                0 AS unknown_sizes, COALESCE(SUM(retrieved),0) AS retrieved_files,
                COALESCE(SUM(searched),0) AS searched_files,
                COALESCE(SUM(CASE WHEN retrieved=0 THEN bytes ELSE 0 END),0) AS remaining_bytes,
                COALESCE(SUM(queued),0) AS in_campaign,
                COALESCE(SUM(encoding='gzip'),0) AS gzip_files,
                COUNT(*)-COUNT(DISTINCT project||':'||observation||':'||sap||':'||beam||':'||part)
                    AS duplicate_beam_keys
                FROM catalogue_progress WHERE {condition}""", params).fetchone())
            obs = dict(db.execute(f"""SELECT COUNT(*) AS observations,
                COALESCE(SUM(bytes),0) AS all_archive_bytes,
                COUNT(CASE WHEN NOT EXISTS (SELECT 1 FROM catalogue_saps s
                    WHERE s.project=o.project AND s.observation=o.observation) THEN 1 END)
                    AS unmapped_observations
                FROM catalogue_observations o WHERE {condition}""", params).fetchone())
            row.update(obs, key=key, label=label, remaining_beams=row['files'] - row['searched_files'])
            row['sap_states'] = dict(db.execute(f'SELECT state, COUNT(*) FROM catalogue_saps WHERE {condition} GROUP BY state', params))
            row['file_states'] = dict(db.execute(f'SELECT state, COUNT(*) FROM catalogue_progress WHERE {condition} GROUP BY state', params))
            row['saps'] = sum(row['sap_states'].values())
            row['queued_saps'] = row['saps'] - row['sap_states'].get('not_queued', 0)
            row['searched_saps'] = row['sap_states'].get('searched', 0)
            row['kinds'] = dict(db.execute(f'SELECT kind, COUNT(*) FROM catalogue_observations WHERE {condition} GROUP BY kind', params))
            row['projections'] = [projection(row, rate) for rate in lta_windows]
            return row

        scopes.extend([
            summarize('lt5', 'LT5_004 survey', "kind='survey' AND project='LT5_004'"),
            summarize('survey', 'All catalogued survey observations', "kind='survey'"),
            summarize('all', 'All LOTAAS projects'),
        ])
        for project, in db.execute('SELECT DISTINCT project FROM catalogue_observations ORDER BY project'):
            row = summarize(project, project, 'project=?', (project,))
            row['project'] = project
            projects.append(row)
    for scope in scopes:
        scope['projections'] = [projection(scope, rate) for rate in lta_windows]
    early = spider_scope(db, cfg, now)
    if early:
        scopes.append(early)
        projects.append(early)
    report = {'time': now, 'catalogue': {k: v for k, v in source.items() if k != 'stamp'},
              'scopes': scopes, 'rates': windows, 'charts': charts, 'first_completion': first,
              'exclude_beams': cfg.exclude_beams, 'projects': projects}
    with db:
        put(db, 'forecast', report)
    return report
