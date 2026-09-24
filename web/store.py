"""The web layer's own databases.

web.sqlite is an index: every table in it is rebuilt from the campaign state
database, the ledger and the results tree, so it can be deleted at any time.
reviews.sqlite holds verdicts, the only record here that nothing else can
rebuild; it is kept apart so the index can be thrown away freely.
"""
from contextlib import contextmanager
import sqlite3
from urllib.parse import quote

INDEX = '''
CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS saps (key TEXT PRIMARY KEY, position INTEGER, files INTEGER, state TEXT,
    detail TEXT, sap_dir TEXT, run_name TEXT, updated REAL);
CREATE TABLE IF NOT EXISTS files (surl TEXT PRIMARY KEY, name TEXT, sap_key TEXT, beam INTEGER,
    state TEXT, request_id INTEGER, submissions INTEGER, failures INTEGER, locality TEXT,
    checked REAL, detail TEXT, fil TEXT, updated REAL);
CREATE INDEX IF NOT EXISTS files_sap ON files(sap_key);
CREATE INDEX IF NOT EXISTS files_state ON files(state, updated);
CREATE INDEX IF NOT EXISTS files_request ON files(request_id);
CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY, sap_key TEXT, submitted REAL, files INTEGER,
    status TEXT, checked REAL, final_seen REAL);
CREATE TABLE IF NOT EXISTS events (source_rowid INTEGER PRIMARY KEY, time REAL, kind TEXT,
    subject TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS events_time ON events(time);
CREATE INDEX IF NOT EXISTS events_subject ON events(subject);
-- State changes seen between passes, timed by the source's own 'updated'.
CREATE TABLE IF NOT EXISTS transitions (id INTEGER PRIMARY KEY, time REAL, kind TEXT, subject TEXT,
    old TEXT, new TEXT);
CREATE INDEX IF NOT EXISTS transitions_new ON transitions(kind, new, time);
CREATE INDEX IF NOT EXISTS transitions_subject ON transitions(subject);

CREATE TABLE IF NOT EXISTS runs (fingerprint TEXT PRIMARY KEY, pilot INTEGER, metadata TEXT, created REAL);
CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, item TEXT, stage TEXT, fingerprint TEXT,
    status TEXT, started REAL, finished REAL, seconds REAL, host TEXT, device TEXT, log TEXT, error TEXT);
CREATE INDEX IF NOT EXISTS attempts_item ON attempts(item, stage);
CREATE INDEX IF NOT EXISTS attempts_stage ON attempts(stage, status, finished);
CREATE TABLE IF NOT EXISTS beam_runs (id INTEGER PRIMARY KEY, beam_id TEXT, item TEXT,
    observation_date TEXT, processing_timestamp TEXT, outcome TEXT, num_candidates INTEGER,
    num_redetections INTEGER, highest_snr REAL, output_dir TEXT, run_name TEXT, fp16 TEXT,
    error_message TEXT, code_version TEXT);
CREATE INDEX IF NOT EXISTS beam_runs_item ON beam_runs(item);
CREATE TABLE IF NOT EXISTS detections (id INTEGER PRIMARY KEY, beam_id TEXT, item TEXT, key TEXT,
    candidate_dm REAL, snr REAL, width_samples INTEGER, detection_type TEXT, pulsar_name TEXT,
    classification_probability REAL, beam_run_id INTEGER, time_seconds REAL, sample_number INTEGER);
CREATE INDEX IF NOT EXISTS detections_key ON detections(key);
CREATE INDEX IF NOT EXISTS detections_item ON detections(item);
CREATE INDEX IF NOT EXISTS detections_run ON detections(beam_run_id);
CREATE TABLE IF NOT EXISTS slack (key TEXT PRIMARY KEY, kind TEXT, beam_id TEXT, plot_path TEXT,
    slack_file_id TEXT, channel TEXT, sent REAL);
CREATE TABLE IF NOT EXISTS archive_beams (uri TEXT, raw_path TEXT, item TEXT, observation TEXT,
    sap INTEGER, beam INTEGER, PRIMARY KEY(uri, raw_path));
CREATE TABLE IF NOT EXISTS archive_receipts (uri TEXT PRIMARY KEY, item TEXT, bytes INTEGER);
CREATE INDEX IF NOT EXISTS receipts_item ON archive_receipts(item);
CREATE TABLE IF NOT EXISTS catalogue_files (uri TEXT PRIMARY KEY, project TEXT, observation TEXT,
    kind TEXT, name TEXT, bytes INTEGER, sap INTEGER, beam INTEGER, part INTEGER, encoding TEXT);
CREATE INDEX IF NOT EXISTS catalogue_scope ON catalogue_files(kind, project, beam);
CREATE TABLE IF NOT EXISTS catalogue_observations (project TEXT, observation TEXT, kind TEXT,
    files INTEGER, bytes INTEGER, PRIMARY KEY(project, observation));
CREATE TABLE IF NOT EXISTS catalogue_saps (project TEXT, observation TEXT, sap INTEGER,
    kind TEXT, files INTEGER, bytes INTEGER, retrieved_files INTEGER, searched_files INTEGER,
    in_campaign INTEGER, state TEXT, queue_key TEXT, PRIMARY KEY(project, observation, sap));
CREATE TABLE IF NOT EXISTS obs_sap (observation TEXT, sap INTEGER, key TEXT, PRIMARY KEY(observation, sap));

CREATE TABLE IF NOT EXISTS result_dirs (path TEXT PRIMARY KEY, mtime REAL, scanned REAL);
CREATE TABLE IF NOT EXISTS beams (dir TEXT PRIMARY KEY, item TEXT, fp16 TEXT, run_name TEXT, node TEXT,
    fingerprint TEXT, pilot INTEGER, observation TEXT, sap INTEGER, beam INTEGER, pointing TEXT,
    ra_deg REAL, dec_deg REAL, observed TEXT, tstart_mjd REAL, tsamp REAL, nu_min REAL, nu_max REAL,
    elapsed REAL, sp_complete INTEGER, clusters INTEGER, max_cluster_snr REAL,
    periodic_folds INTEGER, periodic_candidates INTEGER, mtime REAL);
CREATE INDEX IF NOT EXISTS beams_item ON beams(item);
CREATE INDEX IF NOT EXISTS beams_obs ON beams(observation, sap);
CREATE TABLE IF NOT EXISTS plots (id INTEGER PRIMARY KEY, dir TEXT, item TEXT, kind TEXT, name TEXT,
    path TEXT UNIQUE, key TEXT);
CREATE INDEX IF NOT EXISTS plots_dir ON plots(dir);
CREATE INDEX IF NOT EXISTS plots_key ON plots(key);
CREATE TABLE IF NOT EXISTS periodic (key TEXT PRIMARY KEY, dir TEXT, item TEXT, fp16 TEXT, pilot INTEGER,
    rank INTEGER, dm REAL, period REAL, statistic REAL, harmonics INTEGER, fold_chi2 REAL,
    rfi_like INTEGER, catalogue TEXT, plot TEXT, fold_data TEXT, row TEXT);
CREATE INDEX IF NOT EXISTS periodic_item ON periodic(item);
-- Folds of one observation that share a period, across its beams and SAPs.
CREATE TABLE IF NOT EXISTS periodic_families (key TEXT PRIMARY KEY, family TEXT, beams INTEGER, saps INTEGER,
    dm_min REAL, dm_max REAL);
CREATE INDEX IF NOT EXISTS periodic_families_family ON periodic_families(family);
CREATE TABLE IF NOT EXISTS periodic_triage (key TEXT PRIMARY KEY, signature TEXT, score REAL,
    status TEXT, representative TEXT, members INTEGER, evidence TEXT, reasons TEXT);

CREATE TABLE IF NOT EXISTS sap_info (key TEXT PRIMARY KEY, observation TEXT, sap INTEGER, pointing TEXT,
    ra_deg REAL, dec_deg REAL, observed TEXT, beams_searched INTEGER, fingerprints TEXT,
    candidates INTEGER, max_snr REAL);
CREATE TABLE IF NOT EXISTS candidates (key TEXT PRIMARY KEY, id TEXT UNIQUE, kind TEXT, type TEXT,
    item TEXT, sap_key TEXT, dm REAL, snr REAL, width INTEGER, time REAL, probability REAL,
    pulsar TEXT, period REAL, statistic REAL, fp16 TEXT, run_name TEXT, pilot INTEGER,
    plot_id INTEGER, dir TEXT, slack_sent REAL, snippet TEXT, detections INTEGER, found TEXT);
CREATE INDEX IF NOT EXISTS candidates_type ON candidates(type, snr);
CREATE TABLE IF NOT EXISTS snippets (id TEXT PRIMARY KEY, key TEXT, path TEXT, meta TEXT);
-- A single-pulse candidate and the events at its moment in other beams of its observation.
CREATE TABLE IF NOT EXISTS sp_coincidence (key TEXT PRIMARY KEY, beams INTEGER, saps INTEGER,
    dm_min REAL, dm_max REAL, consistent INTEGER);
-- Every catalogued pulsar near a campaign-searched beam, and what the search found of it.
CREATE TABLE IF NOT EXISTS pulsar_recovery (observation TEXT, psrj TEXT, bname TEXT, dm REAL, period REAL,
    flux_mjy REAL, flux_source TEXT, beam_item TEXT, separation_deg REAL, beams_near INTEGER,
    sp_count INTEGER, sp_best_snr REAL, sp_item TEXT, periodic_count INTEGER, periodic_best REAL,
    periodic_item TEXT, periodic_relation TEXT, PRIMARY KEY(observation, psrj));
DROP TABLE IF EXISTS known_pulsars;
-- Every published LOTAAS source (web.lotaas), how LOTAAS found it, and what this campaign found of it.
CREATE TABLE IF NOT EXISTS lotaas_sources (psrj TEXT PRIMARY KEY, bname TEXT, dm REAL, period REAL,
    flux_mjy REAL, flux_source TEXT, discovery INTEGER, reference TEXT, rrat INTEGER, lotaas_mode TEXT,
    lotaas_note TEXT, observation TEXT, beam_item TEXT, separation_deg REAL, beams_near INTEGER,
    sp_count INTEGER, sp_best_snr REAL, sp_item TEXT, periodic_count INTEGER, periodic_best REAL,
    periodic_item TEXT, periodic_relation TEXT);
'''

REVIEWS = '''
CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY, key TEXT NOT NULL, reviewer TEXT NOT NULL,
    label TEXT NOT NULL, note TEXT, dm REAL, created REAL NOT NULL, slack_ts TEXT);
CREATE INDEX IF NOT EXISTS reviews_key ON reviews(key, created);
-- Explicit, audited dashboard removals. Keep these outside the rebuildable index.
CREATE TABLE IF NOT EXISTS candidate_removals (key TEXT PRIMARY KEY, reason TEXT NOT NULL,
    evidence TEXT NOT NULL, removed_by TEXT NOT NULL, created REAL NOT NULL);
'''

LABELS = {'rfi': 'RFI', 'noise': 'Noise', 'known': 'Known source', 'astro': 'Astrophysical',
          'unsure': 'Unsure / follow-up'}


def _connect(path, schema):
    # URI mode, so the sources can be ATTACHed read-only with ?mode=ro.
    db = sqlite3.connect(f'file:{quote(str(path))}', uri=True, timeout=60, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA busy_timeout=60000')
    db.executescript(schema)
    return db


def index(cfg):
    return _connect(cfg.index_db, INDEX)


def reviews(cfg):
    return _connect(cfg.reviews_db, REVIEWS)


@contextmanager
def reading(cfg):
    """The index with the reviews attached, for the pages."""
    # Read-write handles used only for reading: a read-only handle on a WAL
    # database fails whenever its -shm file is absent.
    db = sqlite3.connect(cfg.index_db, timeout=60, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA busy_timeout=60000')
    db.execute('ATTACH DATABASE ? AS r', (str(cfg.reviews_db),))
    try:
        yield db
    finally:
        db.close()
