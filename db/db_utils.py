import json
import sqlite3
import datetime
import os
from pathlib import Path
from db.initialize_db import initialize_database, probability_value

DB_PATH = os.environ.get("LOTAAS_DB_PATH", "db/processing.db")
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
initialize_database(DB_PATH)

def insert_beam_run(beam_id, observation_date, output_dir, log_file, code_version="v1.0"):
    """
    Inserts a new beam run with status 'processing'
    Returns the inserted row id.
    """
    conn = sqlite3.connect(DB_PATH, timeout=120)
    c = conn.cursor()
    processing_timestamp = datetime.datetime.utcnow().isoformat()
    c.execute("""
        INSERT INTO beam_runs (
            beam_id, observation_date, processing_timestamp, outcome,
            num_candidates, num_redetections, highest_snr,
            output_dir, log_file, error_message, code_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        beam_id, observation_date, processing_timestamp, "processing",
        None, None, None, output_dir, log_file, None, code_version
    ))
    row_id = c.lastrowid
    conn.commit()
    conn.close()
    return row_id

def update_beam_run(row_id, outcome, num_candidates=None, num_redetections=None,
                    highest_snr=None, error_message=None):
    """
    Updates the beam run row with final outcome and stats.
    """
    conn = sqlite3.connect(DB_PATH, timeout=120)
    c = conn.cursor()
    c.execute("""
        UPDATE beam_runs
        SET outcome = ?, num_candidates = ?, num_redetections = ?, highest_snr = ?, error_message = ?
        WHERE id = ?
    """, (
        outcome, num_candidates, num_redetections, highest_snr, error_message, row_id
    ))
    conn.commit()
    conn.close()

def insert_detection(beam_id, candidate_dm, snr, width_samples, detection_type, pulsar_name=None, classification_probability=None, beam_run_id=None, time_seconds=None, sample_number=None, model_probabilities=None, own_snr=None, dispersion_ratio=None, dm_galactic=None):
    """
    Inserts a detection record. model_probabilities: {model: probability} of each FETCH model;
    own_snr: its S/N on its own data (own_data.measure); dispersion_ratio: own_data.dispersion_ratio;
    dm_galactic: the Milky Way's largest DM along the beam's line of sight (NE2001, YMW16).
    """
    conn = sqlite3.connect(DB_PATH, timeout=120)
    c = conn.cursor()
    c.execute("""
        INSERT INTO detections (
            beam_id, candidate_dm, snr, width_samples, detection_type, pulsar_name, classification_probability, beam_run_id, time_seconds, sample_number, model_probabilities, own_snr, dispersion_ratio, dm_galactic
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        beam_id, float(candidate_dm), float(snr), int(width_samples), detection_type, pulsar_name,
        probability_value(classification_probability), beam_run_id,
        None if time_seconds is None else float(time_seconds),
        None if sample_number is None else int(sample_number),
        None if model_probabilities is None else json.dumps(
            {name: round(probability_value(p), 4) for name, p in sorted(model_probabilities.items())}),
        None if own_snr is None else float(own_snr),
        None if dispersion_ratio is None else float(dispersion_ratio),
        None if dm_galactic is None else float(dm_galactic),
    ))
    conn.commit()
    conn.close()
