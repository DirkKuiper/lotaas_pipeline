import sqlite3
import struct


def probability_value(value):
    # Older NumPy float32 bindings were stored as SQLite blobs on x86 nodes.
    if isinstance(value, bytes):
        if len(value) not in (4, 8):
            raise ValueError('Unrecognized stored classifier probability')
        value=struct.unpack('<f' if len(value)==4 else '<d',value)[0]
    if value is not None:
        value=float(value)
        if not 0 <= value <= 1:raise ValueError('Classifier probability outside [0,1]')
    return value

def initialize_database(db_path="db/processing.db"):
    conn = sqlite3.connect(db_path, timeout=120)
    c = conn.cursor()
    c.execute("BEGIN IMMEDIATE")

    # Table summarizing each beam
    c.execute("""
        CREATE TABLE IF NOT EXISTS beam_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            beam_id TEXT,
            observation_date TEXT,
            processing_timestamp TEXT,
            outcome TEXT,
            num_candidates INTEGER,
            num_redetections INTEGER,
            highest_snr REAL,
            output_dir TEXT,
            log_file TEXT,
            error_message TEXT,
            code_version TEXT
        )
    """)

    # Table with each detection
    c.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            beam_id TEXT,
            candidate_dm REAL,
            snr REAL,
            width_samples INTEGER,
            detection_type TEXT,  -- "candidate", "known_pulsar" or "rejected"
            pulsar_name TEXT,
            classification_probability REAL,
            beam_run_id INTEGER REFERENCES beam_runs(id)
        )
    """)

    columns = {r[1] for r in c.execute("PRAGMA table_info(detections)")}
    if "beam_run_id" not in columns:
        c.execute("ALTER TABLE detections ADD COLUMN beam_run_id INTEGER REFERENCES beam_runs(id)")
    for name, kind in [("time_seconds", "REAL"), ("sample_number", "INTEGER")]:
        if name not in columns:
            c.execute(f"ALTER TABLE detections ADD COLUMN {name} {kind}")
    for row_id,value in c.execute("SELECT id,classification_probability FROM detections WHERE typeof(classification_probability)='blob'").fetchall():
        c.execute('UPDATE detections SET classification_probability=? WHERE id=?',(probability_value(value),row_id))

    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path}")

if __name__ == "__main__":
    initialize_database()
