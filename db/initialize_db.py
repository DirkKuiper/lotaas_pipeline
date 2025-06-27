import sqlite3

def initialize_database(db_path="db/processing.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

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
            detection_type TEXT,  -- "known_pulsar" or "candidate"
            pulsar_name TEXT,
            classification_probability REAL
        )
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path}")

if __name__ == "__main__":
    initialize_database()