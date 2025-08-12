import sqlite3

conn = sqlite3.connect("db/processing.db")
c = conn.cursor()

print("Tables:")
c.execute("SELECT name FROM sqlite_master WHERE type='table';")
for row in c.fetchall():
    print("-", row[0])

print("\nSchema for beam_runs:")
c.execute("PRAGMA table_info(beam_runs);")
for row in c.fetchall():
    print(row)

print("\nSchema for detections:")
c.execute("PRAGMA table_info(detections);")
for row in c.fetchall():
    print(row)

conn.close()