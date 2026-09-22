"""Merge copied node ledgers into the campaign database without losing attempts."""
import argparse
import hashlib
from pathlib import Path
import sqlite3
from euroflash.ledger import Ledger
from db.initialize_db import initialize_database, probability_value


def collect(source, destination, node):
    ledger = Ledger(destination)
    initialize_database(str(destination))
    src = sqlite3.connect(source)
    src.row_factory = sqlite3.Row
    tables = {r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    with ledger.connect() as dst:
        dst.execute('CREATE TABLE IF NOT EXISTS imports (source TEXT, tablename TEXT, source_id INTEGER, destination_id INTEGER, PRIMARY KEY(source,tablename,source_id))')
        if 'runs' in tables:
            for row in src.execute('SELECT * FROM runs'):
                dst.execute('INSERT OR IGNORE INTO runs VALUES (?,?,?,?)', tuple(row))
        # Read a copied consistent SQLite snapshot, never an active remote DB file.
        for table in ['attempts', 'beam_runs', 'detections']:
            if table not in tables:
                continue
            for row in src.execute(f'SELECT * FROM {table}'):
                existing = dst.execute('SELECT destination_id FROM imports WHERE source=? AND tablename=? AND source_id=?',
                                       (node, table, row['id'])).fetchone()
                values = dict(row)
                del values['id']
                if table == 'detections':
                    values['classification_probability']=probability_value(values.get('classification_probability'))
                if table == 'detections' and values.get('beam_run_id') is not None:
                    mapped = dst.execute('SELECT destination_id FROM imports WHERE source=? AND tablename=? AND source_id=?',
                                         (node, 'beam_runs', values['beam_run_id'])).fetchone()
                    if mapped is None:
                        raise ValueError('Detection references an unimported beam run')
                    values['beam_run_id'] = mapped[0]
                columns = list(values)
                if existing:
                    assignments = ','.join(f'"{k}"=?' for k in columns)
                    dst.execute(f'UPDATE {table} SET {assignments} WHERE id=?', [values[k] for k in columns]+[existing[0]])
                else:
                    columns_sql=','.join(f'"{k}"' for k in columns)
                    placeholders=','.join('?' for k in columns)
                    cursor=dst.execute(f'INSERT INTO {table}({columns_sql}) VALUES ({placeholders})', list(values.values()))
                    dst.execute('INSERT INTO imports VALUES (?,?,?,?)', (node,table,row['id'],cursor.lastrowid))
    src.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('destination',type=Path)
    p.add_argument('--source-id',required=True,help='Unique node/run identifier, reused for incremental imports')
    a=p.parse_args();collect(a.source,a.destination,a.source_id)


if __name__=='__main__':
    main()
