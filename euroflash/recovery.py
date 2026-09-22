"""Check a reference pulse against search candidates and recorded detections."""
import argparse
import csv
import json
from pathlib import Path
import sqlite3


def candidate_matches(path, dm, seconds, dm_tolerance, time_tolerance, clustered=False):
    matches = []
    count = 0
    with path.open() as stream:
        if clustered:
            rows = csv.DictReader(stream, delimiter='\t')
        else:
            rows = (dict(zip(['DM', 'S/N', 'Time', 'Sample', 'Filter_Width'], line.split()))
                    for line in stream if line.strip() and not line.startswith('#'))
        for row in rows:
            count += 1
            if abs(float(row['DM'])-dm) <= dm_tolerance and abs(float(row['Time'])-seconds) <= time_tolerance:
                matches.append({key: float(row[key]) for key in ['DM', 'S/N', 'Time', 'Sample', 'Filter_Width']})
    matches.sort(key=lambda row: row['S/N'], reverse=True)
    return {'total_rows': count, 'matching_rows': len(matches), 'strongest_matches': matches[:20]}


def report(target, work, dm_tolerance=.5, time_tolerance=.25):
    beam = f"downsampled_{target['observation']}_SAP{target['sap']:03d}_BEAM{target['beam']:03d}_32bit_ff"
    # Require an unambiguous result, rather than accidentally accepting an old run.
    products = list(work.glob(f'processed/{beam}/*/all_detected_candidates.cands'))
    if len(products) != 1:
        raise ValueError(f'Expected exactly one result for {beam}, found {len(products)}')
    directory = products[0].parent
    run = json.loads((work/'run.json').read_text())
    if directory.name != run['fingerprint'][:16]:
        raise ValueError('Candidate directory does not match run fingerprint')
    with sqlite3.connect(f'file:{work / "ledger-snapshot.sqlite"}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        stages = [dict(r) for r in db.execute(
            'SELECT stage,status,seconds FROM attempts WHERE item=? AND fingerprint=? ORDER BY id',
            (beam, run['fingerprint']))]
        detections = [dict(r) for r in db.execute('''SELECT d.* FROM detections d
            JOIN beam_runs b ON b.id=d.beam_run_id WHERE b.output_dir LIKE ?
            AND abs(d.candidate_dm-?)<=? AND abs(d.time_seconds-?)<=?''',
            ('%/'+beam+'/'+directory.name+'/%', target['target_dm'], dm_tolerance,
             target['target_time_seconds'], time_tolerance))]
    latest = {row['stage']: row['status'] for row in stages}
    complete = all(latest.get(stage) == 'success' for stage in ['dedisperse', 'classify'])
    args = (target['target_dm'], target['target_time_seconds'], dm_tolerance, time_tolerance)
    raw = candidate_matches(products[0], *args)
    clustered = candidate_matches(directory/'clustered_candidates.txt', *args, clustered=True)
    return {'reference': target, 'run_fingerprint': run['fingerprint'],
            'dm_tolerance': dm_tolerance, 'time_tolerance_seconds': time_tolerance,
            'stages': stages, 'pipeline_completed': complete,
            'threshold_crossings': raw, 'clustered_candidates': clustered,
            'accepted_detections': detections,
            'reference_pulse_recovered': complete and bool(detections),
            'note': 'A completed run alone does not establish pulse recovery. Reference S/N and FETCH probabilities are comparison values, not imposed results.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True, help='Collected node work directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dm-tolerance', type=float, default=.5)
    parser.add_argument('--time-tolerance', type=float, default=.25)
    args = parser.parse_args()
    result = report(json.loads(args.target.read_text()), args.work,
                    args.dm_tolerance, args.time_tolerance)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if not result['reference_pulse_recovered']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
