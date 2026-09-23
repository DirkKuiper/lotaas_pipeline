#!/usr/bin/python3
"""Read-only campaign health sampling, retained locally for unattended diagnosis."""
import concurrent.futures
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

ROOT = Path('/shared/results/dkuiper/lotaas/campaign')


def node_health(node):
    control = Path.home()/'.ssh/control'/('lotaas-' + node.removeprefix('efc-').replace('-', ''))
    command = "uptime; df -BG --output=avail /home | tail -1"
    if 'gpu' in node:
        command += '; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'
    try:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                                 '-S', str(control), node, command], capture_output=True, text=True, timeout=20)
        return node, {'ok': result.returncode == 0, 'status': result.stdout.strip(),
                      'error': result.stderr.strip() if result.returncode else None}
    except subprocess.TimeoutExpired:
        return node, {'ok': False, 'error': 'health probe timed out'}


def main():
    now = time.time()
    report = {'time': now, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(now)),
              'free_tb': shutil.disk_usage(ROOT).free/1e12}
    for name in ('status', 'supervisor'):
        path = ROOT/f'{name}.json'
        try:
            report[name] = json.loads(path.read_text())
            report[name + '_age_seconds'] = now-path.stat().st_mtime
        except (OSError, ValueError):
            report[name] = None
    pid = (report['supervisor'] or {}).get('driver_pid')
    try:
        report['driver_alive'] = bool(pid and b'euroflash.campaign' in
                                      (Path('/proc')/str(pid)/'cmdline').read_bytes())
    except OSError:
        report['driver_alive'] = False
    with sqlite3.connect(f'file:{ROOT}/campaign-state.sqlite?mode=ro', uri=True) as db:
        report['saps'] = dict(db.execute('SELECT state,COUNT(*) FROM saps GROUP BY state'))
        report['files'] = dict(db.execute('SELECT state,COUNT(*) FROM files GROUP BY state'))
        report['recent_events'] = list(db.execute('SELECT time,kind,subject,detail FROM events ORDER BY time DESC LIMIT 6'))
    with sqlite3.connect(f'file:{ROOT.parent}/campaign.sqlite?mode=ro', uri=True) as db:
        report['rates'] = []
        for hours in (1, 6, 24):
            counts = dict(db.execute("""SELECT stage,COUNT(*) FROM (
                SELECT stage,item,MIN(finished) finished FROM attempts
                WHERE stage IN ('retrieve','classify') AND status='success'
                GROUP BY stage,item) WHERE finished>? GROUP BY stage""", (now-hours*3600,)))
            report['rates'].append({'hours': hours, **counts})
    nodes = ['efc-gpu-00', 'efc-gpu-01'] + [f'efc-cpu-{i:02d}' for i in range(7)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:
        report['nodes'] = dict(pool.map(node_health, nodes))
    temporary = ROOT/'health.json.partial'
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(ROOT/'health.json')
    with (ROOT/'health-history.jsonl').open('a') as stream:
        stream.write(json.dumps(report) + '\n')
    print(json.dumps({key: report[key] for key in ('utc', 'driver_alive', 'saps', 'rates')}))


if __name__ == '__main__':
    main()
