"""Is the campaign moving? The driver, its dispatch, the SSH master it needs, and the disk.

Everything is read from /proc, the campaign directory and `ssh -O check`,
which asks the local ControlMaster socket and touches no network.
"""
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import time


def processes():
    """Command line of every process that can be read, by pid."""
    found = {}
    for proc in Path('/proc').iterdir():
        if proc.name.isdigit():
            try:
                argv = (proc / 'cmdline').read_bytes().split(b'\0')
            except OSError:
                continue
            found[int(proc.name)] = [a.decode(errors='replace') for a in argv if a]
    return found


def started(pid):
    """When a process started, as a UNIX time."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        boot = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines()
                    if line.startswith('btime'))
        return boot + int(fields[19]) / os.sysconf('SC_CLK_TCK')
    except (OSError, IndexError, StopIteration, ValueError):
        return None


def options(argv):
    """'--flag value ...' pairs from a command line; flags without values are True."""
    parsed, key = {}, None
    for arg in argv:
        if arg.startswith('--'):
            key = arg[2:]
            parsed[key] = []
        elif key is not None:
            parsed[key].append(arg)
    return {k: (True if not v else v[0] if len(v) == 1 else v) for k, v in parsed.items()}


def control_socket(control_dir, node):
    """The ControlPath euroflash.cluster.ssh_args uses for a node."""
    return Path(control_dir) / ('lotaas-' + node.removeprefix('efc-').replace('-', ''))


def ssh_master(socket, node):
    if not socket.exists():
        return {'ok': False, 'detail': f'no control socket at {socket}'}
    try:
        result = subprocess.run(['ssh', '-O', 'check', '-S', str(socket), node],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'ok': False, 'detail': str(error)}
    return {'ok': result.returncode == 0, 'detail': (result.stderr or result.stdout).strip()}


STAGE = re.compile(r'/(?:pipeline|lotaas_reprocessing)/(\w+)\.py\s+(\S+)(?:.*--search\s+(\S+))?')


def stage_processes(socket, node, run_name):
    """The search processes a dispatch has running on a node, with how long each has run."""
    if not socket.exists() or not run_name:
        return None
    try:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-S', str(socket), node, 'ps -eo etimes=,args='],
                                capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    found = []
    for line in result.stdout.splitlines():
        seconds, _, args = line.strip().partition(' ')
        if f'/lotaas-runs/{run_name}/' not in args or 'python' not in args:
            continue
        match = STAGE.search(args)
        if not match:
            continue
        beam = next((part for part in Path(match[2]).parts if part.startswith('downsampled_')), match[2])
        found.append({'seconds': int(seconds), 'script': match[1], 'search': match[3], 'beam': beam})
    return sorted(found, key=lambda r: -r['seconds'])


def files_in(directory, pattern):
    count = size = exclusive = 0
    for path in Path(directory).glob(pattern):
        try:
            stat = path.stat()
        except OSError:
            continue
        count += 1
        size += stat.st_size
        # A link nobody else holds is space only we keep allocated.
        exclusive += stat.st_size if stat.st_nlink == 1 else 0
    return {'count': count, 'bytes': size, 'exclusive_bytes': exclusive}


def check(cfg):
    now = time.time()
    running = processes()
    report = {'checked': now}
    drivers = [(pid, argv) for pid, argv in running.items()
               if 'euroflash.campaign' in argv and 'run' in argv]
    driver_options = {}
    if drivers:
        pid, argv = drivers[0]
        driver_options = options(argv)
        report['driver'] = {'running': True, 'pid': pid, 'started': started(pid), 'options': driver_options}
    else:
        report['driver'] = {'running': False}
    dispatches = [(pid, argv) for pid, argv in running.items() if 'euroflash.cluster' in argv]
    report['dispatch'] = [{'pid': pid, 'started': started(pid), 'run_name': options(argv).get('run-name')}
                          for pid, argv in dispatches]
    try:
        status = json.loads((cfg.campaign_root / 'status.json').read_text())
        report['status_time'] = status.get('time_unix')
    except (OSError, ValueError):
        report['status_time'] = None
    report['stop_file'] = (cfg.campaign_root / 'STOP').exists()
    control_dir = Path(driver_options.get('control-dir') or cfg.control_dir)
    nodes = driver_options.get('dispatch-nodes') or ['efc-gpu-01']
    nodes = [nodes] if isinstance(nodes, str) else nodes
    report['ssh'] = {node: ssh_master(control_socket(control_dir, node), node) for node in nodes}
    for run in report['dispatch']:
        run['stages'] = {node: stage_processes(control_socket(control_dir, node), node, run['run_name'])
                         for node in nodes if report['ssh'][node]['ok']}
    try:
        usage = shutil.disk_usage(cfg.campaign_root)
        report['disk'] = {'free_tb': usage.free / 1e12, 'total_tb': usage.total / 1e12,
                          'min_free_tb': float(driver_options.get('min-free-tb', 5.0))}
    except OSError:
        report['disk'] = None
    report['held'] = files_in(cfg.held, '*/*.fil')
    report['snippets'] = files_in(cfg.snippets, '*.fil')
    return report
