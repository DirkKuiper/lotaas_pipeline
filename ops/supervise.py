#!/usr/bin/python3
"""Cron-started singleton supervisor. A STOP file remains an explicit stop."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def existing_driver(root):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc/'cmdline').read_bytes().split(b'\0')
            if (b'euroflash.campaign' in args and b'--root' in args
                    and args[args.index(b'--root') + 1] == str(root).encode()):
                return int(proc.name)
        except (OSError, IndexError):
            pass
    return None


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    root = Path(config['root'])
    with (root/'.supervisor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        restarts = 0
        with (root/'driver.log').open('ab', buffering=0) as log:
            while not (root/'STOP').exists():
                # A manually started driver may already own the campaign.
                with (root/'.campaign.lock').open('a') as campaign_lock:
                    try:
                        fcntl.flock(campaign_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        write(root/'supervisor.json', {'time': time.time(), 'supervisor_pid': os.getpid(),
                              'driver_pid': existing_driver(root), 'restarts': restarts,
                              'checkout': config['checkout'], 'adopted': True})
                        time.sleep(10)
                        continue
                env = dict(os.environ, PATH='/usr/local/bin:/usr/bin:/bin', PYTHONUNBUFFERED='1')
                process = subprocess.Popen(config['command'], cwd=config['checkout'], env=env,
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                           start_new_session=True)
                while process.poll() is None:
                    write(root/'supervisor.json', {'time': time.time(), 'supervisor_pid': os.getpid(),
                          'driver_pid': process.pid, 'restarts': restarts, 'checkout': config['checkout']})
                    time.sleep(5)
                write(root/'supervisor.json', {'time': time.time(), 'supervisor_pid': os.getpid(),
                      'driver_pid': None, 'exit_code': process.returncode, 'restarts': restarts})
                if (root/'STOP').exists() or process.returncode == 0:
                    return
                restarts += 1
                time.sleep(30)


if __name__ == '__main__':
    main()
