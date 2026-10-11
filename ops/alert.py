#!/usr/bin/python3
"""Slack alerts for what needs a person, run every five minutes from cron.

Only conditions the campaign cannot heal by itself, each once it has lasted a
while: a crash-looping or absent driver, dead SSH control masters (they need
the user's forwarded agent), no SAP searched for hours while work waits, a run
stuck for hours, a full disk, the web server down. From 9 to 11 October 2026
the driver crash-looped for 39 hours while health.log recorded it and nobody
was told.

Not spam: everything a tick has to say goes in one message; a condition is
announced once when it starts, again only if it still holds a day later, and
once when it has cleared; at most MAX_PER_DAY messages in any 24 hours.

    alert.py              check, post what is new, remember it
    alert.py --dry-run    show the conditions and the message; post nothing, remember nothing
    alert.py --test       post a test message
"""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/shared/results/dkuiper/lotaas/campaign')
CONFIG = Path.home()/'.config'/'lotaas'/'campaign.json'
SLACKRC = Path.home()/'.config'/'lotaas'/'slackrc'
CHECKOUT = '/home/dkuiper/lotaas-production'
WEB_PORT = 8000

REMIND_SECONDS = 24 * 3600
CLEAR_SECONDS = 15 * 60          # a condition must stay clear this long before it is called resolved
MAX_PER_DAY = 8
CRASHES_PER_HOUR = 3             # driver restarts within an hour that make a crash loop
NO_PROGRESS_SECONDS = 4 * 3600   # no SAP searched while SAPs wait
STUCK_DISPATCH_SECONDS = 6 * 3600
NODE_MIN_FREE_GB = 100


def condition(key, active, title, detail='', fix='', grace=15 * 60):
    return {'key': key, 'active': bool(active), 'title': title, 'detail': detail, 'fix': fix, 'grace': grace}


def checks(now, obs, state):
    """The conditions this tick, from observations (see observe) and the restart history in state."""
    out = []
    history = [h for h in state.get('restarts', []) if now - h[0] <= 3600 and h[1] == obs['supervisor_pid']]
    crashes = (obs['restarts'] - min(h[2] for h in history)) if history and obs['restarts'] is not None else 0
    out.append(condition(
        'driver_crashing', not obs['stop'] and crashes >= CRASHES_PER_HOUR,
        f'the campaign driver crashed {crashes} times in the last hour',
        f'Last error: {obs["driver_error"] or "see driver.log"}',
        f'Read {ROOT}/driver.log; once fixed, the supervisor restarts the driver within 30 s.', grace=0))
    out.append(condition(
        'driver_down', not obs['stop'] and not obs['driver_alive'] and crashes < CRASHES_PER_HOUR,
        'no campaign driver is running',
        f'supervisor.json last written {obs["supervisor_age"]:.0f} s ago' if obs['supervisor_age'] is not None else
        'no supervisor.json',
        f'crontab -l should start {CHECKOUT}/ops/supervise.py every minute; see {ROOT}/supervisor.log.'))
    down = obs['masters_down']
    out.append(condition(
        'masters_down', down, f'SSH control masters down for {len(down)} node(s): nothing can be dispatched there',
        ', '.join(down),
        f'Forward your SSH agent, then run {CHECKOUT}/ops/open-masters.sh {" ".join(down)} '
        '(after a reboot also /shared/results/dkuiper/lotaas/benchmarks/throughput-2026-10-06/bfc-ready.sh for '
        'the GPU nodes, and SIGTERM the driver to clear its dispatch back-off).'))
    idle = None if obs['last_searched'] is None else now - obs['last_searched']
    out.append(condition(
        'no_progress', obs['driver_alive'] and obs['waiting_saps'] and (idle is None or idle > NO_PROGRESS_SECONDS),
        f'no SAP searched for {idle / 3600:.1f} h while {obs["waiting_saps"]} SAPs wait'
        if idle is not None else f'no SAP searched yet while {obs["waiting_saps"]} SAPs wait',
        obs['last_failure'] or '', 'Check the recent dispatch logs in campaign/logs/ and the GPU nodes.', grace=0))
    stuck = obs['stuck_dispatches']
    out.append(condition(
        'stuck_dispatch', stuck, f'{len(stuck)} dispatch(es) running for more than '
        f'{STUCK_DISPATCH_SECONDS // 3600} h', ', '.join(stuck),
        'Read results/<run>/cpu-tier.log and logs/<run>.log; killing the run\'s euroflash.cluster process '
        'returns its SAPs to the queue.', grace=0))
    if obs['head_free_tb'] is not None and obs['min_free_tb'] is not None:
        out.append(condition(
            'head_disk', obs['head_free_tb'] < obs['min_free_tb'] + 0.5,
            f'{obs["head_free_tb"]:.1f} TB free on /shared/results: staging pauses below {obs["min_free_tb"]:g} TB',
            '', 'Free space (kept beams, held snippets, old benchmarks).', grace=3600))
    full = [f'{node} {gb} GB' for node, gb in sorted(obs['node_free_gb'].items()) if gb < NODE_MIN_FREE_GB]
    out.append(condition(
        'node_disk', full, f'{len(full)} node(s) below {NODE_MIN_FREE_GB} GB free in /home', ', '.join(full),
        'Look under ~/lotaas-runs on the node: run directories untouched for a day are removed at the next batch.',
        grace=30 * 60))
    out.append(condition(
        'web_down', not obs['web_up'], f'the web server does not answer on port {WEB_PORT}', '',
        '~/lotaas-web/web/ctl.sh start'))
    return out


def evaluate(now, obs, state):
    """(message or None, items, new state). Items: (kind, condition) with kind new/reminder/resolved."""
    state = json.loads(json.dumps(state or {}))
    active = state.setdefault('conditions', {})
    items = []
    for c in checks(now, obs, state):
        known = active.get(c['key'])
        if c['active']:
            known = known or {'since': now, 'sent': None}
            known.update(title=c['title'], clear_since=None)
            active[c['key']] = known
            if known['sent'] is None and now - known['since'] >= c['grace']:
                items.append(('new', c))
            elif known['sent'] is not None and now - known['sent'] >= REMIND_SECONDS:
                items.append(('reminder', c))
        elif known is not None:
            if known['sent'] is None:
                del active[c['key']]
                continue
            known['clear_since'] = known.get('clear_since') or now
            if now - known['clear_since'] >= CLEAR_SECONDS:
                items.append(('resolved', dict(c, title=known['title'], since=known['since'])))
    if obs['supervisor_pid'] is not None and obs['restarts'] is not None:
        state['restarts'] = [h for h in state.get('restarts', []) if now - h[0] <= 7200] + [
            [now, obs['supervisor_pid'], obs['restarts']]]
    state['messages'] = [t for t in state.get('messages', []) if now - t < 86400]
    if not items:
        return None, [], state
    lines = []
    for kind, c in items:
        if kind == 'resolved':
            lines.append(f':white_check_mark: Resolved: {c["title"]} (after {(now - c["since"]) / 3600:.1f} h)')
            continue
        lead = ':rotating_light: *LOTAAS campaign needs you*' if kind == 'new' else ':rotating_light: *Still unresolved*'
        lines.append(f'{lead}: {c["title"]}')
        if c['detail']:
            lines.append(f'    {c["detail"]}')
        if c['fix']:
            lines.append(f'    _Fix:_ {c["fix"]}')
    return '\n'.join(lines), items, state


def mark_sent(now, items, state):
    for kind, c in items:
        if kind == 'resolved':
            state['conditions'].pop(c['key'], None)
        else:
            state['conditions'][c['key']]['sent'] = now
    state['messages'] = state.get('messages', []) + [now]
    return state


def command_values(command, flag):
    if flag not in command:
        return []
    values = []
    for value in command[command.index(flag) + 1:]:
        if value.startswith('--'):
            break
        values.append(value)
    return values


def control_socket(control, node):
    return Path(control)/('lotaas-' + node.removeprefix('efc-').replace('-', ''))


def master_alive(control, node):
    try:
        return subprocess.run(['ssh', '-S', str(control_socket(control, node)), '-O', 'check', node],
                              capture_output=True, timeout=20).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def node_free_gb(control, node):
    try:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-S',
                                 str(control_socket(control, node)), node, 'df -BG --output=avail /home | tail -1'],
                                capture_output=True, text=True, timeout=30)
        return int(result.stdout.split()[-1].rstrip('G')) if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def driver_error(root):
    """The last exception line of driver.log ('' when none in its tail)."""
    try:
        with (root/'driver.log').open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 20000))
            tail = stream.read().decode(errors='replace').splitlines()
    except OSError:
        return ''
    for line in reversed(tail):
        head = line.split(':', 1)[0]
        if head and not line.startswith(' ') and (head.endswith('Error') or head.endswith('Exception')):
            return line.strip()[:300]
    return ''


def web_up(port=WEB_PORT):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=5):
            return True
    except OSError:
        return False


def observe(now, root=ROOT, config_path=CONFIG):
    config = json.loads(config_path.read_text())
    command = config['command']
    control = (command_values(command, '--control-dir') or [str(Path.home()/'.ssh'/'control')])[0]
    nodes = command_values(command, '--dispatch-nodes') + command_values(command, '--cpu-nodes')
    obs = {'stop': (root/'STOP').exists(), 'supervisor_pid': None, 'restarts': None, 'supervisor_age': None,
           'driver_alive': False, 'driver_error': driver_error(root)}
    try:
        supervisor = json.loads((root/'supervisor.json').read_text())
        obs.update(supervisor_pid=supervisor.get('supervisor_pid'), restarts=supervisor.get('restarts'),
                   supervisor_age=now - (root/'supervisor.json').stat().st_mtime)
        pid = supervisor.get('driver_pid')
        obs['driver_alive'] = bool(pid) and b'euroflash.campaign' in (Path('/proc')/str(pid)/'cmdline').read_bytes()
    except (OSError, ValueError):
        pass
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(nodes))) as pool:
        alive = dict(zip(nodes, pool.map(lambda n: master_alive(control, n), nodes)))
        up = [n for n in nodes if alive[n]]
        free = dict(zip(up, pool.map(lambda n: node_free_gb(control, n), up)))
    obs['masters_down'] = [n for n in nodes if not alive[n]]
    obs['node_free_gb'] = {n: gb for n, gb in free.items() if gb is not None}
    with sqlite3.connect(f'file:{root}/campaign-state.sqlite?mode=ro', uri=True, timeout=60) as db:
        obs['last_searched'] = db.execute("SELECT MAX(time) FROM events WHERE kind='searched'").fetchone()[0]
        obs['waiting_saps'] = db.execute("SELECT COUNT(*) FROM saps WHERE state IN "
                                         "('prepared','dispatched','flatfielding')").fetchone()[0]
        failure = db.execute("SELECT subject,detail FROM events WHERE kind='dispatch_failed' AND time>? "
                             "ORDER BY time DESC LIMIT 1", (now - NO_PROGRESS_SECONDS,)).fetchone()
        obs['last_failure'] = f'Last dispatch failure: {failure[0]}: {failure[1]}' if failure else ''
        obs['stuck_dispatches'] = [
            f'{run} ({(now - started) / 3600:.0f} h)' for run, started in db.execute(
                "SELECT s.run_name, MIN(e.time) FROM saps s JOIN events e ON e.kind='dispatched' AND e.subject=s.run_name "
                "WHERE s.state='dispatched' GROUP BY s.run_name")
            if now - started > STUCK_DISPATCH_SECONDS]
    obs['min_free_tb'] = float((command_values(command, '--min-free-tb') or ['nan'])[0])
    obs['head_free_tb'] = shutil.disk_usage(root).free / 1e12
    obs['web_up'] = web_up()
    return obs


def slack_config(path=SLACKRC):
    values = {}
    for line in path.read_text().splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip()
    return values['bot_token'], values.get('alert_channel_id') or values['channel_id']


def post(text, path=SLACKRC):
    token, channel = slack_config(path)
    request = urllib.request.Request(
        'https://slack.com/api/chat.postMessage', method='POST',
        data=json.dumps({'channel': channel, 'text': text, 'unfurl_links': False}).encode(),
        headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json; charset=utf-8'})
    with urllib.request.urlopen(request, timeout=30) as response:
        answer = json.load(response)
    if not answer.get('ok'):
        raise RuntimeError(f'Slack refused the message: {answer.get("error")}')


def main(argv=sys.argv[1:]):
    now = time.time()
    stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(now))
    if '--test' in argv:
        post(':white_check_mark: LOTAAS campaign alerts are set up. This channel hears from them only when the '
             'campaign needs a manual fix (crash loop, dead SSH masters, no progress, stuck runs, full disks, web down).')
        print(stamp, 'test message posted')
        return
    state_path = ROOT/'alert-state.json'
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError):
        state = {}
    obs = observe(now)
    message, items, state = evaluate(now, obs, state)
    if '--dry-run' in argv:
        print(json.dumps({c['key']: c['active'] for c in checks(now, obs, state)}))
        print(message or '(nothing to post)')
        return
    if message:
        if len(state['messages']) >= MAX_PER_DAY:
            print(stamp, f'suppressed ({MAX_PER_DAY} messages in 24 h):', message.replace('\n', ' | '))
        else:
            try:
                post(message)
                state = mark_sent(now, items, state)
                print(stamp, 'posted:', message.replace('\n', ' | '))
            except Exception as error:
                print(stamp, 'not posted, retried next tick:', error)
    temporary = state_path.with_suffix('.partial')
    temporary.write_text(json.dumps(state, indent=1) + '\n')
    os.replace(temporary, state_path)


if __name__ == '__main__':
    main()
