"""Slack alerts: only what needs a person, once, together, and never a flood."""
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('alert', REPO/'ops'/'alert.py')
alert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alert)

T0 = 1_800_000_000.0


def healthy(**changes):
    obs = {'stop': False, 'supervisor_pid': 7, 'restarts': 0, 'supervisor_age': 3.0, 'driver_alive': True,
           'driver_error': '', 'masters_down': [], 'node_free_gb': {'efc-gpu-00': 700}, 'last_searched': T0 - 600,
           'waiting_saps': 12, 'last_failure': '', 'stuck_dispatches': [], 'head_free_tb': 40.0, 'min_free_tb': 10.0,
           'web_up': True}
    obs.update(changes)
    return obs


def run(ticks):
    """[(seconds after T0, obs)] -> [(seconds, message)] of what would be posted, sending everything."""
    state, posted = {}, []
    for seconds, obs in ticks:
        message, items, state = alert.evaluate(T0 + seconds, obs, state)
        if message:
            state = alert.mark_sent(T0 + seconds, items, state)
            posted.append((seconds, message))
    return posted


def test_a_healthy_campaign_says_nothing():
    assert run([(s, healthy(last_searched=T0 + s - 600)) for s in range(0, 86400, 300)]) == []


def test_a_condition_is_announced_after_its_grace_then_reminded_daily_then_resolved_once():
    def down(s):
        return healthy(masters_down=['efc-bfc-01'], node_free_gb={}, last_searched=T0 + s - 600)
    ticks = ([(s, down(s)) for s in range(0, 2 * 86400, 300)]
             + [(s, healthy(last_searched=T0 + s - 600)) for s in range(2 * 86400, 2 * 86400 + 3600, 300)])
    posted = run(ticks)
    assert [s for s, _ in posted] == [900, 900 + 86400, 2 * 86400 + 900]
    assert 'needs you' in posted[0][1] and 'efc-bfc-01' in posted[0][1] and 'open-masters.sh efc-bfc-01' in posted[0][1]
    assert 'Still unresolved' in posted[1][1]
    assert 'Resolved' in posted[2][1]


def test_a_short_blip_is_never_announced():
    ticks = [(0, healthy(web_up=False)), (300, healthy(web_up=False)), (600, healthy())]
    assert run(ticks) == []


def test_a_condition_that_returns_within_the_clear_period_is_not_resolved_and_announced_again():
    down, up = healthy(web_up=False), healthy()
    ticks = [(s, down) for s in range(0, 1800, 300)] + [(1800, up), (2100, down), (2400, down), (2700, up)]
    posted = run(ticks)
    assert len(posted) == 1, 'flapping must not produce resolved/new pairs'


def test_a_reboot_is_one_message_not_three():
    reboot = healthy(driver_alive=False, masters_down=['efc-gpu-00', 'efc-cpu-00'], web_up=False, node_free_gb={})
    posted = run([(s, reboot) for s in range(0, 1200, 300)])
    assert len(posted) == 1
    assert all(text in posted[0][1] for text in ('no campaign driver', 'SSH control masters', 'web server'))


def test_a_crash_loop_is_told_from_a_single_crash():
    # One crash and a restart: the supervisor heals that, nobody is told.
    once = [(0, healthy(restarts=0)), (300, healthy(restarts=1)), (600, healthy(restarts=1))]
    assert run(once) == []
    loop = [(s, healthy(restarts=s // 300, driver_alive=False, driver_error='sqlite3.OperationalError: no such table'))
            for s in range(0, 1500, 300)]
    posted = run(loop)
    assert len(posted) == 1 and 'crashed 3 times' in posted[0][1] and 'no such table' in posted[0][1]
    assert 'no campaign driver' not in posted[0][1], 'the crash loop explains the missing driver'


def test_a_stop_file_silences_the_driver_alerts():
    stopped = healthy(stop=True, driver_alive=False, restarts=None, supervisor_pid=None)
    assert run([(s, stopped) for s in range(0, 7200, 300)]) == []


def test_no_progress_needs_waiting_work():
    stalled = healthy(last_searched=T0 - 5 * 3600, last_failure='Last dispatch failure: run: exit 255')
    assert 'no SAP searched for 5.0 h' in run([(0, stalled)])[0][1]
    assert run([(0, healthy(last_searched=T0 - 5 * 3600, waiting_saps=0))]) == [], 'nothing waits: tape-bound, not stuck'


def test_old_messages_leave_the_daily_count():
    state = {'messages': [T0 - 90000, T0 - 100, T0 - 50]}
    _, _, state = alert.evaluate(T0, healthy(), state)
    assert state['messages'] == [T0 - 100, T0 - 50]


def test_observe_reads_the_campaign(tmp_path, monkeypatch):
    root = tmp_path/'campaign'
    root.mkdir()
    config = tmp_path/'campaign.json'
    config.write_text(json.dumps({'root': str(root), 'checkout': '/x',
                                  'command': ['python3', '-m', 'euroflash.campaign', 'run', '--min-free-tb', '10',
                                              '--control-dir', str(tmp_path)]}))
    (root/'supervisor.json').write_text(json.dumps({'supervisor_pid': 5, 'driver_pid': None, 'restarts': 4017}))
    (root/'driver.log').write_text('Traceback (most recent call last):\n  File "x"\n'
                                   'sqlite3.OperationalError: no such table: attempts\n')
    with sqlite3.connect(root/'campaign-state.sqlite') as db:
        db.execute('CREATE TABLE events (time REAL, kind TEXT, subject TEXT, detail TEXT)')
        db.execute('CREATE TABLE saps (key TEXT, state TEXT, run_name TEXT)')
        db.executemany('INSERT INTO events VALUES (?,?,?,?)', [
            (T0 - 8 * 3600, 'dispatched', 'campaign-old', 'efc-bfc-01: A,B'),
            (T0 - 3600, 'dispatched', 'campaign-new', 'efc-gpu-00: C'),
            (T0 - 7200, 'searched', 'Z', '73/73'), (T0 - 60, 'dispatch_failed', 'campaign-x', 'exit 255')])
        db.executemany('INSERT INTO saps VALUES (?,?,?)', [('A', 'dispatched', 'campaign-old'),
                                                         ('C', 'dispatched', 'campaign-new'), ('D', 'prepared', None)])
    monkeypatch.setattr(alert, 'web_up', lambda port=8000: True)
    obs = alert.observe(T0, root, config)
    assert obs['restarts'] == 4017 and obs['driver_alive'] is False
    assert obs['driver_error'] == 'sqlite3.OperationalError: no such table: attempts'
    assert obs['last_searched'] == T0 - 7200 and obs['waiting_saps'] == 3
    assert obs['stuck_dispatches'] == ['campaign-old (8 h)']
    assert obs['last_failure'] == 'Last dispatch failure: campaign-x: exit 255'
    assert obs['min_free_tb'] == 10.0 and obs['masters_down'] == []


def test_the_slack_config_prefers_an_alert_channel(tmp_path):
    rc = tmp_path/'slackrc'
    rc.write_text('bot_token = xoxb-1\nchannel_id = C1\n')
    assert alert.slack_config(rc) == ('xoxb-1', 'C1')
    rc.write_text('bot_token = xoxb-1\nchannel_id = C1\nalert_channel_id = C2\n')
    assert alert.slack_config(rc) == ('xoxb-1', 'C2')
