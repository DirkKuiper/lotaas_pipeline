"""Reply to a candidate's own Slack post with a reviewer's verdict.

Off unless the configuration sets slack_threads, and even then sent only when
the reviewer ticks the box for that verdict. The post's message is found from
the file ID postproc.notify_candidates recorded, through files.info, so no
change to the notifier is needed.
"""
import sqlite3

from web.store import LABELS


def thread_of(slack, file_id, channel):
    """The ts of the message that shared the file in the channel."""
    shares = slack.call('files.info', file=file_id)['file'].get('shares') or {}
    for scope in ('public', 'private'):
        messages = (shares.get(scope) or {}).get(channel) or []
        if messages:
            return messages[0]['ts']
    raise LookupError(f'File {file_id} is not shared in {channel}')


def post_verdict(cfg, key, label, note, reviewer, dm=None, slack=None):
    db = sqlite3.connect(cfg.index_db)
    try:
        row = db.execute('SELECT slack_file_id, channel FROM slack WHERE key=?', (key,)).fetchone()
    finally:
        db.close()
    if not row or not row[0]:
        raise LookupError('This candidate was never posted to Slack')
    if slack is None:
        from postproc.slack_client import Slack
        slack = Slack(channel=row[1])
    ts = thread_of(slack, row[0], row[1])
    text = f'*Review:* {LABELS[label]} — {reviewer}'
    if dm is not None:
        text += f' (inspected at DM {float(dm):.2f})'
    if note:
        text += f'\n{note}'
    return slack.call('chat.postMessage', channel=row[1], thread_ts=ts, text=text)['ts']
