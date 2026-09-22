"""Slack uploads for candidate plots, using only the standard library.

The runtime image has no ``slack_sdk``. Adding one would mean rebuilding
``euroflash-runtime.sif``, and the image identity is part of the run
fingerprint that records how every completed beam was searched, so a rebuild
would invalidate resume state and the provenance of finished work. Notification
is not part of the science path and must not force that, so this speaks to the
Slack Web API directly over ``urllib``.

Credentials follow the StageIT convention in ``staging/client.py``: an
environment variable, otherwise a private file under ``~/.config/lotaas``.
Tokens are never read from the repository and never copied to compute nodes.
"""
import configparser
import json
import logging
import mimetypes
import os
from pathlib import Path
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "~/.config/lotaas/slackrc"
API = "https://slack.com/api/"
RETRIES = 4
# Slack rejects uploads above this size; a diagnostic plot is far smaller.
MAX_BYTES = 1024 * 1024 * 1024


class SlackError(RuntimeError):
    """A Slack call failed after exhausting retries."""


def load_credentials(config=None):
    """Return (token, channel), preferring the environment over the config file.

    Returns (None, channel) when no token is configured, so that callers can
    stay quiet rather than fail: a missing token disables notification, it does
    not break a search.
    """
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    channel = os.environ.get("SLACK_CHANNEL_ID", "").strip()
    path = Path(config or os.environ.get("LOTAAS_SLACK_CONFIG", DEFAULT_CONFIG)).expanduser()
    if (not token or not channel) and path.is_file():
        mode = path.stat().st_mode & 0o077
        if mode:
            logger.warning("%s is readable beyond its owner; run: chmod 600 %s", path, path)
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string("[slack]\n" + path.read_text())
        token = token or parser["slack"].get("bot_token", "").strip()
        channel = channel or parser["slack"].get("channel_id", "").strip()
    if token and not token.startswith("xoxb-"):
        raise ValueError("Expected a Slack bot token beginning with xoxb-")
    return (token or None), (channel or None)


class Slack:
    """Minimal Slack Web API client for posting messages and plot images."""

    def __init__(self, token=None, channel=None, config=None, verify=True, timeout=60):
        if token is None or channel is None:
            found_token, found_channel = load_credentials(config)
            token = token or found_token
            channel = channel or found_channel
        self.token = token
        self.channel = channel
        self.timeout = timeout
        # Verified TLS is the default and works from the head node. The escape
        # hatch exists only for hosts with a broken certificate store; it is
        # never selected implicitly.
        self.context = None if verify else ssl._create_unverified_context()

    @property
    def enabled(self):
        return bool(self.token and self.channel)

    def _request(self, request, parse=True):
        """Send a prepared request, retrying transient failures and rate limits."""
        for attempt in range(RETRIES):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout, context=self.context) as response:
                    payload = response.read()
                if not parse:
                    return payload
                body = json.loads(payload.decode())
                if body.get("ok"):
                    return body
                error = body.get("error", "unknown_error")
                # Only rate limiting is worth retrying; a bad token or a channel
                # the bot cannot post to will fail the same way every time.
                if error != "ratelimited" or attempt == RETRIES - 1:
                    raise SlackError(f"Slack API error: {error}")
                delay = 2 ** attempt
            except urllib.error.HTTPError as problem:
                if problem.code == 429:
                    delay = int(problem.headers.get("Retry-After", 2 ** attempt))
                elif problem.code >= 500 and attempt < RETRIES - 1:
                    delay = 2 ** attempt
                else:
                    raise SlackError(f"Slack HTTP {problem.code}: {problem.reason}") from problem
            except urllib.error.URLError as problem:
                if attempt == RETRIES - 1:
                    raise SlackError(f"Slack unreachable: {problem.reason}") from problem
                delay = 2 ** attempt
            logger.warning("Slack call retrying in %ss", delay)
            time.sleep(delay)
        raise SlackError("Slack call exhausted retries")

    def call(self, method, **params):
        data = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None}).encode()
        return self._request(urllib.request.Request(
            API + method, data=data,
            headers={"Authorization": "Bearer " + self.token,
                     "Content-Type": "application/x-www-form-urlencoded; charset=utf-8"}))

    def call_json(self, method, payload):
        return self._request(urllib.request.Request(
            API + method, data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + self.token,
                     "Content-Type": "application/json; charset=utf-8"}))

    def check(self):
        """Confirm the token authenticates and the bot can post to the channel."""
        identity = self.call("auth.test")
        channel = self.call("conversations.info", channel=self.channel)["channel"]
        if channel.get("is_archived"):
            raise SlackError(f"Channel #{channel.get('name')} is archived")
        if not channel.get("is_member"):
            raise SlackError(f"Bot is not a member of #{channel.get('name')}; invite it before uploading files")
        return {"team": identity.get("team"), "bot": identity.get("user"),
                "channel": "#" + channel.get("name", ""), "channel_id": self.channel}

    def post_message(self, text):
        if not self.enabled:
            logger.info("Slack disabled; not sent: %s", text)
            return None
        return self.call("chat.postMessage", channel=self.channel, text=text)["ts"]

    def upload(self, path, title=None, comment=None):
        """Upload one file and share it to the channel.

        Uses the external upload flow: reserve a URL, PUT the bytes, then
        complete, which is what ``files_upload_v2`` does in ``slack_sdk``.
        """
        path = Path(path)
        size = path.stat().st_size
        if not size:
            raise SlackError(f"Refusing to upload empty file: {path}")
        if size > MAX_BYTES:
            raise SlackError(f"File exceeds Slack's upload limit: {path}")
        if not self.enabled:
            logger.info("Slack disabled; not uploaded: %s", path)
            return None
        ticket = self.call("files.getUploadURLExternal", filename=path.name, length=size)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._request(urllib.request.Request(
            ticket["upload_url"], data=path.read_bytes(), method="POST",
            headers={"Content-Type": content_type, "Content-Length": str(size)}), parse=False)
        result = self.call_json("files.completeUploadExternal", {
            "files": [{"id": ticket["file_id"], "title": title or path.name}],
            "channel_id": self.channel,
            **({"initial_comment": comment} if comment else {})})
        return result["files"][0]["id"]
