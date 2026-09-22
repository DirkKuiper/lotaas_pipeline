"""StageIT client usable on the host without third-party Python packages."""
import base64
import binascii
import configparser
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def private_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2)
    path.chmod(0o600)


class StageIT:
    def __init__(self, config=None):
        path = Path(config or os.environ.get("LOTAAS_STAGING_CONFIG", "~/.config/lotaas/stagingrc")).expanduser()
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string("[staging]\n" + path.read_text())
        self.token = parser["staging"]["api_token"].strip()
        hostname = parser["staging"].get("hostname", "sdc.astron.nl")
        self.base = f"https://{hostname}/stageit/api/staging/"

    def call(self, endpoint, data=None, method=None):
        request = Request(self.base + endpoint,
                          data=json.dumps(data).encode() if data is not None else None,
                          headers={"Authorization": "Token " + self.token,
                                   "Content-Type": "application/json"}, method=method)
        with urlopen(request, timeout=60) as response:
            return json.load(response)

    def query(self, query):
        result = self.call("graphql/", {"query": query})
        if result.get("errors"):
            raise RuntimeError("StageIT GraphQL: " + "; ".join(e["message"] for e in result["errors"]))
        return result["data"]

    def submit(self, surls):
        if not surls or any(urlsplit(s).scheme != "srm" for s in surls):
            raise ValueError("Expected a nonempty list of SRM URLs")
        return self.call("requests/", {"surls": surls, "notify": False, "request_type": "stage"})["id"]

    def status(self, request_id):
        return self.query('{ request(id: %d) { id currentStatus response } }' % int(request_id))["request"]

    def downloads(self, request_id):
        request_id = int(request_id)
        tokens = self.query('{ request(id: %d) { macaroons { content validUntil ltaSite { name } } } }' % request_id)["request"]["macaroons"]
        urls = self.call(f"requests/{request_id}/convert2webdav/")
        if not isinstance(urls, list) or any(not isinstance(u, str) for u in urls):
            raise RuntimeError("Unexpected StageIT WebDAV response")
        return {"request_id": request_id, "urls": urls, "macaroons": tokens}


SITES = {"SURF": ("grid.sara.nl", "grid.surfsara.nl", "surf.nl"), "JUELICH": ("fz-juelich.de",),
         "PSNC": ("man.poznan.pl", "psnc.pl"), "POZNAN": ("man.poznan.pl", "psnc.pl")}


def macaroon_paths(token):
    """Path prefixes a macaroon is restricted to, or none if it is unscoped.

    Macaroon caveats are cleartext: only the signature needs the issuing key.
    dCache encodes them in the libmacaroons binary format, base64url and
    often unpadded, so the identifiers can be read here without adding a
    macaroon library to the runtime image.
    """
    padded = token + "=" * (-len(token) % 4)
    raw = None
    for decoder in (base64.urlsafe_b64decode, base64.b64decode):
        try:
            raw = decoder(padded)
            break
        except (binascii.Error, ValueError):
            continue
    if raw is None:
        return []
    text = raw.decode("latin-1")
    root = ""
    for match in re.finditer(r"root:([^\x00-\x1f,;]*)", text):
        root = match.group(1).strip().rstrip("/")
    paths = [m.group(1).strip() for m in re.finditer(r"path:([^\x00-\x1f,;]*)", text)]
    if not paths:
        return [root] if root else []
    return [("/" + root.strip("/") + "/" + p.strip("/")).replace("//", "/").rstrip("/") or "/"
            if root else "/" + p.strip("/") for p in paths]


def tokens_for_url(manifest, url):
    """Every macaroon that could authorise this URL, most likely first.

    Selecting only on the latest expiry was wrong for a request spanning
    several observations: StageIT issues a macaroon per path, so the one
    expiring last is scoped to whichever directory it happens to cover and
    dCache answers 403 'Permission denied for GET on path ...' for the rest.
    Scoped macaroons whose path contains the URL are preferred, longest
    prefix first; unscoped ones follow; ones scoped elsewhere come last but
    are still offered, since caveat parsing is a best effort.
    """
    split = urlsplit(url)
    host, path = split.hostname or "", split.path
    matches = [entry for entry in manifest["macaroons"]
               if any(host == suffix or host.endswith("." + suffix)
                      for suffix in SITES.get(entry["ltaSite"]["name"].upper(), ()))]
    if not matches:
        raise ValueError(f"No site-specific macaroon for {host}")

    def rank(entry):
        prefixes = macaroon_paths(entry["content"])
        covering = [p for p in prefixes if p == "/" or path == p or path.startswith(p + "/")]
        if covering:
            return (0, -max(len(p) for p in covering), entry["validUntil"])
        if not prefixes:
            return (1, 0, entry["validUntil"])
        return (2, 0, entry["validUntil"])

    return [entry["content"] for entry in
            sorted(matches, key=lambda e: (rank(e)[0], rank(e)[1], _descending(rank(e)[2])))]


def _descending(value):
    # validUntil sorts as a string; invert it so later expiry ranks first.
    return tuple(-ord(character) for character in str(value))


def token_for_url(manifest, url):
    return tokens_for_url(manifest, url)[0]
