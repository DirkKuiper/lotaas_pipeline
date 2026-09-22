"""StageIT client usable on the host without third-party Python packages."""
import configparser
import json
import os
from pathlib import Path
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


def token_for_url(manifest, url):
    host = urlsplit(url).hostname or ""
    sites = {"SURF": ("grid.sara.nl", "grid.surfsara.nl", "surf.nl"), "JUELICH": ("fz-juelich.de",),
             "PSNC": ("man.poznan.pl", "psnc.pl"), "POZNAN": ("man.poznan.pl", "psnc.pl")}
    matches = []
    for entry in manifest["macaroons"]:
        site = entry["ltaSite"]["name"].upper()
        if any(host == suffix or host.endswith("." + suffix) for suffix in sites.get(site, ())):
            matches.append(entry)
    if not matches:
        raise ValueError(f"No site-specific macaroon for {host}")
    return max(matches, key=lambda x: x["validUntil"])["content"]
