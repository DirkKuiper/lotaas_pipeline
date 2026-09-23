"""Ask dCache where a file actually is, instead of trusting a staging status.

StageIT reported request 991619 as "success" with all 666 files online ten
minutes after submission, while dCache still held every sampled file on tape
only. The WebDAV door then answers GET with 403 "Permission denied". The
namespace API reports the truth per file, and the request macaroon (scoped to
the whole projects tree) is accepted for it. Directory listings are refused,
so this is one query per file.
"""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

API = os.environ.get('LOTAAS_DCACHE_API', 'https://dcacheview.grid.surfsara.nl:22880/api/v1/namespace')
ONLINE = {'ONLINE', 'ONLINE_AND_NEARLINE'}


def pnfs_path(url):
    """The namespace path of an SRM or WebDAV URL for a file under /pnfs."""
    path = urlsplit(url).path
    if '/pnfs/' not in path:
        raise ValueError(f'Not a dCache namespace URL: {url}')
    return path[path.index('/pnfs/'):]


def locality(url, token, timeout=60):
    """'ONLINE', 'NEARLINE', 'ONLINE_AND_NEARLINE', or 'HTTP<code>' / 'UNREACHABLE'."""
    request = Request(API + quote(pnfs_path(url)) + '?locality=true',
                      headers={'Authorization': 'Bearer ' + token, 'Accept': 'application/json'})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response).get('fileLocality') or 'UNKNOWN'
    except HTTPError as error:
        return f'HTTP{error.code}'
    except (URLError, TimeoutError, OSError):
        return 'UNREACHABLE'


def is_online(value):
    return value in ONLINE
