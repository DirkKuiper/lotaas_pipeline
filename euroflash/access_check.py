"""Probe download authorisation for a staging request, without moving data.

Request 991619 returned HTTP 403 'Permission denied for GET on path ...' from
SURF while StageIT reported its files online, and a byte-range GET on an
earlier request still succeeded. That pattern is what a path-scoped macaroon
produces when the token presented belongs to a different directory: a request
spanning several observations carries one macaroon per path, and the
downloader used to choose whichever expired last.

This reports, per directory in a request, which macaroon is selected and
whether a 1 KiB range request is accepted. It prints path caveats and HTTP
status only. No token is printed, logged or written.
"""
import argparse
import collections
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from staging.client import StageIT, macaroon_paths, tokens_for_url


def probe(url, token, probe_bytes):
    request = Request(url, headers={'Authorization': 'Bearer ' + token,
                                    'Range': f'bytes=0-{probe_bytes - 1}'})
    try:
        with urlopen(request, timeout=60) as response:
            return {'status': response.status,
                    'received_probe_bytes': len(response.read(probe_bytes))}
    except HTTPError as error:
        return {'status': error.code, 'error': error.read().decode('utf-8', 'replace')[:200]}
    except URLError as error:
        return {'status': None, 'error': str(error.reason)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('request_id', type=int)
    p.add_argument('--output', type=Path)
    p.add_argument('--probe-bytes', type=int, default=1024)
    p.add_argument('--per-directory', type=int, default=1,
                   help='URLs to probe from each distinct directory')
    a = p.parse_args()

    manifest = StageIT().downloads(a.request_id)
    report = {'request_id': a.request_id, 'urls': len(manifest['urls']),
              'macaroons': [{'site': m['ltaSite']['name'], 'valid_until': m['validUntil'],
                             'path_caveats': macaroon_paths(m['content'])}
                            for m in manifest['macaroons']],
              'directories': []}

    grouped = collections.OrderedDict()
    for url in manifest['urls']:
        grouped.setdefault(str(Path(urlsplit(url).path).parent), []).append(url)

    for directory, urls in grouped.items():
        entry = {'directory': directory, 'urls': len(urls), 'probes': []}
        for url in urls[:a.per_directory]:
            ordered = tokens_for_url(manifest, url)
            result = probe(url, ordered[0], a.probe_bytes)
            result.update(url=url, macaroons_available=len(ordered),
                          selected_caveats=macaroon_paths(ordered[0]))
            if result['status'] in (401, 403) and len(ordered) > 1:
                # Report whether any other macaroon would have been accepted:
                # that distinguishes a selection fault from a real denial.
                alternates = [probe(url, token, a.probe_bytes)['status'] for token in ordered[1:]]
                result['alternate_statuses'] = alternates
            entry['probes'].append(result)
        report['directories'].append(entry)

    accepted = sum(1 for d in report['directories'] for pr in d['probes']
                   if pr['status'] in (200, 206))
    report['directories_probed'] = len(report['directories'])
    report['directories_accepted'] = accepted
    report['note'] = ('A probe checks authorisation only. It does not measure tape '
                      'availability or sustained throughput.')
    text = json.dumps(report, indent=2)
    if a.output:
        a.output.write_text(text + '\n')
    print(text)
    if accepted != len(report['directories']):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
