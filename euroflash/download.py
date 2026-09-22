"""Download staged LTA tar files with private tokens and verified HTTP lengths."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
from urllib.parse import urlsplit
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from staging.client import tokens_for_url


def open_with_any(url, tokens, headers):
    """Open the URL, trying each macaroon until one is accepted.

    A request spanning several observations carries a macaroon per path, and
    dCache answers 403 'Permission denied for GET on path ...' when the one
    presented is scoped to a different directory. Caveats are ranked before
    this point, but that parse is a best effort, so an authorisation failure
    falls through to the remaining tokens rather than ending the download.
    """
    if isinstance(tokens, str):
        tokens = [tokens]
    if not tokens:
        raise ValueError('No macaroon available for ' + url)
    refusals = []
    for token in tokens:
        try:
            return urlopen(Request(url, headers=dict(headers, Authorization='Bearer ' + token)),
                           timeout=120)
        except HTTPError as error:
            if error.code not in (401, 403):
                raise
            refusals.append(f'{error.code} {error.reason}')
    raise PermissionError(
        f'All {len(tokens)} macaroons were refused for {url}: ' + '; '.join(refusals))


def download(url, token, target, max_bytes):
    target = Path(target)
    receipt = target.with_suffix(target.suffix + '.receipt.json')
    if target.exists() and receipt.exists():
        old = json.loads(receipt.read_text())
        if old['url'] == url and target.stat().st_size == old['bytes']:
            return old
    part = Path(str(target) + '.partial')
    offset = part.stat().st_size if part.exists() else 0
    headers = {}
    if offset:
        headers['Range'] = f'bytes={offset}-'
    started = time.monotonic()
    with open_with_any(url, token, headers) as response:
        if offset and response.status != 206:
            offset = 0  # Server declined range; restart instead of appending duplicate bytes.
        if response.status == 206 and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
            raise RuntimeError('Server returned a mismatched byte range')
        length = int(response.headers['Content-Length'])
        total = offset + length
        if total > max_bytes:
            raise ValueError(f'Archive size {total} exceeds configured cap {max_bytes}')
        if shutil.disk_usage(target.parent).free < length + 2**30:
            raise OSError('Insufficient disk space for download')
        with part.open('ab' if offset else 'wb') as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    if part.stat().st_size != total:
        raise IOError('Incomplete HTTP body; rerun to resume the partial download')
    digest = hashlib.sha256()
    with part.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    part.replace(target)
    result = {'url': url, 'bytes': total, 'transferred_bytes': length,
              'sha256': digest.hexdigest(), 'seconds': time.monotonic() - started}
    receipt.write_text(json.dumps(result, indent=2))
    return result


def extract(archive, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as stream:
        # Extract only regular FITS files; disallow links, absolute paths and traversal.
        members = [m for m in stream if m.isfile() and m.name.lower().endswith('.fits')]
        if not members:
            raise ValueError('No PSRFITS members in archive')
        if shutil.disk_usage(directory).free < sum(m.size for m in members) + 2**30:
            raise OSError('Insufficient space for extracted FITS files')
        for member in members:
            name = Path(member.name)
            if name.is_absolute() or '..' in name.parts:
                raise ValueError('Unsafe archive member path')
        stream.extractall(directory, members=members, filter='data')
    return list(directory.rglob('*.fits'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('directory', type=Path)
    p.add_argument('--max-gib-per-file', type=float, default=64)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    a.directory.mkdir(parents=True, exist_ok=True)
    for url in manifest['urls']:
        target = a.directory / Path(urlsplit(url).path).name
        result = download(url, tokens_for_url(manifest, url), target, int(a.max_gib_per_file * 2**30))
        print(json.dumps(result), flush=True)
        print('Extracted:', [str(x) for x in extract(target, a.directory / target.stem)], flush=True)


if __name__ == '__main__':
    main()
