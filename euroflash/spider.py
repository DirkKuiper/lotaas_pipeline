"""Beams from SPIDER: early-cycle LOTAAS filterbanks already converted there.

/project/euflash/Data/EC_LOTAAS on SURF's SPIDER holds one tar per beam,
<obs>/SAP*/L<obs>_SAP<n>_BEAM<n>_beam_data.tar, each with the beam's
downsampled 32-bit filterbank (648 channels, 7.864 ms) and an RFI plot. That is
what conversion makes of LTA PSRFITS here, so a SPIDER beam skips staging and
conversion: its tar is streamed over SSH, the filterbank is written where
conversion would have put it, and the tar's size and SHA-256 are recorded.
Nothing on SPIDER is changed and no tar is kept.

The early-cycle converter (early_cycle_lotaas_pipeline, preproc/psrfits2fil_rfi.py)
wrote fch1 as OBSFREQ + BW/2, the top edge of the band, where SIGPROC and our
conversion (the mean of the channels' DAT_FREQ) give the first channel's
centre: half a channel lower. The header is corrected on the way in.

Inventory lines are ssh://<destination>/<path>, <destination> as ssh knows it.
Only 1-h survey SAPs (beams 0-73) are listed; confirmation observations
(a ring of up to 128 beams around a known source) are left out unless asked.

  python3 -m euroflash.spider inventory --destination spider -o ec_lotaas-inventory.txt
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import struct
import subprocess
import sys
import time
from urllib.parse import urlsplit

ROOT = '/project/euflash/Data/EC_LOTAAS'
TAR = re.compile(r'^L(\d+)_SAP(\d+)_BEAM(\d+)_beam_data\.tar$')
MEMBER = re.compile(r'^downsampled_(L\d+)_SAP(\d+)_BEAM(\d+)_32bit\.fil$')
SURVEY_BEAMS = range(0, 74)
CHUNK = 8 * 2**20
INTS = {'telescope_id', 'machine_id', 'data_type', 'nchans', 'nbits', 'nifs', 'barycentric',
        'pulsarcentric', 'nbeams', 'ibeam'}
DOUBLES = {'tstart', 'tsamp', 'fch1', 'foff', 'src_raj', 'src_dej', 'az_start', 'za_start', 'refdm', 'period'}


class Unreachable(ConnectionError):
    """ssh itself failed (network, key, agent): nothing is wrong with the file."""


def is_spider(url):
    return url.startswith('ssh://')


def parse(url):
    """The inventory row of one SPIDER beam URL, or None if the name is not a beam tar."""
    name = Path(urlsplit(url).path).name
    match = TAR.match(name)
    if not match:
        return None
    return {'surl': url, 'name': name, 'archive_obs': match[1], 'sap': int(match[2]), 'beam': int(match[3])}


def sigproc_header(buffer):
    """(header, header bytes, {key: offset of its value}) of a SIGPROC filterbank."""
    def string(offset):
        (n,) = struct.unpack_from('<i', buffer, offset)
        if not 0 < n < 80:
            raise ValueError('not a SIGPROC header')
        return buffer[offset + 4:offset + 4 + n].decode('ascii'), offset + 4 + n
    key, offset = string(0)
    if key != 'HEADER_START':
        raise ValueError('not a SIGPROC header')
    header, where = {}, {}
    while True:
        key, offset = string(offset)
        if key == 'HEADER_END':
            return header, offset, where
        where[key] = offset
        if key in INTS:
            (header[key],) = struct.unpack_from('<i', buffer, offset)
            offset += 4
        elif key in DOUBLES:
            (header[key],) = struct.unpack_from('<d', buffer, offset)
            offset += 8
        else:
            header[key], offset = string(offset)


class _Hashing:
    """A read-only stream that hashes and counts what passes through it."""

    def __init__(self, stream):
        self.stream, self.sha256, self.bytes = stream, hashlib.sha256(), 0

    def read(self, n=-1):
        data = self.stream.read(n)
        self.sha256.update(data)
        self.bytes += len(data)
        return data

    def drain(self):
        while self.read(CHUNK):
            pass


def _write_filterbank(source, size, output, edge_fch1):
    """Copy one filterbank member, checking its shape and correcting fch1; returns what was found."""
    head = source.read(min(size, 4096))
    header, length, where = sigproc_header(head)
    if header.get('nbits') != 32 or header.get('nifs', 1) != 1 or not header.get('nchans'):
        raise ValueError(f'expected a 32-bit single-IF filterbank, found {header}')
    frame = header['nchans'] * 4
    if (size - length) % frame:
        raise ValueError(f'filterbank of {size} bytes is not whole spectra of {header["nchans"]} channels (truncated?)')
    samples = (size - length) // frame
    seconds = samples * header['tsamp']
    if not 600 <= seconds <= 7200:
        raise ValueError(f'{seconds:.0f} s of data; a survey beam is about an hour')
    fch1 = header['fch1']
    if edge_fch1:
        # The band's top edge to the first channel's centre (foff is negative).
        fch1 = header['fch1'] + header['foff'] / 2
        head = head[:where['fch1']] + struct.pack('<d', fch1) + head[where['fch1'] + 8:]
    partial = output.with_name(output.name + '.partial')
    with partial.open('wb') as out:
        out.write(head)
        copied = len(head)
        while copied < size:
            data = source.read(min(CHUNK, size - copied))
            if not data:
                break
            out.write(data)
            copied += len(data)
    if copied != size:
        partial.unlink(missing_ok=True)
        raise IOError(f'filterbank ended after {copied} of {size} bytes')
    return partial, {'member_bytes': size, 'samples': samples, 'duration_s': round(seconds, 3),
                     'nchans': header['nchans'], 'tsamp': header['tsamp'], 'source_name': header.get('source_name'),
                     'fch1_archive': header['fch1'], 'fch1': fch1}


def fetch(url, output, receipt_path, destination=None, edge_fch1=True, ssh_options=()):
    """Stream one beam tar from SPIDER into `output` (the converted-filterbank path).

    Raises Unreachable when ssh cannot connect or authenticate, and another
    error when the file itself is missing, truncated or not a survey beam.
    Returns the receipt, also written to receipt_path.
    """
    import tarfile
    row = parse(url)
    if row is None:
        raise ValueError(f'not a SPIDER beam tar: {url}')
    parts = urlsplit(url)
    output, receipt_path = Path(output), Path(receipt_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 4 * 2**30:
        raise OSError('Insufficient disk space for a SPIDER beam')
    remote = shlex.quote(parts.path)
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30', *ssh_options,
               destination or parts.netloc, f'stat -c %s -- {remote} && exec cat -- {remote}']
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    partial, found = None, None
    try:
        announced = process.stdout.readline().strip()
        stream = _Hashing(process.stdout)
        if announced.isdigit():
            with tarfile.open(fileobj=stream, mode='r|') as archive:
                for member in archive:
                    match = MEMBER.match(Path(member.name).name)
                    if not (member.isfile() and match):
                        continue
                    if partial is not None:
                        raise ValueError('more than one filterbank in the archive')
                    ids = (match[1], int(match[2]), int(match[3]))
                    if ids != ('L' + row['archive_obs'], row['sap'], row['beam']):
                        raise ValueError(f'archive member {member.name} is not beam {ids}')
                    partial, found = _write_filterbank(archive.extractfile(member), member.size, output, edge_fch1)
            stream.drain()
        else:
            process.stdout.read()
        error = process.stderr.read().decode(errors='replace').strip()
        code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        if partial is not None:
            partial.unlink(missing_ok=True)
        raise
    if code == 255:
        raise Unreachable(f'ssh to {destination or parts.netloc} failed: {error[-500:]}')
    if code != 0 or not announced.isdigit():
        raise FileNotFoundError(f'SPIDER could not read {parts.path} (exit {code}): {error[-500:]}')
    if stream.bytes != int(announced):
        raise IOError(f'received {stream.bytes} of {int(announced)} bytes of {row["name"]}')
    if partial is None:
        raise ValueError(f'no downsampled 32-bit filterbank in {row["name"]}')
    receipt = {'url': url, 'bytes': stream.bytes, 'sha256': stream.sha256.hexdigest(),
               'seconds': round(time.monotonic() - started, 3), 'fil': str(output), **found}
    tmp = receipt_path.with_name(receipt_path.name + '.partial')
    tmp.write_text(json.dumps(receipt, indent=2))
    os.replace(partial, output)
    os.replace(tmp, receipt_path)
    return receipt


LISTING = r"""find {root} -mindepth 1 -maxdepth 3 \( -name _work -o -name _logs -o -name results \) -prune -o \
  -type f -regextype posix-extended -regex '.*/SAP[0-9]+/L[0-9]+_SAP[0-9]+_BEAM[0-9]+_beam_data\.tar' \
  -printf '%s %p\n'"""


def inventory(destination, root=ROOT, confirmation=False, ssh_options=()):
    """(urls, skipped) for every beam tar under root on SPIDER: survey SAPs only unless confirmation."""
    listing = subprocess.run(['ssh', '-o', 'BatchMode=yes', *ssh_options, destination,
                              LISTING.format(root=shlex.quote(root))],
                             check=True, capture_output=True, text=True).stdout
    saps = defaultdict(list)
    for line in listing.splitlines():
        size, path = line.split(' ', 1)
        row = parse('ssh://' + destination + path)
        if row:
            saps[(row['archive_obs'], row['sap'])].append((row, int(size)))
    urls, skipped = [], []
    for key, beams in sorted(saps.items()):
        survey = all(r['beam'] in SURVEY_BEAMS for r, _ in beams)
        if survey or confirmation:
            urls += [r['surl'] for r, _ in sorted(beams, key=lambda b: b[0]['beam'])]
        else:
            skipped.append(f'L{key[0]}_SAP{key[1]:03d}: {len(beams)} beams, a confirmation observation')
    return urls, skipped


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    inv = sub.add_parser('inventory', help='List SPIDER beam tars as a campaign inventory')
    inv.add_argument('--destination', default='spider', help='ssh destination of SPIDER (default: the spider alias)')
    inv.add_argument('--root', default=ROOT)
    inv.add_argument('--confirmation', action='store_true', help='Include confirmation observations')
    inv.add_argument('-o', '--output', type=Path, required=True)
    get = sub.add_parser('fetch', help='Fetch one beam into a directory (for checks by hand)')
    get.add_argument('url')
    get.add_argument('directory', type=Path)
    get.add_argument('--destination')
    get.add_argument('--keep-edge-fch1', action='store_true', help='Leave fch1 as the archive has it')
    a = p.parse_args(argv)
    if a.command == 'inventory':
        urls, skipped = inventory(a.destination, a.root, a.confirmation)
        partial = a.output.with_name(a.output.name + '.partial')
        partial.write_text(f'# SPIDER {a.destination}:{a.root}, listed {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}\n'
                           + ''.join(u + '\n' for u in urls))
        os.replace(partial, a.output)
        for line in skipped:
            print('skipped', line, file=sys.stderr)
        print(f'{len(urls)} beams in {len({(parse(u)["archive_obs"], parse(u)["sap"]) for u in urls})} SAPs -> {a.output}')
    else:
        row = parse(a.url)
        obs = 'L' + row['archive_obs']
        name = f'downsampled_{obs}_SAP{row["sap"]:03d}_BEAM{row["beam"]:03d}_32bit.fil'
        receipt = fetch(a.url, a.directory/name, a.directory/(row['name'] + '.receipt.json'),
                        a.destination, edge_fch1=not a.keep_edge_fch1)
        print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
