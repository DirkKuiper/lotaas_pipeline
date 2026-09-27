#!/usr/bin/python3
"""Rewrite a filterbank's src_raj in place when it lies outside 0-24 h (lotaas_reprocessing.coordinates).

Beams converted before the converter wrapped RA carry LOFAR's negative angles
west of 0 h (and the early-cycle converter's beyond 24 h near the pole). Only
the 8 bytes of the value change. Without --apply it lists what it would change.

  python3 ops/wrap_ra.py [--apply] FILE...
"""
import argparse
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from euroflash.spider import sigproc_header  # noqa: E402
from lotaas_reprocessing.coordinates import packed_ra  # noqa: E402


def wrap(path, apply=False):
    """(old, new) src_raj of one filterbank, rewritten when apply; None when it needs nothing."""
    with open(path, 'r+b' if apply else 'rb') as stream:
        header, _, where = sigproc_header(stream.read(4096))
        old = header.get('src_raj')
        if old is None or 0 <= old < 240000:
            return None
        new = packed_ra(old)
        if apply:
            stream.seek(where['src_raj'])
            stream.write(struct.pack('<d', new))
        return old, new


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--apply', action='store_true', help='Rewrite the headers (default: list only)')
    p.add_argument('files', nargs='+', type=Path)
    a = p.parse_args()
    for path in a.files:
        change = wrap(path, a.apply)
        if change:
            print(f"{'wrapped' if a.apply else 'would wrap'} {path}: src_raj {change[0]:.4f} -> {change[1]:.4f}")


if __name__ == '__main__':
    main()
