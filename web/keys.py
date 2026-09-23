"""Stable names for beams and candidates, shared by the index, the snippets and the pages.

A single-pulse candidate's key is the one postproc.notify_candidates gives
its Slack post, so a post, a review page and a verdict name the same event.
It leaves out the fingerprint on purpose: the same event found again by a
later search is the same candidate.
"""
import hashlib
import re

ITEM = re.compile(r'(L\d+)_SAP(\d+)_BEAM(\d+)')


def item_of(beam_id):
    """'downsampled_L559289_SAP000_BEAM025_32bit_ff.fil' -> the stem the results use."""
    return beam_id[:-4] if beam_id and beam_id.endswith('.fil') else beam_id


def parse_item(item):
    """(observation, sap, beam) from a beam name, or None."""
    match = ITEM.search(item or '')
    return (match[1], int(match[2]), int(match[3])) if match else None


def sp_key(kind, item, dm, width, snr):
    return f'{kind}|{item}|DM{float(dm):.3f}|W{int(width)}|SN{float(snr):.3f}'


def short_id(key):
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def beam_label(item):
    parsed = parse_item(item)
    return f'{parsed[0]} SAP{parsed[1]:03d} B{parsed[2]:03d}' if parsed else item
