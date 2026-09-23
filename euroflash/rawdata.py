"""Remove archive data once it has been converted, keeping its provenance.

A beam's archive tar (~4.9 GB) and extracted PSRFITS (~4.5 GB) are only needed
until the filterbank exists. Keeping both put ~11 GB per beam on the head's
pool, which filled after ~5,500 of LT5_004's 64,507 beams. What must survive is
the evidence of which bytes were processed: the download receipt (URL, size,
SHA256) and the extraction marker. Both stay; the marker records the deletion
so provenance reconciliation and retrieval resume know the data is gone on
purpose rather than lost.
"""
import json
import os
from pathlib import Path
import time


def extraction_marker(fits_path):
    """The `<archive>.extracted.json` marker for a FITS extracted by the retriever."""
    path = Path(fits_path).resolve()
    for directory in path.parents:
        marker = directory.parent / (directory.name + '.extracted.json')
        if marker.is_file():
            return marker, directory
    return None, None


def raw_deleted(marker):
    try:
        return bool(json.loads(Path(marker).read_text()).get('raw_deleted'))
    except (OSError, ValueError):
        return False


def _record(marker, deleted):
    value = json.loads(marker.read_text())
    value['raw_deleted'] = True
    value.setdefault('deleted_paths', [])
    value['deleted_paths'] = sorted(set(value['deleted_paths']) | {str(p) for p in deleted})
    value['raw_deleted_unix'] = time.time()
    partial = marker.with_name(marker.name + '.partial')
    partial.write_text(json.dumps(value, indent=2))
    os.replace(partial, marker)


def delete_archive(tar_path):
    """Remove a downloaded tar after extraction; its receipt and marker remain."""
    tar_path = Path(tar_path)
    if not tar_path.is_file():
        return []
    tar_path.unlink()
    return [tar_path]


def delete_converted(fits_path):
    """Delete one converted PSRFITS, its archive tar, and empty extraction folders.

    Returns the paths removed. Files the retriever did not produce (no marker)
    are still removed when asked, since the caller has a successful conversion.
    """
    fits_path = Path(fits_path)
    removed = []
    marker, extraction = extraction_marker(fits_path)
    if fits_path.is_file():
        fits_path.unlink()
        removed.append(fits_path)
    if marker is None:
        return removed
    remaining = [p for p in extraction.rglob('*.fits')] if extraction.is_dir() else []
    if not remaining:
        removed += delete_archive(extraction.parent / (extraction.name + '.tar'))
        for directory in sorted((d for d in extraction.rglob('*') if d.is_dir()), key=lambda d: -len(d.parts)):
            try:
                directory.rmdir()
            except OSError:
                pass
        try:
            extraction.rmdir()
        except OSError:
            pass
    _record(marker, removed)
    return removed
