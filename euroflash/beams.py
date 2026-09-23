"""Beams the search leaves out.

In LOTAAS each SAP carries 74 beams: 73 coherent tied-array beams and one
incoherent sum of the stations. The archive stores that one under
`incoherentstokes/`; in every SAP retrieved so far (432 archive members over
four SAPs of L1163405 and L1163866) it is beam 12. Its sensitivity, beam
pattern and RFI response are unlike the coherent beams, it is not part of the
central flatfield (13-73), and flatfielded against that mean it overran the
single-pulse candidate guard. It is excluded from staging, conversion and
search. The archive layout is checked too, so an incoherent beam numbered
differently in another observation is still recognised.
"""
from pathlib import Path
import re

INCOHERENT_BEAMS = (12,)
_BEAM = re.compile(r'_(?:BEAM|B)(\d{1,3})(?:_|\.)')


def beam_number(name):
    """Beam number from an archive, PSRFITS or filterbank name, or None."""
    match = _BEAM.search(Path(name).name)
    return int(match.group(1)) if match else None


def is_incoherent(path):
    """True for a PSRFITS the archive filed under incoherentstokes/."""
    return 'incoherentstokes' in {part.lower() for part in Path(path).parts}


def excluded(path, beams=INCOHERENT_BEAMS):
    """Whether this beam is left out, by number or by archive layout."""
    return beam_number(path) in set(beams) or is_incoherent(path)
