"""Right ascension as SIGPROC headers store it: packed hhmmss.s, from 0 to 24 h.

LOFAR writes a tied-array beam west of 0 h as a negative angle ('-00:07:35.0'),
and the early-cycle converter wrote beams near the pole beyond 24 h
('32:41:08'). SIGPROC has neither. Read back as packed values, -735.0 came out
as '-1:92:65', which every stage that places a beam on the sky refused: four
beams of L1276887 SAP000 (L606840, Dec +64) and five more SAPs failed their
single-pulse classification and folds on 25-26 September.

Standard library only: the campaign driver on the head reads headers too.
"""


def ra_hours(value):
    """Hours of right ascension, wrapped into 0-24 h, from 'hh:mm:ss.s' or a packed hhmmss.s.

    A negative value is negative as a whole: '-00:07:35.0' and -735.0 are both
    7 min 35 s west of 0 h.
    """
    if isinstance(value, str):
        text = value.strip()
        sign = -1.0 if text.startswith('-') else 1.0
        hours, minutes, seconds = ([float(part) for part in text.lstrip('+-').split(':')] + [0.0, 0.0])[:3]
    else:
        sign = -1.0 if value < 0 else 1.0
        packed = abs(float(value))
        hours, minutes, seconds = packed // 10000, packed % 10000 // 100, packed % 100
    return sign * (hours + minutes / 60 + seconds / 3600) % 24


def packed_ra(value):
    """Right ascension ('hh:mm:ss.s', packed hhmmss.s or out of range) as a packed hhmmss.s within 0-24 h."""
    seconds = round(ra_hours(value) * 3600, 4) % 86400
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return hours * 10000 + minutes * 100 + seconds
