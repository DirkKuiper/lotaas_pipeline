"""The ATNF pulsar catalogue on the head node, without numpy.

The campaign driver and the web layer run on the head's own Python, which has
no numpy, pandas or psrqpy. psrqpy keeps the catalogue it downloads on the
compute nodes as a tarball in ~/.cache/psrqpy; its psrcat.db is plain text,
one "KEY VALUE [error] [reference]" line per parameter and records separated
by "@-----". This reads it once per process.

Harmonic matching is deliberately narrow. Sidelobe folds of a bright pulsar
turn up at its period, at integer fractions of it down to 1/32 (J0323+3944
and B2016+28 both appeared at 1/17) and at small multiples; any ratio m/n
with large m and n would match most periods by chance, so those are not tried
and every match needs the DM too.
"""
import math
import os
from pathlib import Path
import tarfile

DEFAULT = Path.home()/'.cache'/'psrqpy'/'psrcat_pkg.tar.gz'
# Topocentric against catalogue period: Earth's orbit shifts it by up to 1e-4.
PERIOD_TOLERANCE = 3e-4
HARMONICS = 32          # P/n for n up to this
MULTIPLES = 8           # m*P for m up to this
FRACTIONS = [(m, n) for n in range(2, 5) for m in range(2, 5) if m % n and n % m]   # 2/3, 3/2, 3/4, ...

_cache = {}


def _sexagesimal(text, hours):
    sign = -1 if text.strip().startswith('-') else 1
    parts = [abs(float(p)) for p in text.strip().lstrip('+-').split(':')]
    value = sum(p / 60 ** i for i, p in enumerate(parts))
    return sign * value * (15 if hours else 1)


def _records(text):
    record = {}
    for line in text.splitlines():
        if line.startswith('@'):
            if record:
                yield record
            record = {}
            continue
        if not line.strip() or line.startswith('#'):
            continue
        fields = line.split()
        if len(fields) >= 2 and fields[0] not in record:
            record[fields[0]] = fields[1]
    if record:
        yield record


def parse(text):
    """Pulsars with a position, a DM and a period: dicts of name, ra, dec (deg), dm, period, f1, pepoch."""
    pulsars = []
    for r in _records(text):
        try:
            ra, dec = _sexagesimal(r['RAJ'], True), _sexagesimal(r['DECJ'], False)
            dm = float(r['DM'])
            if 'F0' in r:
                f0 = float(r['F0'])
            elif 'P0' in r:
                f0 = 1 / float(r['P0'])
            else:
                continue
        except (KeyError, ValueError, ZeroDivisionError):
            continue
        f1 = 0.0
        try:
            f1 = float(r['F1']) if 'F1' in r else -float(r['P1']) * f0 ** 2 if 'P1' in r else 0.0
        except ValueError:
            pass
        try:
            pepoch = float(r['PEPOCH']) if 'PEPOCH' in r else None
        except ValueError:
            pepoch = None
        pulsars.append({'name': r.get('PSRJ', r.get('PSRB', '?')), 'bname': r.get('PSRB'),
                        'ra': ra, 'dec': dec, 'dm': dm, 'f0': f0, 'f1': f1, 'pepoch': pepoch})
    return pulsars


def load(path=None):
    """Every pulsar in the cached catalogue ([] when there is none)."""
    path = Path(path or os.environ.get('LOTAAS_PSRCAT') or DEFAULT)
    key = str(path)
    if key not in _cache:
        try:
            if path.suffix == '.db':
                text = path.read_text(errors='replace')
            else:
                with tarfile.open(path) as archive:
                    member = next(m for m in archive.getmembers() if m.name.endswith('psrcat.db'))
                    text = archive.extractfile(member).read().decode(errors='replace')
            _cache[key] = parse(text)
        except (OSError, StopIteration, tarfile.TarError):
            _cache[key] = []
    return _cache[key]


def separation(ra1, dec1, ra2, dec2):
    """Angle between two positions in degrees (haversine)."""
    ra1, dec1, ra2, dec2 = map(math.radians, (ra1, dec1, ra2, dec2))
    h = math.sin((dec2 - dec1) / 2) ** 2 + math.cos(dec1) * math.cos(dec2) * math.sin((ra2 - ra1) / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(h))))


def cone(pulsars, ra, dec, radius):
    """Pulsars within radius degrees, nearest first, each with its separation."""
    found = []
    for p in pulsars:
        if abs(p['dec'] - dec) > radius:
            continue
        s = separation(ra, dec, p['ra'], p['dec'])
        if s <= radius:
            found.append(dict(p, separation_deg=s))
    return sorted(found, key=lambda p: p['separation_deg'])


def period_at(pulsar, mjd=None):
    """The pulsar's period at an epoch, from F0 and F1."""
    f = pulsar['f0']
    if mjd is not None and pulsar.get('pepoch') is not None and pulsar.get('f1'):
        f += pulsar['f1'] * (mjd - pulsar['pepoch']) * 86400
    return 1 / f


def relations():
    """(label, ratio) with ratio = fold period / pulsar period."""
    out = [(f'1/{n}', 1 / n) for n in range(1, HARMONICS + 1)]
    out += [(f'{m}', float(m)) for m in range(2, MULTIPLES + 1)]
    out += [(f'{m}/{n}', m / n) for m, n in FRACTIONS]
    return out


RELATIONS = relations()


def match(period, dm, pulsars, mjd=None, dm_tolerance=(3.0, 0.1)):
    """The first pulsar (nearest first) whose period this is, at its DM: (pulsar, relation) or None."""
    for p in pulsars:
        if abs(dm - p['dm']) > max(dm_tolerance[0], dm_tolerance[1] * p['dm']):
            continue
        base = period_at(p, mjd)
        for label, ratio in RELATIONS:
            if abs(period / (base * ratio) - 1) <= PERIOD_TOLERANCE:
                return p, label
    return None
