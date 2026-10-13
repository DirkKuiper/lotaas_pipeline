"""Sources this campaign tells us something new about, and the leads that may turn out to be such sources.

A finding is a source with at least one claim the published record does not already hold
(CLAIMS): a new source, a detection before its discovery, LOFAR's first detection of it, a
position, period or DM better than any published one, or a value that conflicts with the
catalogue. People record findings (reviews.sqlite findings, latest row per source); 'not new'
and 'not real' keep a check on record so the same lead is not checked twice.

Leads are what the index finds by itself (indexer.derive_findings), for a person to check:

- blind folds near catalogued pulsars whose entries are loose (no timing solution, a position
  worse than 3', or a period known to worse than 1e-5 of itself), compared with everything
  published of the same pulsar;
- catalogued pulsars found here that no LOFAR publication reports (catalogue_pulsars).

"Published" is the newest ATNF catalogue the web layer has (Config.compare_catalogue, fetched
with `python -m web.findings fetch`; the pipeline itself stays on its pinned version) together
with data/literature.tsv, results of papers the catalogue does not hold yet. On 13 October 2026
ATNF 2.8.1 still listed J2347+02 at 23:47 +02 with P 1.38347 s, while Olszanski et al. 2025 and
I-LOFAR (2026) had timed it as J2347+0300: a lead is not a finding until the literature is checked.

Our own uncertainties are deliberately loose. A position is the S/N-weighted centre of the
beams that saw the source, +-0.25 deg / sqrt(beams) and never better than 5': where published
timing exists, earlier TAB fits here were 3-6' off while claiming 3-7'. A period is good to a
quarter of the search's frequency resolution, P^2 / (4 T); a DM to 0.3 (statistic >= 50) or 0.5.

Standard library only; fetch() alone uses the network.
"""
from collections import OrderedDict
import csv
import json
import math
from pathlib import Path
import re
import sys
import tarfile
import urllib.request

CLAIMS = OrderedDict([
    ('new', 'New source'),
    ('precovery', 'Detected before its discovery'),
    ('first_lofar', 'First LOFAR detection'),
    ('position', 'Better position'),
    ('period', 'Better period'),
    ('dm', 'Better DM'),
    ('conflict', 'Conflicts with the catalogue'),
    ('other', 'Other'),
])
STATUSES = OrderedDict([
    ('finding', 'Finding'),
    ('follow_up', 'Needs follow-up'),
    ('not_new', 'Not new'),
    ('not_real', 'Not real'),
])
LITERATURE = Path(__file__).resolve().parent / 'data' / 'literature.tsv'
ATNF_DOWNLOADS = 'https://www.atnf.csiro.au/research/pulsar/psrcat/downloads/'

# A loose catalogue entry, worth comparing a detection with.
LOOSE_POSITION_DEG = 3 / 60
LOOSE_PERIOD = 1e-5
# Folds considered: neither RFI-like nor below this statistic.
MIN_STATISTIC = 10.0
# A fold matches a loose entry within this period window (P, 2P, P/2) and DM window.
PERIOD_WINDOW = 0.02
HARMONIC_WINDOW = 0.003
MATCH_RADIUS_DEG = 1.5
# Our own uncertainties (see the module docstring).
TAB_HALF_DEG = 0.25
MIN_POSITION_DEG = 5 / 60
DM_ERROR_STRONG, DM_ERROR_WEAK = 0.3, 0.5
STRONG_STATISTIC = 50.0
# A claim needs our uncertainty this many times smaller than the best published one.
POSITION_GAIN, PERIOD_GAIN, DM_GAIN = 2.0, 3.0, 2.0
CONFLICT_SIGMA = 4.0
YEAR = re.compile(r'(\d\d)[a-z]?$')
DISCOVERY_KEYS = ('PSRJ', 'RAJ', 'DECJ', 'P0', 'F0', 'DM')


# ---------------------------------------------------------------- catalogue
def last_unit(text):
    """One unit of the last quoted digit of a number: psrcat quotes errors in these."""
    parts = text.lower().split('e')
    mantissa, exponent = parts[0], int(parts[1]) if len(parts) > 1 else 0
    decimals = len(mantissa.split('.')[1]) if '.' in mantissa else 0
    return 10.0 ** (exponent - decimals)


def ref_year(ref):
    """The year in a psrcat reference code ('mss+20' -> 2020), or None."""
    found = YEAR.search(ref or '')
    if not found:
        return None
    year = int(found.group(1))
    return 1900 + year if year > 60 else 2000 + year


def sexagesimal(text, hours):
    """(degrees, one unit of the last field in degrees) of '19:21', '+34:07:12.5' and the like."""
    sign = -1 if text.startswith('-') else 1
    parts = text.lstrip('+-').split(':')
    value = sum(float(p) / 60 ** i for i, p in enumerate(parts))
    unit = (1.0, 1 / 60, 1 / 3600)[len(parts) - 1] * last_unit(parts[-1])
    scale = 15.0 if hours else 1.0
    return sign * value * scale, unit * scale


def _value(raw, key):
    """(value text, absolute error or None, reference) of one catalogue parameter."""
    value, error, ref = raw[key]
    if error is None:
        return value, None, ref
    return value, float(error) * last_unit(value), ref


def parse(text):
    """Catalogue records with their uncertainties.

    Positions come from RAJ/DECJ or ELONG/ELAT; an unquoted error counts one unit of the last
    field, so '23:47 +02' is +-15' in RA and +-1 deg in Dec. 'timed' means an epoch and a spin-down.
    """
    out, current = [], None
    for line in text.splitlines():
        if line.startswith('@'):
            if current:
                out.append(current)
            current = None
            continue
        if not line.strip() or line.startswith('#'):
            continue
        fields = line.split()
        key, rest = fields[0], fields[1:]
        if key == 'PSRJ':
            current = {'name': rest[0] if rest else '', 'raw': {}}
            current['raw']['PSRJ'] = (rest[0] if rest else '', None, rest[1] if len(rest) > 1 else None)
            continue
        if current is None or not rest or key in current['raw']:
            continue
        value, error, ref = rest[0], None, None
        if len(rest) >= 3:
            error, ref = rest[1], rest[2]
        elif len(rest) == 2:
            try:
                float(rest[1])
                error = rest[1]
            except ValueError:
                ref = rest[1]
        current['raw'][key] = (value, error, ref)
    if current:
        out.append(current)
    return [entry for entry in (_complete(e) for e in out) if entry]


def _complete(entry):
    raw = entry.pop('raw')
    entry['bname'] = raw.get('PSRB', (None,))[0]
    entry['refs'] = {k: v[2] for k, v in raw.items() if v[2]}
    entry['survey'] = raw.get('SURVEY', ('',))[0]
    entry['rrat'] = 'RRAT' in raw.get('TYPE', ('',))[0].upper()
    entry['timed'] = 'PEPOCH' in raw and ('F1' in raw or 'P1' in raw)
    try:
        if 'RAJ' in raw and 'DECJ' in raw:
            entry['ra'], ra_unit = sexagesimal(raw['RAJ'][0], True)
            entry['dec'], dec_unit = sexagesimal(raw['DECJ'][0], False)
            ra_err = (float(raw['RAJ'][1]) if raw['RAJ'][1] else 1.0) * ra_unit
            dec_err = (float(raw['DECJ'][1]) if raw['DECJ'][1] else 1.0) * dec_unit
            entry['pos_err'] = math.hypot(ra_err * math.cos(math.radians(entry['dec'])), dec_err)
            entry['pos_text'] = f"{raw['RAJ'][0]} {raw['DECJ'][0]}"
            entry['pos_ref'] = raw['RAJ'][2]
        elif 'ELONG' in raw and 'ELAT' in raw:
            lam, bet, eps = (math.radians(float(raw['ELONG'][0])), math.radians(float(raw['ELAT'][0])),
                             math.radians(23.4392911))
            dec = math.asin(math.sin(bet) * math.cos(eps) + math.cos(bet) * math.sin(eps) * math.sin(lam))
            ra = math.atan2(math.sin(lam) * math.cos(eps) - math.tan(bet) * math.sin(eps), math.cos(lam))
            entry['ra'], entry['dec'] = math.degrees(ra) % 360, math.degrees(dec)
            entry['pos_err'] = 1 / 3600
            entry['pos_text'] = f"{format_ra(entry['ra'])} {format_dec(entry['dec'])}"
            entry['pos_ref'] = raw['ELONG'][2]
    except (ValueError, IndexError):
        pass
    if 'ra' not in entry:
        return None
    try:
        if 'DM' in raw:
            value, error, ref = _value(raw, 'DM')
            entry['dm'], entry['dm_err'], entry['dm_ref'] = float(value), error or last_unit(value), ref
        if 'P0' in raw:
            value, error, ref = _value(raw, 'P0')
            entry['period'], entry['p_err'], entry['p_ref'] = float(value), error or last_unit(value), ref
        elif 'F0' in raw:
            value, error, ref = _value(raw, 'F0')
            f0 = float(value)
            entry['period'], entry['p_err'], entry['p_ref'] = 1 / f0, (error or last_unit(value)) / f0 ** 2, ref
    except ValueError:
        pass
    years = [ref_year(entry['refs'].get(k)) for k in DISCOVERY_KEYS if entry['refs'].get(k)]
    entry['year'] = min((y for y in years if y), default=None)
    entry['source'] = 'catalogue'
    return entry


_loaded = {}


def load(path):
    """The catalogue package (psrcat_pkg tarball or a bare psrcat.db) at path, parsed, cached by mtime."""
    path = Path(path)
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return [], None
    cached = _loaded.get(str(path))
    if cached and cached[0] == stamp:
        return cached[1], cached[2]
    if path.suffix == '.db':
        text = path.read_text(errors='replace')
    else:
        with tarfile.open(path) as archive:
            member = next(m for m in archive.getmembers() if m.name.endswith('psrcat.db'))
            text = archive.extractfile(member).read().decode(errors='replace')
    version = next((line.split()[1] for line in text.splitlines()[:5] if line.startswith('#CATALOGUE')), None)
    entries = parse(text)
    _loaded[str(path)] = (stamp, entries, version)
    return entries, version


def literature(path=None):
    """Rows of data/literature.tsv as catalogue-like entries (source 'literature').

    Columns: name, aliases (comma-separated), reference, url, ra, dec (sexagesimal), pos_err_arcsec,
    period_s, period_err_s, dm, dm_err, lofar (1: a LOFAR detection), note. Empty cells are unknown.
    """
    try:
        text = Path(path or LITERATURE).read_text()
    except OSError:
        return []
    out = []
    for row in csv.DictReader((line for line in text.splitlines() if line.strip() and not line.startswith('#')),
                              delimiter='\t'):
        entry = {'name': row['name'].strip(), 'aliases': [a.strip() for a in (row.get('aliases') or '').split(',')
                                                          if a.strip()],
                 'reference': row['reference'].strip(), 'url': (row.get('url') or '').strip(),
                 'lofar': (row.get('lofar') or '').strip() == '1', 'note': (row.get('note') or '').strip(),
                 'source': 'literature', 'timed': False, 'refs': {}, 'year': None}
        if row.get('ra') and row.get('dec'):
            entry['ra'] = sexagesimal(row['ra'].strip(), True)[0]
            entry['dec'] = sexagesimal(row['dec'].strip(), False)[0]
            entry['pos_err'] = float(row.get('pos_err_arcsec') or 1.0) / 3600
            entry['pos_text'] = f"{row['ra'].strip()} {row['dec'].strip()}"
            entry['pos_ref'] = entry['reference']
        if row.get('period_s'):
            entry['period'] = float(row['period_s'])
            entry['p_err'] = float(row.get('period_err_s') or last_unit(row['period_s']))
            entry['p_ref'] = entry['reference']
            entry['timed'] = entry['p_err'] < 1e-8
        if row.get('dm'):
            entry['dm'] = float(row['dm'])
            entry['dm_err'] = float(row.get('dm_err') or last_unit(row['dm']))
            entry['dm_ref'] = entry['reference']
        found = re.search(r'(19|20)\d\d', entry['reference'])
        entry['year'] = int(found[0]) if found else None
        out.append(entry)
    return out


# ------------------------------------------------------------------- helpers
def separation(ra1, dec1, ra2, dec2):
    a1, d1, a2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    cosine = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(a1 - a2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def format_ra(deg):
    h = (deg % 360) / 15
    m = (h % 1) * 60
    return f'{int(h):02d}:{int(m):02d}:{(m % 1) * 60:04.1f}'


def format_dec(deg):
    sign = '-' if deg < 0 else '+'
    d = abs(deg)
    m = (d % 1) * 60
    return f'{sign}{int(d):02d}:{int(m):02d}:{round((m % 1) * 60):02d}'


def angle(deg):
    """'62′' or '1.0°' for an uncertainty in degrees."""
    return f'{deg * 60:.0f}′' if deg < 1.5 else f'{deg:.1f}°'


def names_of(entry):
    return {n for n in [entry.get('name'), entry.get('bname'), *entry.get('aliases', [])] if n}


def loose(entry):
    return bool(entry.get('period') and entry.get('dm') is not None and (
        not entry['timed'] or entry['pos_err'] > LOOSE_POSITION_DEG or entry['p_err'] / entry['period'] > LOOSE_PERIOD))


def same_source(entry, ra, dec, period, dm, names=()):
    """Whether a published entry describes the source measured at ra, dec, period, dm (or is named so)."""
    if names_of(entry) & set(names):
        return True
    if 'ra' not in entry or not entry.get('period') or entry.get('dm') is None:
        return False
    if separation(entry['ra'], entry['dec'], ra, dec) > max(MATCH_RADIUS_DEG, 2 * entry['pos_err'] + 0.2):
        return False
    return (abs(entry['period'] / period - 1) < 2e-3
            and abs(entry['dm'] - dm) <= max(2.0, 3 * entry.get('dm_err', 0), 0.1 * dm))


def best(entries, value, error):
    """The entry with the smallest error on one parameter, or None."""
    known = [e for e in entries if e.get(value) is not None and e.get(error) is not None]
    return min(known, key=lambda e: e[error] / (e[value] if value == 'period' else 1)) if known else None


# --------------------------------------------------------------------- leads
def measure(hits):
    """Our position, period and DM from the folds of one source (see the module docstring)."""
    top = max(hits, key=lambda h: h['statistic'])
    same = [h for h in hits if h['observation'] == top['observation'] and h['sap'] == top['sap']]
    beams = {}
    for h in same:
        if h['item'] not in beams or h['statistic'] > beams[h['item']]['statistic']:
            beams[h['item']] = h
    weight = sum(h['statistic'] for h in beams.values())
    ra = sum(h['ra'] * h['statistic'] for h in beams.values()) / weight
    dec = sum(h['dec'] * h['statistic'] for h in beams.values()) / weight
    fundamental = max((h for h in hits if h['ratio'] == 1), key=lambda h: h['statistic'], default=None)
    out = {'ra': ra, 'dec': dec, 'pos_err': max(MIN_POSITION_DEG, TAB_HALF_DEG / math.sqrt(len(beams))),
           'beams': len(beams), 'observation': top['observation'], 'sap': top['sap'], 'observed': top['observed'],
           'statistic': top['statistic'], 'dm': top['dm'],
           'dm_err': DM_ERROR_STRONG if top['statistic'] >= STRONG_STATISTIC else DM_ERROR_WEAK}
    if fundamental:
        seconds = fundamental.get('seconds') or 3600.0
        out['period'] = fundamental['period']
        out['p_err'] = fundamental['period'] ** 2 / (4 * seconds)
    return out


def claims_against(ours, published, lofar_seen):
    """[(type, text)] of what our measurement adds to the published entries of the same source."""
    out = []
    where = best(published, 'ra', 'pos_err')
    if where and where['pos_err'] > POSITION_GAIN * ours['pos_err']:
        out.append(('position', f"{format_ra(ours['ra'])} {format_dec(ours['dec'])} ±{angle(ours['pos_err'])} "
                                f"({ours['beams']} beam{'s' if ours['beams'] != 1 else ''}); published "
                                f"{where['pos_text']} ±{angle(where['pos_err'])} ({describe(where, 'pos_ref')})"))
    spin = best(published, 'period', 'p_err')
    if spin and ours.get('period'):
        combined = math.hypot(spin['p_err'], ours['p_err'])
        off = ours['period'] - spin['period']
        if spin['p_err'] > PERIOD_GAIN * ours['p_err']:
            out.append(('period', f"P {ours['period']:.6f} ± {ours['p_err']:.1g} s; published "
                                  f"{spin['period']:.7g} ± {spin['p_err']:.1g} s ({describe(spin, 'p_ref')})"))
        if not spin['timed'] and abs(off) > CONFLICT_SIGMA * combined:
            out.append(('conflict', f"P {ours['period']:.6f} s is {off / spin['period'] * 100:+.2f}% "
                                    f"({off / combined:+.0f}σ) from the published {spin['period']:.7g} s "
                                    f"({describe(spin, 'p_ref')})"))
    disp = best(published, 'dm', 'dm_err')
    if disp and disp['dm_err'] > DM_GAIN * ours['dm_err']:
        out.append(('dm', f"DM {ours['dm']:.1f} ± {ours['dm_err']:.1f}; published {disp['dm']:g} ± "
                          f"{disp['dm_err']:.2g} ({describe(disp, 'dm_ref')})"))
    if not any(seen_by_lofar(e, lofar_seen) for e in published):
        out.append(('first_lofar', 'no LOFAR publication known here reports it'))
    year = min((e['year'] for e in published if e.get('year')), default=None)
    observed = int(str(ours['observed'])[:4]) if ours.get('observed') else None
    if year and observed and observed < year and out:
        out.append(('precovery', f'seen here in {observed}; first published {year}'))
    return out


def describe(entry, ref_key):
    if entry['source'] == 'literature':
        return entry['reference']
    return f"ATNF {entry['name']}{', ' + entry[ref_key] if entry.get(ref_key) else ''}"


def fold_leads(folds, catalogue, published_extra, lofar_seen):
    """Leads from blind folds near loose catalogue entries.

    folds: dicts with key, item, dm, period, statistic, seconds, ra, dec, observation, sap, observed.
    Returns {source: lead}, the source named by its best published entry.
    """
    by_dec = {}
    for f in folds:
        by_dec.setdefault(int(f['dec'] // 2), []).append(f)
    everything = catalogue + published_extra
    leads = {}
    for entry in catalogue:
        if not loose(entry):
            continue
        band = int(entry['dec'] // 2)
        hits = []
        for f in (f for b in (band - 1, band, band + 1) for f in by_dec.get(b, ())):
            if separation(f['ra'], f['dec'], entry['ra'], entry['dec']) > max(MATCH_RADIUS_DEG, 2 * entry['pos_err']):
                continue
            if abs(f['dm'] - entry['dm']) > max(2.0, 3 * entry['dm_err'], 0.12 * entry['dm']):
                continue
            for ratio in (1, 2, 0.5):
                off = f['period'] / (entry['period'] * ratio) - 1
                if abs(off) <= (PERIOD_WINDOW if ratio == 1 else HARMONIC_WINDOW):
                    hits.append(dict(f, ratio=ratio, period=f['period'] / ratio))
                    break
        if not hits:
            continue
        ours = measure(hits)
        if not ours.get('period'):
            continue
        # Far beyond a precise position, a period only percent-close is chance (FAST's _P candidates).
        if (abs(ours['period'] / entry['period'] - 1) > HARMONIC_WINDOW
                and separation(ours['ra'], ours['dec'], entry['ra'], entry['dec']) > 2 * entry['pos_err'] + 0.2):
            continue
        published = [e for e in everything
                     if e is entry or same_source(e, ours['ra'], ours['dec'], ours['period'], ours['dm'],
                                                  names_of(entry))]
        claims = claims_against(ours, published, lofar_seen)
        if not [c for c in claims if c[0] != 'precovery']:
            continue
        name = min(published, key=lambda e: (not e['timed'], e.get('pos_err', 9), len(e['name'])))['name']
        lead = leads.setdefault(name, {'source': name, 'names': set(), 'claims': [], 'evidence': [], 'ours': ours,
                                       'published': {}})
        lead['names'] |= set().union(*(names_of(e) for e in published))
        if ours['statistic'] >= lead['ours']['statistic']:
            lead['ours'], lead['claims'] = ours, claims
        strongest = {}
        for h in hits:
            strongest[h['key']] = max(strongest.get(h['key'], 0.0), h['statistic'])
        lead.setdefault('strength', {}).update(strongest)
        lead['evidence'] = sorted(lead['strength'], key=lambda k: -lead['strength'][k])[:12]
        for key, value, error in (('position', 'ra', 'pos_err'), ('period', 'period', 'p_err'), ('dm', 'dm', 'dm_err')):
            top = best(published, value, error)
            if top:
                lead['published'][key] = describe(top, {'ra': 'pos_ref', 'period': 'p_ref', 'dm': 'dm_ref'}[value])
    for lead in leads.values():
        lead['names'] = sorted(lead['names'] - {lead['source']})
        lead.pop('strength', None)
    return leads


def seen_by_lofar(entry, lofar_seen):
    return bool(names_of(entry) & lofar_seen or entry.get('lofar')
                or 'lotaas' in (entry.get('survey') or '').lower().split(','))


def lofar_leads(found, catalogue, published_extra, lofar_seen, existing):
    """Leads from catalogued pulsars found here that no LOFAR publication reports (indexer.derive_pulsars).

    found: dicts with psrj, bname, period, dm, ra, dec, evidence (candidate keys), statistic, observed, how.
    Any entry for the same source (by name, or by position, period and DM: J2316+5619 is LOTAAS's J2315+58)
    that LOFAR has seen settles it.
    """
    covered = {n for lead in existing.values() for n in [lead['source'], *lead['names']]}
    out = {}
    for row in found:
        names = {row['psrj'], row.get('bname')} - {None}
        if names & (lofar_seen | covered):
            continue
        if any(seen_by_lofar(e, lofar_seen) and (names & names_of(e) or (
                row.get('ra') is not None and same_source(e, row['ra'], row['dec'], row['period'], row['dm'])))
               for e in catalogue + published_extra):
            continue
        out[row['psrj']] = {'source': row['psrj'], 'names': sorted(names - {row['psrj']}),
                            'claims': [('first_lofar', 'no LOFAR publication known here reports it; '
                                        'found here ' + row['how'])],
                            'evidence': row['evidence'], 'published': {},
                            'ours': {'statistic': row.get('statistic'), 'observed': row.get('observed')}}
    return out


def lead_row(lead, catalogue_version, now):
    ours = dict(lead['ours'])
    return (lead['source'], json.dumps(lead['names']), json.dumps(lead['claims']), json.dumps(ours, default=str),
            json.dumps(lead['published']), json.dumps(lead['evidence']), ours.get('statistic'),
            str(ours.get('observed') or '')[:10] or None, catalogue_version, now)


# ---------------------------------------------------------------- recording
def latest(db, prefix='r.'):
    """The latest finding row per source: {source: row dict with claims and evidence parsed}."""
    out = {}
    for row in db.execute(f"""SELECT f.* FROM {prefix}findings f WHERE f.created = (SELECT MAX(g.created)
            FROM {prefix}findings g WHERE g.source = f.source) ORDER BY f.created DESC"""):
        row = dict(row)
        row['claims'] = json.loads(row['claims'] or '[]')
        row['evidence'] = json.loads(row['evidence'] or '[]')
        row['names'] = json.loads(row['names'] or '[]')
        out[row['source']] = row
    return out


def checked_names(findings):
    """Every name a recorded finding (of any status) covers: a lead under one of them is already checked."""
    return {n for f in findings.values() for n in [f['source'], *f['names']]}


# --------------------------------------------------------------------- fetch
def fetch(directory, version=None):
    """Download the current ATNF catalogue package into directory as psrcat_pkg.tar.gz (by hand:
    python -m web.findings fetch DIRECTORY [VERSION]). Keeps the version's own file beside it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = [f'psrcat_pkg.v{version}.tar.gz'] if version else ['psrcat_pkg.tar.gz']
    for name in names:
        target = directory / name
        urllib.request.urlretrieve(ATNF_DOWNLOADS + name, target)
        entries, found = load(target)
        if not entries:
            raise RuntimeError(f'{name} holds no catalogue')
        kept = directory / f'psrcat_pkg.v{found}.tar.gz'
        if target != kept:
            target.replace(kept)
        current = directory / 'psrcat_pkg.tar.gz'
        if current.exists() or current.is_symlink():
            current.unlink()
        current.symlink_to(kept.name)
        return found, len(entries)


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == 'fetch':
        print(*fetch(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None))
    else:
        sys.exit('usage: python -m web.findings fetch DIRECTORY [VERSION]')
