"""The published LOTAAS sources, and how the original LOTAAS search found them.

The ATNF catalogue's SURVEY field lists every survey that detected a pulsar,
the discovery survey first. 'lotaas' in it marks a published LOTAAS
detection, and 'lotaas' first a LOTAAS discovery: 383 and 58 in catalogue
2.8.1. How LOTAAS found each is not in the catalogue, so it comes from the
papers:

- Michilli et al. 2018 (MNRAS 480, 3457), Table 4: seven discoveries made
  through their single pulses by the LOTAAS Single-pulse Searcher. J0139+33
  was never seen in the periodicity search; J0301+20 and J0317+13 were
  found in it later; J0454+45, J1340+65, J1404+11 and J1849+15 were bright
  enough for both (Sanidas et al. 2019, A&A 626, A104).
- The RRATs LOTAAS detected: single pulses.
- Every other source: the periodicity search, whose discoveries and
  redetections of known pulsars Sanidas et al. 2019 list.

Standard library only, beside euroflash.psrcat, which reads the same file.
"""
import os
from pathlib import Path
import tarfile

from euroflash import psrcat

MICHILLI_2018 = 'Michilli et al. 2018, Table 4'
# Name in the paper, period (s), DM, whether the periodicity search also found it.
SINGLE_PULSE_DISCOVERIES = [('J0139+33', 1.248, 21.2, False), ('J0301+20', 1.207, 19.0, True),
                            ('J0317+13', 1.974, 12.9, True), ('J0454+45', 1.389, 20.8, True),
                            ('J1340+65', 1.394, 30.0, True), ('J1404+11', 2.650, 18.5, True),
                            ('J1849+15', 2.233, 77.4, True)]
PERIODIC = 'periodic'
SINGLE = 'single pulse'
BOTH = 'single pulse and periodic'


def catalogue_text(path=None, member='psrcat.db'):
    """A file of the cached catalogue package: psrcat.db, or psrcat_ref for its references."""
    path = Path(path or os.environ.get('LOTAAS_PSRCAT') or psrcat.DEFAULT)
    try:
        if path.suffix == '.db':
            return (path if member == 'psrcat.db' else path.with_name(member)).read_text(errors='replace')
        with tarfile.open(path) as archive:
            found = next(m for m in archive.getmembers() if m.name.endswith(member))
            return archive.extractfile(found).read().decode(errors='replace')
    except (OSError, StopIteration, tarfile.TarError):
        return ''


def attributes(text):
    """{PSRJ: {'surveys': [...], 'type': str, 'reference': str}} from the raw records."""
    found, record = {}, {}

    def close():
        if record.get('PSRJ'):
            found[record['PSRJ']] = {'surveys': [s.strip().lower() for s in record.get('SURVEY', '').split(',') if s.strip()],
                                     'type': record.get('TYPE', ''), 'reference': record.get('ref', '')}

    for line in text.splitlines():
        if line.startswith('@'):
            close()
            record = {}
            continue
        if not line.strip() or line.startswith('#'):
            continue
        fields = line.split()
        if len(fields) >= 2 and fields[0] not in record:
            record[fields[0]] = fields[1]
            if fields[0] == 'PSRJ' and len(fields) > 2:
                record['ref'] = fields[2]
    close()
    return found


def sources(path=None):
    """Every published LOTAAS detection, as euroflash.psrcat pulsars with how LOTAAS found it."""
    text = catalogue_text(path)
    if not text:
        return []
    found = attributes(text)
    out = []
    for p in psrcat.parse(text):
        a = found.get(p['name'])
        if not a or 'lotaas' not in a['surveys']:
            continue
        period = 1 / p['f0']
        mode, note = PERIODIC, 'periodicity search (Sanidas et al. 2019)'
        if 'RRAT' in a['type'].upper():
            mode, note = SINGLE, 'RRAT: single pulses'
        for name, p_paper, dm_paper, periodic in SINGLE_PULSE_DISCOVERIES:
            # Named from the discovery position (J1340+65 is now J1343+6634): match period, DM and declination.
            if (abs(period - p_paper) < 0.01 and abs(p['dm'] - dm_paper) < 1.5
                    and abs(p['dec'] - float(name[5:])) < 3.5):
                mode, note = (BOTH if periodic else SINGLE), f'{name} in {MICHILLI_2018}'
        out.append(dict(p, period=period, discovery=a['surveys'][0] == 'lotaas', reference=a['reference'],
                        surveys=a['surveys'], rrat='RRAT' in a['type'].upper(), lotaas_mode=mode, lotaas_note=note))
    return out
