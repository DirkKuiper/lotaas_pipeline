"""Which catalogued pulsars LOFAR has already detected, and which it never has.

A pulsar counts as seen by LOFAR on any of:

- 'lotaas' in the ATNF catalogue's SURVEY field;
- a catalogue parameter cited to a paper whose title names LOFAR or LOTAAS
  (psrcat_ref): the censuses, the rotation-measure census and grid, LOTAAS
  timing, single international stations (FR606, I-LOFAR) and the Fermi
  searches. Karako-Argaman et al. 2015 followed GBNCC RRATs up with LOFAR
  without the catalogue saying which it saw; all of them count;
- the tables in data/lofar-pulsars.tsv (build() below), because the catalogue
  keeps one flux per band and so forgets most census detections: Bilous et
  al. 2016 (HBA census), Bilous et al. 2020 (LBA census), Pilia et al. 2016
  (profiles of 100 pulsars) and Sobey et al. 2019 (rotation-measure census).
  The two censuses also list pulsars LOFAR observed and did not detect, with
  a flux limit; those are kept as limits, not detections.

The rule is generous on purpose: a pulsar left over is one no LOFAR
publication known here reports, and a campaign detection of it still needs
the literature checked before it is called the first. Not covered: the
Kondratiev et al. 2016 millisecond-pulsar census beyond the 37 fluxes the
catalogue keeps (its VizieR table does not exist), and the redetections in
Coenen et al. 2014, Noutsos et al. 2015 and Donner et al. 2020, bright
pulsars and millisecond pulsars listed elsewhere too. NenuFAR is not LOFAR.

Standard library only; build() alone uses the network.
"""
import csv
import math
from pathlib import Path
import re
import sys
import urllib.request

from web import lotaas

TABLE = Path(__file__).resolve().parent / 'data' / 'lofar-pulsars.tsv'
TITLE = re.compile(r'LOFAR|LOTAAS|Low[- ]Frequency Array', re.I)
SOURCES = [
    # (reference, URL, format, name column, upper-limit column)
    ('Bilous et al. 2016', 'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/A%2BA/591/A134/obssum'
     '&-out.max=unlimited&-out=PSR&-out=l_Flux&-out=Flux', 'tsv', 'PSR', 'l_Flux'),
    ('Bilous et al. 2020', 'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/A%2BA/635/A75/obssum'
     '&-out.max=unlimited&-out=Name&-out=l_S&-out=S', 'tsv', 'Name', 'l_S'),
    ('Pilia et al. 2016', 'https://cdsarc.cds.unistra.fr/ftp/J/A+A/586/A92/tableb1.dat', (0, 10), None, None),
    ('Sobey et al. 2019', 'https://cdsarc.cds.unistra.fr/ftp/J/MNRAS/484/3646/tablea1.dat', (11, 21), None, None),
]


def citations(ref_text):
    """{code: 'Surname et al. year'} of the catalogue references whose titles name LOFAR or LOTAAS."""
    out = {}
    for block in ref_text.split('***')[1:]:
        words = block.split()
        if not words:
            continue
        code, entry = words[0], ' '.join(words[1:])
        if not TITLE.search(entry):
            continue
        year = re.search(r'(19|20)\d\d', entry)
        surname = entry.split(':', 1)[-1].strip().split(',')[0]
        out[code] = f'{surname} et al. {year[0] if year else ""}'.strip()
    return out


def _records(db_text):
    """(fields {key: first value}, every token of the record) per catalogue record."""
    record, tokens = {}, set()
    for line in db_text.splitlines():
        if line.startswith('@'):
            if record:
                yield record, tokens
            record, tokens = {}, set()
            continue
        if not line.strip() or line.startswith('#'):
            continue
        parts = line.split()
        record.setdefault(parts[0], parts[1] if len(parts) > 1 else '')
        tokens.update(parts[1:])
    if record:
        yield record, tokens


def resolver(names):
    """A function from a name as a paper gives it (J, B, or a short J name) to the catalogue's PSRJ."""
    exact = {}
    for psrj, bname in names:
        exact[psrj] = psrj
        if bname:
            exact[bname] = psrj

    def resolve(name):
        name = name.strip()
        if name in exact:
            return exact[name]
        # 'J0033+57' against 'J0033+5700', 'B1508+55' written without the B.
        matches = {p for n, p in exact.items() if n.startswith(name) or n[1:] == name}
        if not matches and re.fullmatch(r'J\d{4}[+-]\d{4}', name):
            # A position refined since: J0636+5129 is now J0636+5128.
            matches = {p for n, p in exact.items() if re.fullmatch(r'J\d{4}[+-]\d{4}', n)
                       and n[:8] == name[:8] and abs(int(n[8:]) - int(name[8:])) <= 1}
        return matches.pop() if len(matches) == 1 else None
    return resolve


def scattering_ms(dm, mhz=135.0):
    """Typical scattering time at mhz from the DM (Bhat et al. 2004), uncertain by a factor of about 4 either way."""
    if not dm or dm <= 0:
        return None
    x = math.log10(dm)
    return 10 ** (-6.46 + 0.154 * x + 1.07 * x * x - 3.86 * math.log10(mhz / 1000))


def table(path=None):
    """Rows of data/lofar-pulsars.tsv: name, reference, detected, limit_mjy."""
    try:
        text = Path(path or TABLE).read_text()
    except OSError:
        return []
    return list(csv.DictReader((line for line in text.splitlines() if not line.startswith('#')), delimiter='\t'))


def evidence(path=None, table_path=None):
    """{PSRJ: {'seen': [references], 'limits': [(reference, mJy)]}} for every pulsar LOFAR is known to have observed."""
    db_text = lotaas.catalogue_text(path)
    if not db_text:
        return {}
    codes = citations(lotaas.catalogue_text(path, 'psrcat_ref'))
    out, names = {}, []
    for record, tokens in _records(db_text):
        psrj = record.get('PSRJ') or record.get('PSRB')
        if not psrj:
            continue
        names.append((psrj, record.get('PSRB')))
        seen = sorted({codes[t] for t in tokens & codes.keys()})
        if 'lotaas' in record.get('SURVEY', '').lower().split(','):
            seen.insert(0, 'LOTAAS')
        if seen:
            out.setdefault(psrj, {'seen': [], 'limits': []})['seen'] += seen
    resolve = resolver(names)
    for row in table(table_path):
        psrj = resolve(row['name'])
        if psrj is None:
            continue
        entry = out.setdefault(psrj, {'seen': [], 'limits': []})
        if row['detected'] == '1':
            if row['reference'] not in entry['seen']:
                entry['seen'].append(row['reference'])
        else:
            entry['limits'].append((row['reference'], float(row['limit_mjy']) if row['limit_mjy'] else None))
    return out


def build(out=sys.stdout):
    """Write data/lofar-pulsars.tsv from VizieR and the CDS archive (run by hand: python -m web.lofar)."""
    writer = csv.writer(out, delimiter='\t', lineterminator='\n')
    out.write('# LOFAR pulsar detections and census non-detections, built by python -m web.lofar\n'
              '# from VizieR J/A+A/591/A134, J/A+A/635/A75, J/A+A/586/A92 and J/MNRAS/484/3646.\n')
    writer.writerow(['name', 'reference', 'detected', 'limit_mjy'])
    for reference, url, form, name_column, limit_column in SOURCES:
        text = urllib.request.urlopen(url, timeout=120).read().decode(errors='replace')
        if form == 'tsv':
            lines = [line for line in text.splitlines() if line and not line.startswith('#')]
            header = lines[0].split('\t')
            for line in lines[3:]:
                row = dict(zip(header, line.split('\t')))
                name = row.get(name_column, '').strip()
                if not name:
                    continue
                upper = row.get(limit_column, '').strip() == '<'
                flux = [v.strip() for k, v in row.items() if k in ('Flux', 'S')]
                writer.writerow([name, reference, 0 if upper else 1, flux[0] if upper and flux else ''])
        else:
            start, end = form
            for line in text.splitlines():
                if line[start:end].strip():
                    writer.writerow([line[start:end].strip(), reference, 1, ''])


if __name__ == '__main__':
    build()
