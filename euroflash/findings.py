"""Which searched beams keep their filterbank, judged across the observation.

A beam's flatfielded filterbank (1.2 GB) is kept for review when FETCH
accepted one of its single-pulse candidates, or when one of its periodic folds
is worth a look. A FETCH candidate stops holding its beam once its latest
verdict in the web layer's reviews.sqlite says it is nothing to keep data for
(SETTLED: RFI, noise or a known source, given by a person or by web.triage):
on 25 September 316 of the 489 beams kept overnight held nothing else, 219 of
them the whole of L605714 for an undispersed burst at sunset. Judged beam by beam, a fold was worth a look unless the fold
itself was flagged, and in the first night under the per-SAP veto 44% of beams
were kept: 847 of the 849 for folds, most of them interference that one beam
cannot recognise or a known pulsar seen away from its own beam. A fold is not
worth keeping a filterbank for when it is

- flagged as RFI, or vetoed across the beams of its batch;
- below DM 2, where terrestrial signals live, or above DM 1000, where the
  periodic search no longer looks (scattering at 135 MHz smears any period
  there) and where the first night's uncapped folds were red noise;
- a catalogued pulsar within 5 degrees, at its DM, at its period, a
  harmonic 1/2..1/32, a multiple 2..8 or a small fraction (euroflash.psrcat);
- an interference line of other observations pointed more than 10 degrees
  away (at least two, and a tenth of them): a frequency that formed a family
  there, across beams at scattered DMs. No pulsar does that; the 3.38, 5.93
  and 11.66 s lines recur all over the survey;
- one of a family: the same frequency in four beams of the observation, or
  three across two SAPs, unless the family is one pulsar at one DM
  (periodicity_veto.dm_consistent). The per-SAP veto sees one SAP at a time;
  this sees every SAP searched so far.

Each beam's periodic evidence (its folds, and the peaks the veto removed) is
recorded once in periodic-index.jsonl in the campaign root, so judging costs
one read of a small file however large the campaign grows.

Standard library only: the campaign driver runs on the head's own Python.
"""
import bisect
import json
import math
from pathlib import Path
import re
import sqlite3

from euroflash import psrcat

INDEX = 'periodic-index.jsonl'
# Latest verdicts under which a FETCH candidate no longer holds its beam's filterbank.
SETTLED = ('rfi', 'noise', 'known')
# Why a searched beam is kept while none of its run's periodic results can be read yet.
AWAITING = 'periodic results not read yet'
ITEM = re.compile(r'(L\d+)_SAP(\d+)_BEAM(\d+)')
MIN_DM = 2.0
MAX_DM = 1000.0
CATALOGUE_RADIUS_DEG = 5.0
FAMILY_BINS = 1.5
FAMILY_BEAMS = 4
FAMILY_BEAMS_TWO_SAPS = 3
# An interference line: a family of this many beams at scattered DMs. A fold
# recurs as one when it lies within RECURRENCE_BINS of lines in at least
# RECURRENCE_OBSERVATIONS other observations pointed RECURRENCE_AWAY_DEG away,
# and in RECURRENCE_FRACTION of all of those: lines crowd the lowest
# frequencies of every observation, so a fixed count would match a slow pulsar
# by chance ever more often as the campaign grows. Real lines repeat to a
# quarter of a bin; on the first 21 observations a random period of 1-10 s
# matched by chance 1.2% of the time, of 10-30 s 4.6%.
LINE_BEAMS = 6
RECURRENCE_BINS = 0.75
RECURRENCE_OBSERVATIONS = 2
RECURRENCE_FRACTION = 0.1
RECURRENCE_AWAY_DEG = 10.0


def _position(info):
    try:
        return (psrcat._sexagesimal(info['RA (J2000)'], True), psrcat._sexagesimal(info['DEC (J2000)'], False))
    except (KeyError, TypeError, ValueError):
        return None, None


def beam_summary(output, run_name):
    """The periodic evidence of one searched beam: its folds and its vetoed peaks."""
    output = Path(output)
    item = output.parent.name
    match = ITEM.search(item)
    if not match:
        return None
    try:
        metadata = json.loads((output/'metadata.json').read_text())
    except (OSError, ValueError):
        metadata = {}
    ra, dec = _position(metadata.get('observation_info') or {})
    folds = []
    path = output/'periodicity_folded_candidates.jsonl'
    if path.is_file():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            frequency = r.get('refined_frequency_hz') or r.get('frequency_hz') or 0.0
            period = r.get('refined_period_seconds') or r.get('period_seconds') or (1 / frequency if frequency else 0.0)
            folds.append({'f': frequency, 'period': period,
                          'resolution': r.get('frequency_resolution_hz') or 1 / 3600., 'dm': r.get('dm', 0.0),
                          'statistic': r.get('statistic'), 'harmonics': r.get('harmonic_count'),
                          'rfi_like': bool(r.get('rfi_like')), 'multibeam_rfi': bool(r.get('multibeam_rfi')),
                          'catalogue': [m.get('name') for m in r.get('catalogue_matches') or []],
                          'plot': r.get('plot')})
    vetoed = []
    path = output/'periodicity_veto.json'
    if path.is_file():
        try:
            for v in json.loads(path.read_text()).get('vetoed') or []:
                vetoed.append({'f': v['frequency_hz'], 'dm': v['dm']})
        except (ValueError, KeyError):
            pass
    return {'item': item, 'observation': match[1], 'sap': int(match[2]), 'beam': int(match[3]),
            'run': run_name, 'fp': output.name, 'ra': ra, 'dec': dec, 'mjd': metadata.get('tstart_mjd'),
            'folds': folds, 'vetoed': vetoed, 'sp_candidates': 0}


def run_candidates(run_dir):
    """FETCH-accepted single-pulse candidates per beam stem in one run's ledger snapshots."""
    return {stem: len(keys) for stem, keys in run_candidate_keys(run_dir).items()}


def periodic_items(run_dir):
    """Beam stems whose periodic search succeeded in one run, by its ledger snapshots."""
    items = set()
    for snapshot in Path(run_dir).glob('*/ledger-snapshot.sqlite'):
        db = sqlite3.connect(f'file:{snapshot}?mode=ro', uri=True)
        try:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if 'attempts' in tables:
                items |= {Path(item).stem for (item,) in db.execute(
                    "SELECT DISTINCT item FROM attempts WHERE stage='periodicity' AND status='success'")}
        finally:
            db.close()
    return items


def candidate_key(beam_id, dm, width, snr):
    """The web layer's key for a FETCH candidate (web.keys.sp_key): its verdicts carry it.

    None when the ledger lacks a value: no verdict can reach that candidate, so it holds its beam.
    """
    if None in (dm, width, snr):
        return None
    return f'candidate|{Path(beam_id).stem}|DM{float(dm):.3f}|W{int(width)}|SN{float(snr):.3f}'


def run_candidate_keys(run_dir):
    """{beam stem: [key]} of the FETCH-accepted single-pulse candidates in one run's ledger snapshots."""
    keys = {}
    for snapshot in Path(run_dir).glob('*/ledger-snapshot.sqlite'):
        db = sqlite3.connect(f'file:{snapshot}?mode=ro', uri=True)
        try:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if 'detections' in tables:
                for beam, dm, width, snr in db.execute("SELECT beam_id, candidate_dm, width_samples, snr "
                                                       "FROM detections WHERE detection_type='candidate'"):
                    keys.setdefault(Path(beam).stem, []).append(candidate_key(beam, dm, width, snr))
        finally:
            db.close()
    return keys


class Index:
    """periodic-index.jsonl: one line per searched beam and run; the latest run of a beam wins.

    A run whose periodic results could not be read yet is recorded without beams
    and read again at each backfill until they can: the driver judges a run when
    its dispatch exits, and on 25 September the CPU tier's results of 112 runs
    appeared on the head about a minute later, so those runs held no beams, their
    folds kept nothing and their kept beams could never be judged again.
    """

    def __init__(self, root):
        self.path = Path(root)/INDEX
        self.beams, self.runs, self.empty = {}, set(), set()
        self._keys = {}
        if self.path.is_file():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self._take(json.loads(line))

    def _take(self, entry):
        self.runs.add(entry['run'])
        if not entry.get('item'):
            self.empty.add(entry['run'])
            return
        self.empty.discard(entry['run'])
        current = self.beams.get(entry['item'])
        if current is None or entry['run'] >= current['run']:
            self.beams[entry['item']] = entry

    def add_run(self, run_dir, record_empty=True):
        """Record every beam a run searched; returns their entries."""
        run_dir = Path(run_dir)
        candidates = run_candidates(run_dir)
        entries = []
        outputs = {p.parent for name in ('periodicity_search_summary.json', 'periodicity_folded_candidates.jsonl')
                   for p in run_dir.glob(f'*/processed/*/*/{name}')}
        for output in sorted(outputs):
            entry = beam_summary(output, run_dir.name)
            if entry is None:
                continue
            entry['sp_candidates'] = candidates.get(entry['item'], 0)
            entries.append(entry)
        with self.path.open('a') as stream:
            for entry in entries:
                stream.write(json.dumps(entry, sort_keys=True) + '\n')
            if not entries and record_empty:
                stream.write(json.dumps({'run': run_dir.name, 'item': None}) + '\n')
        for entry in entries:
            self._take(entry)
        if not entries:
            self.empty.add(run_dir.name)
        self.runs.add(run_dir.name)
        return entries

    def backfill(self, results):
        """Add every run under results/ not indexed yet, and those whose results could not be read before."""
        added = []
        for run_dir in sorted(Path(results).iterdir()):
            if run_dir.is_dir() and (run_dir.name not in self.runs or run_dir.name in self.empty):
                added += self.add_run(run_dir, record_empty=run_dir.name not in self.runs)
        return added

    def read(self, item, run_name):
        """Whether the index holds this run's results of the beam."""
        entry = self.beams.get(item)
        return entry is not None and entry['run'] == run_name

    def observation(self, observation):
        return [b for item, b in self.beams.items() if item and b['observation'] == observation]

    def candidate_keys(self, beam):
        """The keys of a beam's FETCH candidates in its run, read from the run's ledger snapshots once."""
        if beam['run'] not in self._keys:
            self._keys[beam['run']] = run_candidate_keys(self.path.parent/'results'/beam['run'])
        return self._keys[beam['run']].get(beam['item'], [])


def periodic_key(beam, fold):
    """The web layer's key for a fold (web.indexer): its reviews carry it."""
    import hashlib
    return 'periodicity|' + hashlib.sha256(f"{beam['item']}|{beam.get('fp')}|{fold.get('plot')}".encode()).hexdigest()


def latest_verdicts(path):
    """{key: label} of each candidate's latest verdict in reviews.sqlite; {} without one."""
    path = Path(path) if path else None
    if not path or not path.is_file():
        return {}
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        return dict(db.execute('SELECT key, label FROM reviews r WHERE created = '
                               '(SELECT MAX(created) FROM reviews v WHERE v.key = r.key)').fetchall())
    except sqlite3.Error:
        return {}
    finally:
        db.close()


def settled_keys(verdicts):
    """Candidates whose latest verdict says there is nothing to keep data for (SETTLED)."""
    return frozenset(key for key, label in verdicts.items() if label in SETTLED)


def reviewed_items(path, index, labels=('astro', 'unsure'), verdicts=None):
    """Beams with a candidate whose latest verdict is among labels: never released."""
    latest = latest_verdicts(path) if verdicts is None else verdicts
    wanted = {key for key, label in latest.items() if label in labels}
    items = {key.split('|')[1] for key in wanted if not key.startswith('periodicity|') and key.count('|') >= 1}
    for item, beam in index.beams.items():
        if item and any(periodic_key(beam, fold) in wanted for fold in beam['folds']):
            items.add(item)
    return items


class Judge:
    """Decides, fold by fold, what keeps a filterbank; built once per decision round."""

    def __init__(self, index, pulsars=None, settled=frozenset()):
        self.index = index
        self.settled = settled
        self.pulsars = psrcat.load() if pulsars is None else pulsars
        self._cones = {}
        # Interference lines: frequencies that formed a family across beams at
        # scattered DMs in their own observation. Only those count as evidence
        # that a period recurs elsewhere; single peaks crowd the lowest
        # frequencies of every observation and would match a slow pulsar by chance.
        from lotaas_reprocessing.periodicity_veto import dm_consistent
        by_observation = {}
        for b in index.beams.values():
            if b.get('item'):
                resolution = next((f['resolution'] for f in b['folds']), None) or 1 / 3600.
                for p in b['folds'] + b['vetoed']:
                    by_observation.setdefault(b['observation'], []).append((p['f'], (b['sap'], b['beam']), p['dm'], resolution))
        lines = {}
        for observation, peaks in by_observation.items():
            peaks.sort()
            freqs = [p[0] for p in peaks]
            for f, _, _, resolution in peaks:
                tolerance = FAMILY_BINS * resolution
                near = peaks[bisect.bisect_left(freqs, f - tolerance):bisect.bisect_right(freqs, f + tolerance)]
                if len({p[1] for p in near}) >= LINE_BEAMS and not dm_consistent([p[2] for p in near]):
                    centre = sorted(p[0] for p in near)[len(near) // 2]
                    lines[(observation, round(centre / resolution))] = (centre, observation, resolution)
        self.peaks = sorted(lines.values())
        self.frequencies = [p[0] for p in self.peaks]
        self.pointings = {}
        for b in index.beams.values():
            if b.get('item') and b['ra'] is not None:
                self.pointings.setdefault(b['observation'], (b['ra'], b['dec']))
        self._families = {}

    def cone(self, beam):
        key = (round(beam['ra'], 2), round(beam['dec'], 2))
        if key not in self._cones:
            self._cones[key] = psrcat.cone(self.pulsars, beam['ra'], beam['dec'], CATALOGUE_RADIUS_DEG)
        return self._cones[key]

    def recurring(self, fold, beam):
        """Other observations pointed far away where this frequency is an interference line."""
        tolerance = RECURRENCE_BINS * fold['resolution']
        lo = bisect.bisect_left(self.frequencies, fold['f'] - tolerance)
        hi = bisect.bisect_right(self.frequencies, fold['f'] + tolerance)
        here = (beam['ra'], beam['dec']) if beam['ra'] is not None else None
        elsewhere = set()
        for f, observation, _ in self.peaks[lo:hi]:
            if observation != beam['observation'] and self.far(here, observation):
                elsewhere.add(observation)
        return sorted(elsewhere)

    def far(self, here, observation):
        there = self.pointings.get(observation)
        return not (here and there) or psrcat.separation(*here, *there) > RECURRENCE_AWAY_DEG

    def recurrence_needed(self, beam):
        here = (beam['ra'], beam['dec']) if beam['ra'] is not None else None
        away = sum(1 for observation in self.pointings if observation != beam['observation'] and self.far(here, observation))
        return max(RECURRENCE_OBSERVATIONS, math.ceil(RECURRENCE_FRACTION * away))

    def family(self, fold, beam):
        """Beams of this observation holding the frequency: (beams, SAPs, DMs)."""
        members = self._families.get(beam['observation'])
        if members is None:
            members = sorted((p['f'], b['sap'], b['beam'], p['dm'])
                             for b in self.index.observation(beam['observation'])
                             for p in b['folds'] + b['vetoed'])
            self._families[beam['observation']] = members
        tolerance = FAMILY_BINS * fold['resolution']
        freqs = [m[0] for m in members]
        lo, hi = bisect.bisect_left(freqs, fold['f'] - tolerance), bisect.bisect_right(freqs, fold['f'] + tolerance)
        near = members[lo:hi]
        beams = {(m[1], m[2]) for m in near} | {(beam['sap'], beam['beam'])}
        saps = {s for s, _ in beams}
        dms = [fold['dm']] + [m[3] for m in near if (m[1], m[2]) != (beam['sap'], beam['beam'])]
        return beams, saps, dms

    def reason(self, fold, beam):
        """Why this fold keeps no filterbank, or None when it does."""
        from lotaas_reprocessing.periodicity_veto import dm_consistent
        if fold['rfi_like'] or fold['multibeam_rfi']:
            return 'rfi'
        if fold['dm'] < MIN_DM:
            return 'dm<2'
        if fold['dm'] > MAX_DM:
            return 'dm>1000'
        if fold['catalogue']:
            return 'catalogue ' + fold['catalogue'][0]
        if beam['ra'] is not None and self.pulsars:
            found = psrcat.match(fold['period'], fold['dm'], self.cone(beam), mjd=beam.get('mjd'))
            if found:
                return f"catalogue {found[0]['name']} {found[1]}"
        elsewhere = self.recurring(fold, beam)
        if len(elsewhere) >= self.recurrence_needed(beam):
            return f'recurs in {len(elsewhere)} other observations'
        beams, saps, dms = self.family(fold, beam)
        if ((len(beams) >= FAMILY_BEAMS or (len(saps) >= 2 and len(beams) >= FAMILY_BEAMS_TWO_SAPS))
                and not dm_consistent(dms, home=fold['dm'])):
            return f'family of {len(beams)} beams in {len(saps)} SAPs'
        return None

    def open_candidates(self, beam):
        """How many of a beam's FETCH candidates hold it: those without a settling verdict.

        All of them when the run's ledger snapshots no longer say which they were.
        """
        count = beam.get('sp_candidates') or 0
        if not count or not self.settled:
            return count
        keys = self.index.candidate_keys(beam)
        return sum(1 for key in keys if key not in self.settled) if keys else count

    def keep(self, beam):
        """(keep?, why): FETCH candidates without a settling verdict keep; else the first fold worth a look."""
        held = self.open_candidates(beam)
        if held:
            return True, f"{held} FETCH candidate(s)"
        reasons = [f"{beam['sp_candidates']} FETCH candidate(s) settled"] if beam.get('sp_candidates') else []
        for fold in beam['folds']:
            why = self.reason(fold, beam)
            if why is None:
                return True, f"fold P={fold['period']:.6g} s DM {fold['dm']:g}"
            reasons.append(why)
        return False, '; '.join(sorted(set(reasons))) or 'no folds'
