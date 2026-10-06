"""Folds at the catalogue ephemeris of every known pulsar near a beam: is it seen, and how strongly.

The blind periodic search reports only what crosses its threshold, so a known
pulsar it misses leaves no trace: a beam too faint for it and a search that
lost it looked the same. The redetection audit of 6 October 2026 found that
half of the LOTAAS pulsars missed near a beam centre had no peak above
threshold at all. This fold answers the question directly for every
catalogued pulsar within the catalogue cone (periodicity_folding.
catalogue_matches: 1 degree, period at the observation's epoch).

Each is folded on the periodic trials of the beam, before they are pruned:
- the period is scanned over PERIOD_RANGE around the catalogue value (Earth's
  orbit shifts the topocentric period by up to 1e-4), widened by a binary's
  largest orbital Doppler shift (binary_doppler, 2 pi a sin i / c Pb; at most
  MAX_DOPPLER): B0655+64 was seen at -3.6e-4, outside 1.5e-4, though the blind
  search found it at statistic 573 in the same beam. Steps move the last turn
  by half a bin;
- then the DM trials within DM_RANGE of the catalogue DM at the best period;
- then the period again at the best DM.
The series is first high-passed (a running mean of max(HIGHPASS_SECONDS,
HIGHPASS_TURNS turns) removed) and clipped at CLIP_SIGMA, so slow drifts and
interference spikes do not fold into a profile.

The profile's chi-square over its bins gives -log10 p for one trial; the scan's
trials are taken off (log10 p corrected). The same scan at CONTROL_FACTORS times
the period, related to it by no small harmonic, measures what this beam gives
a pulsar that is not there. 'seen' needs both: p below DETECTED_P after trials
and above every control. A pulsar too fast for MIN_BINS bins at the trial's
sampling is listed as such and not folded.
"""
import json
import math
from pathlib import Path

import numpy as np

PERIOD_RANGE = 1.5e-4
MAX_DOPPLER = 2e-3
DM_RANGE = (1.0, 0.01)
MIN_BINS = 4
MAX_BINS = 64
MAX_PERIOD_TRIALS = 4001
HIGHPASS_SECONDS = 20.0
HIGHPASS_TURNS = 10
CLIP_SIGMA = 6.0
CONTROL_FACTORS = (1.0371, 1.0713, 0.9417, 0.9613)
DETECTED_P = 1e-3
SCHEMA = 'lotaas.catalogue_fold.v1'


def log10_chi2_sf(chi2, dof):
    """log10 P(X >= chi2) for X ~ chi-square(dof), through the Wilson-Hilferty normal approximation;
    finite far into the tail, where scipy's survival function underflows to zero."""
    if chi2 <= 0:
        return 0.0
    k = float(dof)
    z = ((chi2 / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    if z < 30:
        return math.log10(max(0.5 * math.erfc(z / math.sqrt(2)), 1e-300))
    # Mills ratio: 1 - Phi(z) ~ phi(z) / z
    return (-z * z / 2 - math.log(z * math.sqrt(2 * math.pi))) / math.log(10)


def prepared(values, dt, period):
    """The series high-passed and clipped, in units of its own robust sigma."""
    values = np.asarray(values, dtype=np.float64)
    window = max(3, int(round(max(HIGHPASS_SECONDS, HIGHPASS_TURNS * period) / dt)))
    if window < len(values):
        padded = np.concatenate([np.full(window // 2, values[:window].mean()), values,
                                 np.full(window - window // 2, values[-window:].mean())])
        cumulative = np.concatenate([[0.0], np.cumsum(padded)])
        values = values - (cumulative[window:window + len(values)] - cumulative[:len(values)]) / window
    else:
        values = values - values.mean()
    sigma = 1.4826 * np.median(np.abs(values - np.median(values)))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(values)) or 1.0
    return np.clip(values / sigma, -CLIP_SIGMA, CLIP_SIGMA)


class Folder:
    """Folds of one prepared series: chi-square of the profile at a frequency."""

    def __init__(self, values, dt):
        self.values = values
        self.time = np.arange(len(values)) * dt
        self.variance = float(np.var(values)) or 1.0
        self.duration = len(values) * dt

    def chi2(self, frequency, bins):
        phase = (np.remainder(self.time * frequency, 1.0) * bins).astype(np.int64)
        counts = np.bincount(phase, minlength=bins)
        sums = np.bincount(phase, weights=self.values, minlength=bins)
        good = counts > 0
        mean = sums.sum() / counts.sum()
        return float(np.sum((sums[good] - mean * counts[good]) ** 2 / counts[good]) / self.variance)

    def profile(self, frequency, bins):
        phase = (np.remainder(self.time * frequency, 1.0) * bins).astype(np.int64)
        counts = np.bincount(phase, minlength=bins)
        sums = np.bincount(phase, weights=self.values, minlength=bins)
        return (sums - sums.sum() / counts.sum() * counts) / np.sqrt(np.maximum(counts, 1) * self.variance)

    def scan(self, centre, bins, spread=PERIOD_RANGE):
        """(best chi2, its frequency, trials) over a fractional spread around a frequency."""
        step = 1.0 / (2 * bins * self.duration)
        n = min(MAX_PERIOD_TRIALS, 2 * int(math.ceil(spread * centre / step)) + 1)
        best = (-1.0, centre)
        for frequency in np.linspace(centre * (1 - spread), centre * (1 + spread), n):
            value = self.chi2(frequency, bins)
            if value > best[0]:
                best = (value, float(frequency))
        return best[0], best[1], n


def fold_pulsar(pulsar, trials, load, dt_of):
    """The fold record of one catalogued pulsar. trials: [(dm, name)] of searchable periodic trials;
    load(name) -> valid samples of that trial; dt_of(name) -> its sampling in seconds."""
    period, dm = pulsar['period_seconds'], pulsar['dm']
    record = dict(pulsar)
    near = sorted((abs(t_dm - dm), t_dm, name) for t_dm, name in trials
                  if abs(t_dm - dm) <= max(DM_RANGE[0], DM_RANGE[1] * dm))
    if not near:
        return dict(record, status='no trial within the DM range')
    _, first_dm, first = near[0]
    dt = dt_of(first)
    bins = min(MAX_BINS, int(period / dt))
    if bins < MIN_BINS:
        return dict(record, status=f'too fast for {dt * 1000:.1f} ms samples', bins=bins)
    folders = {}

    def folder(name):
        if name not in folders:
            folders[name] = Folder(prepared(load(name), dt_of(name), period), dt_of(name))
        return folders[name]

    centre = 1.0 / period
    spread = PERIOD_RANGE + min(MAX_DOPPLER, float(pulsar.get('binary_doppler') or 0.0))
    best, frequency, period_trials = folder(first).scan(centre, bins, spread)
    best_name, best_dm = first, first_dm
    dm_trials = 0
    for _, t_dm, name in near[1:]:
        if dt_of(name) != dt:
            continue
        dm_trials += 1
        value = folder(name).chi2(frequency, bins)
        if value > best:
            best, best_name, best_dm = value, name, t_dm
    if best_name != first:
        best, frequency, more = folder(best_name).scan(centre, bins, spread)
        period_trials += more
    trials_total = period_trials * (dm_trials + 1)
    log10p = log10_chi2_sf(best, bins - 1)
    controls = []
    for factor in CONTROL_FACTORS:
        value, _, _ = folder(first).scan(centre / factor, bins, spread)
        controls.append(log10_chi2_sf(value, bins - 1))
    corrected = min(0.0, log10p + math.log10(trials_total))
    seen = corrected <= math.log10(DETECTED_P) and log10p < min(controls)
    profile = folder(best_name).profile(frequency, bins)
    return dict(record, status='folded', trial=best_name, trial_dm=best_dm, bins=bins, period_spread=spread,
                period_found=1.0 / frequency, period_offset=1.0 / (frequency * period) - 1.0,
                chi2=best, log10p=log10p, trials=trials_total, log10p_corrected=corrected,
                control_log10p=controls, seen=bool(seen), profile=[round(float(v), 3) for v in profile])


def fold_catalogue(trial_dir, output_dir, metadata, catalogue=None):
    """Fold every catalogued pulsar near the beam; writes catalogue_folds.json and returns its records."""
    from .periodicity import valid_trial
    from .periodicity_folding import catalogue_matches
    from .preprocess import searchable
    from .trials import trial_specs
    trial_dir, output_dir = Path(trial_dir), Path(output_dir)
    if catalogue is None:
        catalogue, status = catalogue_matches(metadata)
    else:
        status = 'given'
    plan = metadata.get('periodicity_dm_plan') or metadata['dedispersion_plan']
    specs = {name: spec for name, spec in trial_specs(metadata, plan).items()
             if searchable(spec['dm'], metadata.get('preprocessing')) and (trial_dir / name).is_file()}
    trials = [(spec['dm'], name) for name, spec in specs.items()]

    def load(name):
        return valid_trial(trial_dir / name, metadata, specs[name]['dm'], specs[name]['downsample'])[0]

    def dt_of(name):
        return float(metadata['tsamp']) * specs[name]['downsample']

    records = []
    for pulsar in catalogue:
        try:
            records.append(fold_pulsar(pulsar, trials, load, dt_of))
        except Exception as error:   # one pulsar's failure must not lose the others
            records.append(dict(pulsar, status=f'failed: {type(error).__name__}: {error}'))
    result = {'schema': SCHEMA, 'catalogue_status': status, 'pulsars': records}
    path = output_dir / 'catalogue_folds.json'
    partial = path.with_name(path.name + '.partial')
    partial.write_text(json.dumps(result, indent=1, allow_nan=False))
    partial.replace(path)
    return records
