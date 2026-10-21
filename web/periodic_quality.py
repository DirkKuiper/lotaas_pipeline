"""Conservative review prioritisation of retained periodic folds.

These are diagnostic heuristics, not calibrated false-alarm probabilities or
an astrophysical classifier. Failing them only defers a fold; it never removes
search products. In particular, nulling, intermittent or weak sources can be
deferred. The original period was fitted to the full data: held-out *phase
windows* below do not make this an independent detection experiment.
"""
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from web.keys import parse_item

VERSION = 3
MIN_REPEATABILITY = 5.0
MIN_PERSISTENCE = 0.75
# LOTAAS 2-bit rows: 512 samples of 7.864 ms. Their residual steps fold at whole numbers of rows: folds at
# 6.00-6.03 rows (24.15-24.27 s) in L543273, L549893 and L543465, at unrelated DMs (29 September 2026).
ROW_SECONDS = 512 * 0.00786432
ROW_TOLERANCE = 0.003
# A fold's profile beyond its fundamental (harmonics 2-4 of the folded profile), as chi-squared with 6 degrees of
# freedom: below this (the 99th percentile of noise) it holds nothing but a sinusoid.
SINUSOID_CHI2 = 16.81
ROW_HARMONIC_CHI2 = 50.0
# A family in one beam and its neighbours (the first ring of tied-array beams sits 0.24-0.25 deg out), in one
# SAP, is a source on the sky: interference reaches beams all over the observation. Such a family is not
# interference and not deferred for a low DM. The 41 s PSR J0311+1402 (DM 1.4 at 135 MHz) was found at
# statistic 85 in L606916 SAP001 beam 53 and three neighbours, and deferred for both (20 October 2026).
# Across all 95,511 families this lets 3 more folds reach the queue.
LOCAL_RADIUS_DEG = 0.35
LOCAL_BEAMS = 7
LOW_DM_REASONS = ('Best DM is close to zero', 'Search response remains strong close to DM zero')
# A fold's 32 subintegrations are 112 s each; at a period above 14 s they hold fewer than MIN_TURNS turns, one
# null or weak spell leaves one empty, and above about 100 s some phase bins stay empty ('Incomplete phase
# coverage', so the fold was never assessed). Such a fold's intervals are merged until each holds MIN_TURNS turns,
# never into fewer than MIN_ROWS, before the time-split checks. Over 20,000 deferred folds at P >= 5 s of the
# archive (benchmarks/slow-pass-pilot-2026-10-21/slow_triage.py) that let 1 more pass, made the 17,754 never
# assessed assessable (none passed), and turned away 13 of the 55 that passed, every one at 23.7-25.3 s, 49-51 s
# or 75.6 s: the 2-bit requantiser's 24.16 s blocks. J0311+1402 and the pulsars' subharmonics passed as before.
MIN_TURNS = 8
MIN_ROWS = 8
# Off since 21 October 2026 ~11:40: in the live index the merged intervals let through dozens of high-DM families
# (DM 500-1000, 46-100 s, five or more beams at one DM: the 'blob' class), which the 32 intervals had deferred.
# The 20,000-fold sample had held one of them.
MERGE_INTERVALS = False
# Folds measured again per index pass after the checks change (their signature); the others keep their verdict
# until their turn, so a change does not stall one pass on every fold of the archive.
REMEASURE_PER_PASS = 50000
# A slow pulsar that nulls or flickers fails the time splits however bright its pulses are, and one longer than
# its sweep reads as low DM; its single pulses are independent evidence. A fold whose only objections are those
# is shown when the beam's clusters at its DM (within max(PULSE_DM_TOLERANCE)) arrive on its pulse phase (within
# max(PULSE_WINDOW_TURNS, the profile's half width)) in PULSE_TURNS or more turns, at binomial log10 p <=
# PULSE_RESCUE_LOG10P against the clusters at that DM; a low DM only when those pulses are dispersed (median DM
# >= PULSE_DISPERSED_DM). In the slow-pass pilot (21 October 2026) that showed J0311+1402 (18 of its 19 pulses on
# phase, log10 p -23.9) and two injected pulsars the splits deferred (-50, -135); of 836 archive folds deferred
# by those objections only, none had it (benchmarks/slow-pass-pilot-2026-10-21/fold_pulses.py).
SPLIT_REASONS = ('Pulse window does not repeat strongly in every time split',
                 'Pulse phase is not supported in enough time intervals')
PULSE_RESCUE_LOG10P = -4.0
PULSE_TURNS = 3
PULSE_WINDOW_TURNS = 0.02
PULSE_DM_TOLERANCE = (1.0, 0.03)
PULSE_MIN_SNR = 6.0
PULSE_DISPERSED_DM = 1.0
PULSE_MERGE_SECONDS = 1.0
PULSES_PER_PASS = 20000
# Measured on folds at P >= 5 s only (the archive's 20,000 sampled); faster folds keep the time splits.
PULSE_MIN_PERIOD = 5.0


def log10_binomial_tail(k, n, p):
    """log10 P(X >= k), X ~ Binomial(n, p)."""
    import math
    if k <= 0:
        return 0.0
    if k > n or p <= 0:
        return -math.inf
    if p >= 1:
        return 0.0
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p)
             + (n - i) * math.log1p(-p) for i in range(k, n + 1)]
    top = max(terms)
    return (top + math.log(sum(math.exp(t - top) for t in terms))) / math.log(10)


def pulse_phase(sums, counts):
    """(phase of the profile's peak, its half width at half maximum) in turns, from a fold's whole hour."""
    with np.errstate(invalid='ignore', divide='ignore'):
        profile = np.nansum(sums, axis=0) / np.nansum(counts, axis=0)
    profile = np.nan_to_num(profile - np.nanmedian(profile))
    nbins = len(profile)
    peak = int(np.argmax(profile))
    width = 1
    while (width < nbins // 2 and profile[(peak + width) % nbins] > profile[peak] / 2
           and profile[(peak - width) % nbins] > profile[peak] / 2):
        width += 1
    return (peak + 0.5) / nbins, width / nbins


def pulse_evidence(fold, quality, events):
    """The beam's single pulses on the fold's pulse phase: {at_dm, on (turns), dm_median, log10p}.

    events: the beam's clusters (dm, snr, time, width); times and fold phases both count from the start of the
    data at the top of the band, as the search dedisperses both."""
    period, dm = float(fold.get('refined_period_seconds') or fold['period_seconds']), float(fold['dm'])
    ev = np.asarray(events, dtype=float).reshape(-1, 4)
    ev = ev[(np.abs(ev[:, 0] - dm) <= max(PULSE_DM_TOLERANCE[0], PULSE_DM_TOLERANCE[1] * dm)) & (ev[:, 1] >= PULSE_MIN_SNR)]
    if not len(ev) or quality.get('peak_phase') is None:
        return {'at_dm': int(len(ev)), 'on': 0, 'dm_median': None, 'log10p': 0.0}
    ev = ev[np.argsort(ev[:, 2])]
    ev = ev[np.r_[True, np.diff(ev[:, 2]) > PULSE_MERGE_SECONDS]]        # one cluster per moment
    phi = float(quality['peak_phase'])
    window = max(PULSE_WINDOW_TURNS, float(quality.get('pulse_half_width') or 0))
    distance = np.abs(((ev[:, 2] / period) % 1.0 - phi + 0.5) % 1.0 - 0.5)
    on = ev[distance <= window]
    turns = len(np.unique(np.floor(on[:, 2] / period + 0.5 - phi)))  # one per rotation
    log10p = log10_binomial_tail(turns, len(ev), min(1.0, 2 * window)) if turns >= PULSE_TURNS else 0.0
    return {'at_dm': int(len(ev)), 'on': int(turns), 'dm_median': float(np.median(on[:, 0])) if len(on) else None,
            'log10p': round(float(log10p), 2)}


def localised(beams, saps, radius):
    """Whether a periodic family lies in one beam and its neighbours, in more than one beam."""
    return saps == 1 and 2 <= (beams or 0) <= LOCAL_BEAMS and radius is not None and radius <= LOCAL_RADIUS_DEG


def normalise_rows(values):
    values = np.asarray(values, dtype=float)
    centred = values - np.median(values, axis=1, keepdims=True)
    scales = np.median(np.abs(centred), axis=1, keepdims=True) / 0.67448975
    # Avoid promoting a constant or numerically degenerate fold.
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError('Degenerate fold noise scale')
    return (values - values.mean(axis=1, keepdims=True)) / scales


def merges(fold, rows=32):
    """Whether a fold of this many subintegrations has its intervals merged (merged_intervals)."""
    period = float(fold.get('refined_period_seconds') or fold.get('period_seconds') or 0)
    seconds = float(fold.get('observation_seconds') or 3600.0)
    return (MERGE_INTERVALS and period > 0 and seconds / rows / period < MIN_TURNS
            and max(MIN_ROWS, int(seconds / (MIN_TURNS * period))) < rows)


def merged_intervals(fold, sums, counts):
    """(sums, counts) with the subintegrations merged so each holds MIN_TURNS turns of the fold's period."""
    period = float(fold.get('refined_period_seconds') or fold.get('period_seconds') or 0)
    seconds = float(fold.get('observation_seconds') or 3600.0)
    rows = sums.shape[0]
    if not MERGE_INTERVALS or period <= 0 or seconds / rows / period >= MIN_TURNS:
        return sums, counts
    groups = max(MIN_ROWS, int(seconds / (MIN_TURNS * period)))
    if groups >= rows:
        return sums, counts
    starts = np.unique(np.linspace(0, rows, groups + 1)[:-1].round().astype(int))
    return np.add.reduceat(sums, starts, axis=0), np.add.reduceat(counts, starts, axis=0)


def assess(fold, data):
    """Cross-check a pulse window chosen in different time subsets.

    Window phase and width are chosen on training rows, then evaluated without
    adjustment on the other rows, in both directions. Do this for alternating
    subintegrations and for first/second halves. Row normalisation stops a few
    noisy intervals dominating the profile. Broad-band agreement, when present,
    uses the same phase window, without fitting each channel separately.
    """
    reasons = []
    sums = np.asarray(data['subintegration_sums'], dtype=float)
    counts = np.asarray(data['subintegration_counts'], dtype=float)
    if sums.shape != counts.shape or sums.ndim != 2 or sums.shape[0] < 16 or sums.shape[1] < 8:
        raise ValueError('Need at least 16 time intervals and 8 phase bins')
    peak_phase, half_width = pulse_phase(sums, counts)
    sums, counts = merged_intervals(fold, sums, counts)
    if not np.all(np.isfinite(sums)) or not np.all(np.isfinite(counts)) or np.any(counts <= 0):
        raise ValueError('Incomplete phase coverage')
    sampling_bins = float(fold['refined_period_seconds']) / float(fold['effective_sampling_seconds'])
    if not np.isfinite(sampling_bins) or sampling_bins < 8:
        reasons.append('Fewer than 8 independent samples per period')
    norm = normalise_rows(sums / counts)
    nrows, nbins = norm.shape
    responses = [(width, sum(np.roll(norm, -i, axis=1) for i in range(width)) / np.sqrt(width))
                 for width in (1, 2, 4, 8, 16) if width <= nbins // 4]
    alternating = (np.arange(0, nrows, 2), np.arange(1, nrows, 2))
    contiguous = (np.arange(nrows // 2), np.arange(nrows // 2, nrows))
    checks, supports, windows = [], [], []
    for train, test in (alternating, alternating[::-1], contiguous, contiguous[::-1]):
        options = []
        for width, response in responses:
            phase = int(np.argmax(response[train].mean(axis=0)))
            options.append((float(response[train, phase].mean()), width, phase, response))
        _, width, phase, response = max(options, key=lambda option: option[0])
        held = response[test, phase]
        deviation = float(held.std(ddof=1))
        if deviation <= 1e-10:
            raise ValueError('Degenerate time variability')
        checks.append(float(held.mean() / deviation * np.sqrt(len(held))))
        supports.append(float(np.mean(held > 0)))
        windows.append((width, phase))
    repeatability, persistence = min(checks), min(supports)
    # Slow variations fold into a sinusoid with nothing at the profile's harmonics; a pulse of a pulsar or a
    # long-period transient is narrow and has them (a duty cycle under a third keeps harmonic 2 at a quarter
    # or more of the fundamental's power). Red noise folded at 11.69 s (L543465 SAP000 B062), 2.00 s and 2.29 s
    # passed every check above with one search harmonic.
    profile = norm.mean(axis=0)
    spectrum = np.abs(np.fft.rfft(profile)) ** 2 / (nbins / 2 / nrows)
    beyond = float(spectrum[2:5].sum()) if nbins >= 10 else None
    if beyond is not None and int(fold.get('harmonic_count') or 1) <= 1 and beyond < SINUSOID_CHI2:
        reasons.append('Folded profile is a sinusoid: a slow variation, not a pulse')
    # Within the fold's own period resolution (P / T: 0.7% at 24 s in an hour), and only for a profile with
    # little beyond its fundamental, so that a narrow pulse at such a period still reaches the queue.
    period = float(fold.get('refined_period_seconds') or fold.get('period_seconds') or 0)
    rows = period / ROW_SECONDS
    tolerance = max(ROW_TOLERANCE, period / float(fold.get('observation_seconds') or 3600))
    if (rows >= 0.97 and abs(rows / round(rows) - 1) <= tolerance and beyond is not None
            and beyond < ROW_HARMONIC_CHI2):
        reasons.append(f'Period is {round(rows)} 2-bit rows of 4.03 s: instrumental')
    if repeatability < MIN_REPEATABILITY:
        reasons.append('Pulse window does not repeat strongly in every time split')
    if persistence < MIN_PERSISTENCE:
        reasons.append('Pulse phase is not supported in enough time intervals')

    curve = np.asarray(data.get('dm_curve', []), dtype=float).reshape(-1, 2)
    curve = curve[np.all(np.isfinite(curve), axis=1)]
    dm_radius = max(0.5, 2 * float(fold.get('dm_step', 0)))
    nearby = curve[np.abs(curve[:, 0] - float(fold['dm'])) <= dm_radius] if len(curve) else curve
    dm_trials = len(np.unique(nearby[:, 0])) if len(nearby) else 0
    if dm_trials < 3:
        reasons.append('Too few neighbouring DM trials support this period')
    if float(fold['dm']) < 2:
        reasons.append('Best DM is close to zero')
    low_dm = curve[curve[:, 0] < 2, 1] if len(curve) else []
    zero_ratio = float(max(low_dm) / max(curve[:, 1])) if len(low_dm) and max(curve[:, 1]) > 0 else None
    if zero_ratio is not None and zero_ratio >= 0.9:
        reasons.append('Search response remains strong close to DM zero')

    subbands = np.asarray(data.get('subbands', []), dtype=float)
    band_support = None
    if subbands.size:
        if subbands.ndim != 2 or subbands.shape[1] != nbins or subbands.shape[0] < 4:
            reasons.append('Frequency diagnostic has insufficient coverage')
        else:
            valid = np.all(np.isfinite(subbands), axis=1) & (np.std(subbands, axis=1) > 0)
            if np.count_nonzero(valid) < 4:
                reasons.append('Frequency diagnostic has insufficient coverage')
            else:
                band = normalise_rows(subbands[valid])
                width, phase = windows[0]
                on = sum(np.roll(band, -i, axis=1)[:, phase] for i in range(width)) / np.sqrt(width)
                band_support = float(np.mean(on > 0))
                if band_support < 0.6:
                    reasons.append('Pulse phase does not agree across the frequency band')
    return {'version': VERSION, 'score': max(0.0, repeatability), 'strong': not reasons,
            'repeatability': repeatability, 'time_checks': checks, 'persistence': persistence,
            'dm_trials': dm_trials, 'zero_dm_ratio': zero_ratio, 'band_support': band_support,
            'harmonic_chi2': beyond,
            'sampling_bins': sampling_bins, 'peak_phase': peak_phase, 'pulse_half_width': half_width, 'reasons': reasons}


def measure(fold, path):
    try:
        with np.load(path, allow_pickle=False) as data:
            return assess(fold, data)
    except (OSError, EOFError, BadZipFile, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        return {'version': VERSION, 'score': 0.0, 'strong': False,
                'reasons': ['Fold diagnostics unavailable or invalid'], 'error': str(error)}


def same_signal(left, right):
    """Fixed-reference grouping; never chain unrelated neighbours together."""
    a, b = left['fold'], right['fold']
    dm_tolerance = max(0.5, 2 * max(a.get('dm_step', 0), b.get('dm_step', 0)),
                       0.05 * min(a['dm'], b['dm']))
    if abs(a['dm'] - b['dm']) > dm_tolerance:
        return False
    periods = sorted((left['period'], right['period']))
    if periods[0] <= 0:
        return False
    ratio = periods[1] / periods[0]
    harmonic = round(ratio)
    return 1 <= harmonic <= 32 and abs(ratio / harmonic - 1) <= 5e-4


def sync(db):
    """Cache numerical diagnostics; regroup current candidates every index pass."""
    cached = {r['key']: dict(r) for r in db.execute('SELECT * FROM periodic_triage')}
    candidates = []
    budget = REMEASURE_PER_PASS
    pulse_budget = PULSES_PER_PASS
    clusters = {}
    for row in db.execute('''SELECT p.*, c.id, f.beams, f.saps, f.dm_min, f.dm_max, f.radius
            FROM periodic p JOIN candidates c ON c.key=p.key
            LEFT JOIN periodic_families f ON f.key=p.key'''):
        row = dict(row)
        fold = json.loads(row['row'])
        path = Path(row['dir']) / (row['fold_data'] or '')
        try:
            stat = path.stat()
            stamp = [stat.st_mtime_ns, stat.st_size]
        except OSError:
            stamp = None
        # Version 3 changed only folds whose intervals are merged: the others keep their version-2 verdicts.
        version = VERSION if merges(fold) else 2
        signature = hashlib.sha256(json.dumps([version, row['row'], str(path), stamp]).encode()).hexdigest()
        old = cached.get(row['key'])
        if old and old['signature'] == signature:
            quality = json.loads(old['evidence'])
        elif old and budget <= 0:
            quality, signature = json.loads(old['evidence']), old['signature']      # measured again in a later pass
        else:
            quality = measure(fold, path)
            budget -= bool(old)
        # Compute family flags afresh: evidence can change as other beams arrive.
        local = localised(row['beams'], row['saps'], row['radius'])
        reasons = [r for r in quality['reasons'] if not (local and r in LOW_DM_REASONS)]
        multibeam = ((row['beams'] or 0) >= 4 or (row['saps'] or 0) >= 2) and not (
            (row['dm_min'] or 0) >= 2 and row['dm_max'] - row['dm_min'] <= max(2, 0.1 * row['dm_max'])) and not local
        if (reasons and set(reasons) <= set(SPLIT_REASONS) | set(LOW_DM_REASONS) and not (multibeam or row['rfi_like'])
                and (row['period'] or 0) >= PULSE_MIN_PERIOD):
            pulses = quality.get('pulses')
            if pulses is None and pulse_budget > 0:
                if quality.get('peak_phase') is None:          # measured before the phase was kept
                    quality = measure(fold, path)
                if row['dir'] not in clusters:
                    from web.indexer import read_clusters
                    events = []
                    read_clusters(row['dir'], events)
                    clusters[row['dir']] = events
                pulses = quality['pulses'] = pulse_evidence(fold, quality, clusters[row['dir']])
                pulse_budget -= 1
            if pulses and pulses['log10p'] <= PULSE_RESCUE_LOG10P:
                dispersed = (pulses['dm_median'] or 0) >= PULSE_DISPERSED_DM
                reasons = [r for r in reasons if r not in SPLIT_REASONS and not (dispersed and r in LOW_DM_REASONS)]
        if multibeam or row['rfi_like']:
            reasons.append('Period is flagged as interference')
        candidates.append(dict(row, fold=fold, quality=quality, signature=signature,
                               strong=not reasons, reasons=reasons))

    groups = {}
    for candidate in sorted(candidates, key=lambda c: (-int(c['strong']), -int(bool(c['catalogue'])),
                                                       -c['quality']['score'], c['key'])):
        parsed = parse_item(candidate['item'])
        # Keep production and pilot queues separate. One group per observation/SAP.
        scope = (*parsed[:2], candidate['pilot'], parsed[2] == 12) if parsed else (candidate['item'], candidate['pilot'])
        family = next((g for g in groups.setdefault(scope, []) if same_signal(candidate, g[0])), None)
        if family is None:
            groups[scope].append([candidate])
        else:
            family.append(candidate)
    result = []
    for families in groups.values():
        for group in families:
            representative = group[0]
            for candidate in group:
                status = 'related' if candidate is not representative else 'strong' if candidate['strong'] else 'deferred'
                result.append((candidate['key'], candidate['signature'], candidate['quality']['score'],
                               status, representative['key'], len(group), json.dumps(candidate['quality']),
                               json.dumps(candidate['reasons'])))
    db.execute('DELETE FROM periodic_triage')
    db.executemany('INSERT INTO periodic_triage VALUES (?,?,?,?,?,?,?,?)', result)
