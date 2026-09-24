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

VERSION = 1
MIN_REPEATABILITY = 5.0
MIN_PERSISTENCE = 0.75


def normalise_rows(values):
    values = np.asarray(values, dtype=float)
    centred = values - np.median(values, axis=1, keepdims=True)
    scales = np.median(np.abs(centred), axis=1, keepdims=True) / 0.67448975
    # Avoid promoting a constant or numerically degenerate fold.
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError('Degenerate fold noise scale')
    return (values - values.mean(axis=1, keepdims=True)) / scales


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
            'sampling_bins': sampling_bins, 'reasons': reasons}


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
    for row in db.execute('''SELECT p.*, c.id, f.beams, f.saps, f.dm_min, f.dm_max
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
        signature = hashlib.sha256(json.dumps([VERSION, row['row'], str(path), stamp]).encode()).hexdigest()
        old = cached.get(row['key'])
        quality = json.loads(old['evidence']) if old and old['signature'] == signature else measure(fold, path)
        # Compute family flags afresh: evidence can change as other beams arrive.
        reasons = list(quality['reasons'])
        multibeam = ((row['beams'] or 0) >= 4 or (row['saps'] or 0) >= 2) and not (
            (row['dm_min'] or 0) >= 2 and row['dm_max'] - row['dm_min'] <= max(2, 0.1 * row['dm_max']))
        if multibeam or row['rfi_like']:
            reasons.append('Period is flagged as interference')
        candidates.append(dict(row, fold=fold, quality=quality, signature=signature,
                               strong=quality['strong'] and not multibeam and not row['rfi_like'], reasons=reasons))

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
