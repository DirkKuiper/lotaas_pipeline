"""Fast folding algorithm (FFA) search of a time series for long periods.

The FFT search sums at most 32 harmonics of a period, whitens against a running median of the spectrum and
loses a slow, narrow pulse to the red noise at its few lowest harmonics: of 15 slow pulsars injected at 60-200 s
(benchmarks/slow-pass-pilot-2026-10-21) the slow pass's FFT found 4. The FFA (Staelin 1969; Morello et al. 2020,
riptide) folds the series at every period of a range at once, so the matched filter is the pulse itself.

For a series of m turns of p samples (m rows), the FFA gives the m profiles folded at periods p + s/(m - 1)
samples, s = 0 .. m - 1, in m p log2 m additions: the rows split into a head and a tail, each folded alone, and
the profile at drift s adds the head's at its drift and the tail's at its drift, rolled so the tail ends at s.
Drifts are rounded at each level, so a row may sit a bin from where an exact fold puts it. Each profile is then
searched with boxcars of WIDTHS bins at every phase.

Periods from period_min to period_max are covered an octave at a time: at downsampling factor f the base periods
are bins_min .. 2 bins_min - 1 samples of f dt. The series is first freed of what is slower than a few turns
(a running median of DETREND_TURNS x the octave's longest period) and normalised to unit scatter.
"""
import math

import numpy as np

try:
    from numba import njit
except ImportError:                  # the host's python has no numba; the search runs in the container
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

WIDTHS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
DETREND_TURNS = 4.0


@njit(cache=True)
def _transform(x):
    """FFA of x (m rows of p samples): row s is the profile at a drift of s samples over the m rows."""
    m, p = x.shape
    if m == 1:
        return x.copy()
    h = m // 2
    head = _transform(x[:h])
    tail = _transform(x[h:])
    out = np.empty_like(x)
    for s in range(m):
        sh = int(round(s * (h - 1) / (m - 1))) if h > 1 else 0
        st = int(round(s * (m - h - 1) / (m - 1))) if m - h > 1 else 0
        d = s - st                   # the tail ends exactly at the drift: its rows never run past it
        for b in range(p):
            out[s, b] = head[sh, b] + tail[st, (b + d) % p]
    return out


@njit(cache=True)
def _boxcar_snr(profiles, rows, widths):
    """Best boxcar S/N of each profile and its width, for profiles summed over `rows` unit-scatter rows."""
    m, p = profiles.shape
    best = np.zeros(m)
    width = np.zeros(m, dtype=np.int64)
    cum = np.empty(2 * p + 1)
    for s in range(m):
        mean = 0.0
        for b in range(p):
            mean += profiles[s, b]
        mean /= p
        cum[0] = 0.0
        for b in range(2 * p):
            cum[b + 1] = cum[b] + profiles[s, b % p] - mean
        for w in widths:
            if w >= p:
                continue
            norm = math.sqrt(w * rows * (1.0 - w / p))
            for b in range(p):
                v = (cum[b + w] - cum[b]) / norm
                if v > best[s]:
                    best[s] = v
                    width[s] = w
    return best, width


def running_median(x, window):
    """A running median over `window` samples (at least 3), evaluated on blocks of a tenth of it, interpolated."""
    window = max(3, int(window))
    step = max(1, window // 10)
    centres = np.arange(0, len(x), step)
    half = window // 2
    medians = np.array([np.median(x[max(0, c - half):c + half + 1]) for c in centres])
    return np.interp(np.arange(len(x)), centres, medians)


def downsample(x, factor):
    n = len(x) // factor
    return x[:n * factor].reshape(n, factor).sum(axis=1)


def search(series, dt, period_min=4.0, period_max=300.0, bins_min=128, threshold=7.0, widths=WIDTHS):
    """Peaks of S/N >= threshold over trial periods: [{period_seconds, snr, width_bins, bins, turns, downsample}].

    Per octave, a peak is the best trial in each run of trials above threshold.
    """
    x = np.asarray(series, dtype=np.float64)
    peaks = []
    factor = 1
    while bins_min * factor * dt < period_max:
        lo, hi = bins_min * factor * dt, 2 * bins_min * factor * dt
        if hi > period_min:
            y = downsample(x, factor)
            y = y - running_median(y, DETREND_TURNS * hi / (factor * dt))
            scale = 1.4826 * np.median(np.abs(y - np.median(y)))
            if scale > 0:
                y = y / scale
                peaks += _octave(y, dt * factor, factor, bins_min, max(lo, period_min), min(hi, period_max),
                                 threshold, np.asarray(widths, dtype=np.int64))
        factor *= 2
    return peaks


def _octave(y, dt, factor, bins_min, period_lo, period_hi, threshold, widths):
    out = []
    for p in range(bins_min, 2 * bins_min):
        m = len(y) // p
        if m < 2 or (p + 1) * dt <= period_lo or p * dt >= period_hi:
            continue
        profiles = _transform(y[:m * p].reshape(m, p).copy())
        snr, width = _boxcar_snr(profiles, m, widths)
        periods = (p + np.arange(m) / (m - 1)) * dt
        above = np.flatnonzero((snr >= threshold) & (periods >= period_lo) & (periods < period_hi))
        if not len(above):
            continue
        runs = np.split(above, np.flatnonzero(np.diff(above) > 1) + 1)
        for run in runs:
            k = run[np.argmax(snr[run])]
            out.append({'period_seconds': float(periods[k]), 'snr': float(snr[k]), 'width_bins': int(width[k]),
                        'bins': int(p), 'turns': int(m), 'downsample': int(factor)})
    return out
