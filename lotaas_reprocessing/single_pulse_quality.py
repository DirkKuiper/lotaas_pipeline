"""Conservative channel masking and a local check of clustered pulse S/N.

These are noise diagnostics, not probabilities of astrophysical origin. Keep
the evidence and low-significance events so thresholds can be audited later.
"""
import numpy as np


def persistent_channels(data, threshold=4.0, sample_size=8192):
    """Indices of persistently noisy channels; data is (channel, time).

    Compare temporal MAD with the median of 33 neighbouring channels. The
    median over time protects short, even bright or band-limited, bursts.
    Random sampling avoids locking to periodic interference and bounds memory.
    """
    if not np.isfinite(threshold) or threshold <= 1:
        raise ValueError('Persistent-channel threshold must be finite and greater than one')
    values = np.asarray(data)
    if values.shape[1] > sample_size:
        idx = np.random.default_rng(0).choice(values.shape[1], sample_size, replace=False)
        values = values[:, idx]
    centre = np.nanmedian(values, axis=1)
    noise = 1.4826 * np.nanmedian(np.abs(values - centre[:, None]), axis=1)
    radius = min(16, len(noise) // 2)
    typical = np.nanmedian(np.lib.stride_tricks.sliding_window_view(
        np.pad(noise, radius, mode='symmetric'), 2 * radius + 1), axis=1)
    return np.flatnonzero((~np.isfinite(noise)) | (noise <= 0) |
                         ((typical > 0) & (noise > threshold * typical))).tolist()


def candidate_key(dm, time, width):
    return f'{float(dm):.3f}|{float(time):.6f}|{int(width)}'


def local_boxcar_snr(series, centre, width, radius=None):
    """At-event boxcar against non-overlapping, same-width nearby windows.

    Indices are in trial samples. Exclude two widths either side of the
    event, and require 32 non-overlapping reference windows. Missing reference
    data means unknown, never a rejection. No peak-hunting away from the event.
    """
    width = max(1, int(width))
    radius = max(64 * width, int(radius or 0))
    start = int(round(centre - (width - 1) / 2))
    lo, hi = max(0, start - radius), min(len(series), start + width + radius)
    values = np.asarray(series[lo:hi], dtype=float)
    event = start - lo
    result = {'local_snr': None, 'reference_windows': 0}
    if event < 0 or event + width > len(values):
        return result
    references = np.arange(0, max(0, len(values) - width + 1), width)
    references = references[(references + width <= event - 2 * width) |
                            (references >= event + 3 * width)]
    sums = np.array([values[i:i + width].sum() for i in references])
    sums = sums[np.isfinite(sums)]
    result['reference_windows'] = int(len(sums))
    pulse = values[event:event + width].sum()
    if len(sums) < 32 or not np.isfinite(pulse):
        return result
    baseline = np.median(sums)
    scale = 1.4826 * np.median(np.abs(sums - baseline))
    if scale > 0:
        result['local_snr'] = float((pulse - baseline) / scale)
    return result


def measure_clusters(output, metadata):
    """Run before the DM trials leave the GPU node; return portable evidence."""
    from pathlib import Path
    from .dm_plan import dm_label
    from .matched_filter import baseline_window, running_baseline, wrap_contaminated_samples
    sp = metadata.get('single_pulse') or {}
    baseline_seconds = sp.get('baseline_seconds')
    output = Path(output)
    rows = np.loadtxt(output / 'clustered_candidates.txt', skiprows=1, ndmin=2, usecols=range(5))
    evidence = {}
    for dm, snr, time, sample, width in rows:
        entry = next(p for p in metadata['dedispersion_plan'] if p['low_dm'] <= dm < p['high_dm'])
        ds = int(entry['downsample'])
        dt = metadata['tsamp'] * ds
        path = output / 'DM_trials' / f"{Path(metadata['filename']).stem}_DM{dm_label(dm)}.dat"
        series = np.memmap(path, mode='r', dtype='float32')
        trim = wrap_contaminated_samples(dm, metadata.get('nu_min'), metadata.get('nu_max'),
                                         metadata['tsamp'], ds)
        valid = series[:len(series) - trim] if trim else series
        w = max(1, round(width / ds))
        radius = round(10 / dt)
        record = local_boxcar_snr(valid, time / dt, w, radius=radius)
        if baseline_seconds:
            # A whitened search is judged on the series it thresholded, and
            # keeps the unwhitened statistic beside it: broadband RFI episodes
            # look pulse-like once a 2 s baseline is gone, but their +-10 s raw
            # neighbourhood is noisy (search audit, 23 September 2026).
            raw = record['local_snr']
            window = baseline_window(w, metadata['tsamp'], ds, baseline_seconds, sp.get('baseline_widths', 64))
            if window < len(valid) // 2:
                lo = max(0, int(time / dt) - max(64 * w, radius) - 2 * w - 2)
                hi = min(len(valid), int(time / dt) + max(64 * w, radius) + 3 * w + 2)
                local = np.asarray(valid[lo:hi], dtype=np.float32) - running_baseline(valid, window, lo, hi)
                record = local_boxcar_snr(local, time / dt - lo, w, radius=radius)
            record['raw_local_snr'] = raw
        evidence[candidate_key(dm, time, width)] = dict(record, dm=float(dm), time=float(time),
            width_seconds=float(width * metadata['tsamp']), search_snr=float(snr))
    return evidence


def review_route(width_seconds, evidence, limits, fetch_count):
    """Only measured low local significance demotes an event; missing noise does not."""
    minimum = limits.get('min_local_snr')
    local = evidence.get('local_snr')
    if minimum is not None and local is not None and local < minimum:
        return 'unconfirmed'
    # Only a whitened search records the unwhitened statistic beside it.
    raw_minimum = limits.get('min_raw_local_snr')
    raw = evidence.get('raw_local_snr')
    if raw_minimum is not None and raw is not None and raw < raw_minimum:
        return 'unconfirmed'
    maximum = limits.get('max_width_seconds')
    if maximum is not None and width_seconds is not None and width_seconds > maximum:
        return 'unclassified'
    budget = limits.get('max_fetch_candidates')
    return 'unclassified' if budget is not None and fetch_count >= budget else None
