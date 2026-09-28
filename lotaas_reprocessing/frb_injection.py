"""FRB-like bursts injected into a flatfielded LOTAAS beam, to measure what the pipeline finds (euroflash.injections).

A burst as it reaches these data: dispersed (it arrives at each channel's top edge K DM (f^-2 - f_top^-2) after
the band's top channel centre); smeared within each channel by the channel's own sweep and by its intrinsic
width (a boxcar of their quadrature sum); scattered with an exponential tail tau(f) = tau135 (f / 135 MHz)^-4;
shaped by a spectrum (flat, a power law, or a Gaussian band); and integrated exactly over each sample. Its
amplitude is set so that an ideal search reports the drawn S/N: integer-sample dedispersion at the native
resolution, the best boxcar up to MAX_WIDTH_SECONDS over the beam's good channels, against their own robust
scatter. That is the burst's radiometer S/N in this beam, which a fluence becomes through the beam's system
equivalent flux density; the truth records what that conversion needs.

    python -m lotaas_reprocessing.frb_injection SOURCE DEST TRUTH_JSON SEED SETTINGS_YAML|BAD_CHANNELS_JSON [N]

With the pipeline's settings, the channels the search will mask are those the GPU stage masks (bad_channels
and persistent_channels, pipeline_gpu), found before the search runs.

First measured on 28 September 2026 (benchmarks/frb-injection-2026-09-28): the search's own S/N of such bursts
was a median 0.82 of this ideal one with the 8-width baseline, 0.54 with the 64-width baseline.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

from lotaas_reprocessing import sigproc_data as sigproc

K_DM = 4148.808
MAX_WIDTH_SECONDS = 1.0
# Parameter ranges, each drawn log-uniformly: wide enough to map where the pipeline stops finding bursts.
RANGES = {'dm': (100.0, 3000.0), 'tau135': (1e-3, 3.0), 'snr': (6.0, 60.0), 'width': (5e-4, 2e-2)}
SPECTRA = ('flat', 'power_law', 'band')
EDGE_SECONDS = 30.0          # no burst this close to the start, and its whole sweep before the end
SPACING_SECONDS = 30.0       # between two bursts' arrival times


def log_uniform(rng, low, high):
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def sweep_seconds(dm, freqs):
    """The dispersion delay across the band (lowest channel centre relative to the highest)."""
    f = np.asarray(freqs, dtype=float)
    return K_DM * dm * (f.min() ** -2 - f.max() ** -2)


def draw(rng, duration, freqs, n, ranges=None, spectra=SPECTRA):
    """Parameters of n bursts that fit in `duration` seconds, arriving at least SPACING_SECONDS apart."""
    ranges = dict(RANGES, **(ranges or {}))
    f_lo, f_hi = float(np.min(freqs)), float(np.max(freqs))
    bursts, tries = [], 0
    while len(bursts) < n and tries < 1000 * n:
        tries += 1
        b = {name: log_uniform(rng, *ranges[name]) for name in ('dm', 'tau135', 'snr', 'width')}
        b['spectrum'] = spectra[int(rng.integers(len(spectra)))]
        if b['spectrum'] == 'power_law':
            b['index'] = float(rng.uniform(-5.0, 5.0))
        elif b['spectrum'] == 'band':
            b['centre'] = float(rng.uniform(f_lo - 5.0, f_hi + 5.0))
            b['fwhm'] = log_uniform(rng, 4.0, 40.0)
        tail = 12 * b['tau135'] * (f_lo / 135.0) ** -4.0
        latest = duration - EDGE_SECONDS - sweep_seconds(b['dm'], freqs) - min(tail, 60.0)
        if latest <= EDGE_SECONDS:
            continue
        t0 = float(rng.uniform(EDGE_SECONDS, latest))
        if all(abs(t0 - other['t0']) >= SPACING_SECONDS for other in bursts):
            b['t0'] = t0
            bursts.append(b)
    return sorted(bursts, key=lambda b: b['t0'])


def envelope(freqs, burst):
    """The burst's relative amplitude in each channel."""
    f = np.asarray(freqs, dtype=float)
    if burst['spectrum'] == 'power_law':
        return (f / 135.0) ** burst['index']
    if burst['spectrum'] == 'band':
        return np.exp(-0.5 * ((f - burst['centre']) / (burst['fwhm'] / 2.3548)) ** 2)
    return np.ones_like(f)


def profile(freqs, df, tsamp, burst):
    """(first sample, (samples, channels) fraction of a unit-amplitude burst in each sample and channel)."""
    f = np.asarray(freqs, dtype=float)
    top = f.max()
    lo_edge, hi_edge = f - df / 2, f + df / 2
    start = burst['t0'] + K_DM * burst['dm'] * (hi_edge ** -2 - top ** -2)
    smear = np.hypot(K_DM * burst['dm'] * (lo_edge ** -2 - hi_edge ** -2), burst['width'])
    tau = burst['tau135'] * (f / 135.0) ** -4.0
    first = int(math.floor(start.min() / tsamp)) - 1
    last = int(math.ceil((start + smear + 12 * tau).max() / tsamp)) + 2
    edges = (np.arange(first, last + 1) * tsamp)[:, None] - start[None, :]
    B, T = smear[None, :], tau[None, :]
    t = np.maximum(edges, 0.0)
    # The boxcar-exponential's CDF: rising while the boxcar lasts, then the exponential's decay.
    inside = (t - T * (1 - np.exp(-t / T))) / B
    after = 1.0 - (T / B) * (np.exp(-np.maximum(t - B, 0) / T) - np.exp(-t / T))
    cdf = np.where(t < B, inside, after)
    return first, (np.diff(cdf, axis=0) * envelope(f, burst)[None, :]).astype(np.float32)


def ideal_snr(first, frac, freqs, tsamp, dm, good, sigma_sum):
    """(S/N, boxcar samples, boxcar start) an ideal search reports for a unit-amplitude burst."""
    f = np.asarray(freqs, dtype=float)
    shifts = np.round(K_DM * dm * (f ** -2 - f.max() ** -2) / tsamp).astype(int)
    series = np.zeros(frac.shape[0])
    for c in np.flatnonzero(good):
        s = shifts[c]
        if s < frac.shape[0]:
            series[:frac.shape[0] - s] += frac[s:, c]
    cumulative = np.concatenate([[0.0], np.cumsum(series)])
    widths = np.unique(np.geomspace(1, max(1, round(MAX_WIDTH_SECONDS / tsamp)), 24).astype(int))
    best = (0.0, 1, 0)
    for w in widths:
        if w >= len(series):
            break
        sums = cumulative[w:] - cumulative[:-w]
        k = int(np.argmax(sums))
        snr = sums[k] / (sigma_sum * math.sqrt(w))
        if snr > best[0]:
            best = (float(snr), int(w), k)
    return best


def noise(data, good):
    """Each channel's centre and robust scatter, from about 20,000 samples spread over the beam."""
    sample = np.asarray(data[::max(1, data.shape[0] // 20000)], dtype=np.float32)
    centre = np.median(sample, axis=0)
    sigma = 1.4826 * np.median(np.abs(sample - centre), axis=0)
    return centre, sigma


def inject(data, header, bad, rng, n=10, ranges=None):
    """Add n bursts to data ((time, channel) float32 in file order, in place); returns their truth."""
    nt, nf = data.shape
    tsamp = float(header['tsamp'])
    freqs = sigproc.channel_frequencies(header)
    df = abs(float(header['foff']))
    good = np.array([c not in set(bad) for c in range(nf)])
    centre, sigma = noise(data, good)
    good &= sigma > 0
    # The flatfielded data are in units of the system's own power: a burst's fluence in these units, divided
    # by this level, times the beam's system-equivalent flux density, is its fluence in Jy s.
    level = float(np.median(centre[good]))
    sigma_sum = math.sqrt(float((sigma[good] ** 2).sum()))
    truth = []
    for i, burst in enumerate(draw(rng, nt * tsamp, freqs, n, ranges)):
        first, frac = profile(freqs, df, tsamp, burst)
        unit, w, k = ideal_snr(first, frac, freqs, tsamp, burst['dm'], good, sigma_sum)
        if unit <= 0:
            continue
        amplitude = burst['snr'] / unit
        lo, hi = max(0, first), min(nt, first + frac.shape[0])
        data[lo:hi] += amplitude * frac[lo - first:hi - first]
        # Per good channel, the burst's mean fluence in data units x s, and the channels' mean
        # scatter per sample: with a system-equivalent flux density these give its fluence in Jy s.
        fluence = float(amplitude * frac[:, good].sum() * tsamp / good.sum())
        truth.append(dict(burst, index=i, snr_ideal=burst['snr'], amplitude=float(amplitude),
                          peak_time=(first + k + (w - 1) / 2) * tsamp, ideal_width_s=w * tsamp,
                          fluence_units=fluence, level=level, sigma_mean=float(sigma[good].mean()),
                          good_channels=int(good.sum()),
                          tsamp=tsamp, channel_mhz=df))
    return truth


def search_mask(data, settings):
    """The channels (file order) the GPU stage masks in every sample: configured, and persistently deviant."""
    from lotaas_reprocessing.single_pulse_quality import persistent_channels
    nchan = data.shape[1]
    reversed_order = data[:, ::-1].T                 # the GPU stage's (channel, time), lowest frequency first
    normalised = reversed_order / np.median(reversed_order) - 1
    automatic = persistent_channels(normalised, settings.get('persistent_channel_threshold', 4.0))
    return sorted({int(c) for c in settings.get('bad_channels') or []} | {nchan - 1 - int(c) for c in automatic})


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    source, dest, truth_path, seed = Path(argv[0]), Path(argv[1]), Path(argv[2]), int(argv[3])
    n = int(argv[5]) if len(argv) > 5 else 10
    header, mapped = sigproc.open_data(source)
    data = np.array(mapped, dtype=np.float32)
    bad = []
    if len(argv) > 4 and argv[4].endswith(('.yaml', '.yml')):
        import yaml
        bad = search_mask(data, yaml.safe_load(Path(argv[4]).read_text()))
    elif len(argv) > 4:
        bad = json.loads(Path(argv[4]).read_text())
    truth = inject(data, header, bad, np.random.default_rng(seed), n)
    partial = dest.with_name(dest.name + '.partial')
    sigproc.write(partial, header, data)
    truth_path.write_text(json.dumps({'source': str(source), 'twin': dest.name, 'seed': seed, 'bad_channels': bad,
                                      'bursts': truth}, indent=1))
    partial.replace(dest)
    print(dest.name, 'with', len(truth), 'bursts')


if __name__ == '__main__':
    main()
