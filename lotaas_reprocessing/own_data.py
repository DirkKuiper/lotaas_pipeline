"""A single-pulse candidate's S/N on its own data, measured as the search measures it.

The review page shows it (web.dynspec, whose Snippet is an OwnData) and the
classifier asks it before FETCH (classify, limits min_own_snr and
min_own_fraction). The stretch of the flatfielded beam around the candidate
(stretch, as the page's snippet is cut) is scaled channel by channel on its
noise off the pulse; the search's RFI block mask and a limit no pulse reaches
in one cell drop interference the search never saw; then come a zero-DM
filter, dedispersion, the search's running baseline, and the boxcar at the
candidate's own time and width against same-width windows within +-10 s.

Measured on 27-28 September 2026: pulses injected into pilot beams read 0.81
to 1.41 times the pipeline's own local S/N this way. Of 444 pulses of B0301+19
and B0329+54 at S/N >= 8, none read below both 4 and half their search S/N;
of the candidates FETCH rejected, 78% did.
"""
import math
import warnings

import numpy as np

from lotaas_reprocessing import sigproc_data as sigproc
from lotaas_reprocessing.baseline import baseline_window, running_baseline
from lotaas_reprocessing.single_pulse_quality import local_boxcar_snr, persistent_channels

K_DM = 4148.808
# The single-pulse search since 27 September 2026 (pipeline settings preprocessing.zero_dm and
# single_pulse.baseline_seconds) subtracts the mean over channels at each sample and a 2 s running
# baseline; the local S/N is measured the same way, so the page and the search agree on a pulse.
SEARCH_BASELINE_SECONDS = 2.0
# ... doubled until it spans this many boxcar widths (single_pulse.baseline_widths).
SEARCH_BASELINE_WIDTHS = 64
# The search's RFI mask judges blocks of this many native samples (settings rfi_block_size).
RFI_BLOCK_SAMPLES = 1000
# No pulse reaches this in one channel and sample (S/N 1000 one sample wide: 40).
MAX_CELL_SIGMA = 50.0


def sweep_seconds(dm, freqs):
    """Arrival delay of each frequency relative to the highest one."""
    f = np.asarray(freqs, dtype=float)
    return K_DM * dm * (1 / f ** 2 - 1 / f.max() ** 2)


def delays(dm, freqs, tsamp):
    return np.round(sweep_seconds(dm, freqs) / tsamp).astype(np.int64)


def dedisperse(data, freqs, tsamp, dm):
    """Each channel shifted back by its delay; samples from beyond the data are NaN."""
    nt, nf = data.shape
    out = np.full((nt, nf), np.nan, dtype=np.float32)
    for channel, shift in enumerate(delays(max(dm, 0.0), freqs, tsamp)):
        if shift < nt:
            out[:nt - shift, channel] = data[shift:, channel]
    return out


def channel_scale(data):
    """Robust per-channel centre and noise; flat or empty channels get a NaN scale."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        centre = np.nanmedian(data, axis=0)
        scale = 1.4826 * np.nanmedian(np.abs(data - centre), axis=0)
    scale = np.where(scale > 0, scale, np.nan)
    return centre, scale


def _clipped(z, sigma=3.0):
    """Cells of z (channel, block) within sigma of their block's mean over channels, clipped
    until nothing changes (lotaas_reprocessing.numpy_utils.sigmaclip_2d)."""
    keep = np.ones_like(z, dtype=bool)
    while True:
        values = np.where(keep, z, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mean, std = np.nanmean(values, axis=0), np.nanstd(values, axis=0)
            now = (values >= mean - sigma * std) & (values <= mean + sigma * std)
        if now.sum() == keep.sum():
            return now
        keep = now


def rfi_mask(data, block, phase=0):
    """The search's RFI mask of (time, channel) data: channel blocks whose spread, skewness or
    kurtosis stand out among the channels of their block (numpy_utils.compute_rfi_mask, without
    scipy). A channel quiet but for strong interference has a small robust scale, so its bursts
    reach thousands of sigma once normalised; the search never sees them, and the local S/N
    measured as the search measures must not either. `phase`: samples of the observation's
    block before the first one here, so blocks fall where the search's did. The search had data
    where a snippet's edge blocks run past it; they are padded with each channel's median, not
    its mean, which a burst drags until the padding itself looks like interference."""
    x = np.asarray(data, dtype=np.float64).T
    nchan, nsamp = x.shape
    lead = int(phase) % block
    blocks = -(-(lead + nsamp) // block)
    if lead or blocks * block > lead + nsamp:
        # np.pad's 'median', with each channel's median taken once rather than once a side.
        padded = np.empty((nchan, blocks * block))
        padded[:] = np.median(x, axis=1, keepdims=True)
        padded[:, lead:lead + nsamp] = x
        x = padded
    x = x.reshape(nchan, blocks, block)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        x = x / x.mean(axis=2, keepdims=True)
        d = x - x.mean(axis=2, keepdims=True)
        d2 = d * d          # products, not powers: numpy's pow is several times slower
        m2, m3, m4 = d2.mean(axis=2), (d2 * d).mean(axis=2), (d2 * d2).mean(axis=2)
        stats = (np.sqrt(m2), m3 / m2 ** 1.5, m4 / m2 ** 2 - 3.0)
    good = _clipped(stats[0]) & _clipped(stats[1]) & _clipped(stats[2])
    return np.repeat(~good, block, axis=1)[:, lead:lead + nsamp].T


def downsample_for(dm, plan):
    """The time resolution the search used at this DM, from its dedispersion plan."""
    for step in plan or []:
        if step['low_dm'] <= dm < step['high_dm']:
            return int(step['downsample'])
    return int(plan[-1]['downsample']) if plan else 1


def stretch(source, dm, tcand, width_samples, plan):
    """The samples around a candidate that the review page cuts and the classifier measures.

    The dispersion sweep either side as `your` reads it plus some off-pulse margin, in
    blocks of k native samples on the search's own grid for this DM, so a decimated
    sample means what it meant to the search. Returns (header, block, start, k, delay,
    margin, search_downsample): block is (time, channel) float32 in file order and
    start its first native sample.
    """
    header, data = sigproc.open_data(source)
    freqs = sigproc.channel_frequencies(header)
    tsamp = float(header['tsamp'])
    dm = float(dm)
    tcand = float(tcand)
    width = max(1, int(width_samples))
    search_downsample = downsample_for(dm, plan)
    # Retain at least eight samples across broad events, aligned to the search grid.
    k = search_downsample * max(1, width // (8 * search_downsample))
    delay = K_DM * dm * (1 / freqs.min() ** 2 - 1 / freqs.max() ** 2)
    # Off-pulse room on both sides, and data for trial DMs above the candidate's.
    margin = max(10.0, 64 * width * tsamp)
    start = int(math.floor((tcand - delay - margin) / (tsamp * k))) * k
    stop = int(math.ceil((tcand + delay + margin) / (tsamp * k))) * k
    total = data.shape[0]
    # Do not manufacture constant off-pulse noise beyond the observation.
    start, stop = max(start, 0), min(stop, total // k * k)
    if stop <= start:
        raise ValueError(f'{source} holds no samples near t={tcand:.3f} s')
    block = np.empty(((stop - start) // k, data.shape[1]), dtype=np.float32)
    batch = max(1, 262144 // (k * data.shape[1]))
    for out_start in range(0, len(block), batch):
        out_stop = min(len(block), out_start + batch)
        raw = np.asarray(data[start + out_start * k:start + out_stop * k])
        block[out_start:out_stop] = raw.reshape(-1, k, data.shape[1]).mean(axis=1)
    return header, block, start, k, delay, margin, search_downsample


class OwnData:
    """A candidate's stretch, scaled and masked for measuring as the search measures.

    data is (time, channel) in file order at tsamp; t0 the time of its first sample relative
    to the candidate (arrival at the highest frequency); width the candidate's boxcar in these
    samples; first its first sample's index on the observation's grid at this resolution
    (k native samples each); bad the search's channel mask in file order.

    Noise is always measured over one fixed stretch around the candidate, the analysis
    window, whatever is displayed: 'guard < |t| <= analysis', where the guard keeps the
    pulse and its immediate surroundings out.
    """

    def __init__(self, data, freqs, tsamp, t0, dm, width, first=0, bad=(), k=1):
        self.data = np.asarray(data, dtype=np.float32)
        self.freqs = np.asarray(freqs, dtype=float)
        self.tsamp, self.t0, self.dm = float(tsamp), float(t0), float(dm)
        self.width = max(1, int(width))
        self.first = int(first)
        self.guard = max(2 * self.width * self.tsamp, 3 * self.tsamp)
        self.analysis = max(64 * self.width * self.tsamp, 10.0)
        # Measure each channel on the same local off-pulse interval after accounting for
        # the candidate's dispersion sweep, not on its entire (potentially 2000-second) stretch.
        aligned = dedisperse(self.data, self.freqs, self.tsamp, self.dm)
        off = self.off_pulse(self.times)
        self.centre, self.scale = channel_scale(aligned[off])
        known = [c for c in bad if 0 <= c < self.data.shape[1]]
        self.flat = np.isnan(self.scale)
        self.known_bad = sorted(set(known) | set(np.flatnonzero(self.flat).tolist()))
        self.automatic_bad = persistent_channels(self.data.T)
        block = max(8, round(RFI_BLOCK_SAMPLES / int(k)))
        self.rfi_mask = rfi_mask(self.data, block, self.first % block)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.normalised = (self.data - self.centre) / self.scale

    @property
    def times(self):
        return self.t0 + np.arange(self.data.shape[0]) * self.tsamp

    def off_pulse(self, times, width_seconds=0.0):
        guard = max(self.guard, 2 * width_seconds)
        return (np.abs(times) > guard + width_seconds / 2) & (np.abs(times) <= self.analysis)

    def masked(self, extra=(), auto_mask=True):
        data = self.normalised.copy()
        data[:, sorted(set(self.known_bad) | set(extra) | (set(self.automatic_bad) if auto_mask else set()))] = np.nan
        return data

    def search_series(self, dm, extra=(), auto_mask=True, baseline_seconds=SEARCH_BASELINE_SECONDS,
                      baseline_widths=SEARCH_BASELINE_WIDTHS):
        """The band series as the search measures it: RFI-masked cells, and cells no pulse could
        reach, at their channel's level; zero-DM filtered, dedispersed, running baseline removed.

        A pulse of S/N 1000 one sample wide is 40 sigma in each cell. Beyond that it is
        interference the search's mask removes at full resolution: a broadband burst scaled
        by a quiet channel's noise reached 1,300 sigma, and its remainder after the zero-DM
        filter halved an injected pulse's S/N beside it (27 September 2026)."""
        data = self.masked(extra, auto_mask)
        data[(self.rfi_mask | (np.abs(data) > MAX_CELL_SIGMA)) & np.isfinite(data)] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            data = data - np.nanmean(data, axis=1, keepdims=True)
            aligned = dedisperse(data, self.freqs, self.tsamp, dm)
            usable = max(1, int(np.isfinite(data).any(axis=0).sum()))
            series = np.where(np.isfinite(aligned).sum(axis=1) >= usable, np.nanmean(aligned, axis=1), np.nan)
        finite = np.isfinite(series)
        window = baseline_window(self.width, self.tsamp, 1, baseline_seconds, baseline_widths)
        if finite.sum() > 2 * window:
            filled = np.where(finite, series, np.nanmedian(series))
            # Baseline blocks on the observation's grid, as the search's over the whole trial.
            lead = self.first % max(1, window // 2)
            padded = np.concatenate([np.full(lead, np.nanmedian(series), dtype=np.float32), filled])
            series = np.where(finite, filled - running_baseline(padded, window)[lead:], np.nan)
        return series

    def local_snr(self, dm=None, extra=(), auto_mask=True, baseline_seconds=SEARCH_BASELINE_SECONDS,
                  baseline_widths=SEARCH_BASELINE_WIDTHS):
        """The boxcar at the candidate's own time and width against same-width windows nearby.

        The event window is fixed, never a peak selected elsewhere in a large snippet.
        """
        series = self.search_series(self.dm if dm is None else dm, extra, auto_mask, baseline_seconds, baseline_widths)
        return local_boxcar_snr(series, -self.t0 / self.tsamp, self.width,
                                radius=round(self.analysis / self.tsamp))['local_snr']


def measure(source, dm, tcand, width_samples, plan, bad=(), baseline_seconds=None, baseline_widths=None):
    """A candidate's local S/N on its own data in a flatfielded beam, measured as the search measures it."""
    header, block, start, k, _, _, _ = stretch(source, dm, tcand, width_samples, plan)
    tsamp = float(header['tsamp'])
    own = OwnData(block, sigproc.channel_frequencies(header), tsamp * k, start * tsamp - float(tcand), dm,
                  round(int(width_samples) / k), start // k, bad, k)
    return own.local_snr(baseline_seconds=baseline_seconds or SEARCH_BASELINE_SECONDS,
                         baseline_widths=baseline_widths or SEARCH_BASELINE_WIDTHS)
