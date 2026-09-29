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
import functools
import math
import os
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
# Trial DMs, as fractions of the candidate's, at which a dispersed pulse must fade (dispersion_ratio).
DISPERSION_FACTORS = (0.5, 0.75)


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


def native_rfi_mask(source, start, stop, k, block=RFI_BLOCK_SAMPLES):
    """The search's RFI mask over native samples start..stop of a beam, on the grid of k-sample cells (a cell
    is masked where any of its samples is): rfi_mask at native resolution in blocks on the observation's
    grid, as the search computed it (pipeline_gpu). Each block's mask is kept for the beam's other candidates.

    The same statistics over k-sample means are another test. In early-cycle beams levelled per 32-sample
    2-bit row, every pair of 16-sample means mirrors about the channel's level, so each channel's skewness
    is 0 to 3e-5 and any signal stands out: at k 16 (DM 1010-2015) the mask took the whole track of every
    FRB-like burst injected there, and the classifier called 65 of 66 'unconfirmed' (29 September 2026)."""
    stat = os.stat(source)
    identity = (str(source), stat.st_size, stat.st_mtime_ns, block)
    nchan = sigproc.read_header(source)[0]['nchans']
    native = np.zeros((stop - start, nchan), dtype=bool)
    for index in range(start // block, -(-stop // block)):
        a = index * block
        lo, hi = max(a, start), min(a + block, stop)
        cells = np.unpackbits(_block_mask(identity, index), axis=1, count=nchan).astype(bool)
        native[lo - start:hi - start] = cells[lo - a:hi - a]
    return native[:(stop - start) // k * k].reshape(-1, k, nchan).any(axis=1)


@functools.lru_cache(maxsize=2048)
def _block_mask(identity, index):
    """One block's RFI mask ((time, channel), bits packed along channels) of the beam `identity` names."""
    source, _, _, block = identity
    data = sigproc.open_data(source)[1]
    return np.packbits(rfi_mask(np.asarray(data[index * block:(index + 1) * block]), block), axis=1)


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

    Where k-sample means hold almost no noise, k is divided by 4 until they do: early-cycle
    beams levelled per 32-sample 2-bit row have every row's mean at its channel's level, so
    at k 32 (the search's above DM 2015) a quarter of the channels were exactly flat and the
    rest nearly, and an FRB-like burst injected there measured S/N -0.5 (29 September 2026).
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
    block = _means(data, start, stop, k)
    if k >= LEVELLED_MIN_K:
        native = _scale(np.asarray(data[start:min(stop, start + 2048)], dtype=np.float32))
        while k >= LEVELLED_MIN_K and _scale(block[:2048]) < LEVELLED_FRACTION * native / math.sqrt(k):
            k //= 4
            block = _means(data, start, stop, k)
    return header, block, start, k, delay, margin, search_downsample


LEVELLED_MIN_K = 8          # k-sample means are checked for noise from here
LEVELLED_FRACTION = 0.3     # of white noise's scale at k: below it, the means are levelled rows


def _means(data, start, stop, k):
    """Means of k native samples of (time, channel) data from start to stop (multiples of k)."""
    block = np.empty(((stop - start) // k, data.shape[1]), dtype=np.float32)
    batch = max(1, 262144 // (k * data.shape[1]))
    for out_start in range(0, len(block), batch):
        out_stop = min(len(block), out_start + batch)
        raw = np.asarray(data[start + out_start * k:start + out_stop * k])
        block[out_start:out_stop] = raw.reshape(-1, k, data.shape[1]).mean(axis=1)
    return block


def _scale(block):
    """The median over channels of each channel's robust scale."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        centre = np.median(block, axis=0)
        return float(np.nanmedian(1.4826 * np.median(np.abs(block - centre), axis=0)))


class OwnData:
    """A candidate's stretch, scaled and masked for measuring as the search measures.

    data is (time, channel) in file order at tsamp; t0 the time of its first sample relative
    to the candidate (arrival at the highest frequency); width the candidate's boxcar in these
    samples; first its first sample's index on the observation's grid at this resolution
    (k native samples each); bad the search's channel mask in file order; mask the search's RFI
    mask on this grid (native_rfi_mask), without which it is approximated on these samples.

    Noise is always measured over one fixed stretch around the candidate, the analysis
    window, whatever is displayed: 'guard < |t| <= analysis', where the guard keeps the
    pulse and its immediate surroundings out.
    """

    def __init__(self, data, freqs, tsamp, t0, dm, width, first=0, bad=(), k=1, mask=None):
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
        if mask is not None and np.shape(mask) == self.data.shape:
            self.rfi_mask = np.asarray(mask, dtype=bool)      # native_rfi_mask: the search's own
        else:
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


def _window_sums(series, width):
    """Sums of every width-sample window of series; NaN where a window holds a non-finite sample."""
    finite = np.isfinite(series)
    sums = np.concatenate([[0.0], np.cumsum(np.where(finite, series, 0.0), dtype=np.float64)])
    counts = np.concatenate([[0], np.cumsum(finite)])
    return np.where(counts[width:] - counts[:-width] == width, sums[width:] - sums[:-width], np.nan)


def dispersion_ratio(own, factors=DISPERSION_FACTORS, baseline_seconds=SEARCH_BASELINE_SECONDS,
                     baseline_widths=SEARCH_BASELINE_WIDTHS):
    """How much of a candidate's signal survives dedispersion at too low a DM: (ratio, S/N at its DM).

    At f x DM each channel's signal lands (1 - f) x its delay later than at DM, so the strongest
    same-width boxcar from the candidate's time to (1 - f) x the sweep after it is compared with
    the one at the candidate's time and DM, both against the scatter of same-width sums outside
    those windows. A dispersed pulse is smeared over the sweep error and fades; interference in a
    few channels only moves and keeps its S/N. On 28 September 2026 the ratio (the larger over
    f = 0.5 and 0.75) was 0.23-0.47 (quartiles) for injected FRB-like bursts FETCH rejected and
    0.86-1.7 for real clusters FETCH rejected at DM >= 300 (benchmarks/frb-injection-2026-09-28).
    None when the stretch leaves too little noise to judge.
    """
    w, dt = own.width, own.tsamp
    sweep = float(sweep_seconds(own.dm, own.freqs).max())
    centres = own.t0 + (np.arange(own.data.shape[0] - w + 1) + (w - 1) / 2) * dt
    slack = max(2 * w * dt, 0.1)
    windows = {1.0: np.abs(centres) <= slack}
    for f in factors:
        windows[f] = (centres >= -slack) & (centres <= (1 - f) * sweep + slack)
    outside = ~np.logical_or.reduce(list(windows.values()))
    snrs = {}
    for f, window in windows.items():
        sums = _window_sums(own.search_series(f * own.dm, (), True, baseline_seconds, baseline_widths), w)
        reference = sums[outside & np.isfinite(sums)]
        inside = sums[window & np.isfinite(sums)]
        if len(reference) < 32 or not len(inside):
            return None, None
        centre = np.median(reference)
        scale = 1.4826 * np.median(np.abs(reference - centre))
        if scale <= 0:
            return None, None
        snrs[f] = float((inside.max() - centre) / scale)
    if snrs[1.0] <= 0:
        return None, snrs[1.0]
    return max(snrs[f] for f in factors) / snrs[1.0], snrs[1.0]


def smoothness(own, span=16):
    """The fraction of the candidate's on-pulse spectrum left by a running median over `span` channels.

    Dedispersed at its DM, each channel's excess in the boxcar over its own level around it (10 widths
    either side, the pulse and 2 widths out), in frequency order: about 1 for a burst, whose spectrum is
    smooth over many channels even when band-limited; small when the signal sits in scattered single
    channels, as a channel's level jump for seconds does. On 29 September 2026, 428 injected FRB-like
    bursts the route queued read 0.83 or more in 90% (all but one >= 0.6); the route's 159 real
    candidates of the night before, almost all such bars, read a median 0.37 (47 >= 0.6).
    0 when the excess sums to nothing; None when the stretch does not reach 10 widths either side.
    """
    data = own.masked()
    data[(own.rfi_mask | (np.abs(data) > MAX_CELL_SIGMA)) & np.isfinite(data)] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        data = data - np.nanmean(data, axis=1, keepdims=True)
        aligned = dedisperse(data, own.freqs, own.tsamp, own.dm)
        w = own.width
        start = int(round(-own.t0 / own.tsamp - (w - 1) / 2))
        lo, hi = start - 10 * w, start + 11 * w
        if lo < 0 or hi > aligned.shape[0]:
            return None
        # Each channel's excess over its own level around the candidate (the pulse and 2 widths either side out).
        near = np.concatenate([aligned[lo:start - 2 * w], aligned[start + 3 * w:hi]])
        on = np.nanmean(aligned[start:start + w], axis=0) - np.nanmedian(near, axis=0)
        order = np.argsort(own.freqs)
        spectrum = on[order]
        half = span // 2
        smooth = np.array([np.nanmedian(spectrum[max(0, i - half):i + half]) for i in range(len(spectrum))])
        total = np.nansum(spectrum)
        # Nothing standing above its surroundings is no burst either.
        return float(np.nansum(smooth) / total) if total > 0 else 0.0


def near_edge(source, dm, tcand, seconds):
    """Whether a candidate, with its whole sweep, comes within `seconds` of the start or end of its beam: where the
    data begin and end, levels settle and circular dedispersion wraps, and the route's junk gathered."""
    header, data = sigproc.open_data(source)
    duration = data.shape[0] * float(header['tsamp'])
    return tcand < seconds or tcand + float(sweep_seconds(dm, sigproc.channel_frequencies(header)).max()) > duration - seconds


def load(source, dm, tcand, width_samples, plan, bad=()):
    """The OwnData of a candidate in a flatfielded beam: the stretch the page cuts and the classifier measures."""
    header, block, start, k, _, _, _ = stretch(source, dm, tcand, width_samples, plan)
    tsamp = float(header['tsamp'])
    mask = native_rfi_mask(source, start, start + len(block) * k, k) if k > 1 else None
    return OwnData(block, sigproc.channel_frequencies(header), tsamp * k, start * tsamp - float(tcand), dm,
                   round(int(width_samples) / k), start // k, bad, k, mask)


def measure(source, dm, tcand, width_samples, plan, bad=(), baseline_seconds=None, baseline_widths=None):
    """A candidate's local S/N on its own data in a flatfielded beam, measured as the search measures it."""
    return load(source, dm, tcand, width_samples, plan, bad).local_snr(
        baseline_seconds=baseline_seconds or SEARCH_BASELINE_SECONDS,
        baseline_widths=baseline_widths or SEARCH_BASELINE_WIDTHS)
