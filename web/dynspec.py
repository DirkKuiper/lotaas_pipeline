"""Dynamic-spectrum maths for the candidate viewer.

Arrays are (time, channel) in file order. Dispersion is referenced to the
highest frequency, as the search and the classifier reference it, so a
dedispersed pulse sits at the candidate time whatever the DM, and the sweep
runs to later times toward the bottom of the band.

Every S/N here is measured against the snippet's own off-pulse samples with a
robust (median, MAD) scale. Dedispersed LOTAAS series are red, so a boxcar S/N
is normalised by the scatter of boxcar sums of the same width, not by the
per-sample noise times sqrt(width).
"""
import json
import math
from pathlib import Path
import warnings

import numpy as np

from lotaas_reprocessing.single_pulse_quality import persistent_channels, local_boxcar_snr

from web import sigproc

# Dispersion constant in s MHz^2 pc^-1 cm^3; the classifier and your use 4148808 ms.
K_DM = 4148.808


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


def scrunch(x, tfactor=1, ffactor=1):
    """Average blocks of tfactor samples and ffactor channels, ignoring NaN; remainders dropped."""
    nt, nf = x.shape
    nt2, nf2 = nt // tfactor, nf // ffactor
    x = x[:nt2 * tfactor, :nf2 * ffactor].reshape(nt2, tfactor, nf2, ffactor)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(x, axis=(1, 3))


def robust_snr(series, off):
    """The series in units of its robust scatter over the off-pulse samples."""
    reference = series[off]
    reference = reference[np.isfinite(reference)]
    if reference.size < 8:
        return np.full(series.shape, np.nan)
    centre = np.median(reference)
    scale = 1.4826 * np.median(np.abs(reference - centre))
    if not scale > 0:
        return np.full(series.shape, np.nan)
    return (series - centre) / scale


def boxcar(series, width):
    """Sums over each complete width-sample window, placed at the window centre."""
    width = max(1, int(width))
    series = np.asarray(series, dtype=float)
    valid = np.isfinite(series)
    sums = np.concatenate([[0.], np.cumsum(np.where(valid, series, 0.))])
    counts = np.concatenate([[0], np.cumsum(valid)])
    windows = np.where(counts[width:] - counts[:-width] == width, sums[width:] - sums[:-width], np.nan)
    out = np.full(series.size, np.nan)
    start = (width - 1) // 2
    out[start:start + windows.size] = windows
    return out


def expected_fraction(ddm, width_ms, bandwidth_mhz, centre_ghz):
    """S/N kept by a real pulse of this width dedispersed ddm away from its DM.

    Cordes & McLaughlin (2003), eq. 12: (sqrt(pi)/2) erf(z) / z with
    z = 6.91e-3 ddm bandwidth / (width centre^3), in MHz, ms and GHz.
    """
    zeta = 6.91e-3 * np.abs(np.atleast_1d(ddm)) * bandwidth_mhz / (width_ms * centre_ghz ** 3)
    return np.array([1.0 if z < 1e-9 else math.sqrt(math.pi) / 2 * math.erf(z) / z for z in zeta])


def half_width_dm(width_ms, bandwidth_mhz, centre_ghz):
    """DM offset at which a real pulse of this width keeps half its S/N (z = 1.77)."""
    return 1.7725 * width_ms * centre_ghz ** 3 / (6.91e-3 * bandwidth_mhz)


def parse_mask(text, nchans):
    """Channel indices from '10-20,138'; out-of-range entries are ignored."""
    channels = set()
    for part in (text or '').replace(' ', '').split(','):
        if not part:
            continue
        low, _, high = part.partition('-')
        try:
            low, high = int(low), int(high or low)
        except ValueError:
            continue
        channels.update(c for c in range(min(low, high), max(low, high) + 1) if 0 <= c < nchans)
    return sorted(channels)


def channel_smearing_ms(dm, freq_mhz, channel_mhz):
    """Dispersion smearing within one channel, 8.3 us DM dnu_MHz / nu_GHz^3, in ms."""
    return 8.3e-3 * dm * abs(channel_mhz) / (np.asarray(freq_mhz, dtype=float) / 1e3) ** 3


WIDTHS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)

# Subbands the viewer offers: divisors of LOTAAS's 648 channels.
SUBBANDS = (4, 6, 8, 12, 18, 27, 36, 54, 81, 108, 162, 324, 648)
# Per-pixel S/N from which a pulse stands out in a waterfall by eye.
VISIBLE_SIGMA = 2.5


def pixel_snr(snr, nsub, tscrunch, width):
    """S/N a flat-spectrum pulse puts in one displayed pixel.

    Its S/N is split over nsub subbands and, while the time bins are narrower
    than the pulse, over width / tscrunch columns.
    """
    if snr is None or not np.isfinite(snr):
        return None
    return float(snr) / math.sqrt(max(1, int(nsub)) * max(1.0, float(width) / max(1, int(tscrunch))))


def suggested_view(snr, width, nchans):
    """Subbands and time bins at which a pulse of this S/N reaches VISIBLE_SIGMA per pixel.

    At the former defaults, 81 subbands and native samples, a pulse of local
    S/N 7 on real LOTAAS noise was 0.6 sigma per pixel and invisible; at 8
    subbands and bins of its own width it was 3.1 (search audit, 23 September
    2026). Bins of the boxcar width, and the most subbands, up to 81, that
    keep the pulse visible.
    """
    tscrunch = max(1, int(width))
    limit = (float(snr) / VISIBLE_SIGMA) ** 2 if snr and np.isfinite(snr) and snr > 0 else 81
    choices = [n for n in SUBBANDS if n <= nchans and nchans % n == 0] or [nchans]
    fitting = [n for n in choices if n <= min(limit, 81)]
    return (max(fitting) if fitting else min(choices)), tscrunch


# The first display: fine enough to show what is in the data (narrowband streaks,
# broadband bursts, the edges of the pulse), smoothed by about a pixel so that a
# faint pulse still builds up for the eye.
DETAIL_SUBBANDS = 64
DETAIL_BINS_PER_WIDTH = 4
SMOOTH_PIXELS = 1.0


def detail_view(width, nchans):
    """Subbands and time bins of the first display: about 64 subbands, a quarter of the pulse per bin.

    The display used to open at bins of the whole boxcar and the fewest
    subbands that kept an S/N 7 pulse at 2.5 sigma per pixel (suggested_view):
    for a 190 ms candidate an 8 x 21 grid of blocks, which hid the
    interference, sweeps and edges a reviewer judges by. That view stays
    available as the 'matched' preset.
    """
    counts = set(SUBBANDS) | {nchans} | {2 ** k for k in range(2, 12)}
    choices = sorted(n for n in counts if n <= nchans and nchans % n == 0)
    nsub = max([n for n in choices if n <= DETAIL_SUBBANDS] or [min(choices)])
    return nsub, max(1, int(round(int(width) / DETAIL_BINS_PER_WIDTH)))


def smoothing_gain(sigma):
    """How much Gaussian smoothing by sigma pixels on both axes raises the per-pixel S/N of an
    extended signal over white noise: the noise falls by sqrt(2 sqrt(pi) sigma) on each axis."""
    return 2 * math.sqrt(math.pi) * sigma if sigma and sigma > 0 else 1.0


def smooth_image(image, sigma):
    """Gaussian smoothing by sigma pixels on both axes of a (time, subband) image, ignoring NaN.

    Masked subbands stay masked; their neighbours are averaged over what is left.
    """
    if not sigma or sigma <= 0:
        return image
    radius = max(1, int(math.ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    valid = np.isfinite(image)
    data, weight = np.where(valid, image, 0.0), valid.astype(float)
    for axis in (0, 1):
        data = np.apply_along_axis(np.convolve, axis, data, kernel, mode='same')
        weight = np.apply_along_axis(np.convolve, axis, weight, kernel, mode='same')
    with np.errstate(invalid='ignore', divide='ignore'):
        smoothed = data / weight
    smoothed[~valid] = np.nan
    return smoothed


class Snippet:
    """A candidate's filterbank snippet with its sidecar, normalised once per channel.

    Noise is always measured over one fixed stretch around the candidate, the
    analysis window, whatever is displayed: 'guard < |t| <= analysis', where the
    guard keeps the pulse and its immediate surroundings out.
    """

    def __init__(self, path):
        path = Path(path)
        self.path = path
        self.header, raw = sigproc.open_data(path)
        self.data = np.array(raw, dtype=np.float32)
        self.meta = json.loads(path.with_suffix('.json').read_text())
        self.freqs = sigproc.channel_frequencies(self.header)
        self.tsamp = float(self.header['tsamp'])
        self.dm = float(self.meta['dm'])
        # Time of sample 0 relative to the candidate (arrival at the highest frequency).
        self.t0 = float(self.meta['t0_relative'])
        self.width = max(1, int(round(self.meta['width_samples'] / self.meta['downsample'])))
        self.guard = max(2 * self.width * self.tsamp, 3 * self.tsamp)
        self.analysis = max(64 * self.width * self.tsamp, 10.0)
        # Measure each channel on the same local off-pulse interval after
        # accounting for the candidate's dispersion sweep, not on its entire
        # (potentially 2000-second) raw snippet.
        aligned = dedisperse(self.data, self.freqs, self.tsamp, self.dm)
        off = self.off_pulse(self.times)
        self.centre, self.scale = channel_scale(aligned[off])
        known = [c for c in self.meta.get('bad_channels', []) if 0 <= c < self.data.shape[1]]
        self.flat = np.isnan(self.scale)
        self.known_bad = sorted(set(known) | set(np.flatnonzero(self.flat).tolist()))
        self.automatic_bad = persistent_channels(self.data.T)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.normalised = (self.data - self.centre) / self.scale
        self.channel_mhz = abs(self.header['foff'])
        self.bandwidth = self.channel_mhz * self.data.shape[1]
        self.centre_ghz = float(np.mean(self.freqs)) / 1e3
        self.per_dm = float(sweep_seconds(1.0, self.freqs).max())
        # A trial above the candidate DM needs, at the far edge of the analysis
        # window, samples of the lowest channel this much later; past the
        # snippet's end there are none.
        end = self.t0 + self.data.shape[0] * self.tsamp
        self.dm_limit = max(self.dm, (end - self.guard - 3 * self.width * self.tsamp) / self.per_dm)

    @property
    def default_window(self):
        # Wide enough to compare the pulse with twenty times its width of data on each side.
        return max(1.5, 20 * self.width * self.tsamp, 16 * self.tsamp)

    @property
    def times(self):
        return self.t0 + np.arange(self.data.shape[0]) * self.tsamp

    def masked(self, extra=(), auto_mask=True):
        data = self.normalised.copy()
        data[:, sorted(set(self.known_bad) | set(extra) | (set(self.automatic_bad) if auto_mask else set()))] = np.nan
        return data

    def smearing_ms(self, dm=None):
        """Smearing within one channel at the bottom and top of the band."""
        dm = self.dm if dm is None else dm
        return (float(channel_smearing_ms(dm, self.freqs.min(), self.channel_mhz)),
                float(channel_smearing_ms(dm, self.freqs.max(), self.channel_mhz)))

    def band_series(self, data, dm, coverage=0.9):
        """Band average of the dedispersed data where enough channels are present."""
        valid = np.isfinite(data)
        filled = np.where(valid, data, 0.0)
        usable = max(1, int(valid.any(axis=0).sum()))
        nt = data.shape[0]
        total = np.zeros(nt)
        count = np.zeros(nt)
        for channel, shift in enumerate(delays(dm, self.freqs, self.tsamp)):
            if shift < nt:
                total[:nt - shift] += filled[shift:, channel]
                count[:nt - shift] += valid[shift:, channel]
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(count >= coverage * usable, total / count, np.nan)

    def off_pulse(self, times, width_seconds=0.0):
        guard = max(self.guard, 2 * width_seconds)
        return (np.abs(times) > guard + width_seconds / 2) & (np.abs(times) <= self.analysis)

    def snr(self, series, times, width):
        """Boxcar S/N at this width against the scatter of boxcar sums off the pulse."""
        tsamp = times[1] - times[0] if len(times) > 1 else self.tsamp
        return robust_snr(boxcar(series, width), self.off_pulse(times, width * tsamp))

    def best_width(self, series, times):
        """The boxcar width (in samples of `times`) that maximises S/N at the candidate."""
        tsamp = times[1] - times[0]
        best = (None, 1)
        for width in sorted(set(WIDTHS) | {self.width}):
            if width * tsamp > self.analysis / 4:
                break
            snr = self.snr(series, times, width)
            near = np.abs(times) <= width * tsamp / 2 + tsamp / 2
            if np.isfinite(snr[near]).any():
                peak = float(np.nanmax(snr[near]))
                if best[0] is None or peak > best[0]:
                    best = (peak, width)
        return best

    def view(self, dm=None, tscrunch=1, nsub=None, window=-1, mask=(), clip=99.0,
             max_columns=2048, auto_mask=True, smooth=0.0):
        """Measure once at saved resolution; scrunch only the displayed arrays."""
        dm = self.dm if dm is None else max(0.0, float(dm))
        nf = self.data.shape[1]
        ffactor = max(1, nf // int(nsub)) if nsub else 1
        masked = sorted(set(self.known_bad) | set(mask) | (set(self.automatic_bad) if auto_mask else set()))
        aligned = dedisperse(self.masked(mask, auto_mask), self.freqs, self.tsamp, dm)
        times = self.times
        off = self.off_pulse(times, self.width * self.tsamp)
        usable = max(1, nf - len(masked))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            counts = np.isfinite(aligned).sum(axis=1)
            covered = counts >= usable
            series = np.where(covered, np.nanmean(aligned, axis=1), np.nan)
        per_sample = robust_snr(series, off)
        boxcar_snr = self.snr(series, times, self.width)
        # This is a fixed event window, never a peak selected elsewhere in a
        # large snippet. The local noise uses independent same-width windows.
        evidence = local_boxcar_snr(series, -self.t0 / self.tsamp, self.width,
                                   radius=round(self.analysis / self.tsamp))
        peak = evidence['local_snr']
        # The same measurement without the reviewer's own mask. Channels picked
        # while looking at the pulse raise its S/N by chance alone: dropping
        # the 10% of subbands that are lowest at the pulse lifts a 5-sigma
        # noise event to about 6.8.
        unmasked = peak
        if mask:
            plain = dedisperse(self.masked((), auto_mask), self.freqs, self.tsamp, dm)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                plain_counts = np.isfinite(plain).sum(axis=1)
                plain_usable = max(1, nf - len(set(self.known_bad) | (set(self.automatic_bad) if auto_mask else set())))
                plain_series = np.where(plain_counts >= plain_usable, np.nanmean(plain, axis=1), np.nan)
            unmasked = local_boxcar_snr(plain_series, -self.t0 / self.tsamp, self.width,
                                        radius=round(self.analysis / self.tsamp))['local_snr']
        best_snr, best_width = self.best_width(series, times)

        # On/reference spectra use the measured scatter of same-width sums in
        # each subband, including correlated noise; sqrt(N) is not assumed.
        subbands = scrunch(aligned, 1, ffactor)
        on_index = int(np.argmin(np.abs(times - ((self.width - 1) % 2) * self.tsamp / 2)))
        references = np.flatnonzero(off & (times < 0) & covered)
        ref_index = references[-1] if references.size else None
        spectrum_on, spectrum_off = [], []
        for channel in subbands.T:
            z = self.snr(channel, times, self.width)
            spectrum_on.append(z[on_index])
            spectrum_off.append(z[ref_index] if ref_index is not None else np.nan)
        freqs = scrunch(self.freqs[None, :], 1, ffactor)[0]
        window = self.default_window if window is not None and window < 0 else window
        keep = np.abs(times) <= float(window) if window else covered
        # Whole snippet means the interval with the full usable band; delay
        # padding and the curved incomplete-band tail are not a pulse profile.
        chosen = np.flatnonzero(keep)
        if not chosen.size:
            chosen = np.array([int(np.argmin(np.abs(times)))])
        first, stop = chosen[0], chosen[-1] + 1
        factor = max(1, int(tscrunch))
        while (stop - first) // factor > max_columns:
            factor *= 2
        factor = min(factor, stop - first)
        display_times = scrunch(times[first:stop, None], factor, 1)[:, 0]
        coarse = scrunch(subbands[first:stop], factor, 1)
        # Four subband profiles at the candidate width: a real dispersed pulse
        # rises at one time in several of them; interference often in one.
        quarters = scrunch(aligned, 1, max(1, nf // 4))
        profiles = [scrunch(self.snr(q, times, self.width)[first:stop, None], factor, 1)[:, 0]
                    for q in quarters.T]
        profile_freqs = scrunch(self.freqs[None, :], 1, max(1, nf // 4))[0]
        # Colour contrast is independent of which temporal interval is shown.
        reference = subbands[off]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            centre = np.nanmedian(reference, axis=0)
            scale = 1.4826 * np.nanmedian(np.abs(reference - centre), axis=0)
            image = (coarse - centre) / np.where(scale > 0, scale, np.nan)
            image = smooth_image(image, smooth)
            # Each displayed pixel in sigma of its own time bins and smoothing, measured
            # off the pulse; the channel normalisation above is at the saved resolution.
            quiet = self.off_pulse(display_times, self.width * self.tsamp)
            if quiet.sum() >= 8:
                reference = image[quiet]
                pixel = 1.4826 * np.nanmedian(np.abs(reference - np.nanmedian(reference)))
                if np.isfinite(pixel) and pixel > 0:
                    image = image / pixel
        finite = image[np.isfinite(image)]
        low, high = np.percentile(finite, [100 - clip, clip]) if finite.size else (-1., 1.)
        return {'dm': dm, 'tsamp': self.tsamp * factor, 'analysis_tsamp': self.tsamp,
                'tscrunch': factor, 'nsub': image.shape[1], 'window': window,
                'times': display_times, 'freqs': freqs, 'image': image.T.astype(np.float32),
                'zmin': float(low), 'zmax': float(high),
                'series': scrunch(per_sample[first:stop, None], factor, 1)[:, 0],
                'boxcar': scrunch(boxcar_snr[first:stop, None], factor, 1)[:, 0],
                'peak_snr': peak, 'width': self.width, 'width_seconds': self.width * self.tsamp,
                'best_snr': best_snr, 'best_width': best_width,
                'reference_windows': evidence['reference_windows'],
                'spectrum_on': np.asarray(spectrum_on), 'spectrum_off': np.asarray(spectrum_off),
                'masked': masked, 'automatic_bad': self.automatic_bad,
                'unmasked_peak_snr': unmasked, 'profiles': np.asarray(profiles, dtype=np.float32),
                'profile_freqs': profile_freqs,
                'pixel_snr': pixel_snr(peak, image.shape[1], factor, self.width),
                'smooth': float(smooth or 0.0),
                'smoothed_pixel_snr': (pixel_snr(peak, image.shape[1], factor, self.width) * smoothing_gain(smooth)
                                       if smooth and pixel_snr(peak, image.shape[1], factor, self.width) is not None
                                       else None),
                'coverage_seconds': [float(times[covered][0]), float(times[covered][-1])] if covered.any() else None}

    def dm_response(self, points=121, mask=(), auto_mask=True):
        """Boxcar S/N near the candidate time over a fine grid around its DM and a
        coarse one from DM 0, with what a real dispersed pulse would keep."""
        data = self.masked(mask, auto_mask)
        times = self.times
        at_dm = self.band_series(data, self.dm)
        best_snr, best_width = self.best_width(at_dm, times)
        width = best_width or self.width
        low_ms, high_ms = self.smearing_ms()
        # A real pulse is no narrower than a sample or its smearing in one channel.
        width_ms = max(width * self.tsamp * 1e3, math.hypot(self.tsamp * 1e3, (low_ms + high_ms) / 2))
        half = half_width_dm(width_ms, self.bandwidth, self.centre_ghz)
        span = max(8 * half, 1.0)
        fine = self.dm + np.linspace(-span, span, points)
        fine = fine[(fine >= 0) & (fine <= self.dm_limit)]
        coarse = np.linspace(0.0, max(0.0, min(2 * self.dm, self.dm_limit)), points)

        def response(dms, track=True):
            window = np.abs(times) <= self.analysis
            plane = np.full((len(dms), int(window.sum())), np.nan)
            peaks = np.full(len(dms), np.nan)
            for row, dm in enumerate(dms):
                trial_series = self.band_series(data, dm)
                snr = self.snr(trial_series, times, width)
                plane[row] = snr[window]
                # The band-averaged centroid moves by half the extra sweep off the true DM.
                drift = (self.dm - dm) * float(np.mean(sweep_seconds(1., self.freqs)))
                near = np.abs(times - drift) <= width * self.tsamp / 2 + self.tsamp
                if not track:
                    # A broad diagnostic of nearby undispersed interference,
                    # explicitly not evidence that this is the same event.
                    near = np.abs(times) <= abs(dm - self.dm) * self.per_dm + 3 * width * self.tsamp
                if np.isfinite(snr[near]).any():
                    peaks[row] = np.nanmax(snr[near])
                if not track:
                    # Undispersed interference can be much narrower than the
                    # dispersed cluster's reported width.
                    for trial_width in sorted({1, 2, self.width} - {width}):
                        narrow = self.snr(trial_series, times, trial_width)
                        if np.isfinite(narrow[near]).any():
                            peaks[row] = np.fmax(peaks[row], np.nanmax(narrow[near]))
            return plane, peaks

        plane, fine_peaks = response(fine)
        _, coarse_peaks = response(coarse, track=False)
        best = int(np.nanargmax(fine_peaks)) if np.isfinite(fine_peaks).any() else None
        reference = fine_peaks[np.argmin(np.abs(fine - self.dm))] if len(fine) else np.nan
        expected = expected_fraction(fine - self.dm, width_ms, self.bandwidth, self.centre_ghz) * reference
        return {'dm': self.dm, 'width_samples': width, 'width_ms': width_ms, 'best_width_snr': best_snr,
                'smearing_ms': [low_ms, high_ms], 'half_width_dm': half,
                'fine_dms': fine, 'fine_snr': fine_peaks, 'expected_snr': expected,
                'coarse_dms': coarse, 'coarse_snr': coarse_peaks,
                'best_dm': float(fine[best]) if best is not None else None,
                'best_snr': float(fine_peaks[best]) if best is not None else None,
                'plane_times': times[np.abs(times) <= self.analysis], 'plane': plane.astype(np.float32)}
