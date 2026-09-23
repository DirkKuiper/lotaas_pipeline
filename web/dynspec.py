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
        self.centre, self.scale = channel_scale(self.data)
        known = [c for c in self.meta.get('bad_channels', []) if 0 <= c < self.data.shape[1]]
        self.flat = np.isnan(self.scale)
        self.known_bad = sorted(set(known) | set(np.flatnonzero(self.flat).tolist()))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.normalised = (self.data - self.centre) / self.scale
        self.channel_mhz = abs(self.header['foff'])
        self.bandwidth = self.channel_mhz * self.data.shape[1]
        self.centre_ghz = float(np.mean(self.freqs)) / 1e3
        self.per_dm = float(sweep_seconds(1.0, self.freqs).max())
        self.guard = max(20 * self.width * self.tsamp, 0.5)
        self.analysis = max(1.5, 30 * self.width * self.tsamp, self.guard + 1.0)
        # A trial above the candidate DM needs, at the far edge of the analysis
        # window, samples of the lowest channel this much later; past the
        # snippet's end there are none.
        end = self.t0 + self.data.shape[0] * self.tsamp
        self.dm_limit = max(self.dm, (end - self.analysis - 3 * self.width * self.tsamp) / self.per_dm)

    @property
    def times(self):
        return self.t0 + np.arange(self.data.shape[0]) * self.tsamp

    def masked(self, extra=()):
        data = self.normalised.copy()
        data[:, sorted(set(self.known_bad) | set(extra))] = np.nan
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
        guard = max(self.guard, 20 * width_seconds)
        return (np.abs(times) > guard) & (np.abs(times) <= max(self.analysis, guard + 1.0))

    def snr(self, series, times, width):
        """Boxcar S/N at this width against the scatter of boxcar sums off the pulse."""
        tsamp = times[1] - times[0] if len(times) > 1 else self.tsamp
        return robust_snr(boxcar(series, width), self.off_pulse(times, width * tsamp))

    def best_width(self, series, times):
        """The boxcar width (in samples of `times`) that maximises S/N at the candidate."""
        tsamp = times[1] - times[0]
        best = (None, 1)
        for width in WIDTHS:
            if width * tsamp > self.analysis / 4:
                break
            snr = self.snr(series, times, width)
            near = np.abs(times) <= 2 * width * tsamp + tsamp
            if np.isfinite(snr[near]).any():
                peak = float(np.nanmax(snr[near]))
                if best[0] is None or peak > best[0]:
                    best = (peak, width)
        return best

    def view(self, dm=None, tscrunch=1, nsub=None, window=None, mask=(), clip=99.0, max_columns=2048):
        """Everything one render of the viewer needs, at the requested DM and resolution."""
        dm = self.dm if dm is None else max(0.0, float(dm))
        nf = self.data.shape[1]
        ffactor = max(1, nf // int(nsub)) if nsub else 1
        dedispersed = dedisperse(self.masked(mask), self.freqs, self.tsamp, dm)
        times = self.times
        tscrunch = max(1, int(tscrunch))
        shown = np.abs(times) <= float(window) if window else np.ones(times.size, bool)
        while shown.sum() // tscrunch > max_columns:
            tscrunch *= 2
        tsamp = self.tsamp * tscrunch
        width = max(1, int(round(self.width / tscrunch)))
        coarse = scrunch(dedispersed, tscrunch, 1)
        times = scrunch(times[:, None], tscrunch, 1)[:, 0]
        off = self.off_pulse(times, width * tsamp)
        on = np.abs(times) <= max(width, 1) * tsamp / 2 + tsamp / 2

        # Band-averaged series of per-channel-normalised data, as S/N per sample
        # and as the boxcar S/N at the candidate's width.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            counts = np.isfinite(coarse).sum(axis=1)
            series = np.where(counts >= 0.9 * max(1, counts.max()), np.nanmean(coarse, axis=1), np.nan)
        per_sample = robust_snr(series, off)
        boxcar_snr = self.snr(series, times, width)
        near = np.abs(times) <= max(3 * width * tsamp, 3 * tsamp)
        peak = float(np.nanmax(boxcar_snr[near])) if np.isfinite(boxcar_snr[near]).any() else None
        best_snr, best_width = self.best_width(series, times)

        # The spectrum over the candidate's own window against one of the same
        # length well before it, both in S/N per channel.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            spectrum_on = np.nanmean(coarse[on], axis=0) * math.sqrt(max(on.sum(), 1))
            before = np.flatnonzero(off & (times < 0))
            reference = before[:max(on.sum(), 1)] if before.size else np.array([], int)
            spectrum_off = (np.nanmean(coarse[reference], axis=0) * math.sqrt(max(reference.size, 1))
                            if reference.size else np.full(nf, np.nan))

        keep = np.abs(times) <= float(window) if window else np.ones(times.size, bool)
        image = scrunch(coarse[keep], 1, ffactor)
        spectrum_on = scrunch(spectrum_on[None, :], 1, ffactor)[0] * math.sqrt(ffactor)
        spectrum_off = scrunch(spectrum_off[None, :], 1, ffactor)[0] * math.sqrt(ffactor)
        freqs = scrunch(self.freqs[None, :], 1, ffactor)[0]
        # Re-normalise each subband over the off-pulse samples so the colour
        # scale means the same thing at every scrunch factor.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            reference = scrunch(coarse[off], 1, ffactor) if off.any() else image
            centre = np.nanmedian(reference, axis=0)
            scale = 1.4826 * np.nanmedian(np.abs(reference - centre), axis=0)
            image = (image - centre) / np.where(scale > 0, scale, np.nan)
        finite = image[np.isfinite(image)]
        low, high = np.percentile(finite, [100 - clip, clip]) if finite.size else (-1.0, 1.0)
        return {'dm': dm, 'tsamp': tsamp, 'tscrunch': tscrunch, 'nsub': image.shape[1],
                'times': times[keep], 'freqs': freqs, 'image': image.T.astype(np.float32),
                'zmin': float(low), 'zmax': float(high),
                'series': per_sample[keep], 'boxcar': boxcar_snr[keep], 'peak_snr': peak, 'width': width,
                'best_snr': best_snr, 'best_width': best_width,
                'spectrum_on': spectrum_on, 'spectrum_off': spectrum_off,
                'masked': sorted(set(self.known_bad) | set(mask))}

    def dm_response(self, points=121, mask=()):
        """Boxcar S/N near the candidate time over a fine grid around its DM and a
        coarse one from DM 0, with what a real dispersed pulse would keep."""
        data = self.masked(mask)
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

        def response(dms):
            window = np.abs(times) <= self.analysis
            plane = np.full((len(dms), int(window.sum())), np.nan)
            peaks = np.full(len(dms), np.nan)
            for row, dm in enumerate(dms):
                snr = self.snr(self.band_series(data, dm), times, width)
                plane[row] = snr[window]
                # The band-averaged centroid moves by half the extra sweep off the true DM.
                drift = abs(dm - self.dm) * self.per_dm / 2
                near = np.abs(times) <= drift + 3 * width * self.tsamp + self.tsamp
                if np.isfinite(snr[near]).any():
                    peaks[row] = np.nanmax(snr[near])
            return plane, peaks

        plane, fine_peaks = response(fine)
        _, coarse_peaks = response(coarse)
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
