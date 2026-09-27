"""Bounded-memory Fourier dedispersion with explicit time-domain scrunching."""
import numpy as np


def backend(name):
    if name not in {'cpu', 'gpu', 'auto'}:
        raise ValueError('backend must be cpu, gpu or auto')
    if name != 'cpu':
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('No CUDA devices visible')
            cp.zeros(1).sum().item()
            return cp
        except (ImportError, RuntimeError) as error:
            if name == 'gpu':
                raise RuntimeError('GPU requested but CUDA is unavailable') from error
    return np


def time_scrunch(data, factor, xp=np):
    if factor < 1 or int(factor) != factor:
        raise ValueError('downsample must be a positive integer')
    count = data.shape[1] // factor
    if count < 2:
        raise ValueError('Too few samples for this downsampling factor')
    return xp.asarray(data[:, :count * factor]).reshape(data.shape[0], count, factor).mean(axis=2)


def iter_dedispersed(data, tsamp, frequencies, dms, downsample=1, xp=np):
    """Yield one real time series per DM; reuse the channel FFT within a range.

    Frequencies are MHz and DM is pc cm^-3. Like the original pipeline this
    is circular Fourier dedispersion, so edge events need downstream review.
    The original dispersion constant is retained.

    Downsampling keeps the channel spectra below the new Nyquist frequency
    and nothing above it: an ideal low-pass. Averaging blocks of samples
    first, as this did until 27 September, passed about a tenth of the
    amplitude just above the new Nyquist frequency and folded it to a low
    one: B2217+47's 9th, 19th and 20th harmonics came out of the x8 and x4
    trials of L543473 SAP001 as periodic candidates at 1.221, 0.286 and
    0.187 s, at DMs 888, 439 and 302 (its own is 43.5).
    """
    if tsamp <= 0 or np.any(np.asarray(frequencies) <= 0):
        raise ValueError('Positive sampling time and frequencies required')
    if downsample < 1 or int(downsample) != downsample:
        raise ValueError('downsample must be a positive integer')
    n = data.shape[1] // int(downsample)
    if n < 2:
        raise ValueError('Too few samples for this downsampling factor')
    if xp is np:
        from scipy import fft
    else:
        fft = xp.fft
    signal = xp.asarray(data[:, :n * int(downsample)]).astype(xp.float32, copy=False)
    spectra = fft.rfft(signal, axis=1)
    del signal
    if downsample > 1:
        # The full-resolution spectra up to the new Nyquist frequency; the inverse
        # transform at the reduced length then scales by the factor, taken out below.
        spectra = spectra[:, :n // 2 + 1].copy()
    f = xp.asarray(np.fft.rfftfreq(n, tsamp * downsample), dtype=xp.float32)
    nu = xp.asarray(frequencies, dtype=xp.float64)
    delays = ((nu ** -2 - nu.max() ** -2) / 2.41e-4).astype(xp.float32)
    for dm in dms:
        phase = xp.exp((2j * xp.pi * float(dm)) * delays[:, None] * f[None, :])
        spectrum = xp.sum(spectra * phase, axis=0)
        yield float(dm), (fft.irfft(spectrum, n=n) / downsample).astype(xp.float32)
