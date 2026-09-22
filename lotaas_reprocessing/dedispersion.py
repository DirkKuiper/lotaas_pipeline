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
    Unlike the original code we average time samples before the FFT, never
    subsample the frequency bins. The original dispersion constant is retained.
    """
    if tsamp <= 0 or np.any(np.asarray(frequencies) <= 0):
        raise ValueError('Positive sampling time and frequencies required')
    signal = time_scrunch(data, downsample, xp).astype(xp.float32, copy=False)
    n = signal.shape[1]
    if xp is np:
        from scipy import fft
    else:
        fft = xp.fft
    spectra = fft.rfft(signal, axis=1)
    del signal
    f = xp.asarray(np.fft.rfftfreq(n, tsamp * downsample), dtype=xp.float32)
    nu = xp.asarray(frequencies, dtype=xp.float64)
    delays = ((nu ** -2 - nu.max() ** -2) / 2.41e-4).astype(xp.float32)
    for dm in dms:
        phase = xp.exp((2j * xp.pi * float(dm)) * delays[:, None] * f[None, :])
        spectrum = xp.sum(spectra * phase, axis=0)
        yield float(dm), fft.irfft(spectrum, n=n).astype(xp.float32)
