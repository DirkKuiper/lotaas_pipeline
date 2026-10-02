"""The data before dedispersion: RFI-masked cells filled, and a zero-DM filter.

Measured on 27 September 2026 on three LT5_004 beams, one of them holding
B2217+47, and an early-cycle beam (benchmarks/snr-discrepancy-2026-09-27):

- The RFI mask flags 7.9 s blocks of single channels. Filled with noise at
  the observation's global mean, each block sat at a level its channel did
  not have, which lifted or dropped the dedispersed series for 7.9 s. The
  brightest sample on such a step made an S/N 7 event with no pulse there,
  as many at negative DMs as at positive ones, and the review page, which
  has no block mask, showed S/N 2. Filled at the channel's own local level
  those events are gone.
- Subtracting at each sample the mean over channels (a zero-DM filter,
  Eatough, Keane & Lyne 2009) removes what arrives at every frequency at
  once: broadband interference and the slow broadband wander of a
  tied-array beam. That wander set the single-pulse noise scale about four
  times above the noise a pulse competes with: injected pulses of
  white-noise S/N 30 came out at 6 or below and were almost never found.
  With the local fill, this filter and the matched filter's 2 s running
  baseline they came out at 26-28 and were found 94 and 96 times of 96.
"""
import warnings

import numpy as np

FILL_MODES = ('global', 'local')


def local_levels(masked, block):
    """Each channel's level at every sample: medians of its unmasked (non-NaN) samples in blocks, interpolated.

    A channel with no unmasked sample is at 0, the normalised data's mean.
    """
    nchan, nsamp = masked.shape
    blocks = -(-nsamp // block)
    padded = np.full((nchan, blocks * block), np.nan, dtype=np.float32)
    padded[:, :nsamp] = masked
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)            # wholly masked blocks
        medians = np.nanmedian(padded.reshape(nchan, blocks, block), axis=2)
    del padded
    centres = (np.arange(blocks) + 0.5) * block
    t = np.arange(nsamp)
    levels = np.zeros((nchan, nsamp), dtype=np.float32)
    for channel in range(nchan):
        ok = np.isfinite(medians[channel])
        if ok.any():
            levels[channel] = np.interp(t, centres[ok], medians[channel][ok])
    return levels


def fill_masked(data, mask, block, rng, mode='global'):
    """Fill the masked cells of data (channel, time; NaN where masked) in place and return it.

    'global': noise at the unmasked data's mean and spread, the original fill.
    'local': each channel's local level (local_levels over RFI blocks) plus
    noise of the typical channel's spread.
    """
    if mode not in FILL_MODES:
        raise ValueError(f'mask_fill must be one of {FILL_MODES}, not {mode!r}')
    if mode == 'global':
        data[mask] = rng.normal(np.nanmean(data), np.nanstd(data), int(mask.sum()))
        return data
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)            # wholly masked channels
        centre = np.nanmedian(data, axis=1, keepdims=True)
        spread = 1.4826 * np.nanmedian(np.abs(data - centre), axis=1)
    spread = spread[np.isfinite(spread)]
    typical = float(np.median(spread)) if spread.size else 0.0
    levels = local_levels(data, block)
    data[mask] = levels[mask] + typical * rng.standard_normal(int(mask.sum())).astype(np.float32)
    return data


def zero_dm(data):
    """Subtract, at each sample, the mean over channels, in place; returns data."""
    data -= data.mean(axis=0, keepdims=True)
    return data


SLOW_CAP_SAMPLES = (16, 32, 64, 128, 256, 512, 1024)      # 0.13 to 8 s of 7.864 ms samples


def cap_cells(data, cell, limit, sigma=None, first=0):
    """Hold each channel's mean over cells of `cell` samples within `limit` sigma, in place; returns data.

    data is (channel, time); sigma each channel's scatter per sample (its robust scale when not given); cells
    lie on the observation's grid, `first` being the index there of the first sample. A cell's mean that
    stands further from the channel's median cell mean than limit x sigma / sqrt(cell), what white noise
    would scatter it by, is brought back to that bound: the excess is taken off every sample of the cell.
    """
    nchan, nsamp = data.shape
    cell = int(cell)
    lead = (-int(first)) % cell
    cells = (nsamp - lead) // cell
    if cell < 1 or cells < 2:
        return data
    if sigma is None:
        sample = data[:, ::max(1, nsamp // 20000)]
        sigma = 1.4826 * np.nanmedian(np.abs(sample - np.nanmedian(sample, axis=1, keepdims=True)), axis=1)
    view = data[:, lead:lead + cells * cell].reshape(nchan, cells, cell)
    mean = view.mean(axis=2)                                       # a wholly masked (NaN) channel stays as it is
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        deviation = mean - np.nanmedian(mean, axis=1, keepdims=True)
    bound = (limit * np.asarray(sigma, dtype=float) / np.sqrt(cell))[:, None]
    with np.errstate(invalid='ignore'):
        excess = np.where(np.abs(deviation) > bound, deviation - np.sign(deviation) * bound, 0.0)
    view -= excess[:, :, None].astype(data.dtype)
    return data


def cap_slow(data, cap, sigma=None, first=0, samples_per_cell=1):
    """The slow cap of the search (settings preprocessing.slow_cap: {sigma, samples}) on (channel, time) data,
    in place; returns data. `samples_per_cell`: native samples in each of data's samples, for data already
    averaged in time (cells shorter than one of them are skipped).

    Measured on 2 October 2026 in 40 levelled LT5 beams (benchmarks/scattered-2026-10-02): after the RFI block
    mask, one to two percent of the one-second means of single channels stand 5 sigma or more from their
    channel's level, in nearly every channel and for up to 40 s. They carry most of the scatter of
    second-long sums of the dedispersed series, which is 1.5 times white noise's at 1 s under the 8-width
    baseline and 2.7 times at 4 s, so a scattered burst is found at half its ideal S/N. A dispersed burst is
    far from the cap: S/N s in w seconds over N channels puts s / sqrt(N) sigma in a w-second cell, 0.5 at
    S/N 12, and reaches 4 sigma only at S/N 100, where the cap leaves it S/N 100.
    """
    if not cap:
        return data
    limit = float(cap.get('sigma', 4.0))
    if sigma is None:
        nsamp = data.shape[1]
        sample = data[:, ::max(1, nsamp // 20000)]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            sigma = 1.4826 * np.nanmedian(np.abs(sample - np.nanmedian(sample, axis=1, keepdims=True)), axis=1)
    for samples in cap.get('samples') or SLOW_CAP_SAMPLES:
        cell = int(round(int(samples) / samples_per_cell))
        if cell >= 1:
            cap_cells(data, cell, limit, sigma, first)
    return data


def searchable(dm, preprocessing=None):
    """Whether a DM trial still carries signal after this preprocessing.

    After the zero-DM filter every sample's mean over channels is zero, so the
    DM 0 trial, their sum, is zero but for rounding. The matched filter scaled
    that residue, largest where bright interference had been subtracted, into
    about 230 events of S/N 7 to 30 per beam (pilot of 27 September 2026).
    Any other trial shifts the channels against each other and keeps its signal.
    """
    return not ((preprocessing or {}).get('zero_dm') and dm == 0)
