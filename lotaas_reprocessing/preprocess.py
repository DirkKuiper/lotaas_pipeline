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


def searchable(dm, preprocessing=None):
    """Whether a DM trial still carries signal after this preprocessing.

    After the zero-DM filter every sample's mean over channels is zero, so the
    DM 0 trial, their sum, is zero but for rounding. The matched filter scaled
    that residue, largest where bright interference had been subtracted, into
    about 230 events of S/N 7 to 30 per beam (pilot of 27 September 2026).
    Any other trial shifts the channels against each other and keeps its signal.
    """
    return not ((preprocessing or {}).get('zero_dm') and dm == 0)
