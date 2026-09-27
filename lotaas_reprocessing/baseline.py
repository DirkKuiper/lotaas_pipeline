"""The running baseline of a dedispersed series (numpy only: the web layer measures with it too)."""
import numpy as np


def running_baseline(series, window, start=0, stop=None):
    """The slow baseline of a dedispersed series, for samples start..stop.

    Medians of consecutive blocks of window/2 samples, counted from sample 0,
    linearly interpolated between block centres and held flat beyond the
    outermost ones. A block median ignores a pulse filling less than half of
    it, so a boxcar of width w keeps its signal when the window is >= 64 w.
    Blocks are aligned to sample 0 whatever stretch is requested, so a local
    stretch gets exactly the values the whole series gets there.
    """
    n = len(series)
    stop = n if stop is None else min(int(stop), n)
    start = max(0, int(start))
    block = max(1, int(window) // 2)
    blocks = n // block
    if blocks < 2:
        return np.full(stop - start, np.median(np.asarray(series)), dtype=np.float32)
    first = max(0, start // block - 1)
    last = min(blocks, (stop - 1) // block + 2)
    medians = np.median(np.asarray(series[first * block:last * block], dtype=np.float32)
                        .reshape(last - first, block), axis=1)
    centres = (np.arange(first, last) + 0.5) * block
    return np.interp(np.arange(start, stop), centres, medians).astype(np.float32)


def baseline_window(width, tsamp, downsample, baseline_seconds, baseline_widths=64):
    """Samples of running baseline removed before a boxcar of `width` trial samples."""
    return max(int(round(baseline_seconds / (tsamp * downsample))), int(baseline_widths) * int(width))
