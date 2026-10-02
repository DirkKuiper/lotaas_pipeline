"""The data before dedispersion: masked cells at the channel's own level, and the zero-DM filter."""
import numpy as np
import pytest

from lotaas_reprocessing.dedispersion import iter_dedispersed
from lotaas_reprocessing.preprocess import cap_cells, cap_slow, fill_masked, local_levels, searchable, zero_dm


def drifting(nchan=16, nsamp=20000, seed=3):
    """Unit noise on slow per-channel levels of a few sigma, as tied-array beams drift."""
    rng = np.random.default_rng(seed)
    t = np.arange(nsamp)
    levels = 3 * np.sin(2 * np.pi * t[None, :] / rng.uniform(2e4, 6e4, (nchan, 1)) + rng.uniform(0, 6, (nchan, 1)))
    return (levels + rng.standard_normal((nchan, nsamp))).astype(np.float32)


def step(series, lo, hi):
    """|mean inside lo:hi - mean of the same length just before|, in units of that difference's noise."""
    inside, before = series[lo:hi], series[2 * lo - hi:lo]
    noise = np.std(series[:lo - (hi - lo)]) * np.sqrt(2 / (hi - lo))
    return abs(inside.mean() - before.mean()) / noise


@pytest.mark.parametrize('mode, stepped', [('global', True), ('local', False)])
def test_a_block_filled_at_its_channels_level_leaves_no_step(mode, stepped):
    # 27 September: a 7.9 s block of 33 channels filled at the global mean lifted the dedispersed
    # series by 6 units for the whole block, and its brightest sample made an S/N 7 event.
    data = drifting()
    mask = np.zeros(data.shape, dtype=bool)
    mask[:4, 12000:13000] = True
    masked = np.where(mask, np.nan, data)
    filled = fill_masked(masked, mask, 1000, np.random.default_rng(0), mode)
    assert np.isfinite(filled).all()
    assert (step(filled.sum(axis=0), 12000, 13000) > 10) == stepped


def test_the_global_fill_is_the_fill_before_28_september():
    data = drifting(4, 3000)
    mask = np.zeros(data.shape, dtype=bool)
    mask[1, 100:900] = True
    masked = np.where(mask, np.nan, data)
    expected = masked.copy()
    expected[mask] = np.random.default_rng(42).normal(np.nanmean(masked), np.nanstd(masked), int(mask.sum()))
    np.testing.assert_array_equal(fill_masked(masked.copy(), mask, 1000, np.random.default_rng(42), 'global'), expected)
    with pytest.raises(ValueError, match='mask_fill'):
        fill_masked(masked.copy(), mask, 1000, np.random.default_rng(0), 'nearest')


def test_a_wholly_masked_channel_is_filled_at_zero_and_levels_follow_the_blocks():
    data = drifting(4, 5000)
    mask = np.zeros(data.shape, dtype=bool)
    mask[2] = True
    masked = np.where(mask, np.nan, data)
    levels = local_levels(masked, 1000)
    assert np.all(levels[2] == 0)
    assert abs(levels[0, 2500] - np.median(data[0, 2000:3000])) < 0.1
    filled = fill_masked(masked, mask, 1000, np.random.default_rng(0), 'local')
    assert abs(filled[2].mean()) < 0.1 and 0.5 < filled[2].std() < 1.5


def boxcar_snr(series, centre, width, off):
    """Boxcar S/N at centre against the boxcar sums away from it."""
    sums = np.convolve(series, np.ones(width), 'valid')
    mask = np.ones(sums.size, dtype=bool)
    mask[max(0, centre - off):centre + off] = False
    return (sums[centre - width // 2] - np.median(sums[mask])) / (1.4826 * np.median(np.abs(sums[mask] - np.median(sums[mask]))))


def test_zero_dm_removes_what_arrives_at_every_frequency_at_once_and_keeps_a_dispersed_pulse():
    rng = np.random.default_rng(1)
    tsamp, nchan, nsamp = 0.0078643, 64, 16384
    freqs = np.linspace(120.0, 151.0, nchan)
    data = rng.standard_normal((nchan, nsamp)).astype(np.float32)
    data[:, 4000:4003] += 1.5                                       # undispersed: interference
    delays = np.round(30.0 * (freqs ** -2 - freqs.max() ** -2) / 2.41e-4 / tsamp).astype(int)
    for c in range(nchan):                                          # a pulse at DM 30
        data[c, 9000 + delays[c]:9000 + delays[c] + 3] += 1.5
    # Undispersed, the burst is seen near DM 0 (at exactly 0 the filtered sum vanishes), smeared over ~20 samples at 1.5.
    before = {dm: s for dm, s in iter_dedispersed(data, tsamp, freqs, [1.5, 30.0])}
    after = {dm: s for dm, s in iter_dedispersed(zero_dm(data.copy()), tsamp, freqs, [1.5, 30.0])}
    assert boxcar_snr(before[1.5], 3991, 20, 80) > 5
    assert boxcar_snr(after[1.5], 3991, 20, 80) < 3
    assert boxcar_snr(after[30.0], 9001, 3, 50) > 0.9 * boxcar_snr(before[30.0], 9001, 3, 50)


def test_after_the_zero_dm_filter_the_dm_zero_trial_holds_only_rounding_and_is_not_searched():
    rng = np.random.default_rng(3)
    nu = np.linspace(119.45, 151.04, 64)
    data = rng.normal(0, 1, (64, 8192)).astype(np.float32)
    data[:, 4000:4003] += 400.                            # bright broadband interference
    zero_dm(data)
    trials = dict(iter_dedispersed(data, .00786, nu, [0.0, 0.5]))
    at_zero, dispersed = (np.abs(np.asarray(trials[dm])).max() for dm in (0.0, 0.5))
    assert at_zero < 1e-3 * dispersed, 'the DM 0 trial should be zero but for rounding'
    assert not searchable(0.0, {'zero_dm': True})
    assert searchable(0.0, {}) and searchable(0.0, None) and searchable(0.5, {'zero_dm': True})


def test_the_slow_cap_takes_a_channel_s_long_excess_and_leaves_noise_and_the_other_channels():
    rng = np.random.default_rng(5)
    data = rng.standard_normal((8, 4096)).astype(np.float32)
    data[3, 1024:1088] += 1.0                               # 64 samples at 1 sigma: 8 sigma for a 64-sample mean
    before = data.copy()
    cap_cells(data, 64, 4.0, np.ones(8))
    cell = data[3, 1024:1088].mean() - np.median(data[3].reshape(-1, 64).mean(axis=1))
    assert abs(cell - 4.0 / 8) < 1e-3                       # held at 4 sigma of such a mean
    assert np.array_equal(np.delete(data, 3, axis=0), np.delete(before, 3, axis=0))
    assert np.array_equal(data[3, :1024], before[3, :1024]) and np.array_equal(data[3, 1088:], before[3, 1088:])
    # Cells lie on the observation's grid: the same samples are a cell for a stretch starting anywhere.
    part = before[:, 1000:].copy()
    cap_cells(part, 64, 4.0, np.ones(8), first=1000)
    assert np.allclose(part[3, 24:88], data[3, 1024:1088], atol=0.05)


def test_the_slow_cap_leaves_a_dispersed_burst_and_holds_a_very_bright_one_far_above_any_threshold():
    rng = np.random.default_rng(6)
    nchan, cell = 64, 128

    def summed(snr):
        """The S/N of a 128-sample burst in every channel, summed over the band, after the cap."""
        data = rng.standard_normal((nchan, 8192)).astype(np.float32)
        data[:, 2048:2048 + cell] += snr / np.sqrt(nchan * cell)
        cap_slow(data, {'sigma': 4.0, 'samples': [32, 64, 128, 256]})
        series = data.sum(axis=0)
        return (series[2048:2048 + cell].sum() - np.median(series) * cell) / np.sqrt(nchan * cell)
    assert abs(summed(12.0) - 12.0) < 1.5                   # 1.5 sigma per channel cell: untouched
    assert summed(300.0) > 4.0 * np.sqrt(nchan) - 2         # 37 sigma per cell: held at 4, S/N 32 here (100 in 647 channels)
    untouched = rng.standard_normal((4, 1000)).astype(np.float32)
    assert cap_slow(untouched.copy(), None).tobytes() == untouched.tobytes()
    # Data already averaged over 32 native samples: cells shorter than one of its samples are skipped.
    assert cap_slow(untouched.copy(), {'sigma': 4.0, 'samples': [8]}, samples_per_cell=32).tobytes() == untouched.tobytes()
