"""The DM-time plane FETCH sees must not change when only its crop is computed."""
import numpy as np
import pytest

your_candidate = pytest.importorskip('your.candidate')
from lotaas_reprocessing.classify import dm_time_plane


class ArrayCandidate(your_candidate.Candidate):
    """A your Candidate over an in-memory chunk, without a file."""

    def __init__(self, data, frequencies, tsamp, dm):
        self.data, self._frequencies, self._tsamp, self.dm = data, frequencies, tsamp, dm
        self.dmt = None

    chan_freqs = property(lambda self: self._frequencies)
    native_tsamp = property(lambda self: self._tsamp)


def reference(candidate, decimate):
    candidate.dmtime(dmsteps=256)
    candidate.decimate(key='dmt', axis=1, pad=True, decimate_factor=decimate, mode='median')
    return your_candidate.crop(candidate.dmt, candidate.dmt.shape[1] // 2 - 128, 256, axis=1)


@pytest.mark.parametrize('dm,decimate,samples', [(30., 1, 20000), (300., 1, 20000), (300., 4, 20001),
                                                  (300., 32, 20000), (12., 1, 2048), (5.5, 1, 20000)])
def test_the_cropped_plane_is_bit_identical(dm, decimate, samples):
    rng = np.random.default_rng(int(dm * 10) + decimate)
    frequencies = np.linspace(151.0, 119.5, 64)
    data = rng.normal(size=(samples, 64)).astype(np.float32)
    expected = reference(ArrayCandidate(data.copy(), frequencies, 0.0078643, dm), decimate)
    actual = dm_time_plane(ArrayCandidate(data.copy(), frequencies, 0.0078643, dm), decimate)
    assert actual.shape == expected.shape == (256, 256)
    assert np.array_equal(actual, expected)


def test_a_crop_reaching_the_decimation_padding_uses_the_original_path():
    rng = np.random.default_rng(1)
    frequencies = np.linspace(151.0, 119.5, 16)
    data = rng.normal(size=(256 * 3 + 1, 16)).astype(np.float32)   # decimated length 257
    expected = reference(ArrayCandidate(data.copy(), frequencies, 0.0078643, 20.), 3)
    assert np.array_equal(dm_time_plane(ArrayCandidate(data.copy(), frequencies, 0.0078643, 20.), 3), expected)


# FETCH's inputs with a width-scaled DM-time range (fetch_bowtie) and a cleaned chunk (fetch_clean).
from lotaas_reprocessing import classify  # noqa: E402

LOTAAS = np.linspace(151.04, 119.45, 648)
TSAMP = 0.0078643


def test_without_a_bowtie_the_range_is_the_old_five_dm():
    assert classify.fetch_range_dm(LOTAAS, TSAMP, 32, None) == 5.0
    assert classify.fetch_range_dm(LOTAAS, TSAMP, 32, 0) == 5.0


def test_the_bowtie_range_follows_the_width_not_the_dm():
    sweep = classify.K_DM * (119.45 ** -2 - 151.04 ** -2)
    # A 252 ms boxcar: 126 ms pixels, 64 of them half-way to the corner.
    assert classify.fetch_range_dm(LOTAAS, TSAMP, 32, 0.5) == pytest.approx(64 * 16 * TSAMP / sweep)
    assert 70 < classify.fetch_range_dm(LOTAAS, TSAMP, 32, 0.5) < 80
    # A one-sample pulse keeps the old +-5 DM floor; the arms then travel further than asked.
    assert classify.fetch_range_dm(LOTAAS, TSAMP, 1, 0.5) == 5.0
    assert classify.fetch_range_dm(LOTAAS, TSAMP, 64, 1.0) == pytest.approx(2 * classify.fetch_range_dm(LOTAAS, TSAMP, 64, 0.5))


def test_the_bowtie_arms_reach_the_asked_share_of_the_half_plane():
    """A delta pulse's arrival at the band's edges at the plane's top row, in pixels from the centre."""
    width, bowtie = 24, 0.5
    r = classify.fetch_range_dm(LOTAAS, TSAMP, width, bowtie)
    sweep = classify.K_DM * (LOTAAS.min() ** -2 - LOTAAS.max() ** -2)
    assert r * sweep / ((width // 2) * TSAMP) == pytest.approx(bowtie * 128)


def test_a_cleaned_chunk_has_channels_on_one_scale():
    rng = np.random.default_rng(3)
    nt, nf = 20000, 64
    data = (100 + rng.normal(size=(nt, nf)) * np.linspace(1, 20, nf)).astype(np.float32)   # 1x to 20x noisier
    data[5000, 10] = 1e6                                                    # one cell no pulse reaches
    good = np.ones(nf, dtype=bool)
    good[[3, 40]] = False
    z = classify.clean_chunk(data, good)
    assert z.dtype == np.float32
    assert np.all(z[:, ~good] == 0)
    assert abs(z[5000, 10]) < classify.FETCH_MAX_CELL_SIGMA
    spread = z[:, good].std(axis=0)
    assert spread.max() / spread.min() < 1.2


def test_a_cleaned_chunk_has_no_broadband_wander():
    rng = np.random.default_rng(5)
    nt, nf = 20000, 64
    wander = np.cumsum(rng.normal(size=nt))[:, None] * 0.5                  # common to every channel
    data = (100 + rng.normal(size=(nt, nf)) + wander).astype(np.float32)
    good = np.ones(nf, dtype=bool)
    good[7] = False
    z = classify.clean_chunk(data, good)
    assert np.allclose(z[:, good].mean(axis=1), 0, atol=1e-4)             # the zero-DM filter
    band = z[:, good].sum(axis=1)
    assert np.abs(np.corrcoef(band[1:], np.diff(wander[:, 0]))[0, 1]) < 0.05


def test_fetch_inputs_is_unchanged_without_the_new_settings(tmp_path, monkeypatch):
    """With neither setting, the planes are those FETCH was given before 30 September 2026."""
    rng = np.random.default_rng(4)
    data = (50 + rng.normal(size=(6000, 64)) * np.linspace(1, 5, 64)).astype(np.float32)

    class FakeCandidate(ArrayCandidate):
        def __init__(self, fp, dm, tcand, width, label, snr, min_samp, device):
            super().__init__(data.copy(), np.linspace(151.0, 119.5, 64), TSAMP, dm)
            self.tcand, self.width, self.snr, self.tsamp = tcand, width, snr, TSAMP
            self.dedispersed = None

        def get_chunk(self):
            pass

    monkeypatch.setattr(classify, 'Candidate', FakeCandidate)
    before = []
    for kwargs in ({}, {'bowtie': None, 'clean': False}):
        cand, X, Y, dec = classify.fetch_inputs('x.fil', 40.0, 20.0, 8, 12.0, [2, 7], **kwargs)
        before.append((X, Y))
        assert cand.fetch_range_dm == 5.0 and dec == 4
    assert np.array_equal(before[0][0], before[1][0]) and np.array_equal(before[0][1], before[1][1])
    cand, X, Y, _ = classify.fetch_inputs('x.fil', 40.0, 20.0, 8, 12.0, [2, 7], bowtie=0.5, clean=True)
    assert cand.fetch_range_dm > 5.0
    assert X.shape == Y.shape == (1, 256, 256, 1) and np.isfinite(X).all() and np.isfinite(Y).all()
    assert not np.array_equal(Y, before[0][1])


def test_the_running_baseline_removes_a_level_bar_but_keeps_a_pulse():
    rng = np.random.default_rng(6)
    nt, nf, width = 12000, 32, 16
    data = (100 + rng.normal(size=(nt, nf))).astype(np.float32)
    data[2000:6000, 5] += 3.0                          # one channel high for 4000 samples (a level bar)
    data[9000:9016, 16:24] += 4.0                      # a 16-sample pulse in a quarter of the band
    good = np.ones(nf, dtype=bool)
    plain = classify.clean_chunk(data, good)
    z = classify.clean_chunk(data, good, baseline_pixels=64, pixel=width // 2)
    assert plain[3000:5000, 5].mean() > 0.8            # left in by the cleaning alone
    assert abs(z[3000:5000, 5].mean()) < 0.1           # gone away from its edges
    assert plain[9000:9016, 16:24].mean() > 2
    assert z[9000:9016, 16:24].mean() > 0.9 * plain[9000:9016, 16:24].mean()
    assert np.all(z[:, ~good] == 0)


def test_the_rfi_mask_zeroes_a_channel_block_the_search_would_flag():
    rng = np.random.default_rng(7)
    nt, nf = 8000, 32
    data = (100 + rng.normal(size=(nt, nf))).astype(np.float32)
    data[3000:4000:7, 9] += 30.0                       # spiky interference below the 50-sigma cell clip
    good = np.ones(nf, dtype=bool)
    plain = classify.clean_chunk(data, good)
    z = classify.clean_chunk(data, good, rfi_mask=True)
    assert plain[3000:4000:7, 9].mean() > 5            # left in without the mask
    assert z[3000:4000, 9].std() < 0.3                 # zeroed: only minus each row's mean over 32 channels
    assert z[5000:6000, 9].std() > 0.8                 # an unflagged block keeps its noise
    assert np.array_equal(z[:3000, 9] != 0, plain[:3000, 9] != 0)               # other blocks untouched
