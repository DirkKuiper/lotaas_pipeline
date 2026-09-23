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
