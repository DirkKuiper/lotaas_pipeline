import json

import numpy as np
import pytest

from web import dynspec, sigproc
from web.tests.conftest import TSAMP, synthetic_filterbank


def snippet(tmp_path, **kwargs):
    path = synthetic_filterbank(tmp_path / 'fake.fil', **kwargs)
    meta = {'dm': kwargs.get('dm', 30.0), 't0_relative': -15.0, 'width_samples': 3, 'downsample': 1,
            'tsamp_native': TSAMP, 'bad_channels': []}
    path.with_suffix('.json').write_text(json.dumps(meta))
    return dynspec.Snippet(path)


def test_round_trip(tmp_path):
    path = synthetic_filterbank(tmp_path / 'a.fil', nsamp=100)
    header, data = sigproc.open_data(path)
    assert data.shape == (100, 64) and header['nbits'] == 32
    assert header['source_name'] == 'LOTAAS-P1254B-SAP0'
    assert np.isclose(header['tsamp'], TSAMP)


def test_dedispersion_recovers_the_pulse_at_its_dm(tmp_path):
    s = snippet(tmp_path)
    view = s.view(window=2.0, nsub=16)
    assert view['peak_snr'] > 8
    assert abs(view['times'][np.nanargmax(view['boxcar'])]) < 3 * TSAMP
    off = s.view(dm=0.0, window=2.0)
    assert off['peak_snr'] < view['peak_snr'] / 2


def test_dm_response_peaks_at_the_injected_dm(tmp_path):
    s = snippet(tmp_path)
    response = s.dm_response(points=61)
    assert abs(response['best_dm'] - 30.0) <= 2 * response['half_width_dm'] / 8 + 0.05
    assert response['width_samples'] in (2, 3, 4)
    fine = response['fine_snr']
    assert np.nanmax(fine) == pytest.approx(response['best_snr'])
    assert fine[0] < response['best_snr'] / 2


def test_undispersed_rfi_shows_at_dm_zero(tmp_path):
    s = snippet(tmp_path, amplitude=0.0, undispersed=15.5)
    response = s.dm_response(points=41)
    assert response['coarse_snr'][0] > 10


def test_expected_fraction():
    assert dynspec.expected_fraction(0.0, 8, 31.6, 0.135)[0] == 1.0
    half = dynspec.half_width_dm(8, 31.6, 0.135)
    assert dynspec.expected_fraction(half, 8, 31.6, 0.135)[0] == pytest.approx(0.5, abs=0.01)


def test_mask_parsing():
    assert dynspec.parse_mask('3-5, 9,bad,700', 648) == [3, 4, 5, 9]
    assert dynspec.parse_mask('', 648) == []


def test_smearing_matches_the_standard_formula():
    # 8.3 us DM dnu / nu^3: DM 131.6, 48.8 kHz at 119.45 MHz is about 31 ms.
    assert dynspec.channel_smearing_ms(131.6, 119.45, 0.048828) == pytest.approx(31.3, abs=0.2)
