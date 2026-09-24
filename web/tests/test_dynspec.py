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


def test_measurements_do_not_depend_on_display_averaging_or_zoom(tmp_path):
    s = snippet(tmp_path)
    small = s.view(window=2., nsub=16)
    whole = s.view(window=None, max_columns=100)
    averaged = s.view(tscrunch=8, window=5., nsub=8)
    assert whole['tsamp'] > small['tsamp']
    for v in (whole, averaged):
        assert v['peak_snr'] == small['peak_snr']
        assert v['best_snr'] == small['best_snr']
        assert v['best_width'] == small['best_width']


def test_whole_view_excludes_the_missing_band_tail(tmp_path):
    s = snippet(tmp_path)
    v = s.view(window=None, max_columns=100000)
    assert v['times'][-1] <= s.times[-1] - dynspec.sweep_seconds(s.dm, s.freqs).max() + s.tsamp


def test_broad_band_limited_pulse_is_preserved_and_window_fits_it(tmp_path):
    s = snippet(tmp_path, width=120, amplitude=0., nsamp=20000)
    s.meta['width_samples'] = 120
    # Write a dispersed ~1 s pulse in one quarter of the band.
    data = s.data.copy()
    for c, delay in enumerate(dynspec.delays(s.dm, s.freqs, s.tsamp)):
        if 16 <= c < 32:
            start = int(round(-s.t0 / s.tsamp)) - 60 + delay
            data[start:start + 120, c] += .7
    sigproc.write(s.path, s.header, data)
    s.path.with_suffix('.json').write_text(json.dumps(s.meta))
    s = dynspec.Snippet(s.path)
    assert not s.automatic_bad
    assert s.default_window > 7
    assert s.view()['peak_snr'] > 7


@pytest.mark.parametrize('snr,nsub', [(7, 6), (8, 8), (10, 12), (15, 36), (30, 81), (None, 81)])
def test_suggested_view_keeps_the_pulse_visible(snr, nsub):
    assert dynspec.suggested_view(snr, 3, 648) == (nsub, 3)
    if snr:
        assert dynspec.pixel_snr(snr, nsub, 3, 3) >= dynspec.VISIBLE_SIGMA or nsub == 4


def test_pixel_snr_spreads_over_subbands_and_unbinned_columns():
    assert dynspec.pixel_snr(9, 81, 1, 1) == pytest.approx(1.0)
    assert dynspec.pixel_snr(9, 9, 1, 4) == pytest.approx(1.5)
    assert dynspec.pixel_snr(9, 9, 4, 4) == pytest.approx(3.0)


def test_view_reports_profiles_pixel_snr_and_the_unmasked_snr(tmp_path):
    s = snippet(tmp_path)
    plain = s.view(window=2.0, nsub=8)
    assert plain['profiles'].shape == (4, len(plain['times']))
    assert plain['unmasked_peak_snr'] == plain['peak_snr']
    assert plain['pixel_snr'] == pytest.approx(dynspec.pixel_snr(plain['peak_snr'], 8, 1, s.width))
    masked = s.view(window=2.0, nsub=8, mask=[5, 6, 7])
    assert masked['unmasked_peak_snr'] == pytest.approx(plain['peak_snr'])


def test_masking_channels_picked_at_the_pulse_inflates_noise(tmp_path):
    """Why the viewer shows the S/N without the reviewer's mask beside it."""
    s = snippet(tmp_path, amplitude=0.0, nsamp=6000)
    at = int(round(-s.t0 / s.tsamp))
    aligned = dynspec.dedisperse(s.normalised, s.freqs, s.tsamp, s.dm)
    on = np.nanmean(aligned[at - 1:at + 2], axis=0)
    lowest = np.argsort(on)[:len(on) // 8].tolist()
    view = s.view(window=2.0, mask=lowest, auto_mask=False)
    assert view['peak_snr'] > view['unmasked_peak_snr'] + 1.0
