"""Injected FRB-like bursts (frb_injection): where they land, and that the pipeline's own measure reads their S/N."""
import json

import numpy as np

from lotaas_reprocessing import frb_injection, own_data, sigproc_data
from test_own_data import FCH1, FOFF, NCHANS, PLAN, TSAMP, write_filterbank


def test_bursts_fit_the_observation_and_keep_apart():
    freqs = FCH1 + np.arange(NCHANS) * FOFF
    bursts = frb_injection.draw(np.random.default_rng(3), 3600.0, freqs, 10)
    assert len(bursts) == 10 and {b['spectrum'] for b in bursts} <= set(frb_injection.SPECTRA)
    times = [b['t0'] for b in bursts]
    assert min(np.diff(times)) >= frb_injection.SPACING_SECONDS and times[0] >= frb_injection.EDGE_SECONDS
    assert all(b['t0'] + frb_injection.sweep_seconds(b['dm'], freqs) < 3600.0 for b in bursts)


def test_an_injected_burst_reads_its_drawn_snr_on_its_own_data(tmp_path):
    source = write_filterbank(tmp_path / 'beam.fil', nsamp=24000)
    header, mapped = sigproc_data.open_data(source)
    data = np.array(mapped, dtype=np.float32)
    ranges = {'dm': (150.0, 200.0), 'tau135': (1e-3, 2e-3), 'snr': (25.0, 26.0), 'width': (1e-3, 2e-3)}
    truth = frb_injection.inject(data, header, [], np.random.default_rng(1), n=2, ranges=ranges)
    twin = tmp_path / 'twin.fil'
    sigproc_data.write(twin, header, data)
    assert np.array_equal(np.asarray(sigproc_data.open_data(twin)[1]), data)
    for b in truth:
        width = max(1, round(b['ideal_width_s'] / TSAMP))
        measured = own_data.measure(twin, b['dm'], b['peak_time'], width, PLAN, (), 2.0)
        assert 0.7 * b['snr_ideal'] < measured < 1.3 * b['snr_ideal'], (b, measured)
        assert b['fluence_units'] > 0 and b['good_channels'] == NCHANS


def test_the_command_line_writes_the_twin_and_its_truth(tmp_path):
    source = write_filterbank(tmp_path / 'beam.fil', nsamp=24000)
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps([3, 7]))
    # The default DM range reaches 3000, whose sweep outlasts this three-minute beam: none may fit.
    frb_injection.main([str(source), str(tmp_path / 'twin.fil'), str(tmp_path / 'truth.json'), '5', str(bad), '3'])
    truth = json.loads((tmp_path / 'truth.json').read_text())
    assert truth['twin'] == 'twin.fil' and truth['bad_channels'] == [3, 7] and (tmp_path / 'twin.fil').exists()
    assert all(b['good_channels'] == NCHANS - 2 for b in truth['bursts'])


def test_with_the_pipeline_settings_the_search_s_channel_mask_is_used(tmp_path):
    source = write_filterbank(tmp_path / 'beam.fil', nsamp=24000)
    settings = tmp_path / 'settings.yaml'
    settings.write_text('bad_channels: [2]\npersistent_channel_threshold: 4.0\n')
    frb_injection.main([str(source), str(tmp_path / 'twin.fil'), str(tmp_path / 'truth.json'), '5', str(settings), '1'])
    assert json.loads((tmp_path / 'truth.json').read_text())['bad_channels'] == [2]


def test_a_twin_of_a_row_levelled_beam_is_levelled_again_as_the_flatfield_levelled_real_bursts(tmp_path):
    from test_own_data import levelled_beam
    source = levelled_beam(tmp_path / 'beam.fil', (0.0, 0.0, 0, 0.0))            # levelled rows of 32, no pulse
    header, mapped = sigproc_data.open_data(source)
    before = np.array(mapped, dtype=np.float32)
    frb_injection.level_rows(again := before.copy(), 32)
    assert np.allclose(again, before, atol=1e-4), 'a levelled beam does not change'
    burst = before.copy()
    burst[3200:3264, :] += 1.0                                                     # two whole rows, one part-row
    burst[3300:3310, :] += 1.0
    frb_injection.level_rows(burst, 32)
    rows = burst[:19968].reshape(-1, 32, NCHANS).mean(axis=1)
    assert np.ptp(rows, axis=0).max() < 1e-3, 'every row back at its level'
    assert np.allclose(burst[3200:3264], before[3200:3264], atol=1e-4), 'a burst filling rows is gone, as in the real beam'
    assert np.isclose((burst[3300:3310] - before[3300:3310]).mean(), 1 - 10 / 32, atol=1e-3), 'a shorter one keeps 22/32'
    (tmp_path / 'bad.json').write_text('[]')
    frb_injection.main([str(source), str(tmp_path / 'twin.fil'), str(tmp_path / 'truth.json'), '5', str(tmp_path / 'bad.json'), '0', '32'])
    assert json.loads((tmp_path / 'truth.json').read_text())['level_rows'] == 32
