import json
import numpy as np
import pytest

from lotaas_reprocessing.single_pulse_quality import (
    persistent_channels, local_boxcar_snr, review_route, measure_clusters, candidate_key)


def test_persistent_mask_ignores_short_bright_band_limited_bursts():
    x = np.random.default_rng(4).normal(size=(64, 20000)).astype('float32')
    x[17] *= 12
    x[38] = 0
    x[40:55, 9000:9500] += 100  # bright pulse, only part of the band
    assert persistent_channels(x) == [17, 38]
    assert persistent_channels(x[::-1]) == [25, 46]


def test_local_noise_demotes_a_globally_significant_gain_excursion():
    x = np.random.default_rng(10).normal(size=60000)
    x[28000:32000] *= 6
    x[29998:30002] += 8
    # A global white-noise estimate calls this significant; its neighbourhood does not.
    assert x[29998:30002].sum() / 2 > 7
    check = local_boxcar_snr(x, 29999.5, 4, radius=1500)
    assert check['local_snr'] < 5
    assert review_route(2., check, {'min_local_snr': 5, 'max_width_seconds': 1}, 0) == 'unconfirmed'


@pytest.mark.parametrize('width,band_fraction', [(0.25, 1.), (1., 1.), (5., .25), (30., .5)])
def test_high_dm_scattered_bursts_survive_channel_and_local_checks(width, band_fraction):
    from lotaas_reprocessing.dedispersion import iter_dedispersed
    dt, dm, arrival = .05, 3000., 150.
    freqs = np.linspace(119.5, 151., 64)
    t = np.arange(16000) * dt
    rng = np.random.default_rng(7)
    x = rng.normal(size=(64, len(t))).astype('float32')
    delays = dm * (freqs ** -2 - freqs.max() ** -2) / 2.41e-4
    channels = int(64 * band_fraction)
    amplitude = 30 / np.sqrt(channels * width / dt)
    for c in range(channels):
        relative = t - arrival - delays[c]
        tau = width * (freqs[c] / 135.) ** -4
        x[c] += amplitude * np.exp(-np.maximum(relative, 0) / tau) * (relative >= 0)
    x[55] += rng.normal(0, 15, len(t))
    bad = persistent_channels(x)
    assert bad == [55]
    x[bad] = 0
    _, trial = next(iter_dedispersed(x, dt, freqs, [dm]))
    valid = trial[:len(trial) - int(np.ceil(delays.max() / dt))]
    w = max(1, round(width / dt))
    check = local_boxcar_snr(valid, arrival / dt + (w - 1) / 2, w, radius=round(10 / dt))
    assert check['local_snr'] is None or check['local_snr'] > 7
    route = review_route(width, check, {'min_local_snr': 5, 'max_width_seconds': 1}, 0)
    assert route == ('unclassified' if width > 1 else None)


def test_insufficient_noise_is_unknown_and_does_not_label_a_candidate_as_noise():
    check = local_boxcar_snr(np.ones(100), 50, 20)
    assert check['local_snr'] is None
    assert review_route(30, check, {'min_local_snr': 5, 'max_width_seconds': 1}, 200) == 'unclassified'


def test_evidence_reads_real_cluster_format_and_does_not_use_wrap_tail(tmp_path):
    trials = tmp_path / 'DM_trials'
    trials.mkdir()
    x = np.random.default_rng(19).normal(size=10000).astype('float32')
    x[4998:5002] += 8
    x[-3000:] = 1e8
    x.tofile(trials / 'beam_DM1000.0.dat')
    (tmp_path / 'clustered_candidates.txt').write_text(
        'DM S/N Time Sample Filter_Width DM_scaled Cluster\n1000 14 249.975 49995 40 20 0\n')
    meta = {'filename': 'beam.fil', 'tsamp': .005, 'nu_min': 120., 'nu_max': 150.,
            'dedispersion_plan': [{'low_dm': 999, 'high_dm': 1001, 'downsample': 10}]}
    result = measure_clusters(tmp_path, meta)
    row = result[candidate_key(1000, 249.975, 40)]
    assert row['local_snr'] > 10 and row['width_seconds'] == .2
    json.dumps(result, allow_nan=False)


def test_empty_cluster_file_is_valid(tmp_path):
    (tmp_path / 'clustered_candidates.txt').write_text('DM S/N Time Sample Filter_Width\n')
    assert measure_clusters(tmp_path, {}) == {}


@pytest.mark.parametrize('ds', [1, 2, 4, 8, 16, 32, 128])
def test_search_never_uses_a_template_above_one_second(tmp_path, ds):
    from lotaas_reprocessing.matched_filter import filter_widths_for, run_matched_filtering
    dt = .00786432
    widths = filter_widths_for(4000, dt, ds, max_duration=1.)
    assert np.all(widths * dt * ds <= 1.)
    x = np.random.default_rng(ds).normal(size=4000).astype('float32')
    x[1000:2000] += 15  # an obvious, very broad event must not generate a broad template
    path = tmp_path / 'trial.dat'
    x.tofile(path)
    found = run_matched_filtering(path, dt, 1000., downsample=ds, max_duration=1.)
    assert np.all(found[3] * dt * ds <= 1.)
    if dt * ds > 1.:
        assert not widths.size and not found[3].size


def test_one_second_cap_flows_through_production_cpu_stage(tmp_path):
    from pipeline.pipeline_cpu import single_pulse
    trials = tmp_path / 'DM_trials'
    trials.mkdir()
    x = np.random.default_rng(27).normal(size=10000).astype('float32')
    x[1998:2003] += 12
    x.tofile(trials / 'beam_DM1000.0.dat')
    plan = [{'low_dm': 1000., 'high_dm': 1001., 'ddm': 1., 'downsample': 1}]
    meta = {'filename': 'beam.fil', 'tsamp': .1, 'samples_processed': len(x),
            'single_pulse': {'max_width_seconds': 1.}, 'dedispersion_plan': plan,
            'nu_min': 120., 'nu_max': 150., 'observation_info': {
                'Object': 'synthetic', 'Telescope': 'synthetic', 'Instrument': 'synthetic',
                'Observation Date': 'synthetic', 'Frequency Range (MHz)': '120-150'}}
    # validate_trials reads each payload's size, not its .inf header.
    single_pulse(tmp_path, meta)
    summary = json.loads((tmp_path / 'single_pulse_summary.json').read_text())
    assert summary['max_width_seconds'] == 1.
    evidence = json.loads((tmp_path / 'single_pulse_evidence.json').read_text())
    assert evidence and max(r['width_seconds'] for r in evidence.values()) <= 1.
    assert max(r['local_snr'] for r in evidence.values() if r['local_snr'] is not None) > 10
