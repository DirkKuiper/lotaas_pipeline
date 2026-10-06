import json
import math

import numpy as np

from lotaas_reprocessing import catalogue_fold
from lotaas_reprocessing.catalogue_fold import fold_catalogue, fold_pulsar, log10_chi2_sf
from lotaas_reprocessing.trials import trial_specs

TSAMP = 0.00786432
SAMPLES = 1 << 18          # 34 minutes


def pulsar_series(period, amplitude, offset=0.0, dm_smear=0.0, seed=1, n=SAMPLES, dt=TSAMP, width=0.02):
    """Noise with a pulse train whose topocentric period is the catalogue period times 1 + offset."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    data = rng.normal(size=n)
    true = period * (1 + offset)
    phase = np.remainder(t / true, 1.0)
    data += amplitude * np.exp(-0.5 * (np.minimum(phase, 1 - phase) * true / (width + dm_smear)) ** 2)
    data += 3.0 * np.sin(2 * np.pi * t / 600.0)           # a slow drift the high-pass must remove
    return data.astype('float32')


def single(series, dm=26.8, dt=TSAMP):
    return [(dm, 'trial')], (lambda name: series), (lambda name: dt)


def test_a_pulsar_off_its_catalogue_period_by_earths_motion_is_seen():
    psr = {'name': 'J0332+5434', 'dm': 26.8, 'period_seconds': 0.714520, 'separation_deg': 0.1}
    record = fold_pulsar(psr, *single(pulsar_series(0.714520, 0.25, offset=8e-5)))
    assert record['status'] == 'folded' and record['seen']
    assert abs(record['period_offset'] - 8e-5) < 2e-5
    assert record['log10p_corrected'] < -3 and record['log10p'] < min(record['control_log10p'])
    assert len(record['profile']) == record['bins'] == 64


def test_noise_and_a_slow_drift_are_not_a_pulsar():
    psr = {'name': 'J0332+5434', 'dm': 26.8, 'period_seconds': 0.714520, 'separation_deg': 0.1}
    record = fold_pulsar(psr, *single(pulsar_series(0.714520, 0.0, seed=7)))
    assert record['status'] == 'folded' and not record['seen']
    assert record['log10p_corrected'] > -3


def test_the_best_dm_trial_is_found_near_the_catalogue_dm():
    psr = {'name': 'J0332+5434', 'dm': 26.8, 'period_seconds': 0.714520, 'separation_deg': 0.1}
    series = {26.8: pulsar_series(0.714520, 0.1, dm_smear=0.03, seed=2),        # same fluence, smeared
              27.3: pulsar_series(0.714520, 0.25, seed=2), 40.0: pulsar_series(0.714520, 0.0, seed=3)}
    trials = [(dm, f'DM{dm}') for dm in series]
    record = fold_pulsar(psr, trials, lambda name: series[float(name[2:])], lambda name: TSAMP)
    assert record['seen'] and record['trial_dm'] == 27.3


def test_a_pulsar_too_fast_for_the_sampling_is_listed_not_folded():
    msp = {'name': 'J1939+2134', 'dm': 71.0, 'period_seconds': 0.00155, 'separation_deg': 0.3}
    assert fold_pulsar(msp, *single(np.zeros(1000, 'float32'), dm=71.0))['status'].startswith('too fast')
    far = {'name': 'J1000+0000', 'dm': 300.0, 'period_seconds': 1.0, 'separation_deg': 0.3}
    assert fold_pulsar(far, *single(np.zeros(1000, 'float32'), dm=26.8))['status'] == 'no trial within the DM range'


def test_the_chi2_tail_stays_finite_far_beyond_double_precision():
    assert math.isclose(log10_chi2_sf(63.0, 63), math.log10(0.4876), abs_tol=0.03)
    assert log10_chi2_sf(20000.0, 63) < -1000


def test_fold_catalogue_reads_the_periodic_trials_and_writes_its_product(tmp_path):
    plan = [{'low_dm': 26.0, 'high_dm': 28.0, 'ddm': 0.5, 'downsample': 1}]
    metadata = {'filename': str(tmp_path/'beam.fil'), 'samples_processed': SAMPLES, 'tsamp': TSAMP,
                'nu_min': 119.4, 'nu_max': 151.0, 'dedispersion_plan': plan}
    trials = tmp_path/'Periodic_DM_trials'; trials.mkdir()
    for name, spec in trial_specs(metadata, plan).items():
        amplitude = 0.25 if spec['dm'] == 27.0 else 0.0
        pulsar_series(0.714520, amplitude, seed=int(spec['dm'] * 10)).tofile(trials/name)
    catalogue = [{'name': 'J0332+5434', 'dm': 26.8, 'period_seconds': 0.714520, 'separation_deg': 0.1},
                 {'name': 'J1939+2134', 'dm': 27.0, 'period_seconds': 0.00155, 'separation_deg': 0.4}]
    records = fold_catalogue(trials, tmp_path, metadata, catalogue)
    written = json.loads((tmp_path/'catalogue_folds.json').read_text())
    assert written['schema'] == catalogue_fold.SCHEMA and len(written['pulsars']) == 2
    assert records[0]['seen'] and records[0]['trial_dm'] == 27.0
    assert records[1]['status'].startswith('too fast')
