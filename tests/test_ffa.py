"""The FFA (lotaas_reprocessing.ffa) and its search of the slow pass's trials (ffa_periodicity)."""
import json
import math

import numpy as np

from lotaas_reprocessing import ffa, ffa_periodicity
from tests.test_slow_periodicity import best, run, settings as slow_settings, slow_beam

DT = 0.0314572788           # the slow pass's 4 x 7.864 ms


def test_the_transform_folds_a_drifting_pulse_at_its_drift():
    # Drifts rounded at each level put a row at most 2 bins from an exact fold's, most within one.
    for m in (37, 64, 101, 333):
        p = 50
        for s in range(m):
            x = np.zeros((m, p))
            for j in range(m):
                x[j, (7 + int(round(j * s / (m - 1)))) % p] = 1.0
            folded = ffa._transform(x)
            assert folded[s, 5:10].sum() == m and folded[s, 6:9].sum() >= 0.9 * m


def train(period, duty, ideal, seconds=1800.0, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds / DT)) * DT
    pulse = np.exp(-0.5 * (((t % period) - 0.3 * period) / (duty * period / 2.355)) ** 2)
    return rng.normal(size=len(t)) + ideal / math.sqrt(np.sum(pulse ** 2)) * pulse


def test_a_slow_pulse_train_is_found_near_its_ideal_snr():
    for period, duty in ((7.31, 0.015), (40.9073, 0.012), (120.7, 0.01)):
        peaks = ffa.search(train(period, duty, 20.0), DT, 4.0, 300.0)
        at = [q for q in peaks if abs(q['period_seconds'] / period - 1) < 3e-3]
        assert at and max(q['snr'] for q in at) >= 14.0


def test_noise_stays_below_the_threshold():
    for seed in range(3):
        noise = np.random.default_rng(20 + seed).normal(size=int(1800 / DT))
        assert all(q['snr'] < 7.0 for q in ffa.search(noise, DT, 4.0, 300.0, threshold=5.5))


def test_the_dm_grid_takes_one_trial_per_step():
    specs = {f'b_DM{d:.2f}.dat': {'dm': d} for d in np.arange(0, 100, 0.25)}
    names = ffa_periodicity.selected(specs, [[0.0, 10.0, 0.5], [10.0, 40.0, 1.0], [40.0, 100.0, 2.0]])
    dms = sorted(specs[n]['dm'] for n in names)
    assert len(dms) == 20 + 30 + 30 and dms[:3] == [0.0, 0.5, 1.0] and dms[-1] == 98.0
    assert ffa_periodicity.on_steps(ffa_periodicity.BLOCK_SECONDS / 4) and not ffa_periodicity.on_steps(40.9073)


def test_the_ffa_folds_a_long_period_the_pipeline_was_given(tmp_path):
    import yaml
    fil = slow_beam(tmp_path / 'downsampled_L000001_SAP000_BEAM010_32bit_ff.fil', period=30.917, width=0.4,
                    amplitude=0.05, nsamp=60000)
    path = slow_settings(tmp_path)
    values = yaml.safe_load(path.read_text())
    values['ffa'] = dict(values['ffa'], enabled=True, dm_steps=[[0.0, 2.0, 0.5]])
    path.write_text(yaml.safe_dump(values))
    out = tmp_path / 'out'
    run('pipeline/pipeline_gpu.py', fil, out, '--settings', path, '--backend', 'cpu')
    metadata = json.loads((out / 'metadata.json').read_text())
    assert metadata['ffa_enabled'] and metadata['ffa']['threshold'] == 7.0
    run('pipeline/pipeline_cpu.py', out, '--search', 'periodicity')
    run('pipeline/pipeline_cpu.py', out, '--search', 'periodicity-fold')
    assert best(out / 'ffa_periodicity_folded_candidates.jsonl', 30.917) > 10
    folds = [json.loads(l) for l in (out / 'ffa_periodicity_folded_candidates.jsonl').read_text().splitlines()]
    assert all(f['plot'].startswith('periodicity_plots/ffa_periodic_') for f in folds)
    assert all(f['statistic_kind'] == 'ffa_boxcar_snr' for f in folds)
