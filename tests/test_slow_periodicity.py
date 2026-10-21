"""The slow-pulsar pass: its own trials from the data before the zero-DM filter, searched, vetoed and folded apart."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from lotaas_reprocessing import periodicity_veto as V
from lotaas_reprocessing.dm_plan import dm_values
from lotaas_reprocessing import sigproc_data, slow_periodicity
from lotaas_reprocessing.trials import product_outputs, validate_products, write_manifest

REPO = Path(__file__).resolve().parents[1]
TSAMP = 0.00786432
NCHAN, FCH1, FOFF = 648, 151.043701171875, -0.048828125
K_DM = 4148.808


def slow_beam(path, period=12.3456, dm=1.0, width=0.5, amplitude=0.05, nsamp=24000, seed=3):
    """Noise and a broadband pulse train at a low DM, each pulse wider than its sweep across the band, as
    PSR J0311+1402's (0.46 s against 0.20 s)."""
    rng = np.random.default_rng(seed)
    data = rng.normal(100.0, 1.0, (nsamp, NCHAN)).astype(np.float32)
    t = np.arange(nsamp) * TSAMP
    freqs = FCH1 + np.arange(NCHAN) * FOFF
    for channel, f in enumerate(freqs):
        delay = K_DM * dm * (f ** -2 - FCH1 ** -2)
        data[((t - delay - 3.0) % period) < width, channel] += amplitude
    header = {'telescope_id': 11, 'machine_id': -1, 'data_type': 1, 'source_name': 'LOTAAS-P0000A-SAP0',
              'src_raj': 31100.0, 'src_dej': 140200.0, 'tstart': 58011.0, 'tsamp': TSAMP, 'fch1': FCH1,
              'foff': FOFF, 'nchans': NCHAN, 'nifs': 1, 'nbits': 32}
    sigproc_data.write(path, header, data)
    return path


def settings(tmp_path, enabled=True):
    values = yaml.safe_load((REPO / 'ops' / 'production-settings.yaml').read_text())
    values['dedispersion_plan'] = [{'low_dm': 0.0, 'high_dm': 2.0, 'ddm': 0.5, 'downsample': 1}]
    values['periodicity'].update(dm_plan=[{'low_dm': 0.0, 'high_dm': 2.0, 'ddm': 0.5, 'downsample': 1}],
                                 catalogue_match=False, catalogue_fold=False, multibeam_veto=False)
    values['slow_periodicity'].update(enabled=enabled,
                                      dm_plan=[{'low_dm': 0.0, 'high_dm': 2.0, 'ddm': 0.5, 'downsample': 4}])
    path = tmp_path / 'settings.yaml'
    path.write_text(yaml.safe_dump(values))
    return path


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=REPO, capture_output=True)


def best(path, period):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    near = [r for r in rows if min(abs(r['period_seconds'] * n / period - 1) for n in range(1, 17)) < 0.01]
    return max((r['statistic'] for r in near), default=0.0)


def test_the_slow_pass_finds_a_pulse_train_the_zero_dm_filter_removes(tmp_path):
    fil = slow_beam(tmp_path / 'downsampled_L000001_SAP000_BEAM010_32bit_ff.fil')
    out = tmp_path / 'out'
    run('pipeline/pipeline_gpu.py', fil, out, '--settings', settings(tmp_path), '--backend', 'cpu')
    metadata = json.loads((out / 'metadata.json').read_text())
    assert metadata['preprocessing']['zero_dm'] and metadata['slow_periodicity_enabled']
    assert metadata['slow_preprocessing']['zero_dm'] is False
    assert metadata['slow_periodicity_config']['harmonics'][-1] == 32
    slow_trials = sorted((out / 'Slow_DM_trials').glob('*.dat'))
    expected = len(dm_values(metadata['slow_periodicity']['dm_plan'][0]))
    assert len(slow_trials) == expected == 4 and all(p.stat().st_size == 4 * (24000 // 4) for p in slow_trials)
    validate_products(out)
    # Never hard-linked to the filtered trials.
    main = {p.name: p.stat().st_ino for p in (out / 'Periodic_DM_trials').glob('*.dat')}
    assert not set(main.values()) & {p.stat().st_ino for p in slow_trials}
    run('pipeline/pipeline_cpu.py', out, '--search', 'periodicity')
    run('pipeline/pipeline_cpu.py', out, '--search', 'periodicity-fold')
    slow_best = best(out / 'slow_periodicity_folded_candidates.jsonl', 12.3456)
    main_best = best(out / 'periodicity_raw_candidates.jsonl', 12.3456)
    assert slow_best > 30 and slow_best > 2 * main_best
    folds = [json.loads(line) for line in (out / 'slow_periodicity_folded_candidates.jsonl').read_text().splitlines()]
    assert all(f['plot'].startswith('periodicity_plots/slow_periodic_') for f in folds)
    assert product_outputs(out, 'slow_periodicity_summary.json')
    # Pruned to what the folds could read: only sifted-best trials are left.
    assert len(list((out / 'Slow_DM_trials').glob('*.dat'))) < 4


def test_without_the_pass_nothing_of_it_is_written(tmp_path):
    fil = slow_beam(tmp_path / 'downsampled_L000001_SAP000_BEAM011_32bit_ff.fil', nsamp=6000)
    out = tmp_path / 'out'
    run('pipeline/pipeline_gpu.py', fil, out, '--settings', settings(tmp_path, enabled=False), '--backend', 'cpu')
    metadata = json.loads((out / 'metadata.json').read_text())
    assert not metadata['slow_periodicity_enabled'] and not (out / 'Slow_DM_trials').exists()
    validate_products(out)


def test_the_slow_pass_is_vetoed_among_its_own_peaks(tmp_path):
    res = 1 / 3600.
    dirs = []
    for beam in (13, 14, 15, 16, 17):
        directory = tmp_path / f'downsampled_L559955_SAP000_BEAM{beam:03d}_32bit_ff' / '0123456789abcdef'
        directory.mkdir(parents=True)
        rows = [{'frequency_hz': 0.05, 'frequency_resolution_hz': res, 'dm': beam * 3.0, 'is_sifted_best': True}]
        (directory / 'slow_periodicity_candidates.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        (directory / 'periodicity_candidates.jsonl').write_text('')
        dirs.append(directory)
    written = V.apply(dirs, prefix=slow_periodicity.PREFIX)
    assert sorted(written.values()) == [1] * 5
    assert all((d / 'slow_periodicity_veto.json').exists() and not (d / 'periodicity_veto.json').exists() for d in dirs)


def test_settings_merge_over_the_main_search():
    slow = slow_periodicity.settings_of({'slow_periodicity': {'enabled': True, 'max_folds': 8}})
    assert slow['enabled'] and slow['max_folds'] == 8 and slow['dm_plan'][0]['downsample'] == 4
    config = slow_periodicity.periodicity_config({'threshold': 12.0, 'max_folds': 16, 'harmonics': [1, 2]}, slow)
    assert config['threshold'] == 12.0 and config['max_folds'] == 8 and config['harmonics'][-1] == 32
    assert 'dm_plan' not in config and 'enabled' not in config
    assert slow_periodicity.preprocessing_of({'zero_dm': True, 'mask_fill': 'local'}, slow) == \
        {'zero_dm': False, 'mask_fill': 'local'}
    assert not slow_periodicity.enabled({})
