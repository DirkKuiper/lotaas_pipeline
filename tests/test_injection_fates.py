"""Following injected bursts through a searched run (injection_fates)."""
import json
import sqlite3

import numpy as np

from lotaas_reprocessing import frb_injection, injection_fates, sigproc_data
from test_own_data import PLAN, TSAMP, write_filterbank


def searched_twin(tmp_path, kind):
    """A run with one twin beam holding one injected burst, which the classifier recorded as `kind`."""
    sources = tmp_path / 'sources'
    sources.mkdir()
    header, mapped = sigproc_data.open_data(write_filterbank(tmp_path / 'beam.fil', nsamp=24000))
    data = np.array(mapped, dtype=np.float32)
    ranges = {'dm': (150.0, 160.0), 'tau135': (1e-3, 2e-3), 'snr': (25.0, 26.0), 'width': (1e-3, 2e-3)}
    truth = frb_injection.inject(data, header, [], np.random.default_rng(1), n=1, ranges=ranges)
    item = 'downsampled_L1_SAP000_BEAM001_32bit_ff'
    sigproc_data.write(sources / f'{item}.fil', header, data)
    (tmp_path / 'truth').mkdir()
    (tmp_path / 'truth' / f'{item}.json').write_text(json.dumps({'twin': f'{item}.fil', 'bursts': truth}))
    beam = tmp_path / 'run' / 'efc-cpu-00' / 'processed' / item / 'ffffffffffffffff'
    beam.mkdir(parents=True)
    b = truth[0]
    width = max(1, round(b['ideal_width_s'] / TSAMP))
    (beam / 'metadata.json').write_text(json.dumps({'tsamp': TSAMP, 'dedispersion_plan': PLAN, 'bad_channels': [],
                                                    'single_pulse': {'baseline_seconds': 2.0, 'baseline_widths': 8}}))
    (beam / 'clustered_candidates.txt').write_text(
        'DM\tS/N\tTime\tSample\tFilter_Width\n'
        f"{b['dm']:.1f}\t24.0\t{b['peak_time']:.6f}\t1\t{width}\n150.0\t9.0\t20.0\t1\t4\n")
    (beam / 'sp_classify_summary.json').write_text('{}')
    with sqlite3.connect(tmp_path / 'run' / 'efc-cpu-00' / 'ledger-snapshot.sqlite') as db:
        db.execute('CREATE TABLE detections (id INTEGER PRIMARY KEY, beam_id TEXT, candidate_dm REAL, snr REAL, '
                   'width_samples INTEGER, detection_type TEXT, classification_probability REAL, time_seconds REAL, '
                   'model_probabilities TEXT, own_snr REAL, dispersion_ratio REAL)')
        db.execute('INSERT INTO detections VALUES (1,?,?,24.0,?,?,0.2,?,NULL,23.0,0.2)',
                   (f'{item}.fil', float(f"{b['dm']:.1f}"), width, kind, float(f"{b['peak_time']:.6f}")))
        db.execute("INSERT INTO detections VALUES (2,?,150.0,9.0,4,'rejected',0.1,20.0,NULL,5.0,NULL)", (f'{item}.fil',))
    return tmp_path, b


def test_a_burst_is_followed_to_the_queue_and_the_rest_counted_apart(tmp_path):
    root, burst = searched_twin(tmp_path, 'dispersed')
    bursts, others = injection_fates.fates(root / 'run', root / 'truth', {'dedispersion_plan': PLAN}, root / 'sources')
    assert len(bursts) == 1 and bursts[0]['stage'] == 'dispersed' and bursts[0]['queued']
    assert bursts[0]['page'] > 15 and not bursts[0]['noise'] and bursts[0]['dispersion_ratio'] == 0.2
    assert [(o['type'], o['dm']) for o in others] == [('rejected', 150.0)]


def test_the_noise_rule_is_the_web_triage_s():
    assert injection_fates.noise_rule(3.9, 10.0) and not injection_fates.noise_rule(6.5, 20.0)
    assert injection_fates.noise_rule(5.0, 10.0) and not injection_fates.noise_rule(6.1, 10.0)
    assert not injection_fates.noise_rule(3.0, 6.5)          # below the search S/N the rule looks at
