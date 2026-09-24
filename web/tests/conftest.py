"""Fixtures: a throwaway campaign, ledger, results tree and filterbanks; nothing real is touched."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from web import config, sigproc  # noqa: E402
from web.dynspec import K_DM  # noqa: E402

TSAMP = 0.007864319719374176
FCH1, NCHANS = 151.043701171875, 64
FOFF = -31.59 / NCHANS


def synthetic_filterbank(path, dm=30.0, t_pulse=15.0, amplitude=1.2, width=3, nsamp=4000, seed=1,
                         undispersed=None):
    """Gaussian noise with one dispersed boxcar pulse, arriving at t_pulse at the top of the band."""
    rng = np.random.default_rng(seed)
    data = rng.normal(10.0, 1.0, (nsamp, NCHANS)).astype(np.float32)
    freqs = FCH1 + np.arange(NCHANS) * FOFF
    for channel, f in enumerate(freqs):
        start = int(round((t_pulse + K_DM * dm * (1 / f ** 2 - 1 / FCH1 ** 2)) / TSAMP))
        data[start:start + width, channel] += amplitude
    if undispersed is not None:
        start = int(round(undispersed / TSAMP))
        data[start:start + 2, :] += 3.0
    header = {'telescope_id': 11, 'machine_id': -1, 'data_type': 1, 'source_name': 'LOTAAS-P1254B-SAP0',
              'src_raj': 84708.0, 'src_dej': 682459.0, 'tstart': 57713.157638888886, 'tsamp': TSAMP,
              'fch1': FCH1, 'foff': FOFF, 'nchans': NCHANS, 'nifs': 1, 'nbits': 32}
    sigproc.write(path, header, data)
    return path


@pytest.fixture
def cfg(tmp_path):
    for name in ('campaign', 'results', 'sources', 'control'):
        (tmp_path / name).mkdir()
    settings = tmp_path / 'settings.yaml'
    settings.write_text('bad_channels: [5]\ndedispersion_plan:\n'
                        '  - {low_dm: 0.0, high_dm: 150.6, ddm: 0.1, downsample: 1}\n'
                        '  - {low_dm: 150.6, high_dm: 300.0, ddm: 0.3, downsample: 2}\n')
    return config.load(path=tmp_path / 'none.toml', campaign_root=tmp_path / 'campaign',
                       ledger=tmp_path / 'campaign.sqlite', data=tmp_path / 'web',
                       result_roots=[tmp_path / 'results'], source_roots=[tmp_path / 'sources'],
                       observation_catalogue=tmp_path / 'catalogue',
                       settings=settings, token_file=tmp_path / 'token', control_dir=tmp_path / 'control')


ITEM = 'downsampled_L559289_SAP000_BEAM025_32bit_ff'


@pytest.fixture
def campaign(cfg):
    """A campaign with one searched SAP, its ledger records and one beam's results on disk."""
    from euroflash.campaign import State
    from euroflash.ledger import Ledger
    from db.initialize_db import initialize_database
    state = State(cfg.state_db)
    prepared = cfg.campaign_root / 'prepared' / 'data' / 'L559289' / 'SAP000'
    with state.db() as db:
        db.execute("INSERT INTO saps VALUES ('L1163405_SAP000', 0, 2, 'searched', NULL, ?, 'run-a', 100.0)",
                   (str(prepared),))
        db.execute("INSERT INTO saps VALUES ('L1163405_SAP001', 1, 1, 'pending', NULL, NULL, NULL, 100.0)")
        for beam, st in ((25, 'searched'), (12, 'excluded')):
            db.execute('INSERT INTO files(surl,name,sap_key,beam,state,request_id,submissions,updated) '
                       'VALUES (?,?,?,?,?,?,?,?)', (f'srm://x/L1163405_SAP000_B{beam:03d}_P000_bf_aa.tar',
                                                    f'L1163405_SAP000_B{beam:03d}_P000_bf_aa.tar',
                                                    'L1163405_SAP000', beam, st, 7, 1, 100.0))
        db.execute("INSERT INTO files(surl,name,sap_key,beam,state,updated) VALUES "
                   "('srm://x/L1163405_SAP001_B000_P000_bf_bb.tar','L1163405_SAP001_B000_P000_bf_bb.tar',"
                   "'L1163405_SAP001',0,'pending',100.0)")
        db.execute("INSERT INTO requests VALUES (7, 'L1163405_SAP000', 50.0, 2, 'success', 60.0, 60.0)")
        db.execute("INSERT INTO events VALUES (90.0, 'searched', 'L1163405_SAP000', '1/1 beams in run-a')")
    ledger = Ledger(cfg.ledger)
    initialize_database(str(cfg.ledger))
    fingerprint = 'f' * 64
    ledger.register_run(fingerprint, {'pilot': False})
    with ledger.connect() as db:
        db.execute("INSERT INTO attempts(item,stage,fingerprint,status,started,finished,seconds,host) VALUES "
                   "(?, 'classify', ?, 'success', 80.0, 85.0, 5.0, 'efc-gpu-01')", (ITEM, fingerprint))
        output = f'/home/dkuiper/lotaas-runs/run-a/work/processed/{ITEM}/ffffffffffffffff/candidate_plots'
        db.execute("INSERT INTO beam_runs(beam_id,observation_date,processing_timestamp,outcome,num_candidates,"
                   "num_redetections,highest_snr,output_dir,log_file,code_version) VALUES "
                   "(?, '2016-11-21 03:47:00.000', '2026-09-23T07:00:00', 'classified', 2, 0, 12.0, ?, 'x', ?)",
                   (ITEM + '.fil', output, fingerprint))
        run = db.execute('SELECT MAX(id) FROM beam_runs').fetchone()[0]
        for dm, snr, kind, p in ((30.0, 12.0, 'candidate', 0.99), (60.0, 7.5, 'rejected', 0.1)):
            db.execute("INSERT INTO detections(beam_id,candidate_dm,snr,width_samples,detection_type,pulsar_name,"
                       "classification_probability,beam_run_id,time_seconds,sample_number) VALUES "
                       "(?,?,?,3,?,NULL,?,?,15.0,1907)", (ITEM + '.fil', dm, snr, kind, p, run))
        db.execute('CREATE TABLE slack_notifications (key TEXT PRIMARY KEY, kind TEXT NOT NULL, beam_id TEXT, '
                   'plot_path TEXT, slack_file_id TEXT, channel TEXT NOT NULL, sent REAL NOT NULL)')
        db.execute("INSERT INTO slack_notifications VALUES (?, 'candidate', ?, NULL, 'F1', 'C1', 95.0)",
                   (f'candidate|{ITEM}|DM30.000|W3|SN12.000', ITEM))
        db.execute("INSERT INTO archive_beams VALUES ('srm://x/L1163405_SAP000_B025_P000_bf_aa.tar', 'x', ?, "
                   "'L559289', 0, 25)", (ITEM,))
    beam_dir = cfg.result_roots[0] / 'run-a' / 'efc-gpu-01' / 'processed' / ITEM / 'ffffffffffffffff'
    (beam_dir / 'candidate_plots').mkdir(parents=True)
    (beam_dir.parents[2] / 'run.json').write_text(json.dumps(
        {'fingerprint': fingerprint, 'pilot': False,
         'settings': '/home/dkuiper/lotaas-runs/run-a/source/campaign-settings.yaml'}))
    (beam_dir / 'metadata.json').write_text(json.dumps({
        'pilot': False, 'tsamp': TSAMP, 'nu_min': 119.45, 'nu_max': 151.04, 'bad_channels': [7],
        'dedispersion_plan': [{'low_dm': 0.0, 'high_dm': 150.6, 'ddm': 0.1, 'downsample': 1}],
        'observation_info': {'RA (J2000)': '08:47:08.00', 'DEC (J2000)': '+68:24:59.00',
                             'Object': 'LOTAAS-P1254B-SAP0', 'Observation Date': '2016-11-21 03:47:00.000'}}))
    (beam_dir / 'clustered_candidates.txt').write_text(
        'DM\tS/N\tTime\tSample\tFilter_Width\tDM_scaled\tCluster\n30.0\t12.0\t15.0\t1907\t3\t1\t1\n')
    (beam_dir / 'candidate_plots' / 'DM30.0_Width3_SNR12.0.png').write_bytes(b'\x89PNG\r\n\x1a\n')
    return {'state': state, 'ledger': ledger, 'prepared': prepared, 'beam_dir': beam_dir}
