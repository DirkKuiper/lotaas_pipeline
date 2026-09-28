"""The injection lane (euroflash.injections): sampling production's SAPs, making twins, searching, recording."""
import json
import sqlite3
from pathlib import Path

from euroflash import injections


def options(tmp_path):
    campaign = tmp_path / 'campaign'
    campaign.mkdir()
    with sqlite3.connect(campaign / 'campaign-state.sqlite') as db:
        db.execute('CREATE TABLE saps (key TEXT, position INTEGER, files INTEGER, state TEXT, detail TEXT, sap_dir TEXT, '
                   'run_name TEXT, updated REAL, source TEXT)')
    return {'root': campaign, 'checkout': tmp_path / 'checkout', 'image': tmp_path / 'runtime.sif',
            'settings': {'lta': tmp_path / 'lt5.yaml', 'spider': tmp_path / 'ec.yaml'}, 'control_dir': str(tmp_path),
            'timeouts': ['sp_classify=1800'], 'cpu_tier_workers': '12'}


def prepared_sap(o, key, obs, state='prepared', source='lta', beams=(0, 12, 22)):
    sap_dir = o['root'] / 'prepared' / 'data' / obs / 'SAP000'
    for beam in beams:
        path = sap_dir / f'B{beam:03d}' / f'downsampled_{obs}_SAP000_BEAM{beam:03d}_32bit_ff.fil'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'x' * 16)
    with sqlite3.connect(o['root'] / 'campaign-state.sqlite') as db:
        db.execute('INSERT INTO saps(key,state,sap_dir,source) VALUES (?,?,?,?)', (key, state, str(sap_dir), source))
    return sap_dir


def lane(tmp_path, clock=None):
    o = options(tmp_path)
    ln = injections.Lane(tmp_path / 'lane', o, now=clock or (lambda: 1000.0))
    ln.code = lambda: (tmp_path / 'snapshot', 'c0ffee', tmp_path / 'image.sif')
    ln.started = []
    ln.remote = lambda node, command, log: ln.started.append((node, command, log)) or True
    return ln, o


def test_campaign_options_come_from_the_supervisor_s_configuration(tmp_path):
    config = tmp_path / 'campaign.json'
    config.write_text(json.dumps({'root': '/r', 'checkout': '/c', 'command': [
        'python3', '-m', 'euroflash.campaign', 'run', '--settings', '/c/lt5.yaml', '--spider-settings', '/c/ec.yaml',
        '--image', '/c/i.sif', '--cpu-tier-workers', '12', '--control-dir', '/ctl', '--stage-timeout', 'sp_classify=1800']}))
    o = injections.campaign_options(config)
    assert o['settings'] == {'lta': Path('/c/lt5.yaml'), 'spider': Path('/c/ec.yaml')} and o['timeouts'] == ['sp_classify=1800']


def test_one_beam_of_each_sap_is_held_by_a_link_never_the_incoherent_one(tmp_path):
    ln, o = lane(tmp_path)
    prepared_sap(o, 'L1_SAP000', 'L10')
    prepared_sap(o, 'L2_SAP000', 'L20', state='searched')          # already gone: never sampled
    assert ln.stage() == ['L1_SAP000'] and ln.stage() == []
    sample = ln.rows('SELECT * FROM samples')[0]
    assert sample['beam'] in (0, 22) and Path(sample['link']).stat().st_nlink == 2
    assert sample['seed'] == injections.Lane(tmp_path / 'lane', o).rows('SELECT seed FROM samples')[0]['seed']


def test_twins_are_made_on_cpu_nodes_and_settled_when_they_end(tmp_path):
    ln, o = lane(tmp_path)
    prepared_sap(o, 'L1_SAP000', 'L10')
    ln.stage()
    ln.make()
    node, command, log = ln.started[0]
    assert node in injections.CPU_NODES and 'lotaas_reprocessing.frb_injection' in command
    assert str(o['settings']['lta']) in command                    # the search's own channel mask
    sample = ln.rows('SELECT * FROM samples')[0]
    Path(sample['twin']).write_bytes(b'twin')
    Path(sample['truth']).write_text('{}')
    Path(str(log) + '.exit').write_text('0\n')
    assert ln.make() == ['L1_SAP000'] and ln.rows('SELECT state FROM samples')[0]['state'] == 'ready'
    assert not Path(sample['link']).exists()                        # the link no longer holds production's beam


def ready_twins(ln, o, n, source='lta', start=0):
    for i in range(start, start + n):
        twin = ln.root / 'twins' / f'downsampled_L{i}_SAP000_BEAM001_32bit_ff.fil'
        twin.write_bytes(b'twin')
        truth = ln.root / 'truth' / f'L{i}_SAP000.json'
        truth.write_text(json.dumps({'twin': twin.name, 'bursts': []}))
        with ln.db:
            ln.db.execute("INSERT INTO samples(sap,source,twin,truth,state,made) VALUES (?,?,?,?,'ready',?)",
                          (f'L{i}_SAP000', source, str(twin), str(truth), 1000.0))


def test_a_full_batch_is_searched_as_production_searches(tmp_path):
    ln, o = lane(tmp_path)
    ready_twins(ln, o, injections.BATCH - 1)
    launched = []
    assert ln.dispatch(popen=lambda *a, **k: launched.append((a, k))) is None      # not enough, not old enough
    ready_twins(ln, o, 2, start=injections.BATCH)
    name = ln.dispatch(popen=lambda *a, **k: launched.append((a, k)))
    script = launched[0][0][0][2]
    assert name.endswith('-lta') and 'euroflash.cluster' in script and str(o['settings']['lta']) in script
    assert '--workers-per-gpu 1' in script and 'sp_classify=1800' in script and launched[0][1]['cwd'] == str(o['checkout'])
    assert len(ln.rows("SELECT * FROM samples WHERE state='dispatched'")) == injections.BATCH
    assert len(list((ln.root / 'batches' / name).glob('*.fil'))) == injections.BATCH
    assert ln.dispatch(popen=lambda *a, **k: None) is None                          # one batch at a time


def test_an_old_twin_goes_alone(tmp_path):
    clock = [1000.0]
    ln, o = lane(tmp_path, clock=lambda: clock[0])
    ready_twins(ln, o, 1, source='spider')
    clock[0] += injections.MAX_WAIT + 1
    assert ln.dispatch(popen=lambda *a, **k: None).endswith('-spider')


def test_a_finished_batch_is_followed_to_its_fates_and_its_twins_go(tmp_path):
    ln, o = lane(tmp_path)
    ready_twins(ln, o, injections.BATCH)
    name = ln.dispatch(popen=lambda *a, **k: None)
    (ln.root / 'logs' / f'{name}.log.exit').write_text('0\n')
    ln.collect()
    node, command, log = ln.started[-1]
    assert 'lotaas_reprocessing.injection_fates' in command and ln.running()[0]['state'] == 'analysing'
    twin = Path(ln.rows('SELECT twin FROM samples LIMIT 1')[0]['twin'])
    burst = {'twin': twin.stem, 'index': 0, 'dm': 500.0, 'tau135': 0.01, 'width': 0.001, 'snr_ideal': 12.0,
             'spectrum': 'flat', 'stage': 'dispersed', 'cluster': {'snr': 10.5}, 'own_snr': 9.0,
             'dispersion_ratio': 0.3, 'fetch': 0.2, 'page': 9.5, 'queued': True}
    (ln.root / 'fates' / f'{name}.json').write_text(json.dumps({'bursts': [burst], 'others': []}))
    Path(str(log) + '.exit').write_text('0\n')
    assert ln.collect() == [name]
    row = ln.rows('SELECT * FROM bursts')[0]
    assert row['queued'] == 1 and row['fingerprint'] == 'c0ffee' and row['sap'] == 'L0_SAP000'
    assert not twin.exists() and not (ln.root / 'batches' / name).exists()
    assert ln.rows('SELECT state FROM batches')[0]['state'] == 'done'
    assert 'reached the review queue' in injections.report(ln)
