"""Continuous staging, raw-data removal and GPU sharing, without the network."""
import io
import json
import sqlite3
import tarfile
import time
from argparse import Namespace
from pathlib import Path

import pytest

from euroflash import campaign as C
from euroflash.ledger import Ledger
from euroflash.rawdata import delete_converted, raw_deleted

WEBDAV = 'https://webdav.grid.surfsara.nl:2882/lt5_004/'
SRM = 'srm://srm.grid.sara.nl:8443/pnfs/grid.sara.nl/data/lofar/ops/projects/lt5_004/'


def surl(obs, sap, beam):
    return f'{SRM}{obs}/L{obs}_SAP{sap:03d}_B{beam:03d}_P000_bf_{beam:08x}.tar'


class FakeStageIT:
    """StageIT whose 'online' list is set by the test, independently of dCache."""

    def __init__(self):
        self.requests, self.online, self.next = {}, set(), 5000
        self.final = False

    def submit(self, surls):
        self.next += 1
        self.requests[self.next] = list(surls)
        return self.next

    def status(self, request_id):
        files = self.requests[request_id]
        online = [s for s in files if C.basename(s) in self.online]
        done = self.final or len(online) == len(files)
        return {'currentStatus': 'success' if done else 'in progress',
                'response': json.dumps({'online': online, 'errors': {}})}

    def downloads(self, request_id):
        return {'request_id': request_id,
                'urls': [WEBDAV + s.split('/lt5_004/')[1] for s in self.requests[request_id]],
                'macaroons': [{'content': 'token', 'validUntil': '2099-01-01', 'ltaSite': {'name': 'SURF'}}]}


class FakeRunner:
    """Conversion and flatfield with the real directory layout and raw deletion."""

    def __init__(self, root):
        self.root = root
        self.flatfielded = []

    def convert(self, raw, delete_raw=False):
        import re
        obs, sap, beam = re.search(r'(L\d+)_SAP(\d+)_BEAM(\d+)', raw.name).groups()
        directory = self.root/'data'/obs/f'SAP{int(sap):03d}'/f'B{int(beam):03d}'
        directory.mkdir(parents=True, exist_ok=True)
        output = directory/f'downsampled_{obs}_SAP{int(sap):03d}_BEAM{int(beam):03d}_32bit.fil'
        output.write_bytes(b'fil')
        if delete_raw:
            delete_converted(raw)
        return directory.parent, output

    def flatfield(self, sap):
        paths = sorted(sap.glob('B*/*_32bit.fil'))
        for path in paths:
            path.with_name(path.stem + '_ff.fil').write_bytes(b'ff')
        self.flatfielded.append(sap)
        return paths


def fake_download(url, tokens, target, max_bytes):
    name = Path(target).name
    obs, sap, beam = C.NAME.match(name).groups()
    payload = b'psrfits'
    with tarfile.open(target, 'w') as archive:
        info = tarfile.TarInfo(f'stokes/SAP{int(sap)}/BEAM{int(beam)}/L{int(obs) - 600000}_SAP{int(sap)}_BEAM{int(beam)}_2bit.fits')
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    receipt = {'url': url, 'bytes': Path(target).stat().st_size, 'sha256': 'f' * 64, 'seconds': 0.1}
    Path(str(target) + '.receipt.json').write_text(json.dumps(receipt))
    return receipt


def build(tmp_path, monkeypatch, saps, locality=None, **overrides):
    inventory = tmp_path/'inventory.txt'
    inventory.write_text('\n'.join(surl(obs, sap, beam) for obs, sap, beams in saps for beam in beams) + '\n')
    options = C.parser().parse_args(['run', '--root', str(tmp_path/'campaign'), '--inventory', str(inventory),
                                     '--ledger', str(tmp_path/'ledger.sqlite'), '--poll-seconds', '0',
                                     '--locality-interval', '0', '--files-per-worker', '100',
                                     '--min-free-tb', '0'])
    for key, value in overrides.items():
        setattr(options, key, value)
    api = FakeStageIT()
    where = locality if locality is not None else {}
    campaign = C.Campaign(options, api=api, locate=lambda s, token: where.get(C.basename(s), 'NEARLINE'),
                          runner=FakeRunner(tmp_path/'campaign'/'prepared'))
    campaign.state.load(C.parse_inventory(inventory), C.missing_central_beams, tuple(options.exclude_beams))
    monkeypatch.setattr('euroflash.download.download', fake_download)
    return campaign, api, where


def settle(campaign):
    """Let background flatfields finish, then take another pass."""
    from concurrent.futures import wait
    wait(list(campaign.flatfield_jobs.values()))
    return campaign.tick()


def put_online(api, where, obs, sap, beams):
    for beam in beams:
        name = C.basename(surl(obs, sap, beam))
        api.online.add(name)
        where[name] = 'ONLINE_AND_NEARLINE'


FULL = list(range(0, 74))


def test_a_rolling_window_keeps_requests_in_flight_and_skips_incomplete_saps(tmp_path, monkeypatch):
    saps = [('1000001', 0, FULL), ('1000001', 1, FULL), ('1000001', 2, [b for b in FULL if b != 40]),
            ('1000002', 0, FULL)]
    campaign, api, _ = build(tmp_path, monkeypatch, saps, max_staging_saps=2)
    campaign.tick()
    states = {r['key']: r['state'] for r in campaign.state.rows('SELECT key,state FROM saps')}
    assert states == {'L1000001_SAP000': 'staging', 'L1000001_SAP001': 'staging',
                      'L1000001_SAP002': 'incomplete', 'L1000002_SAP000': 'pending'}
    assert sorted(len(v) for v in api.requests.values()) == [73, 73], 'beam 12, the incoherent beam, is not staged'
    campaign.tick()
    assert len(api.requests) == 2, 'a request in flight is not submitted twice'


def test_files_move_from_tape_to_prepared_and_raw_data_is_deleted(tmp_path, monkeypatch):
    campaign, api, where = build(tmp_path, monkeypatch, [('1000001', 0, FULL), ('1000001', 1, FULL)],
                                 max_staging_saps=1)
    campaign.tick()
    put_online(api, where, '1000001', 0, FULL)
    campaign.tick()
    settle(campaign)
    root = tmp_path/'campaign'
    sap = campaign.state.rows("SELECT * FROM saps WHERE key='L1000001_SAP000'")[0]
    assert sap['state'] == 'prepared'
    sap_dir = Path(sap['sap_dir'])
    assert len(list(sap_dir.glob('B*/*_ff.fil'))) == 73
    assert not list(sap_dir.glob('B*/*_32bit.fil')), 'unflattened filterbanks are removed'
    assert not list((root/'downloads').glob('*.tar')), 'archives are removed after extraction'
    assert not list((root/'extracted').rglob('*.fits')), 'PSRFITS are removed after conversion'
    markers = list((root/'extracted').glob('*.extracted.json'))
    receipts = list((root/'downloads').glob('*.tar.receipt.json'))
    assert len(markers) == len(receipts) == 73 and all(raw_deleted(m) for m in markers)
    with Ledger(tmp_path/'ledger.sqlite').connect() as db:
        assert db.execute("SELECT COUNT(*) FROM attempts WHERE stage='retrieve' AND status='success'").fetchone()[0] == 73
        assert db.execute('SELECT COUNT(*) FROM archive_receipts').fetchone()[0] == 73
    # The freed window admits the next SAP.
    assert campaign.state.rows("SELECT state FROM saps WHERE key='L1000001_SAP001'")[0]['state'] == 'staging'


def test_stageit_online_is_not_trusted_and_tape_only_files_are_requested_again(tmp_path, monkeypatch):
    campaign, api, where = build(tmp_path, monkeypatch, [('1000001', 0, FULL)], restage_after_hours=0)
    campaign.tick()
    first = set(api.requests)
    api.final = True
    for beam in FULL:                      # StageIT: success; dCache: still on tape
        api.online.add(C.basename(surl('1000001', 0, beam)))
    campaign.tick()
    rows = campaign.state.rows('SELECT state,submissions,locality FROM files')
    assert not any(r['state'] in ('online', 'converted') for r in rows)
    campaign.tick()                        # the re-request is submitted
    assert len(api.requests) == 2 and set(api.requests) > first
    assert all(r['submissions'] == 2 for r in campaign.state.rows("SELECT submissions FROM files WHERE state!='excluded'"))


def test_a_file_that_keeps_failing_to_download_is_given_up_and_blocks_its_sap(tmp_path, monkeypatch):
    campaign, api, where = build(tmp_path, monkeypatch, [('1000001', 0, FULL)], max_failures=2)
    campaign.tick()
    put_online(api, where, '1000001', 0, FULL)
    bad = C.basename(surl('1000001', 0, 20))

    def flaky(url, tokens, target, max_bytes):
        if Path(target).name == bad:
            raise IOError('Incomplete HTTP body')
        return fake_download(url, tokens, target, max_bytes)
    monkeypatch.setattr('euroflash.download.download', flaky)
    campaign.tick()
    campaign.tick()
    campaign.tick()
    row = campaign.state.rows('SELECT state,failures FROM files WHERE name=?', bad)[0]
    assert row == {'state': 'failed', 'failures': 2}
    sap = campaign.state.rows('SELECT state,detail FROM saps')[0]
    assert sap['state'] == 'attention' and '20' in sap['detail']


def test_a_refused_download_waits_for_dcache_instead_of_failing(tmp_path, monkeypatch):
    campaign, api, where = build(tmp_path, monkeypatch, [('1000001', 0, FULL)])
    campaign.tick()
    put_online(api, where, '1000001', 0, [13])

    def refused(*args):
        raise PermissionError('All 1 macaroons were refused')
    monkeypatch.setattr('euroflash.download.download', refused)
    campaign.refresh()
    campaign.retrieve()
    row = campaign.state.rows('SELECT state,failures FROM files WHERE beam=13')[0]
    assert row == {'state': 'requested', 'failures': 0}


def snapshot(path, statuses):
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(path)
    for item, status in statuses.items():
        attempt = ledger.start(item, 'classify', 'fp', 'log', ['cmd'])
        ledger.finish(attempt, error=None if status == 'success' else 'failed')


def test_a_search_removes_prepared_beams_that_succeeded_and_keeps_failures(tmp_path, monkeypatch):
    campaign, api, where = build(tmp_path, monkeypatch, [('1000001', 0, FULL)])
    campaign.tick()
    put_online(api, where, '1000001', 0, FULL)
    campaign.tick()
    settle(campaign)
    sap = campaign.state.rows('SELECT * FROM saps')[0]
    beams = sorted(Path(sap['sap_dir']).glob('B*/*_ff.fil'))
    run_name = 'campaign-test'
    campaign.state.set_sap(sap['key'], state='dispatched', run_name=run_name)
    statuses = {b.stem: 'success' for b in beams}
    statuses[beams[0].stem] = 'failed'
    snapshot(tmp_path/'campaign'/'results'/run_name/'efc-gpu-01'/'ledger-snapshot.sqlite', statuses)
    campaign.dispatch_process = Namespace(run_name=run_name, returncode=1)
    campaign.finish_dispatch()
    assert [b.exists() for b in beams] == [True] + [False] * (len(beams) - 1)
    after = campaign.state.rows('SELECT state,detail FROM saps')[0]
    assert after['state'] == 'attention' and after['detail'].startswith('1 beams')


def test_a_dispatch_starts_one_cluster_run_with_hard_linked_inputs(tmp_path, monkeypatch):
    campaign, api, where = build(tmp_path, monkeypatch, [('1000001', 0, FULL)])
    campaign.tick()
    put_online(api, where, '1000001', 0, FULL)
    campaign.tick()
    settle(campaign)
    launched = []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            launched.append(command)
        def poll(self):
            return None
    monkeypatch.setattr(C.subprocess, 'Popen', FakeProcess)
    campaign.o.dispatch_nodes = ['efc-gpu-01']
    run_name = campaign.dispatch()
    assert run_name and len(launched) == 1
    command = launched[0]
    assert command[1:3] == ['-m', 'euroflash.cluster']
    for flag in ('--skip-trials', '--cleanup-remote'):
        assert flag in command
    assert command[command.index('--workers-per-gpu') + 1] == '3'
    batch = Path(command[command.index('--input') + 1])
    linked = sorted(batch.glob('*_ff.fil'))
    assert len(linked) == 73 and all(p.stat().st_nlink == 2 for p in linked)
    assert command[command.index('--exclude-beams') + 1] == '12'
    assert campaign.state.rows('SELECT state FROM saps')[0]['state'] == 'dispatched'
    assert campaign.dispatch() == 'running', 'one cluster run at a time'


def test_a_dispatch_that_never_ran_puts_saps_back_and_backs_off(tmp_path, monkeypatch):
    campaign, api, where = build(tmp_path, monkeypatch, [('1000001', 0, FULL)])
    campaign.tick()
    put_online(api, where, '1000001', 0, FULL)
    campaign.tick()
    settle(campaign)
    key = campaign.state.rows('SELECT key FROM saps')[0]['key']
    campaign.state.set_sap(key, state='dispatched', run_name='campaign-ssh-down')
    campaign.dispatch_process = Namespace(run_name='campaign-ssh-down', returncode=255)
    campaign.finish_dispatch()
    assert campaign.state.rows('SELECT state FROM saps')[0]['state'] == 'prepared'
    assert campaign.dispatch_retry_after > time.time()


def test_an_interrupted_driver_resumes_its_work(tmp_path, monkeypatch):
    campaign, api, where = build(tmp_path, monkeypatch, [('1000001', 0, FULL)])
    campaign.tick()
    with campaign.state.db() as db:
        db.execute("UPDATE files SET state='working' WHERE beam=13")
    campaign.recover()
    assert campaign.state.rows('SELECT state FROM files WHERE beam=13')[0]['state'] == 'online'


# ----------------------------------------------------------------- run.py

def runner(tmp_path, **extra):
    from euroflash.run import Runner
    image = tmp_path/'image.sif'; image.write_bytes(b'image')
    settings = tmp_path/'settings.yaml'; settings.write_text('dedispersion_plan: []')
    args = Namespace(work=tmp_path/'work', ledger=tmp_path/'ledger.sqlite', settings=settings, image=image,
                     input=tmp_path/'input', backend='gpu', gpus='0,1', cpu_workers=2, pilot=False,
                     max_samples=None, preprocess_workers=2, convert_only=False, **extra)
    (tmp_path/'input').mkdir(exist_ok=True)
    return Runner(args)


def test_several_workers_share_each_gpu(tmp_path):
    import threading
    run = runner(tmp_path, workers_per_gpu=3)
    run.fp = 'abc'
    seen, active, peak, lock = [], {}, {}, threading.Lock()
    release = threading.Event()

    def step(item, stage, command, expected, gpu=None, **kwargs):
        if stage != 'dedisperse':
            return
        with lock:
            seen.append(gpu)
            active[gpu] = active.get(gpu, 0) + 1
            peak[gpu] = max(peak.get(gpu, 0), active[gpu])
            if sum(active.values()) == 6:
                release.set()
        release.wait(5)
        with lock:
            active[gpu] -= 1
    run.step = step
    run.analyze = lambda item, output: None
    beams = []
    for i in range(12):
        path = tmp_path/f'beam{i:02d}_ff.fil'; path.write_bytes(b'x'); beams.append(path)
    run.process(beams)
    assert sorted(set(seen)) == ['0', '1'] and len(seen) == 12
    assert peak == {'0': 3, '1': 3}


def test_prepare_deletes_raw_data_only_after_conversion_and_flatfields_the_whole_sap(tmp_path):
    run = runner(tmp_path, delete_raw=True)
    run.fp = 'abc'
    extraction = tmp_path/'input'/'L1000001_SAP000_B013_P000_bf_0000000d'
    fits = extraction/'stokes'/'SAP0'/'BEAM13'/'L400001_SAP0_BEAM13_2bit.fits'
    fits.parent.mkdir(parents=True)
    fits.write_bytes(b'psrfits')
    tar = extraction.parent/(extraction.name + '.tar'); tar.write_bytes(b'archive')
    marker = extraction.parent/(extraction.name + '.extracted.json')
    url = SRM + '1000001/' + extraction.name + '.tar'
    run.ledger.discover([url], 'lt5_004', 'test')
    marker.write_text(json.dumps({'request_id': 1, 'archive': {'url': url, 'sha256': 'a', 'bytes': 7},
                                  'fits': [str(fits)]}))
    # A beam converted by an earlier run, whose raw data is already gone.
    earlier = run.root/'data'/'L400001'/'SAP000'/'B014'/'downsampled_L400001_SAP000_BEAM014_32bit.fil'
    earlier.parent.mkdir(parents=True); earlier.write_bytes(b'fil')
    calls = []

    def step(item, stage, command, expected, gpu=None, fingerprint_override=None, validator=None):
        calls.append(stage)
        if stage == 'downsample':
            assert fits.exists(), 'raw data must still exist while converting'
            for path in expected:
                Path(path).write_bytes(b'fil')
        else:
            for path in expected:
                Path(path).write_bytes(b'ff')
    run.step = step
    prepared = run.prepare()
    assert calls == ['downsample', 'flatfield']
    assert not fits.exists() and not tar.exists() and not extraction.exists()
    assert raw_deleted(marker)
    assert sorted(p.name for p in prepared) == ['downsampled_L400001_SAP000_BEAM013_32bit_ff.fil',
                                                'downsampled_L400001_SAP000_BEAM014_32bit_ff.fil']


def test_a_failed_conversion_keeps_the_raw_data(tmp_path):
    run = runner(tmp_path, delete_raw=True)
    fits = tmp_path/'input'/'L400001_SAP000_BEAM013.fits'; fits.write_bytes(b'psrfits')

    def step(*args, **kwargs):
        raise RuntimeError('downsample failed')
    run.step = step
    with pytest.raises(RuntimeError):
        run.prepare()
    assert fits.exists()


def test_reconcile_accepts_receipts_whose_raw_data_was_deleted_on_purpose(tmp_path):
    from euroflash.provenance import reconcile
    ledger = Ledger(tmp_path/'ledger.sqlite')
    url = SRM + '1000001/L1000001_SAP000_B013_P000_bf_0000000d.tar'
    ledger.discover([url], 'lt5_004', 'test')
    marker = tmp_path/'L1000001_SAP000_B013_P000_bf_0000000d.extracted.json'
    value = {'request_id': 1, 'archive': {'url': url, 'sha256': 'a', 'bytes': 7},
             'fits': [str(tmp_path/'gone'/'L400001_SAP0_BEAM13_2bit.fits')]}
    marker.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='missing FITS'):
        reconcile(tmp_path, ledger)
    value['raw_deleted'] = True
    marker.write_text(json.dumps(value))
    assert reconcile(tmp_path, ledger) == 1


def test_throttling_backs_off_without_counting_a_failure(tmp_path, monkeypatch):
    campaign, api, where = build(tmp_path, monkeypatch, [('1000001', 0, FULL)], max_failures=1)
    campaign.tick()
    put_online(api, where, '1000001', 0, [13])

    def throttled(*args):
        raise IOError('HTTP Error 429: Too Many Requests')
    monkeypatch.setattr('euroflash.download.download', throttled)
    campaign.refresh()
    campaign.retrieve()
    assert campaign.state.rows('SELECT state,failures FROM files WHERE beam=13')[0] == {'state': 'online', 'failures': 0}
    assert campaign.throttle_until > time.time() and campaign.retrieve() == 0


def test_flatfielding_runs_beside_retrieval_and_frees_the_staging_slot(tmp_path, monkeypatch):
    import threading
    campaign, api, where = build(tmp_path, monkeypatch, [('1000001', 0, FULL), ('1000001', 1, FULL)],
                                 max_staging_saps=1)
    gate = threading.Event()
    slow = campaign.runner.flatfield
    campaign.runner.flatfield = lambda sap: (gate.wait(5), slow(sap))[1]
    campaign.tick()
    put_online(api, where, '1000001', 0, FULL)
    campaign.tick()
    states = {r['key']: r['state'] for r in campaign.state.rows('SELECT key,state FROM saps')}
    assert states == {'L1000001_SAP000': 'flatfielding', 'L1000001_SAP001': 'staging'}
    gate.set()
    settle(campaign)
    assert campaign.state.rows("SELECT state FROM saps WHERE key='L1000001_SAP000'")[0]['state'] == 'prepared'


def test_an_incoherent_member_is_excluded_whatever_its_number(tmp_path, monkeypatch):
    campaign, api, where = build(tmp_path, monkeypatch, [('1000001', 0, FULL)])
    campaign.tick()
    put_online(api, where, '1000001', 0, FULL)
    odd = C.basename(surl('1000001', 0, 7))

    def incoherent_seven(url, tokens, target, max_bytes):
        receipt = fake_download(url, tokens, target, max_bytes)
        if Path(target).name == odd:
            with tarfile.open(target, 'w') as archive:
                info = tarfile.TarInfo('incoherentstokes/SAP0/BEAM7/L400001_SAP0_BEAM7_2bit.fits')
                info.size = 3
                archive.addfile(info, io.BytesIO(b'abc'))
        return receipt
    monkeypatch.setattr('euroflash.download.download', incoherent_seven)
    campaign.tick()
    settle(campaign)
    row = campaign.state.rows('SELECT state FROM files WHERE name=?', odd)[0]
    assert row['state'] == 'excluded'
    assert not list((tmp_path/'campaign'/'extracted').rglob('*BEAM7*.fits'))
    assert campaign.state.rows('SELECT state FROM saps')[0]['state'] == 'prepared'


def test_a_sap_held_only_by_the_incoherent_beam_is_released_on_restart(tmp_path, monkeypatch):
    campaign, api, where = build(tmp_path, monkeypatch, [('1000001', 0, FULL)])
    campaign.tick()
    put_online(api, where, '1000001', 0, FULL)
    campaign.tick()
    settle(campaign)
    sap = campaign.state.rows('SELECT * FROM saps')[0]
    # As an older driver left it: beam 12 converted, searched with the rest, and failed.
    beam12 = Path(sap['sap_dir'])/'B012'/'downsampled_L400001_SAP000_BEAM012_32bit_ff.fil'
    beam12.parent.mkdir(exist_ok=True); beam12.write_bytes(b'ff')
    for other in Path(sap['sap_dir']).glob('B*/*_ff.fil'):
        if other != beam12:
            other.unlink()
    campaign.state.set_sap(sap['key'], state='attention', run_name='campaign-old',
                           detail='1 beams not searched in campaign-old')
    campaign.recover()
    assert not beam12.exists()
    assert campaign.state.rows('SELECT state FROM saps')[0]['state'] == 'searched'


def test_manual_preparation_skips_the_incoherent_beam(tmp_path):
    run = runner(tmp_path)
    run.fp = 'abc'
    coherent = tmp_path/'input'/'stokes'/'L400001_SAP000_BEAM013.fits'
    incoherent = tmp_path/'input'/'incoherentstokes'/'L400001_SAP000_BEAM012.fits'
    for path in (coherent, incoherent):
        path.parent.mkdir(parents=True); path.write_bytes(b'psrfits')
    converted = []

    def step(item, stage, command, expected, gpu=None, fingerprint_override=None, validator=None):
        converted.append(item)
        for path in expected:
            Path(path).write_bytes(b'x')
    run.step = step
    run.prepare()
    assert 'L400001_SAP000_B012' not in converted and 'L400001_SAP000_B013' in converted


def test_beam_numbers_and_layout_are_recognised():
    from euroflash.beams import beam_number, excluded
    assert beam_number('L1163405_SAP000_B012_P000_bf_03604bcb.tar') == 12
    assert beam_number('L559289_SAP0_BEAM12_2bit.fits') == 12
    assert beam_number('downsampled_L559289_SAP000_BEAM012_32bit_ff.fil') == 12
    assert excluded('downsampled_L559289_SAP000_BEAM012_32bit_ff.fil')
    assert not excluded('downsampled_L559289_SAP000_BEAM013_32bit_ff.fil')
    assert excluded('/x/incoherentstokes/SAP0/BEAM3/L1_SAP0_BEAM3_2bit.fits')
    assert not excluded('downsampled_L559289_SAP000_BEAM012_32bit_ff.fil', beams=())


def test_a_cluster_run_that_outlived_its_driver_is_followed_not_repeated(tmp_path, monkeypatch):
    import subprocess, sys
    campaign, api, where = build(tmp_path, monkeypatch, [('1000001', 0, FULL)])
    campaign.tick()
    put_online(api, where, '1000001', 0, FULL)
    campaign.tick()
    settle(campaign)
    key = campaign.state.rows('SELECT key FROM saps')[0]['key']
    campaign.state.set_sap(key, state='dispatched', run_name='campaign-orphan')
    # Stand-in for a euroflash.cluster process started by a previous driver.
    orphan = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)', '-m', 'euroflash.cluster',
                               '--run-name', 'campaign-orphan'])
    try:
        campaign.recover()
        assert campaign.state.rows('SELECT state FROM saps')[0]['state'] == 'dispatched'
        assert campaign.dispatch_process.pid == orphan.pid
        assert campaign.dispatch() == 'running'
    finally:
        orphan.kill(); orphan.wait()
    assert campaign.dispatch_process.poll() is not None


def test_retry_requeues_attention_saps_with_prepared_beams(tmp_path, monkeypatch):
    campaign, api, where = build(tmp_path, monkeypatch, [('1000001', 0, FULL), ('1000001', 1, FULL)], max_staging_saps=2)
    campaign.tick()
    put_online(api, where, '1000001', 0, FULL)
    put_online(api, where, '1000001', 1, FULL)
    campaign.tick()
    settle(campaign)
    first, second = [r['key'] for r in campaign.state.rows('SELECT key FROM saps ORDER BY position')]
    campaign.state.set_sap(first, state='attention', detail='2 beams not searched in run-a')
    campaign.state.set_sap(second, state='attention', detail='central beams not retrieved: [40]')
    for path in Path(campaign.state.rows('SELECT sap_dir FROM saps WHERE key=?', second)[0]['sap_dir']).glob('B*/*_ff.fil'):
        path.unlink()
    assert C.retry_saps(tmp_path/'campaign') == [first]
    assert campaign.state.rows('SELECT state FROM saps WHERE key=?', first)[0]['state'] == 'prepared'
    assert campaign.state.rows('SELECT state FROM saps WHERE key=?', second)[0]['state'] == 'attention'


def test_raw_reclaim_removes_only_archives_with_evidence(tmp_path):
    from euroflash.reclaim import survey_raw
    ledger = Ledger(tmp_path/'ledger.sqlite')
    data = tmp_path/'data'

    def archive(beam):
        stem = f'L1000001_SAP000_B{beam:03d}_P000_bf_{beam:08x}'
        fits = data/stem/'stokes'/f'L400001_SAP0_BEAM{beam}_2bit.fits'
        fits.parent.mkdir(parents=True)
        fits.write_bytes(b'psrfits')
        (data/(stem + '.tar')).write_bytes(b'archive')
        marker = data/(stem + '.extracted.json')
        marker.write_text(json.dumps({'request_id': 1, 'archive': {'url': stem}, 'fits': [str(fits)]}))
        return stem, fits
    converted_stem, _ = archive(13)
    lost_stem, _ = archive(14)            # converted, but its output is gone and it was never searched
    unconverted_stem, _ = archive(15)
    incoherent_stem, _ = archive(12)
    output = tmp_path/'prepared'/'downsampled_L400001_SAP000_BEAM013_32bit.fil'
    output.parent.mkdir(); output.write_bytes(b'fil')
    for beam, path in ((13, output), (14, tmp_path/'prepared'/'gone.fil')):
        attempt = ledger.start(f'L400001_SAP000_B{beam:03d}', 'downsample', 'fp', 'log', ['convert'])
        with ledger.connect() as db:
            db.execute("UPDATE attempts SET status='success',outputs=? WHERE id=?", (json.dumps({str(path): 3}), attempt))
    found = {marker.name.split('.')[0]: reason for fits, tar, size, reason, marker in survey_raw(data, tmp_path/'ledger.sqlite')}
    assert found == {converted_stem: 'converted', incoherent_stem: 'excluded beam'}
