"""What the 9-11 October 2026 outage taught: runners that outlive their run, nodes left full, a CPU tier that never
started, and run directories a reboot left behind."""
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from lotaas_reprocessing.periodicity import atomic_json

REPO = Path(__file__).resolve().parents[1]


def runner(tmp_path):
    from tests.test_tiers import runner as make
    run = make(tmp_path, stages='cpu')
    run.fp = 'f' * 64
    return run


def test_a_cpu_runner_stops_when_its_run_directory_is_removed(tmp_path, monkeypatch):
    run = runner(tmp_path)
    handoff = run.handoff()
    monkeypatch.setattr('euroflash.run.time.sleep', lambda seconds: shutil.rmtree(handoff, ignore_errors=True))
    with pytest.raises(RuntimeError, match='the head gave up on this run'):
        run.process_stream(poll=0)


def test_a_cpu_runner_gives_up_when_nothing_arrives_and_nothing_runs(tmp_path):
    run = runner(tmp_path)
    run.HANDOFF_IDLE_SECONDS = -1
    run.handoff()
    with pytest.raises(RuntimeError, match='the head has gone'):
        run.process_stream(poll=0)


def test_a_cpu_runner_busy_with_a_beam_is_not_idle(tmp_path):
    import threading
    run = runner(tmp_path)
    run.HANDOFF_IDLE_SECONDS = 0.5
    handoff = run.handoff()
    release = threading.Event()
    def search(item, output):
        assert release.wait(5)
        return []
    run.search = search
    run.fold_and_finish = lambda item, output, errors: None
    output = run.root/'processed'/'b1'/'x'
    output.mkdir(parents=True)
    atomic_json(output/'metadata.json', {'periodicity': {'multibeam_veto': False}})
    atomic_json(handoff/'b1.ready', {'item': 'b1', 'fingerprint': run.fp, 'output': 'processed/b1/x', 'input': '/x/b1.fil'})
    def finish_later():
        time.sleep(1.2)                       # longer than the idle limit, while the beam is searched
        atomic_json(handoff/'.done', {})
        release.set()
    threading.Thread(target=finish_later).start()
    run.process_stream(poll=0.01)
    assert (handoff/'b1.searched').exists()


def test_the_runner_kill_stops_the_runner_and_not_the_shell_that_carries_it(tmp_path):
    from euroflash.cluster import runner_kill
    inputs = str(tmp_path/'runs'/'campaign-x'/'input')
    other = str(tmp_path/'runs'/'campaign-y'/'input')
    sleeper = 'import time; time.sleep(60)'
    target = subprocess.Popen([sys.executable, '-c', sleeper, 'euroflash.run', '--prepared', '--input', inputs])
    bystander = subprocess.Popen([sys.executable, '-c', sleeper, 'euroflash.run', '--prepared', '--input', other])
    try:
        time.sleep(0.3)
        result = subprocess.run(['sh', '-c', runner_kill(inputs)])
        assert result.returncode == 0, 'pkill must not kill its own shell'
        assert target.wait(5) == -15
        assert bystander.poll() is None, "another run's runner is left alone"
    finally:
        for process in (target, bystander):
            if process.poll() is None:
                process.kill()


def test_stale_run_directories_go_and_live_ones_stay(tmp_path, monkeypatch):
    from euroflash import cluster
    monkeypatch.setattr(cluster, 'ssh_args', lambda node, control_dir=None: ['sh', '-c'])
    base = tmp_path/'lotaas-runs'
    old = time.time() - 3 * 86400
    for name in ('campaign-old', 'inject-old', 'campaign-current', 'campaign-live', 'validation-old'):
        (base/name/'work').mkdir(parents=True)
        (base/name/'work'/'ledger.sqlite').write_text('x')
    (base/'campaign-live'/'work'/'processed').mkdir()        # a beam added an hour ago
    for name in ('campaign-old', 'inject-old', 'campaign-current', 'validation-old', 'campaign-live'):
        for path in [base/name/'work'/'ledger.sqlite', base/name/'work', base/name]:
            os.utime(path, (old, old))
    removed = cluster.remove_stale_runs('efc-gpu-00', str(base) + '/', str(base/'campaign-current'))
    assert sorted(Path(p).name for p in removed) == ['campaign-old', 'inject-old']
    assert sorted(p.name for p in base.iterdir()) == ['campaign-current', 'campaign-live', 'validation-old']


def test_a_node_without_a_ledger_sends_none_back(tmp_path, monkeypatch):
    from euroflash import cluster
    from euroflash.ledger import Ledger
    monkeypatch.setattr(cluster, 'ssh_args', lambda node, control_dir=None: ['sh', '-c'])
    work = tmp_path/'work'
    work.mkdir()
    with pytest.raises(subprocess.CalledProcessError):
        cluster.snapshot('efc-cpu-00', str(work))
    assert not (work/'ledger-snapshot.sqlite').exists() and not (work/'ledger.sqlite').exists()
    ledger = Ledger(work/'ledger.sqlite')
    ledger.finish(ledger.start('beam', 'classify', 'fp', 'log', ['cmd']))
    cluster.snapshot('efc-cpu-00', str(work))
    assert (work/'ledger-snapshot.sqlite').stat().st_size > 0


# A fake cluster: each node is a directory, ssh runs the command there, and the runner is a stand-in.
FAKE_SSH = '''
import json, os, sys
node, command = sys.argv[1], ' '.join(sys.argv[2:])
if json.loads(os.environ['FAKE_BEHAVIOUR']).get(node) == 'refuse':
    sys.stderr.write('mux_client_request_session: session request failed: Session open refused by peer\\n')
    sys.exit(255)
local = command.replace('/lotaas-fake', os.environ['FAKE_NODES'] + '/' + node)
if 'python3 -m euroflash.run' in local:
    import shlex
    args = shlex.split(local.split('&&', 1)[1])
    os.execv(sys.executable, [sys.executable, os.environ['FAKE_RUNNER'], node] + args)
os.execvp('sh', ['sh', '-c', local])
'''

FAKE_RUNNER = '''
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.environ['FAKE_REPO'])
from euroflash.ledger import Ledger
node, args = sys.argv[1], sys.argv[2:]
stages = args[args.index('--stages') + 1]
inputs, work = Path(args[args.index('--input') + 1]), Path(args[args.index('--work') + 1])
behaviour = json.loads(os.environ['FAKE_BEHAVIOUR'])
handoff = work/'handoff'
if stages == 'gpu':
    if behaviour.get('gpu') == 'hang':
        while True:
            time.sleep(0.1)
    ledger = Ledger(work/'ledger.sqlite')
    handoff.mkdir(parents=True, exist_ok=True)
    for f in sorted(inputs.glob('*.fil')):
        output = work/'processed'/f.stem/'x'
        output.mkdir(parents=True, exist_ok=True)
        (output/'single_pulse.json').write_text('{}')
        head_view = '/lotaas-fake' + str(f)[len(os.environ['FAKE_NODES'] + '/' + node):]
        (handoff/(f.stem + '.ready')).write_text(json.dumps(
            {'item': f.stem, 'fingerprint': 'fp', 'output': 'processed/' + f.stem + '/x', 'input': head_view}))
        ledger.finish(ledger.start(f.stem, 'single_pulse', 'fp', 'log', ['cmd']))
    (handoff/'.done').write_text('{}')
    sys.exit(0)
ledger = Ledger(work/'ledger.sqlite')
while True:
    done = (handoff/'.done').is_file()
    for marker in handoff.glob('*.ready'):
        item = marker.name[:-6]
        if not (handoff/(item + '.searched')).exists():
            ledger.finish(ledger.start(item, 'classify', 'fp', 'log', ['cmd']))
            (handoff/(item + '.searched')).write_text('{}')
    if done:
        break
    time.sleep(0.05)
'''


def fake_cluster(tmp_path, monkeypatch, behaviour):
    from euroflash import cluster
    (tmp_path/'fake_ssh.py').write_text(FAKE_SSH)
    (tmp_path/'fake_runner.py').write_text(FAKE_RUNNER)
    nodes = tmp_path/'nodes'
    nodes.mkdir()
    monkeypatch.setenv('FAKE_NODES', str(nodes))
    monkeypatch.setenv('FAKE_RUNNER', str(tmp_path/'fake_runner.py'))
    monkeypatch.setenv('FAKE_REPO', str(REPO))
    monkeypatch.setenv('FAKE_BEHAVIOUR', json.dumps(behaviour))
    monkeypatch.setattr(cluster, 'ssh_args', lambda node, control_dir=None: [sys.executable, str(tmp_path/'fake_ssh.py'), node])
    monkeypatch.setattr(cluster, 'cpu_health', lambda node, *args: (True, 'probe'))
    monkeypatch.setattr(cluster, 'RELAY_POLL', 0.05)
    inputs = tmp_path/'input'
    inputs.mkdir()
    for beam in (13, 14, 15):
        (inputs/f'downsampled_L1_SAP000_BEAM{beam:03d}_32bit_ff.fil').write_bytes(b'beam')
    (tmp_path/'settings.yaml').write_text('a: 1\n')
    (tmp_path/'image.sif').write_bytes(b'image')
    argv = ['cluster', '--input', str(inputs), '--work', str(tmp_path/'results'), '--ledger', str(tmp_path/'ledger.sqlite'),
            '--nodes', 'efc-gpu-00', '--cpu-nodes', 'efc-cpu-00', 'efc-cpu-01', '--remote-root', '/lotaas-fake/runs',
            '--run-name', 'campaign-test', '--skip-health-check', '--cleanup-remote', '--image', str(tmp_path/'image.sif'),
            '--settings', str(tmp_path/'settings.yaml'), '--cpu-lock-dir', str(tmp_path/'locks')]
    monkeypatch.setattr(sys, 'argv', argv)
    return cluster, nodes


def test_a_cpu_node_refusing_sessions_hands_the_batch_to_another(tmp_path, monkeypatch):
    cluster, nodes = fake_cluster(tmp_path, monkeypatch, {'efc-cpu-00': 'refuse'})
    cluster.main()
    notes = (tmp_path/'results'/'cpu-tier.log').read_text()
    assert 'efc-cpu-00 failed before it took a beam' in notes and 'CPU tier on efc-cpu-01' in notes
    import sqlite3
    with sqlite3.connect(tmp_path/'results'/'efc-cpu-01'/'ledger-snapshot.sqlite') as db:
        searched = {item for item, in db.execute("SELECT item FROM attempts WHERE stage='classify' AND status='success'")}
    assert searched == {f'downsampled_L1_SAP000_BEAM{beam:03d}_32bit_ff' for beam in (13, 14, 15)}
    assert not (nodes/'efc-gpu-00'/'runs'/'campaign-test').exists() and not (nodes/'efc-cpu-01'/'runs'/'campaign-test').exists()


def test_without_any_cpu_node_the_gpu_runner_is_stopped_and_its_node_emptied(tmp_path, monkeypatch):
    cluster, nodes = fake_cluster(tmp_path, monkeypatch, {'efc-cpu-00': 'refuse', 'efc-cpu-01': 'refuse', 'gpu': 'hang'})
    started = time.monotonic()
    with pytest.raises(RuntimeError, match='Pipeline failed on efc-gpu-00'):
        cluster.main()
    assert time.monotonic() - started < 60, 'the hanging GPU runner was stopped, not waited for'
    notes = (tmp_path/'results'/'cpu-tier.log').read_text()
    assert notes.count('failed before it took a beam') == 2 and 'Stopping the GPU runner on efc-gpu-00' in notes
    assert 'GPU node efc-gpu-00 could not be collected' in notes, 'its runner never wrote a ledger'
    assert not (nodes/'efc-gpu-00'/'runs'/'campaign-test').exists(), 'cleaned although the collect failed'
    assert (tmp_path/'results'/'efc-gpu-00.gpu-done').exists()


def fake_catalogue(path, version):
    import io, tarfile
    data = f'#CATALOGUE {version}\n#\nPSRJ     J0000+00\n@-----\n'.encode()
    with tarfile.open(path, 'w:gz') as archive:
        info = tarfile.TarInfo('psrcat_tar/psrcat.db'); info.size = len(data)
        archive.addfile(info, io.BytesIO(data))


def test_every_node_uses_the_pinned_catalogue(tmp_path):
    from lotaas_reprocessing import atnf
    assert atnf.CATALOGUE_VERSION == '2.8.1' and atnf.catalogue_version(atnf.PINNED) == '2.8.1'
    cache = tmp_path/'psrqpy'
    assert atnf.pin_catalogue(cache) is True, 'a node without a cache gets the pinned copy, not a download'
    assert atnf.catalogue_version(cache/'psrcat_pkg.tar.gz') == '2.8.1'
    (cache/'query_latest.pkl').write_bytes(b'parsed')
    assert atnf.pin_catalogue(cache) is False and (cache/'query_latest.pkl').exists(), 'the right version stays'
    fake_catalogue(cache/'psrcat_pkg.tar.gz', '2.9.0')
    assert atnf.pin_catalogue(cache) is True and not (cache/'query_latest.pkl').exists()
    assert atnf.catalogue_version(cache/'psrcat_pkg.tar.gz') == '2.8.1'
    (cache/'psrcat_pkg.tar.gz').write_bytes(b'truncated')
    assert atnf.pin_catalogue(cache) is True, 'a broken cache is repaired'


def test_the_pinned_catalogue_travels_with_the_source_and_the_head_reads_it(tmp_path):
    from euroflash import cluster, psrcat
    settings, image = tmp_path/'settings.yaml', tmp_path/'image.sif'
    settings.write_text('a: 1\n'); image.write_bytes(b'image')
    names = [name for _, name in cluster.freeze_source(cluster.REPO, settings, image)]
    assert 'lotaas_reprocessing/data/psrcat_pkg-2.8.1.tar.gz' in names
    assert psrcat.DEFAULT == psrcat.PINNED and len(psrcat.load(psrcat.PINNED)) == 4213
