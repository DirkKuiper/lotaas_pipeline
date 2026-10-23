"""Ingest on CPU nodes (euroflash.ingest): the node's streaming unpack and conversion, and the head's side."""
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from euroflash import ingest as I


class Response(io.BytesIO):
    """An HTTP body as open_with_any returns it."""

    def __init__(self, data, length=None):
        super().__init__(data)
        self.headers = {'Content-Length': str(len(data) if length is None else length)}


def archive(members):
    """A tar body with {name: bytes} members, as the LTA serves it."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return stream.getvalue()


def test_the_archive_is_unpacked_as_it_arrives_and_hashed_whole(tmp_path):
    body = archive({'stokes/SAP0/BEAM13/L1_SAP0_BEAM13_2bit.fits': b'f' * 5000, 'stokes/README': b'notes'})
    fits, sha256, count = I.stream_extract(Response(body), len(body), tmp_path)
    assert [p.relative_to(tmp_path).as_posix() for p in fits] == ['stokes/SAP0/BEAM13/L1_SAP0_BEAM13_2bit.fits']
    assert fits[0].read_bytes() == b'f' * 5000
    assert (sha256, count) == (hashlib.sha256(body).hexdigest(), len(body))    # the padding too


def test_a_short_body_an_unsafe_member_and_no_fits_are_refused(tmp_path):
    body = archive({'stokes/SAP0/BEAM13/L1_SAP0_BEAM13_2bit.fits': b'f' * 5000})
    with pytest.raises(IOError, match='Incomplete HTTP body'):
        I.stream_extract(Response(body), len(body) + 512, tmp_path/'a')
    with pytest.raises(ValueError, match='Unsafe'):
        I.stream_extract(Response(archive({'../escape.fits': b'x'})), None, tmp_path/'b')
    body = archive({'notes.txt': b'x'})
    with pytest.raises(ValueError, match='No PSRFITS'):
        I.stream_extract(Response(body), len(body), tmp_path/'c')


def fixture_archive(tmp_path, beam=13, incoherent=False):
    sys.path.insert(0, str(Path(__file__).parent))
    from make_fixture import make
    make(tmp_path/'fits')
    raw = tmp_path/'fits'/f'L000000_SAP000_BEAM{beam:03d}.fits'
    folder = 'incoherentstokes' if incoherent else 'stokes'
    return raw, archive({f'{folder}/SAP0/BEAM{beam}/{raw.name}': raw.read_bytes()})


def test_a_node_converts_to_the_same_filterbank_as_the_head_and_leaves_no_scratch(tmp_path, monkeypatch):
    from euroflash.run import CONVERSION_SCRIPT, REPO
    raw, body = fixture_archive(tmp_path)
    monkeypatch.setattr('euroflash.download.open_with_any', lambda url, tokens, headers: Response(body))
    spec = {'url': 'https://lta/L1_SAP000_B013.tar', 'tokens': ['t'], 'item': 'L1_SAP000_B013', 'max_bytes': 2**30,
            'exclude_beams': [12], 'data_root': str(tmp_path/'data'), 'scratch': str(tmp_path/'scratch')}
    result = I.node_job(spec)
    output = tmp_path/'data'/'L000000'/'SAP000'/'B013'/'downsampled_L000000_SAP000_BEAM013_32bit.fil'
    assert result['ok'] and not result['excluded'] and result['outputs'] == [str(output)]
    assert result['receipt']['sha256'] == hashlib.sha256(body).hexdigest()
    assert result['fits'][0]['name'] == raw.name and result['fits'][0]['bytes'] == raw.stat().st_size
    head = tmp_path/'head.fil'
    subprocess.run([sys.executable, str(REPO/CONVERSION_SCRIPT), '-o', str(head), str(raw)], check=True,
                   capture_output=True)
    assert output.read_bytes() == head.read_bytes()
    assert list((tmp_path/'scratch').iterdir()) == []
    assert [p.name for p in output.parent.iterdir()] == [output.name]           # no partial copy left


def test_an_incoherent_beam_is_not_converted(tmp_path, monkeypatch):
    _, body = fixture_archive(tmp_path, beam=14, incoherent=True)
    monkeypatch.setattr('euroflash.download.open_with_any', lambda url, tokens, headers: Response(body))
    result = I.node_job({'url': 'u', 'tokens': ['t'], 'item': 'x', 'max_bytes': 2**30, 'exclude_beams': [12],
                         'data_root': str(tmp_path/'data'), 'scratch': str(tmp_path/'scratch')})
    assert result['excluded'] and result['outputs'] == [] and not (tmp_path/'data').exists()


def test_stale_scratch_is_cleared_and_fresh_scratch_kept(tmp_path):
    import os
    old, new = tmp_path/'old-1', tmp_path/'new-2'
    old.mkdir(); new.mkdir()
    os.utime(old, (0, 0))
    I.clean_stale(tmp_path)
    assert not old.exists() and new.exists()


def completed(stdout='', stderr='', code=0):
    return subprocess.CompletedProcess([], code, stdout.encode(), stderr.encode())


def head(tmp_path, outcome):
    ingest = I.Ingest(['efc-cpu-04', 'efc-cpu-05'], '.', 'settings.yaml', 'image.sif', log_dir=tmp_path/'logs')
    ingest.prepare = lambda node: '/home/u/lotaas-ingest/abc'
    ingest.ssh = lambda node, command, **kwargs: outcome
    return ingest


def test_the_head_reads_the_nodes_result_and_tells_node_from_file_failures(tmp_path):
    ok = {'ok': True, 'receipt': {}, 'fits': [], 'excluded': False, 'outputs': ['/shared/x.fil']}
    result = head(tmp_path, completed('conversion chatter\n' + I.RESULT_PREFIX + json.dumps(ok))).run('efc-cpu-04', {}, 'a')
    assert result['outputs'] == ['/shared/x.fil'] and result['node'] == 'efc-cpu-04'
    refused = {'ok': False, 'kind': 'permission', 'error': 'All 1 macaroons were refused'}
    with pytest.raises(PermissionError, match='macaroons'):
        head(tmp_path, completed(I.RESULT_PREFIX + json.dumps(refused), code=1)).run('efc-cpu-04', {}, 'b')
    broken = {'ok': False, 'kind': 'OSError', 'error': 'Incomplete HTTP body'}
    with pytest.raises(RuntimeError, match='Incomplete HTTP body') as failure:
        head(tmp_path, completed(I.RESULT_PREFIX + json.dumps(broken), code=1)).run('efc-cpu-04', {}, 'c')
    assert not isinstance(failure.value, I.NodeUnavailable)
    with pytest.raises(I.NodeUnavailable):
        head(tmp_path, completed(stderr='mux_client_request_session: session request failed', code=255)).run(
            'efc-cpu-05', {}, 'd')
    assert sorted(p.name for p in (tmp_path/'logs').iterdir()) == ['b.log', 'c.log', 'd.log']


def test_jobs_go_to_the_least_busy_node_and_a_node_down_takes_none(tmp_path):
    ingest = I.Ingest(['efc-cpu-04', 'efc-cpu-05', 'efc-cpu-06'], '.', 's', 'i')
    nodes = [ingest.acquire() for _ in range(4)]
    assert nodes == ['efc-cpu-04', 'efc-cpu-05', 'efc-cpu-06', 'efc-cpu-04']
    ingest.release('efc-cpu-05')
    ingest.back_off('efc-cpu-06')
    assert ingest.acquire() == 'efc-cpu-05'
    ingest.back_off('efc-cpu-04'); ingest.back_off('efc-cpu-05')
    assert ingest.acquire() is None


def test_the_job_runs_the_frozen_code_in_the_image_with_scratch_and_shared_bound():
    ingest = I.Ingest(['efc-cpu-04'], '.', 's', 'i', scratch='/tmp/lotaas-ingest',
                      shared=('/shared/results/dkuiper/lotaas',))
    command = ingest.command('/home/u/lotaas-ingest/abc')
    assert command.startswith('mkdir -p /tmp/lotaas-ingest && cd /home/u/lotaas-ingest/abc && apptainer exec')
    assert '--bind /tmp/lotaas-ingest --bind /shared/results/dkuiper/lotaas' in command
    assert command.endswith('/home/u/lotaas-ingest/abc/containers/runtime.sif python -m euroflash.ingest node')


def test_ingest_runs_through_its_own_master_beside_the_search_tiers():
    from pathlib import Path
    ingest = I.Ingest(['efc-cpu-04'], '.', 's', 'i', control_dir=Path('/home/u/.ssh/control'))
    assert ingest.ssh_args('efc-cpu-04') == ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                                             '-S', '/home/u/.ssh/control/ingest-cpu04', 'efc-cpu-04']
