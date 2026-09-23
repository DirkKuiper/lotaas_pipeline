import json
import os
import numpy as np
import pytest
from pathlib import Path
from preproc.downsample_psrfits2fil_32bit import unpack, convert
from lotaas_reprocessing.dedispersion import iter_dedispersed, time_scrunch
from lotaas_reprocessing.filterbank import FilterbankFile
from lotaas_reprocessing.matched_filter import run_matched_filtering
from euroflash.ledger import Ledger
from euroflash.download import extract
from staging.client import token_for_url


def test_psrfits_two_bit_values():
    actual = unpack(np.array([0b00011011, 0b11100100], dtype=np.uint8), 2, 2, 4)
    np.testing.assert_array_equal(actual, [[0, 1, 2, 3], [3, 2, 1, 0]])


@pytest.mark.parametrize('factor', [1, 2, 4])
def test_dispersed_pulse_recovered_at_correct_time(factor):
    tsamp, dm = .01, 50.
    frequencies = np.linspace(120., 160., 16)
    n = 4097
    arrival = 10.
    t = np.arange(n) * tsamp
    delays = dm * (frequencies**-2 - frequencies.max()**-2) / 2.41e-4
    data = np.exp(-.5 * ((t[None, :] - arrival - delays[:, None]) / .1)**2).astype('float32')
    _, trial = next(iter_dedispersed(data, tsamp, frequencies, [dm], factor))
    assert len(trial) == n // factor
    assert abs(np.argmax(trial)*tsamp*factor - arrival) <= tsamp*factor
    assert trial.max() > 15


def test_zero_dm_is_sum_of_time_scrunched_channels():
    x = np.random.default_rng(7).normal(size=(5, 101)).astype('float32')
    _, trial = next(iter_dedispersed(x, .1, np.arange(120., 125.), [0], 2))
    np.testing.assert_allclose(trial, time_scrunch(x, 2).sum(axis=0), atol=2e-6)


def test_gpu_matches_cpu_when_available():
    cp = pytest.importorskip('cupy')
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA driver unavailable on this host')
    if not count:
        pytest.skip('No visible GPU')
    x = np.random.default_rng(1).normal(size=(8, 1001)).astype('float32')
    freqs = np.linspace(120., 160., 8)
    cpu = list(iter_dedispersed(x, .1, freqs, [0, 20, 150], 2))
    gpu = list(iter_dedispersed(x, .1, freqs, [0, 20, 150], 2, cp))
    for (_, a), (_, b) in zip(cpu, gpu):
        np.testing.assert_allclose(a, cp.asnumpy(b), atol=2e-5, rtol=2e-4)


def test_matched_filter_odd_length_and_constant_input(tmp_path):
    p = tmp_path/'zero.dat'
    for value in [0., 1.]:
        np.full(1001, value, dtype='float32').tofile(p)
        result = run_matched_filtering(p, .1, 10)
        assert all(len(x) == 0 for x in result)


def test_short_noise_observation_does_not_generate_constant_window_detections(tmp_path):
    signal=np.random.default_rng(35).normal(size=8192).astype('float32')
    signal-=signal.mean()
    path=tmp_path/'short-noise.dat';signal.tofile(path)
    _,_,snr,_=run_matched_filtering(path,.00786432,10)
    assert len(snr)<100
    assert np.all(snr<10)


def test_candidate_limit_precedes_large_plot(tmp_path,monkeypatch):
    from lotaas_reprocessing import matched_filter,cluster
    monkeypatch.setattr(cluster,'MAX_CANDIDATES',2)
    monkeypatch.setattr(matched_filter,'run_matched_filtering',lambda *args,**kwargs:tuple(np.ones(3) for _ in range(4)))
    (tmp_path/'beam_DM10.0.dat').write_bytes(b'test')
    with pytest.raises(RuntimeError,match='Too many candidates'):
        matched_filter.run_all_matched_filtering(tmp_path,.1,tmp_path,{},[])
    assert not (tmp_path/'all_matched_filter_overview.png').exists()


def test_the_dedispersion_wrap_tail_is_excluded_from_the_search(tmp_path):
    """Dedispersion advances each channel circularly, so what occupied the
    first samples of the low-frequency channels reappears at the end."""
    from lotaas_reprocessing.matched_filter import (
        run_all_matched_filtering, wrap_contaminated_samples)
    tsamp, n, dm = .01, 4096, 100.
    nu = np.linspace(119.45, 151.04, 64)
    polluted = wrap_contaminated_samples(dm, nu.min(), nu.max(), tsamp)
    assert 0 < polluted < n

    data = np.zeros((len(nu), n), dtype='float32')
    data[:, 0:3] = 50.                       # broadband interference at the start
    _, trial = next(iter_dedispersed(data, tsamp, nu, [dm]))
    trials = tmp_path/'DM_trials'
    trials.mkdir()
    np.asarray(trial).astype('float32').tofile(trials/'beam_DM100.0.dat')
    plan = [{'low_dm': 0., 'high_dm': 200., 'ddm': .1, 'downsample': 1}]
    info = {'Object': 'x', 'Telescope': 'x', 'Instrument': 'x',
            'Observation Date': 'x', 'Frequency Range (MHz)': 'x'}

    def search(**kwargs):
        out = tmp_path/('out' + str(len(list(tmp_path.glob('out*')))))
        out.mkdir()
        run_all_matched_filtering(trials, tsamp, str(out), info, plan, **kwargs)
        rows = [line.split() for line in
                (out/'all_detected_candidates.cands').read_text().splitlines()
                if not line.startswith('#')]
        return np.array([[float(r[2]), float(r[1])] for r in rows]) if rows else np.empty((0, 2))

    untrimmed = search(trim_wrap=False)
    trimmed = search(nu_min=float(nu.min()), nu_max=float(nu.max()))

    tail_start = (n - polluted) * tsamp
    assert np.any(untrimmed[:, 0] >= tail_start), 'expected the wrap artefact to be found'
    assert not np.any(trimmed[:, 0] >= tail_start), 'wrap artefact still reported'


def _cluster(tmp_path, rows, name):
    from lotaas_reprocessing.cluster import cluster_candidates
    import pandas as pd
    source, output = tmp_path/(name+'.cands'), tmp_path/(name+'.tsv')
    source.write_text(''.join(f'{dm} {snr} {time} {round(time/DT)} 1\n'
                              for dm, snr, time in rows))
    cluster_candidates(source, output)
    return pd.read_csv(output, sep='\t')


def test_isolated_events_are_not_collapsed_into_one_candidate(tmp_path):
    """The DBSCAN noise label was treated as a cluster, so every detection in
    a beam without a neighbour was reduced to a single reported row."""
    rows = [(20, 12, 100), (40, 11, 200), (60, 10, 300)]
    assert len(_cluster(tmp_path, rows, 'isolated')) == 3


def test_distant_dms_do_not_collide_on_the_cluster_grid(tmp_path):
    """DM/ddm mapped both DM 50.2 and DM 150.6 to 502."""
    rows = [(50.2, 12, 100), (150.6, 11, 100)]
    assert len(_cluster(tmp_path, rows, 'collision')) == 2


def test_one_pulse_across_a_plan_boundary_stays_one_candidate(tmp_path):
    """DM/ddm mapped adjacent trials DM 150.5 and DM 150.6 to 1505 and 502."""
    rows = [(150.5, 12, 100), (150.6, 11, 100)]
    assert len(_cluster(tmp_path, rows, 'boundary')) == 1


def test_a_source_repeating_every_four_seconds_keeps_its_pulses(tmp_path):
    """A five-second tolerance merged a repeater into one candidate."""
    rows = [(30, 12, 100), (30, 11, 104), (30, 10, 108)]
    assert len(_cluster(tmp_path, rows, 'repeater')) == 3


def test_trial_position_is_monotonic_over_the_whole_plan():
    from lotaas_reprocessing.cluster import dm_trial_position
    position = dm_trial_position(np.linspace(0, 10019, 200000))
    assert np.all(np.diff(position) > 0)
    assert dm_trial_position([150.5])[0] == 1505
    assert dm_trial_position([150.6])[0] == 1506


def test_cell_clustering_matches_dbscan_including_duplicates_and_boundaries():
    from sklearn.cluster import DBSCAN
    from lotaas_reprocessing.cluster import cluster_labels
    rng=np.random.default_rng(71)
    cases=[rng.uniform(-100,100,(2000,2)),
           np.concatenate([rng.normal(size=(800,2)),rng.normal(size=(600,2))+50]),
           np.array([[0,0],[3,4],[8,4],[20,20],[20,20],[99,99],[-3,-4]],dtype=float)]
    for points in cases:
        for eps in [1.,5.,10.]:
            expected=DBSCAN(eps=eps,min_samples=2).fit_predict(points)
            actual=cluster_labels(points,eps)
            np.testing.assert_array_equal(actual,expected)


def test_fingerprint_survives_copying_the_image_and_growing_the_batch(tmp_path):
    """The fingerprint hashed absolute paths, modification times and every
    input in the run, so the head and a node disagreed about identical work
    and adding one beam invalidated the rest of the batch."""
    import shutil
    from euroflash.run import fingerprint
    settings = tmp_path/'settings.yaml'
    settings.write_text('rfi_block_size: 1000\n')
    image = tmp_path/'runtime.sif'
    image.write_bytes(b'image-contents')
    options = {'backend': 'gpu', 'pilot': False}
    baseline = fingerprint(settings, image, options)

    # Same image, copied to another path on another machine's layout.
    elsewhere = tmp_path/'node'/'source'/'containers'/'runtime.sif'
    elsewhere.parent.mkdir(parents=True)
    shutil.copy2(image, elsewhere)
    os.utime(elsewhere, (1, 1))
    assert fingerprint(settings, elsewhere, options) == baseline

    # Different image contents must still be a different search.
    other = tmp_path/'other.sif'
    other.write_bytes(b'image-contents-rebuilt')
    assert fingerprint(settings, other, options) != baseline

    # Settings and options remain part of the identity.
    settings.write_text('rfi_block_size: 2000\n')
    assert fingerprint(settings, image, options) != baseline


def test_image_digest_is_cached_and_rehashes_when_the_image_changes(tmp_path):
    from euroflash.run import image_digest
    image = tmp_path/'runtime.sif'
    image.write_bytes(b'first')
    first = image_digest(image)
    assert (tmp_path/'runtime.sif.sha256').is_file()
    assert image_digest(image) == first
    image.write_bytes(b'second')
    assert image_digest(image) != first


def test_reclaim_removes_only_trials_that_cannot_be_retried(tmp_path):
    """Trials are kept when the CPU stage fails, so it can be retried without
    redoing dedispersion. Unbounded, a burst of failures fills a node disk."""
    import sqlite3, time
    from euroflash.reclaim import survey
    ledger = tmp_path/'ledger.sqlite'
    now = time.time()
    with sqlite3.connect(ledger) as db:
        db.execute('CREATE TABLE attempts(id INTEGER PRIMARY KEY, item TEXT, stage TEXT,'
                   ' fingerprint TEXT, status TEXT, finished REAL)')
        rows = [('done', 'success', now - 10),          # classified: trials are dead
                ('stale', 'failed', now - 30*86400),    # long past retry
                ('recent', 'failed', now - 3600),       # still worth retrying
                ('running', 'running', None)]           # CPU stage in flight
        for index, (item, status, finished) in enumerate(rows):
            db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?)',
                       (index, item, 'classify', 'f'*64, status, finished))
    for item, _, _ in [(r[0], 0, 0) for r in rows]:
        trials = tmp_path/'work'/'processed'/item/('f'*16)/'DM_trials'
        trials.mkdir(parents=True)
        (trials/'trial.dat').write_bytes(b'x'*1024)

    found = survey(tmp_path/'work', ledger, retention_seconds=7*86400, now=now)
    assert {path.parent.parent.name for path, _, _ in found} == {'done', 'stale'}
    assert all(size == 1024 for _, size, _ in found)


def test_resume_rechecks_output_and_records_errors(tmp_path):
    ledger = Ledger(tmp_path/'runs.sqlite')
    output = tmp_path/'result'
    output.write_text('verified')
    job = ledger.start('beam', 'search', 'codehash', tmp_path/'log', ['python'])
    ledger.finish(job, [output])
    assert ledger.completed('beam', 'search', 'codehash')
    assert not ledger.completed('beam', 'search', 'different-code')
    output.write_text('truncated')
    assert not ledger.completed('beam', 'search', 'codehash')
    job = ledger.start('beam', 'search', 'codehash', tmp_path/'log', ['python'])
    ledger.finish(job, error='CUDA failure')
    assert ledger.summary()['errors'][0]['error'] == 'CUDA failure'


def test_site_specific_tokens():
    manifest = {'macaroons': [dict(ltaSite={'name': 'SURF'}, content='surf-token', validUntil='2027'),
                             dict(ltaSite={'name': 'JUELICH'}, content='juelich-token', validUntil='2027')]}
    assert token_for_url(manifest, 'https://webdav.grid.surfsara.nl:2882/data') == 'surf-token'
    with pytest.raises(ValueError):
        token_for_url(manifest, 'https://unrelated.example/data')


def _scoped_macaroon(path, valid_until):
    """A macaroon-shaped blob carrying a readable dCache path caveat."""
    import base64
    body = b'\x02\x01lofar\x00\x02\x01path:' + path.encode() + b'\x00\x06signature'
    return dict(ltaSite={'name': 'SURF'}, validUntil=valid_until,
                content=base64.urlsafe_b64encode(body).decode().rstrip('='))


def test_a_macaroon_is_chosen_by_path_not_by_expiry():
    """A request spanning several observations carries a macaroon per path.
    Choosing the one expiring last gave dCache a token scoped to a different
    directory, which answers 403 'Permission denied for GET on path ...'."""
    from staging.client import tokens_for_url, macaroon_paths
    manifest = {'macaroons': [_scoped_macaroon('/lt5_004/1163405', '2027-12-31'),
                              _scoped_macaroon('/lt5_004/1261459', '2027-01-01')]}
    assert macaroon_paths(manifest['macaroons'][0]['content']) == ['/lt5_004/1163405']

    # The right macaroon wins even though the other one lives longer.
    ordered = tokens_for_url(
        manifest, 'https://webdav.grid.surfsara.nl:2882/lt5_004/1261459/L1261459_B000.tar')
    assert ordered[0] == manifest['macaroons'][1]['content']
    assert len(ordered) == 2  # The other is still offered as a fallback.

    ordered = tokens_for_url(
        manifest, 'https://webdav.grid.surfsara.nl:2882/lt5_004/1163405/L1163405_B013.tar')
    assert ordered[0] == manifest['macaroons'][0]['content']


def test_an_unscoped_macaroon_still_sorts_before_one_scoped_elsewhere():
    from staging.client import tokens_for_url
    unscoped = dict(ltaSite={'name': 'SURF'}, content='plain-token', validUntil='2020')
    manifest = {'macaroons': [_scoped_macaroon('/lt5_004/9999999', '2030'), unscoped]}
    ordered = tokens_for_url(manifest, 'https://webdav.grid.surfsara.nl:2882/lt5_004/1261459/x.tar')
    assert ordered[0] == 'plain-token'


def test_a_refused_macaroon_falls_through_to_the_next(monkeypatch):
    from urllib.error import HTTPError
    from euroflash import download as module
    tried = []

    def fake_urlopen(request, timeout=None):
        token = request.get_header('Authorization')
        tried.append(token)
        if token != 'Bearer good':
            raise HTTPError(request.full_url, 403, 'Permission denied for GET on path', {}, None)
        return 'response'

    monkeypatch.setattr(module, 'urlopen', fake_urlopen)
    assert module.open_with_any('https://host/x', ['bad', 'good'], {}) == 'response'
    assert tried == ['Bearer bad', 'Bearer good']

    with pytest.raises(PermissionError, match='All 2 macaroons were refused'):
        module.open_with_any('https://host/x', ['bad', 'worse'], {})


def test_a_non_authorisation_error_is_not_retried(monkeypatch):
    from urllib.error import HTTPError
    from euroflash import download as module
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(1)
        raise HTTPError(request.full_url, 500, 'Server error', {}, None)

    monkeypatch.setattr(module, 'urlopen', fake_urlopen)
    with pytest.raises(HTTPError):
        module.open_with_any('https://host/x', ['a', 'b'], {})
    assert len(calls) == 1


def test_node_health_refuses_a_host_without_uvm(monkeypatch):
    """Beams were split across the named nodes whether or not they could run.
    efc-gpu-00 has the UVM module blocked, so cuInit returns 999 and its whole
    share of the batch failed after the transfer."""
    import subprocess
    from euroflash import cluster

    replies = {
        'healthy': (0, 'GPU 0: RTX PRO 6000\nGPU 1: RTX PRO 6000\nUVM_PRESENT\n', ''),
        'no-uvm': (1, 'GPU 0: RTX PRO 6000\n', ''),
        'unreachable': (255, '', 'Permission denied (publickey).'),
        'no-gpus': (0, 'UVM_PRESENT\n', ''),
    }

    def fake_run(args, **kwargs):
        node = args[-2]
        code, out, err = replies[node]
        return subprocess.CompletedProcess(args, code, out, err)

    monkeypatch.setattr(cluster.subprocess, 'run', fake_run)
    assert cluster.node_health('healthy')[0] is True
    assert cluster.node_health('no-uvm') == (False, 'no /dev/nvidia-uvm; CUDA cannot initialise on this host')
    assert cluster.node_health('unreachable')[0] is False
    assert cluster.node_health('no-gpus') == (False, 'nvidia-smi listed no GPUs')


def test_node_health_treats_a_hanging_probe_as_unusable(monkeypatch):
    import subprocess
    from euroflash import cluster

    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 120)

    monkeypatch.setattr(cluster.subprocess, 'run', fake_run)
    assert cluster.node_health('slow') == (False, 'health probe timed out')


def test_tar_traversal_rejected(tmp_path):
    import io, tarfile
    p = tmp_path/'bad.tar'
    with tarfile.open(p, 'w') as f:
        member = tarfile.TarInfo('../escape.fits'); member.size = 4
        f.addfile(member, io.BytesIO(b'fits'))
    with pytest.raises(ValueError, match='Unsafe'):
        extract(p, tmp_path/'extract')
    assert not (tmp_path/'escape.fits').exists()


def test_concurrent_prepared_beams_have_separate_outputs(tmp_path):
    from argparse import Namespace
    from euroflash.run import Runner
    image = tmp_path/'image.sif'; image.write_bytes(b'test')
    settings = tmp_path/'settings.yaml'; settings.write_text('dedispersion_plan: []')
    inputs = tmp_path/'inputs'; inputs.mkdir()
    beams = [inputs/'beam13_ff.fil', inputs/'beam15_ff.fil']
    for p in beams: p.write_bytes(b'beam')
    args = Namespace(work=tmp_path/'work', ledger=tmp_path/'ledger.sqlite', settings=settings,
                     image=image, input=inputs, backend='gpu', gpus='0,1', cpu_workers=2,
                     pilot=True, max_samples=None)
    runner=Runner(args);runner.fp='abc'
    outputs={}
    def step(item,stage,command,expected,gpu=None,**kwargs):
        if stage=='dedisperse':
            idx=command.index(str(Path(__file__).resolve().parents[1]/'pipeline/pipeline_gpu.py'))
            outputs[item]=command[idx+2]
    runner.step=step
    runner.analyze=lambda item,output: None
    runner.process(beams)
    assert len(outputs)==2
    assert len(set(outputs.values()))==2
    assert all(str(tmp_path/'work') in p for p in outputs.values())


def test_import_updates_attempt_instead_of_duplicating(tmp_path):
    from euroflash.collect import collect
    source=tmp_path/'source.sqlite';destination=tmp_path/'campaign.sqlite'
    ledger=Ledger(source)
    attempt=ledger.start('beam','search','version','log',['python'])
    collect(source,destination,'gpu01/run1')
    ledger.finish(attempt,error='observed failure')
    collect(source,destination,'gpu01/run1')
    with Ledger(destination).connect() as db:
        rows=db.execute('SELECT * FROM attempts').fetchall()
    assert len(rows)==1
    assert rows[0]['status']=='failed'
    assert rows[0]['error']=='observed failure'


DT = .007864319719374176


def test_boxcar_recovers_the_full_root_w_gain():
    """The superseded tanh-smoothed kernel returned 0.707 of the ideal
    statistic at width one, where most single pulses are found."""
    from lotaas_reprocessing.matched_filter import boxcar_statistic
    n = 8192
    for width in [1, 2, 3, 4, 9, 20]:
        pulse = np.zeros(n)
        pulse[1000:1000 + width] = 1.
        cumulative = np.empty(n + 1)
        cumulative[0] = 0
        np.cumsum(pulse, out=cumulative[1:])
        total = cumulative[width:] - cumulative[:-width]
        assert abs(total.max() / width - 1.) < 1e-9


def test_statistic_is_calibrated_so_a_threshold_keeps_its_meaning():
    from lotaas_reprocessing.matched_filter import boxcar_statistic
    x = np.random.default_rng(7).normal(size=457728)
    cumulative = np.empty(x.size + 1)
    cumulative[0] = 0
    np.cumsum(x, out=cumulative[1:])
    for width in [1, 4, 32, 1024]:
        statistic = boxcar_statistic(cumulative, width, np.random.default_rng(0))
        assert abs(statistic.std() - 1.) < .05
        assert abs(statistic.mean()) < .05


@pytest.mark.parametrize('seconds,amplitude', [(30, 100), (120, 100), (300, 100), (600, 100)])
def test_a_bright_long_pulse_does_not_suppress_itself(tmp_path, seconds, amplitude):
    """Normalising by the standard deviation of a response containing the
    signal put a 100-sigma pulse of 300 s at 4.09, below the threshold of 5."""
    nsamp = 457728
    t = np.arange(nsamp) * DT
    x = np.random.default_rng(390).normal(size=nsamp)
    x[abs(t - 1800) < seconds / 2] += amplitude
    x = (x - x.mean()).astype('float32')
    path = tmp_path / 'broad.dat'
    x.tofile(path)
    _, _, snr, _ = run_matched_filtering(path, DT, 10)
    assert snr.size and snr.max() > 7


def test_an_event_at_the_end_is_not_reported_at_time_zero(tmp_path):
    """Circular convolution reported a pulse in the last ten samples as 81
    detections in the first 0.1 seconds."""
    x = np.random.default_rng(390).normal(size=8192).astype('float32')
    x[-10:] += 100
    x -= x.mean()
    path = tmp_path / 'edge.dat'
    x.tofile(path)
    times, _, strengths, _ = run_matched_filtering(path, DT, 30)
    assert not np.any(times < .1)
    assert strengths.size and times[strengths.argmax()] > 8192 * DT * .9


def test_searching_one_trial_twice_gives_the_same_candidates(tmp_path):
    x = np.random.default_rng(11).normal(size=100000).astype('float32')
    x[50000:50004] += 12
    path = tmp_path / 'repeat.dat'
    x.tofile(path)
    first = run_matched_filtering(path, DT, 10)
    second = run_matched_filtering(path, DT, 10)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_mixed_or_truncated_dm_trials_are_rejected(tmp_path):
    from pipeline.pipeline_cpu import validate_trials
    metadata={'filename':'/data/beam13.fil','samples_processed':101,
              'dedispersion_plan':[dict(low_dm=10.,high_dm=12.,ddm=1.,downsample=2)]}
    for dm in [10,11]:np.zeros(50,dtype='float32').tofile(tmp_path/f'beam13_DM{dm:.1f}.dat')
    validate_trials(metadata,tmp_path)
    extra=tmp_path/'beam15_DM10.0.dat';extra.write_bytes(b'unrelated')
    with pytest.raises(ValueError,match='mixed'):validate_trials(metadata,tmp_path)
    extra.unlink()
    (tmp_path/'beam13_DM10.0.dat').write_bytes(b'truncated')
    with pytest.raises(ValueError,match='incorrect'):validate_trials(metadata,tmp_path)


def test_survey_dm_grid_has_exclusive_endpoints():
    import yaml
    from lotaas_reprocessing.dm_plan import dm_values, dm_label
    plans=yaml.safe_load((Path(__file__).resolve().parents[1]/'settings.yaml').read_text())['dedispersion_plan']
    labels=[]
    for plan in plans:
        dms=dm_values(plan)
        assert dms[0]==plan['low_dm']
        assert all(plan['low_dm']<=dm<plan['high_dm'] for dm in dms)
        labels.extend(map(dm_label,dms))
    assert len(labels)==len(set(labels))==4407
    assert dm_values(dict(low_dm=.1,high_dm=.4,ddm=.1))==[.1,.2,.3]
    assert dm_label(.05)=='0.05'


def test_import_remaps_detection_beam_run_ids(tmp_path):
    import sqlite3
    from euroflash.collect import collect
    from db.initialize_db import initialize_database
    source=tmp_path/'node.sqlite';destination=tmp_path/'head.sqlite'
    initialize_database(str(source));initialize_database(str(destination))
    with sqlite3.connect(destination) as db:
        db.execute("INSERT INTO beam_runs(beam_id) VALUES ('existing')")
    with sqlite3.connect(source) as db:
        db.execute("INSERT INTO beam_runs(beam_id) VALUES ('remote')")
        db.execute("INSERT INTO detections(beam_id,beam_run_id) VALUES ('remote',1)")
    collect(source,destination,'node/run2')
    collect(source,destination,'node/run2')
    with sqlite3.connect(destination) as db:
        rows=db.execute('SELECT b.beam_id FROM detections d JOIN beam_runs b ON b.id=d.beam_run_id').fetchall()
    assert rows==[('remote',)]


def test_legacy_numpy_probability_becomes_sql_real(tmp_path):
    import sqlite3
    from db.initialize_db import initialize_database
    from euroflash.collect import collect
    source=tmp_path/'old.sqlite';destination=tmp_path/'head.sqlite'
    initialize_database(source)
    with sqlite3.connect(source) as db:
        db.execute('INSERT INTO detections(classification_probability) VALUES (?)',(np.float32(.91),))
    collect(source,destination,'old/run')
    initialize_database(source)
    for path in [source,destination]:
        with sqlite3.connect(path) as db:
            value,kind=db.execute('SELECT classification_probability,typeof(classification_probability) FROM detections').fetchone()
        assert kind=='real'
        assert value==pytest.approx(.91)


def test_a_pulsar_degrees_away_no_longer_vetoes_a_candidate(tmp_path, monkeypatch):
    """The veto matched on DM alone across the whole 5-degree query cone, so a
    new source sharing a DM with any pulsar in the field was recorded as a
    redetection of it."""
    import sqlite3
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    monkeypatch.setenv('LOTAAS_DB_PATH', str(tmp_path/'veto.sqlite'))
    from lotaas_reprocessing import classify
    from db import db_utils
    from db.initialize_db import initialize_database
    monkeypatch.setattr(db_utils, 'DB_PATH', str(tmp_path/'veto.sqlite'))
    initialize_database(db_utils.DB_PATH)

    beam = SkyCoord('12:00:00', '+45:00:00', unit=(u.hourangle, u.deg))
    # One pulsar sharing the candidate DM, four degrees off the beam.
    far = beam.directional_offset_by(0*u.deg, 4*u.deg)

    class FakeTable:
        def to_pandas(self):
            import pandas as pd
            return pd.DataFrame({
                'PSRJ': ['J1200+4900'],
                'RAJ': [far.ra.to_string(unit=u.hour, sep=':', pad=True, precision=2)],
                'DECJ': [far.dec.to_string(unit=u.deg, sep=':', alwayssign=True,
                                           pad=True, precision=2)],
                'DM': [42.0]})

    class FakeQuery:
        def __init__(self, **kwargs):
            self.table = FakeTable()

    monkeypatch.setattr(classify, 'QueryATNF', FakeQuery)
    # Stop after the veto decision: reaching FETCH means the veto did not fire.
    monkeypatch.setattr(classify, 'get_model',
                        lambda name: (_ for _ in ()).throw(RuntimeError('reached FETCH')))

    candidates = tmp_path/'cands.tsv'
    candidates.write_text('DM\tS/N\tTime\tSample\tFilter_Width\n42.0\t12.0\t100.0\t12716\t1\n')
    info = {'RA (J2000)': '12:00:00', 'DEC (J2000)': '+45:00:00'}

    with pytest.raises(RuntimeError, match='reached FETCH'):
        classify.classify_candidates('beam.fil', candidates, str(tmp_path/'plots'), info)

    # Inside the veto radius the same pulsar does account for the detection.
    near = beam.directional_offset_by(0*u.deg, 0.2*u.deg)
    FakeTable.to_pandas = lambda self: __import__('pandas').DataFrame({
        'PSRJ': ['J1200+4900'],
        'RAJ': [near.ra.to_string(unit=u.hour, sep=':', pad=True, precision=2)],
        'DECJ': [near.dec.to_string(unit=u.deg, sep=':', alwayssign=True,
                                    pad=True, precision=2)],
        'DM': [42.0]})
    classify.classify_candidates('beam.fil', candidates, str(tmp_path/'plots'), info)
    with sqlite3.connect(db_utils.DB_PATH) as db:
        assert db.execute(
            "SELECT detection_type,pulsar_name FROM detections").fetchall() == [
            ('known_pulsar', 'J1200+4900')]


def test_classifier_records_empty_beams_and_missing_input(tmp_path, monkeypatch):
    import sqlite3
    monkeypatch.setenv('LOTAAS_DB_PATH', str(tmp_path/'classifier.sqlite'))
    from lotaas_reprocessing import classify
    from db import db_utils
    from db.initialize_db import initialize_database
    monkeypatch.setattr(db_utils, 'DB_PATH', str(tmp_path/'classifier.sqlite'))
    initialize_database(db_utils.DB_PATH)
    candidates=tmp_path/'empty.tsv'
    candidates.write_text('DM\tS/N\tTime\tSample\tFilter_Width\n')
    classify.classify_candidates('empty.fil',candidates,str(tmp_path/'plots'))
    with pytest.raises(FileNotFoundError):
        classify.classify_candidates('missing.fil',tmp_path/'missing.tsv',str(tmp_path/'plots'))
    with sqlite3.connect(db_utils.DB_PATH) as db:
        assert db.execute('SELECT outcome FROM beam_runs ORDER BY id').fetchall()==[('no_candidates',),('error',)]
