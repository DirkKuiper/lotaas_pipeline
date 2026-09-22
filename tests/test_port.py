import json
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
    monkeypatch.setattr(matched_filter,'run_matched_filtering',lambda *args:tuple(np.ones(3) for _ in range(4)))
    (tmp_path/'beam_DM10.0.dat').write_bytes(b'test')
    with pytest.raises(RuntimeError,match='Too many candidates'):
        matched_filter.run_all_matched_filtering(tmp_path,.1,tmp_path,{},[])
    assert not (tmp_path/'all_matched_filter_overview.png').exists()


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
    def step(item,stage,command,expected,gpu=None):
        if stage=='dedisperse':
            idx=command.index(str(Path(__file__).resolve().parents[1]/'pipeline/pipeline_gpu.py'))
            outputs[item]=command[idx+2]
    runner.step=step
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
