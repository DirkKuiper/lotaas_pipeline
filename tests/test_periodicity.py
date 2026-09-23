import json

import numpy as np
import pytest
from euroflash.ledger import Ledger

from lotaas_reprocessing.periodicity import (
    dm_grid_assessment,
    resolved_config,
    run_periodicity_search,
    search_trial,
    sift_candidates,
)


def config(**overrides):
    values = {
        "period_min_seconds": 0.5,
        "period_max_seconds": 100.0,
        "harmonics": [1, 2, 4, 8, 16],
        "threshold": 12.0,
        "red_noise_window_bins": 31,
        "rfi_frequencies_hz": [],
        "rfi_tolerance_bins": 2,
        "sift_dm_tolerance": 2.0,
        "sift_period_fraction": 0.01,
        "max_folds": 0,
        "catalogue_match": False,
    }
    values.update(overrides)
    return resolved_config(values)


def periodic_data(dt, period, n=65536, amplitude=2.0, pulse_width=None, seed=4):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    data = rng.normal(size=n)
    if pulse_width is None:
        data += amplitude * np.sin(2 * np.pi * t / period)
    else:
        data[(t % period) < pulse_width] += amplitude
    return data.astype("float32")


def test_sinusoid_period_and_sampling_resolution_are_recorded():
    for dt in (0.01, 0.02):
        candidates = search_trial(periodic_data(dt, 2.56), dt, config())
        strongest = max(candidates, key=lambda row: row["statistic"])
        assert abs(strongest["period_seconds"] - 2.56) <= strongest["frequency_resolution_hz"] * 2.56 ** 2
        assert strongest["effective_sampling_seconds"] == dt
        assert strongest["frequency_bin"] > 0


def test_narrow_pulse_train_requires_harmonic_summing():
    candidates = search_trial(periodic_data(0.01, 2.56, amplitude=3.0, pulse_width=0.08), 0.01, config())
    at_fundamental = [row for row in candidates if abs(row["period_seconds"] - 2.56) < .02]
    assert at_fundamental
    assert max(at_fundamental, key=lambda row: row["statistic"])["harmonic_count"] > 1


def test_noise_is_not_reported_at_a_conservative_threshold():
    candidates = search_trial(np.random.default_rng(42).normal(size=65536), .01,
                              config(threshold=14.0))
    assert len(candidates) <= 2


def test_rfi_like_line_is_retained_and_flagged():
    data = periodic_data(.01, 1.0, amplitude=3.0)
    candidates = search_trial(data, .01, config(rfi_frequencies_hz=[1.0], threshold=8.0))
    assert any(row["rfi_like"] and abs(row["period_seconds"] - 1.0) < .02 for row in candidates)


def test_sifting_keeps_dm_mismatch_and_harmonic_relationships():
    base = {"frequency_hz": .4, "period_seconds": 2.5, "frequency_bin": 10,
            "frequency_resolution_hz": .01, "statistic": 20., "harmonic_count": 1,
            "observation_seconds": 100., "effective_sampling_seconds": .01,
            "rfi_like": False}
    rows = []
    for dm, period, statistic in ((100., 2.5, 20.), (100.5, 2.5, 18.), (101., 5., 15.)):
        row = dict(base, dm=dm, period_seconds=period, frequency_hz=1 / period, statistic=statistic)
        rows.append(row)
    sifted = sift_candidates(rows, config())
    assert len({row["sift_group"] for row in sifted}) == 1
    assert all(row["relationships"]["nearby_dm"] for row in sifted)
    assert any(row["relationships"]["harmonic"] for row in sifted)
    assert sum(row["is_sifted_best"] for row in sifted) == 1


def test_periodic_dm_grid_assessment_exposes_coarse_ranges():
    assessment = dm_grid_assessment([
        {"low_dm": 0, "high_dm": 100, "ddm": .1, "downsample": 1},
        {"low_dm": 100, "high_dm": 200, "ddm": 10, "downsample": 1},
    ], .01, 110, 190)
    assert assessment[0]["adequate"]
    assert not assessment[1]["adequate"]
    assert assessment[1]["residual_samples"] > assessment[0]["residual_samples"]


def test_end_to_end_products_are_atomic_and_machine_readable(tmp_path):
    trials = tmp_path / "Periodic_DM_trials"
    trials.mkdir()
    periodic_data(.01, 2.56, n=65536).tofile(trials / "beam_DM10.0.dat")
    metadata = {
        "tsamp": .01, "filename": "beam.fil", "samples_processed": 65536,
        "nu_min": 120., "nu_max": 160.,
        "dedispersion_plan": [{"low_dm": 10, "high_dm": 11, "ddm": 1, "downsample": 1}],
        "periodicity_dm_plan": [{"low_dm": 10, "high_dm": 11, "ddm": 1, "downsample": 1}],
    }
    summary = run_periodicity_search(trials, tmp_path, metadata, config())
    assert summary["complete"] is True
    assert (tmp_path / "periodicity_summary.json").is_file()
    assert not (tmp_path / "periodicity_summary.json.partial").exists()
    rows = [json.loads(line) for line in (tmp_path / "periodicity_raw_candidates.jsonl").read_text().splitlines()]
    assert rows and {"dm", "period_seconds", "frequency_bin", "effective_sampling_seconds"} <= rows[0].keys()


def test_incomplete_periodicity_products_cannot_resume_as_success(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite")
    outputs = [tmp_path / name for name in
               ("periodicity_raw_candidates.jsonl", "periodicity_candidates.jsonl",
                "periodicity_summary.json")]
    for path in outputs:
        path.write_text("{}\n")
    attempt = ledger.start("beam", "classify", "fingerprint", tmp_path / "run.log", [])
    ledger.finish(attempt, outputs)
    assert ledger.completed("beam", "classify", "fingerprint")
    outputs[-1].unlink()
    assert not ledger.completed("beam", "classify", "fingerprint")


@pytest.mark.parametrize("offset", [0., .1, .25, .5, .73])
def test_fractional_bin_narrow_pulses_keep_harmonic_gain(offset):
    n=65536;dt=.01;cycles=256+offset
    phase=np.arange(n)*cycles/n
    data=(np.random.default_rng(4).normal(size=n)+.6*((phase%1)<.03125)).astype("float32")
    rows=search_trial(data,dt,config(threshold=8,red_noise_window_bins=257))
    fundamental=[r for r in rows if abs(r["frequency_bin"]-cycles)<.1]
    assert fundamental
    assert max(r["statistic"] for r in fundamental)>12
    assert max(fundamental,key=lambda r:r["statistic"])["harmonic_count"]>=4


@pytest.mark.parametrize("boundary", ["minimum", "maximum"])
def test_period_search_includes_boundary_peaks(boundary):
    n=65536;dt=.01;period=n*dt/256
    cfg=config(harmonics=[1],period_min_seconds=period if boundary=="minimum" else period/2,
               period_max_seconds=period if boundary=="maximum" else period*2)
    rows=search_trial(periodic_data(dt,period,amplitude=3),dt,cfg)
    assert any(r["frequency_bin"]==256 for r in rows)


@pytest.mark.parametrize("bad", [np.nan,np.inf])
def test_nonfinite_trial_is_a_failure_not_empty_success(bad):
    x=np.ones(4096);x[1]=bad
    with pytest.raises(ValueError,match="Invalid periodic"):
        search_trial(x,.01,config())


def test_noise_at_actual_default_threshold():
    # Full observation lengths, multiple seeds; not the old 0.5 s period cut.
    cfg=resolved_config({"catalogue_match":False})
    counts=[len(search_trial(np.random.default_rng(seed).normal(size=457728),.0078643197,cfg))
            for seed in range(5)]
    assert max(counts)<=2


def test_gamma_scores_match_scipy_survival():
    from scipy.special import gammaincc
    from lotaas_reprocessing.periodicity import gamma_log10_survival
    for h in [1,2,4,8,16]:
        power=np.array([1.,10.,40.,100.])
        np.testing.assert_allclose(gamma_log10_survival(power,h),-np.log10(gammaincc(h,power)),atol=1e-12)


def test_harmonic_sift_checks_adjacent_log_buckets():
    rows=[dict(dm=10.,period_seconds=p,statistic=s,rfi_like=False) for p,s in [(1.,10.),(2.01,9.)]]
    result=sift_candidates(rows,config())
    assert len({r['sift_group'] for r in result})==1
    assert all(r['relationships']['harmonic'] for r in result)


def test_rfi_in_summed_harmonic_flags_fundamental():
    n=65536;dt=.01;frequency=640/(n*dt)
    rows=search_trial(periodic_data(dt,1/frequency,amplitude=3),dt,
                      config(rfi_frequencies_hz=[frequency]))
    sub=[r for r in rows if r['frequency_bin']==320 and r['harmonic_count']>=2]
    assert sub and all(r['rfi_like'] and 2 in r['rfi_harmonics'] for r in sub)


def test_sifting_has_a_hard_work_limit():
    rows=[dict(dm=10.,period_seconds=1.,statistic=12.,rfi_like=False) for _ in range(100)]
    with pytest.raises(RuntimeError,match='work limit'):
        sift_candidates(rows,config(max_sift_comparisons=100))


def test_input_array_is_not_modified():
    x=periodic_data(.01,2.56).astype(float);copy=x.copy()
    search_trial(x,.01,config())
    np.testing.assert_array_equal(x,copy)


def test_wrap_tail_is_excluded_and_duration_updated(tmp_path):
    from lotaas_reprocessing.periodicity import valid_trial
    from lotaas_reprocessing.matched_filter import wrap_contaminated_samples
    meta={'tsamp':.01,'nu_min':120.,'nu_max':160.}
    tail=wrap_contaminated_samples(100,120,160,.01)
    x=periodic_data(.01,2.56);x[-tail:]+=500
    path=tmp_path/'beam.dat';x.tofile(path)
    valid,removed=valid_trial(path,meta,100,1)
    assert removed==tail and len(valid)==len(x)-tail
    rows=search_trial(valid,.01,config())
    assert rows and all(r['observation_seconds']==len(valid)*.01 for r in rows)


def test_trial_manifest_detects_missing_and_same_size_corruption(tmp_path):
    import yaml
    from lotaas_reprocessing.trials import write_manifest,validate_products,link_trial
    plan=[dict(low_dm=10.,high_dm=11.,ddm=1.,downsample=1)]
    meta={'filename':'beam.fil','samples_processed':100,'dedispersion_plan':plan,
          'periodicity_enabled':True,'periodicity_dm_plan':plan}
    (tmp_path/'metadata.json').write_text(json.dumps(meta))
    source=tmp_path/'DM_trials'/'beam_DM10.0.dat';source.parent.mkdir()
    np.arange(100,dtype='float32').tofile(source);source.with_suffix('.inf').write_text('info')
    target=tmp_path/'Periodic_DM_trials'/source.name
    link_trial(source,target);write_manifest(tmp_path);assert validate_products(tmp_path)
    target.unlink()
    with pytest.raises(ValueError,match='incomplete'):validate_products(tmp_path)
    target.write_bytes(b'x'*400)
    with pytest.raises(ValueError,match='Corrupted'):validate_products(tmp_path)
    link_trial(source,target)
    assert validate_products(tmp_path) and source.stat().st_ino==target.stat().st_ino


def test_dispersed_periodic_train_recovers_with_dm_offset_and_downsampling():
    from lotaas_reprocessing.dedispersion import iter_dedispersed
    from lotaas_reprocessing.matched_filter import wrap_contaminated_samples
    dt=.01;n=32768;period=1.2345;dm=25.35;nu=np.linspace(120,160,16)
    t=np.arange(n)*dt
    delay=dm*(nu**-2-nu.max()**-2)/2.41e-4
    phase=np.remainder((t[None,:]-delay[:,None])/period,1)
    data=np.random.default_rng(17).normal(size=phase.shape)+.6*(phase<.04)
    for ds in [1,2,4]:
        for trial_dm in [25.3,25.4]:
            _,trial=next(iter_dedispersed(data.astype('float32'),dt,nu,[trial_dm],ds))
            tail=wrap_contaminated_samples(trial_dm,120,160,dt,ds)
            rows=search_trial(trial[:-tail],dt*ds,config())
            match=[r for r in rows if abs(r['period_seconds']/period-1)<.001]
            assert match


def test_fold_refinement_and_products_survive_trial_cleanup(tmp_path):
    from lotaas_reprocessing.trials import product_outputs
    n=32768;dt=.01;period=1.2345
    trials=tmp_path/'Periodic_DM_trials';trials.mkdir()
    periodic_data(dt,period,n=n,amplitude=1.,pulse_width=.05).tofile(trials/'beam_DM10.0.dat')
    meta={'tsamp':dt,'filename':'beam.fil','samples_processed':n,'nu_min':120.,'nu_max':160.,
          'dedispersion_plan':[dict(low_dm=10.,high_dm=11.,ddm=1.,downsample=1)]}
    summary=run_periodicity_search(trials,tmp_path,meta,config(max_folds=2))
    assert summary['folded_candidates']>0
    rows=[json.loads(line) for line in (tmp_path/'periodicity_folded_candidates.jsonl').read_text().splitlines()]
    assert abs(rows[0]['refined_period_seconds']/period-1)<.001
    import shutil
    shutil.rmtree(trials)
    files=product_outputs(tmp_path,'periodicity_summary.json')
    assert any(p.suffix=='.png' for p in files) and any(p.suffix=='.npz' for p in files)



def test_independent_search_checkpoints_retry_only_failed_branch(tmp_path):
    from argparse import Namespace
    from euroflash.run import Runner
    import yaml
    image=tmp_path/'runtime.sif';image.write_bytes(b'image')
    settings=tmp_path/'settings.yaml';settings.write_text('periodicity: {enabled: true}')
    args=Namespace(work=tmp_path/'work',ledger=tmp_path/'ledger.sqlite',settings=settings,
                   image=image,input=tmp_path)
    runner=Runner(args);runner.fp='version'
    out=tmp_path/'products';out.mkdir()
    (out/'metadata.json').write_text(json.dumps({'periodicity_enabled':True}))
    calls=[];completed=set()
    def step(item,stage,command,expected,**kwargs):
        if stage in completed:return
        calls.append(stage)
        if stage=='periodicity' and calls.count('periodicity')==1:
            raise RuntimeError('interrupted search')
        completed.add(stage)
    runner.step=step
    with pytest.raises(RuntimeError,match='interrupted'):
        runner.analyze('beam',out)
    runner.analyze('beam',out)
    assert calls==['single_pulse','periodicity','periodicity','classify']
    with runner.ledger.connect() as db:
        assert db.execute("SELECT status FROM attempts WHERE stage='classify'").fetchone()[0]=='failed'


def test_single_pulse_failure_still_runs_periodicity(tmp_path):
    from argparse import Namespace
    from euroflash.run import Runner
    args=Namespace(work=tmp_path/'work',ledger=tmp_path/'ledger.sqlite',settings=tmp_path/'s',
                   image=tmp_path/'i',input=tmp_path)
    runner=Runner(args);runner.fp='version'
    out=tmp_path/'products';out.mkdir();(out/'metadata.json').write_text(json.dumps({'periodicity_enabled':True}))
    stages=[]
    def step(item,stage,*args,**kwargs):
        stages.append(stage)
        if stage=='single_pulse':raise RuntimeError('FETCH failed')
    runner.step=step
    with pytest.raises(RuntimeError,match='FETCH failed'):runner.analyze('beam',out)
    assert stages==['single_pulse','periodicity']


def test_finalize_refuses_to_remove_trials_if_fold_product_is_missing(tmp_path):
    from pipeline.pipeline_cpu import finalize
    from lotaas_reprocessing.periodicity import atomic_json
    (tmp_path/'DM_trials').mkdir();(tmp_path/'Periodic_DM_trials').mkdir()
    (tmp_path/'sp').write_text('ok')
    atomic_json(tmp_path/'single_pulse_summary.json',{'complete':True,'outputs':{'sp':2}})
    atomic_json(tmp_path/'periodicity_summary.json',{'complete':True,'outputs':{'missing.png':100}})
    with pytest.raises(ValueError,match='Missing'):finalize(tmp_path,{'periodicity_enabled':True})
    assert (tmp_path/'DM_trials').is_dir() and (tmp_path/'Periodic_DM_trials').is_dir()


def test_failed_invalid_trial_removes_previous_completion_marker(tmp_path):
    from lotaas_reprocessing.periodicity import atomic_json
    trials=tmp_path/'trials';trials.mkdir()
    np.full(64,np.nan,dtype='float32').tofile(trials/'beam_DM10.0.dat')
    meta={'filename':'beam.fil','samples_processed':64,'tsamp':.01,'nu_min':120.,'nu_max':160.,
          'dedispersion_plan':[dict(low_dm=10.,high_dm=11.,ddm=1.,downsample=1)]}
    atomic_json(tmp_path/'periodicity_summary.json',{'complete':True})
    with pytest.raises(ValueError,match='Nonfinite'):run_periodicity_search(trials,tmp_path,meta,config())
    assert not (tmp_path/'periodicity_summary.json').exists()



@pytest.mark.parametrize("period,duty", [(1.2345,.03),(20.123,.05),(83.17,.03),(83.17,.2)])
def test_red_noise_and_long_period_injections(period,duty):
    n=131072;dt=.01;t=np.arange(n)*dt
    rng=np.random.default_rng(99)
    # White noise plus a smooth, strongly correlated low-frequency component.
    from scipy.ndimage import gaussian_filter1d
    red=gaussian_filter1d(rng.normal(size=n),100)*15
    x=rng.normal(size=n)+red+3.0*(np.remainder(t/period,1)<duty)
    rows=search_trial(x.astype('float32'),dt,config(period_max_seconds=150,red_noise_window_bins=257))
    assert any(abs(r['period_seconds']/period-1)<.005 for r in rows)


def test_benchmark_includes_sifting_and_folding(tmp_path):
    from euroflash.periodicity_benchmark import benchmark
    trials=tmp_path/'trials';trials.mkdir();n=8192
    for dm in [10.,11.]:periodic_data(.01,1.2345,n=n).tofile(trials/f'beam_DM{dm}.dat')
    metadata={'filename':'beam.fil','samples_processed':n,'tsamp':.01,'nu_min':120.,'nu_max':160.,
              'dedispersion_plan':[dict(low_dm=10.,high_dm=12.,ddm=1.,downsample=1)]}
    result=benchmark(trials,metadata,config(max_folds=1))
    assert result['benchmark_scope']=='full_beam' and result['folded_candidates']==1
    assert {'sifting','folding','plots','product_writes'}<=set(result['includes'])
    assert len(list(trials.glob('*.dat')))==2


def test_periodic_notification_is_gated_and_idempotent(tmp_path):
    import yaml
    from postproc.notify_candidates import connect
    from postproc.notify_periodicity import run_once
    from lotaas_reprocessing.periodicity import atomic_json
    out=tmp_path/'processed'/'beam'/'fingerprint';out.mkdir(parents=True)
    row={'plot':'periodic.png','fold_data':'fold.npz','rfi_like':False,'catalogue_matches':[{'name':'J0323+3944'}],
         'refined_period_seconds':3.032,'dm':26.2,'statistic':25.,'harmonic_count':8,'observation_seconds':3600.}
    (out/'periodic.png').write_bytes(b'plot');(out/'fold.npz').write_bytes(b'fold')
    folded=out/'periodicity_folded_candidates.jsonl';folded.write_text(json.dumps(row)+'\n')
    (out/'metadata.json').write_text(json.dumps({'filename':'beam.fil','pilot':True}))
    atomic_json(out/'periodicity_summary.json',{'complete':True,'outputs':{'periodic.png':4,'fold.npz':4,folded.name:folded.stat().st_size}})
    ledger=Ledger(tmp_path/'ledger.sqlite')
    class FakeSlack:
        channel='test';calls=[]
        def upload(self,path,**kwargs):self.calls.append((path,kwargs));return 'Ftest'
    slack=FakeSlack()
    with connect(tmp_path/'ledger.sqlite') as db:
        assert run_once(slack,db,[out])==[]
        with pytest.raises(ValueError,match='No successful'):
            run_once(slack,db,[out],include_pilot=True)
        attempt=ledger.start('beam','periodicity','fingerprint',tmp_path/'log',[])
        ledger.finish(attempt,[out/'periodicity_summary.json'])
        assert len(run_once(slack,db,[out],include_pilot=True,known_only=True))==1
        assert run_once(slack,db,[out],include_pilot=True)==[]
    assert len(slack.calls)==1
    assert 'Pilot validation' in slack.calls[0][1]['comment']



def test_reclaim_protects_active_periodicity_retry_and_counts_hardlinks_once(tmp_path):
    import os,time
    from euroflash.reclaim import survey
    ledger=Ledger(tmp_path/'ledger.sqlite');fp='f'*64
    out=tmp_path/'work'/'processed'/'beam'/fp[:16]
    source=out/'DM_trials'/'x.dat';source.parent.mkdir(parents=True);source.write_bytes(b'x'*1024)
    periodic=out/'Periodic_DM_trials';periodic.mkdir();os.link(source,periodic/'x.dat')
    attempt=ledger.start('beam','classify',fp,tmp_path/'log',[])
    ledger.finish(attempt,error='old failure')
    now=time.time()+10*86400
    retry=ledger.start('beam','periodicity',fp,tmp_path/'retry.log',[])
    assert survey(tmp_path/'work',tmp_path/'ledger.sqlite',86400,now=now)==[]
    ledger.finish(retry,error='failed again')
    found=survey(tmp_path/'work',tmp_path/'ledger.sqlite',86400,now=now)
    assert len(found)==2 and sum(size for _,size,_ in found)==1024



def test_direct_cpu_entrypoint_attempts_both_searches_on_failure(tmp_path,monkeypatch):
    from pipeline import pipeline_cpu
    import sys
    (tmp_path/'metadata.yaml').write_text('periodicity_enabled: true')
    stages=[]
    def single(*args):
        stages.append('single')
        raise RuntimeError('single failed')
    monkeypatch.setattr(pipeline_cpu,'single_pulse',single)
    monkeypatch.setattr(pipeline_cpu,'periodicity',lambda *args:stages.append('periodic'))
    monkeypatch.setattr(pipeline_cpu,'finalize',lambda *args:stages.append('cleanup'))
    monkeypatch.setattr(sys,'argv',['pipeline_cpu.py',str(tmp_path)])
    with pytest.raises(RuntimeError,match='single failed'):pipeline_cpu.main()
    assert stages==['single','periodic']


def test_single_pulse_only_finalization_has_no_periodic_requirement(tmp_path):
    from pipeline.pipeline_cpu import finalize
    from lotaas_reprocessing.periodicity import atomic_json
    (tmp_path/'DM_trials').mkdir();(tmp_path/'single.cands').write_text('ok')
    atomic_json(tmp_path/'single_pulse_summary.json',{'complete':True,'outputs':{'single.cands':2}})
    finalize(tmp_path,{'periodicity_enabled':False})
    assert not (tmp_path/'DM_trials').exists()



def test_coarse_high_dm_rows_do_not_expand_low_dm_sift_work():
    rows=[dict(dm=float(dm),dm_step=.1,period_seconds=1.,statistic=12.,rfi_like=False)
          for dm in np.arange(0,40,.1)]
    rows.append(dict(dm=9800.,dm_step=8.,period_seconds=1.,statistic=12.,rfi_like=False))
    result=sift_candidates(rows,config(max_sift_comparisons=15000))
    assert len({r['sift_group'] for r in result})==2


def test_sifting_connects_neighbours_across_dm_step_boundaries():
    rows=[dict(dm=149.9,dm_step=.1,period_seconds=1.,statistic=12.,rfi_like=False),
          dict(dm=150.6,dm_step=.2,period_seconds=1.,statistic=13.,rfi_like=False),
          dict(dm=9600.,dm_step=8.,period_seconds=1.,statistic=14.,rfi_like=False),
          dict(dm=9608.,dm_step=8.,period_seconds=1.,statistic=15.,rfi_like=False)]
    result=sift_candidates(rows,config())
    assert len({r['sift_group'] for r in result})==2
    assert result[0]['sift_group']==result[1]['sift_group']
    assert result[2]['sift_group']==result[3]['sift_group']
