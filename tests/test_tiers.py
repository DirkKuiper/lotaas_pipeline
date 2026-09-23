"""The two-tier search: the cross-beam veto, trial pruning, FETCH limits, stage limits and dispatch."""
import json
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from lotaas_reprocessing import periodicity_veto as V
from lotaas_reprocessing.periodicity import atomic_json

RES = 1 / 3600.


def peak(f, dm, sap, beam, key=None):
    return {'frequency_hz': f, 'frequency_resolution_hz': RES, 'dm': dm, 'sap': sap, 'beam': beam,
            'key': key or (sap, beam, f)}


def test_a_period_in_many_beams_at_any_dm_is_vetoed_and_a_pulsar_is_not():
    rfi = [peak(0.16857, dm, 0, beam) for beam, dm in ((13, 0.0), (20, 40.7), (31, 700.), (44, 3100.), (52, 1.0))]
    pulsar = [peak(0.32984, 26.2 + 0.1 * (beam % 2), 0, beam) for beam in (50, 51, 52, 53)]
    lone = [peak(0.81, 55.0, 0, 60)]
    vetoed = V.decide(rfi + pulsar + lone, veto_bins=1.1, veto_beams=4)
    assert {k for k in vetoed} == {p['key'] for p in rfi}
    assert all(v['beams'] == 5 and v['dm_max'] == 3100. for v in vetoed.values())


def test_a_period_in_two_saps_of_one_observation_is_vetoed():
    vetoed = V.decide([peak(1.5, 30.0, 0, 20), peak(1.5 + 0.5 * RES, 91.0, 1, 33)], veto_beams=4)
    assert len(vetoed) == 2 and all(v['saps'] == 2 for v in vetoed.values())


def test_crowded_low_frequencies_do_not_veto_by_chance():
    rng = np.random.default_rng(3)
    # Red noise piles peaks into the lowest bins of every beam of a bad observation.
    noise = [peak(float(rng.uniform(2, 60)) * RES, float(rng.uniform(0, 1000)), 0, beam)
             for beam in range(13, 74) for _ in range(4)]
    target = peak(30.2 * RES, 18.0, 0, 70, key='pulsar')
    vetoed = V.decide(noise + [target], veto_beams=4)
    assert 'pulsar' not in vetoed
    # The same crowding with a real family on top of it is still caught.
    family = [peak(45.0 * RES, dm, 0, beam, key=('family', beam)) for beam, dm in
              zip(range(13, 73, 2), np.linspace(0, 900, 30))]
    vetoed = V.decide(noise + family, veto_beams=4)
    assert sum(1 for k in vetoed if isinstance(k, tuple) and k[0] == 'family') >= 25


def beam_dir(root, beam, rows):
    directory = root/f'downsampled_L559955_SAP000_BEAM{beam:03d}_32bit_ff'/'0123456789abcdef'
    directory.mkdir(parents=True)
    (directory/'periodicity_candidates.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return directory


def row(f, dm, best=True):
    return {'frequency_hz': f, 'frequency_resolution_hz': RES, 'dm': dm, 'is_sifted_best': best}


def test_apply_writes_a_veto_beside_each_beam_it_is_asked_to(tmp_path):
    dirs = [beam_dir(tmp_path, beam, [row(0.2, beam * 10.), row(0.2, 1., best=False), row(0.7 + beam / 1000, 20.)])
            for beam in (13, 14, 15, 16, 17)]
    written = V.apply(dirs, write=dirs[:4])
    assert sorted(written.values()) == [1, 1, 1, 1] and not (dirs[4]/'periodicity_veto.json').exists()
    veto = json.loads((dirs[0]/'periodicity_veto.json').read_text())
    assert veto['vetoed_indices'] == [0] and veto['compared_beams'] == 5 and veto['observation'] == 'L559955'


def periodic_data(dt, period, n, amplitude, width, seed=4):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    data = rng.normal(size=n)
    data[(t % period) < width] += amplitude
    return data.astype('float32')


def search_config(**overrides):
    from lotaas_reprocessing.periodicity import resolved_config
    values = {'period_min_seconds': 0.5, 'period_max_seconds': 100.0, 'harmonics': [1, 2, 4, 8, 16],
              'threshold': 12.0, 'red_noise_window_bins': 31, 'rfi_frequencies_hz': [], 'rfi_tolerance_bins': 2,
              'sift_dm_tolerance': 2.0, 'sift_period_fraction': 0.01, 'max_folds': 4, 'catalogue_match': False}
    values.update(overrides)
    return resolved_config(values)


def test_vetoed_peaks_are_not_folded_and_unfoldable_trials_are_pruned(tmp_path):
    from lotaas_reprocessing.periodicity import search_periodicity, fold_periodicity, read_jsonl
    from lotaas_reprocessing.trials import product_outputs
    from pipeline.pipeline_cpu import prune_trials
    n, dt, period = 32768, .01, 1.2345
    trials = tmp_path/'Periodic_DM_trials'; trials.mkdir()
    for dm in (10.0, 11.0, 12.0):
        periodic_data(dt, period, n, 1., .05).tofile(trials/f'beam_DM{dm}.dat')
        (trials/f'beam_DM{dm}.inf').write_text('inf')
    meta = {'tsamp': dt, 'filename': 'beam.fil', 'samples_processed': n, 'nu_min': 120., 'nu_max': 160.,
            'dedispersion_plan': [dict(low_dm=10., high_dm=13., ddm=1., downsample=1)]}
    search_periodicity(trials, tmp_path, meta, search_config())
    rows = read_jsonl(tmp_path/'periodicity_candidates.jsonl')
    best = [i for i, r in enumerate(rows) if r['is_sifted_best']]
    keep = {rows[i]['trial_file'] for i in best}
    prune_trials(tmp_path)
    assert {p.name for p in trials.glob('*.dat')} == keep
    strongest = max(best, key=lambda i: rows[i]['statistic'])
    atomic_json(tmp_path/'periodicity_veto.json', {'schema': 1, 'vetoed_indices': [strongest], 'compared_beams': 5})
    summary = fold_periodicity(trials, tmp_path, meta, search_config())
    folded = read_jsonl(tmp_path/'periodicity_folded_candidates.jsonl')
    assert all(abs(r['frequency_hz'] - rows[strongest]['frequency_hz']) > 1e-12 or r['dm'] != rows[strongest]['dm']
               for r in folded)
    assert summary['multibeam_veto'] == {'applied': True, 'vetoed_best_candidates': 1, 'compared_beams': 5}
    assert 'periodicity_veto.json' in summary['outputs'] and product_outputs(tmp_path, 'periodicity_summary.json')


def classifier(tmp_path, monkeypatch):
    monkeypatch.setenv('LOTAAS_DB_PATH', str(tmp_path/'classifier.sqlite'))
    from lotaas_reprocessing import classify
    from db import db_utils
    from db.initialize_db import initialize_database
    monkeypatch.setattr(db_utils, 'DB_PATH', str(tmp_path/'classifier.sqlite'))
    initialize_database(db_utils.DB_PATH)

    class Empty:
        def __init__(self, **kwargs):
            import pandas
            self.table = type('T', (), {'to_pandas': lambda s: pandas.DataFrame(columns=['PSRJ', 'RAJ', 'DECJ', 'DM'])})()
    monkeypatch.setattr(classify, 'QueryATNF', Empty)
    return classify


def test_wide_clusters_and_those_past_the_budget_are_recorded_not_classified(tmp_path, monkeypatch):
    import sqlite3
    classify = classifier(tmp_path, monkeypatch)
    reached = []
    monkeypatch.setattr(classify, 'get_model', lambda name: (_ for _ in ()).throw(RuntimeError('reached FETCH')))
    candidates = tmp_path/'cands.tsv'
    # 600 s wide at DM 7250 (76288 samples of 7.864 ms), and DM 3 (below the old floor of 10) at 9 samples.
    candidates.write_text('DM\tS/N\tTime\tSample\tFilter_Width\n7249.8\t10.3\t2371.9\t301600\t76288\n'
                          '6000.0\t9.0\t100.0\t12716\t76288\n3.0\t8.0\t50.0\t6358\t9\n')
    info = {'RA (J2000)': '12:00:00', 'DEC (J2000)': '+45:00:00'}
    limits = {'min_dm': 2.0, 'min_snr': 7.0, 'max_width_seconds': 1.0, 'max_fetch_candidates': 0}
    counts = classify.classify_candidates('beam.fil', candidates, str(tmp_path/'plots'), info,
                                          limits=limits, tsamp=0.007864319719374176)
    assert counts == {'fetch': 0, 'known_pulsar': 0, 'unclassified_wide': 2, 'unclassified_budget': 1}
    with sqlite3.connect(tmp_path/'classifier.sqlite') as db:
        assert db.execute("SELECT COUNT(*) FROM detections WHERE detection_type='unclassified'").fetchone()[0] == 3
    # The DM 3 cluster does reach FETCH once the budget allows it.
    limits['max_fetch_candidates'] = 1
    with pytest.raises(RuntimeError, match='reached FETCH'):
        classify.classify_candidates('beam.fil', candidates, str(tmp_path/'plots'), info,
                                     limits=limits, tsamp=0.007864319719374176)


def runner(tmp_path, **extra):
    from euroflash.run import Runner
    image = tmp_path/'image.sif'; image.write_bytes(b'image')
    settings = tmp_path/'settings.yaml'; settings.write_text('dedispersion_plan: []')
    args = Namespace(work=tmp_path/'work', ledger=tmp_path/'ledger.sqlite', settings=settings, image=image,
                     input=tmp_path/'input', backend='gpu', gpus='0,1', cpu_workers=2, pilot=False,
                     max_samples=None, **extra)
    (tmp_path/'input').mkdir(exist_ok=True)
    return Runner(args)


def test_a_stage_past_its_limit_is_killed_with_its_children(tmp_path):
    run = runner(tmp_path)
    run.fp = 'abc'
    marker = tmp_path/'child-alive'
    child = f"import time,pathlib; time.sleep(3); pathlib.Path({str(marker)!r}).write_text('x')"
    command = [sys.executable, '-c', f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(60)"]
    started = time.time()
    with pytest.raises(RuntimeError, match='timed out after 1 s'):
        run.step('beam', 'sp_classify', command, [], timeout=1)
    assert time.time() - started < 40
    time.sleep(4)
    assert not marker.exists(), 'the stage process group was killed, grandchildren included'
    with run.ledger.connect() as db:
        assert db.execute("SELECT status FROM attempts").fetchone()[0] == 'failed'


def test_stage_limits_parse_and_reject_unknown_stages():
    from euroflash.run import parse_timeouts, TIMEOUTS
    assert parse_timeouts(['sp_classify=60'])['sp_classify'] == 60 and TIMEOUTS['sp_classify'] == 2700
    with pytest.raises(ValueError):
        parse_timeouts(['fetch=60'])


def test_a_cpu_node_refuses_beams_prepared_under_another_fingerprint(tmp_path):
    run = runner(tmp_path, stages='cpu')
    run.fp = 'b' * 64
    handoff = run.handoff()
    atomic_json(handoff/'beam.ready', {'item': 'beam', 'fingerprint': 'a' * 64, 'output': 'processed/beam/x',
                                       'input': '/x/beam.fil'})
    atomic_json(handoff/'.done', {})
    with pytest.raises(ValueError, match='source, settings or image differ'):
        run.process_stream(poll=0)


def test_a_cpu_node_searches_beams_as_they_arrive_then_finishes_the_batch(tmp_path):
    run = runner(tmp_path, stages='cpu')
    run.fp = 'c' * 64
    handoff = run.handoff()
    seen, finished = [], []
    run.search = lambda item, output: seen.append(item) or []
    run.finish_batch = lambda searched, pool: finished.extend(item for item, _, _ in searched) or []
    for item in ('b1', 'b2'):
        output = run.root/'processed'/item/'x'
        output.mkdir(parents=True)
        atomic_json(output/'metadata.json', {'periodicity': {'multibeam_veto': True}})
        atomic_json(handoff/f'{item}.ready', {'item': item, 'fingerprint': run.fp, 'output': f'processed/{item}/x',
                                              'input': f'/x/{item}.fil'})
    atomic_json(handoff/'.done', {})
    run.process_stream(poll=0)
    assert sorted(seen) == ['b1', 'b2'] and sorted(finished) == ['b1', 'b2']
    assert (handoff/'b1.searched').exists() and (handoff/'b2.searched').exists()


def test_a_slow_beam_cannot_block_completed_results_without_a_veto(tmp_path):
    import threading
    run = runner(tmp_path, stages='cpu')
    run.fp = 'e' * 64
    completed = threading.Event()
    finished = []
    def search(item, output):
        if item == 'slow':
            assert completed.wait(5), 'fast beam must finalize while slow beam is still running'
        return []
    def finish(item, output, errors):
        finished.append(item)
        if item == 'fast':
            completed.set()
    run.search, run.fold_and_finish = search, finish
    for item in ('fast', 'slow'):
        output = run.root/'processed'/item/'x'
        output.mkdir(parents=True)
        atomic_json(output/'metadata.json', {'periodicity': {'multibeam_veto': False}})
        atomic_json(run.handoff()/f'{item}.ready', {'item': item, 'fingerprint': run.fp,
                    'output': f'processed/{item}/x', 'input': f'/x/{item}.fil'})
    atomic_json(run.handoff()/'.done', {})
    run.process_stream(poll=0)
    assert finished == ['fast', 'slow']
    assert json.loads((run.handoff()/'fast.searched').read_text())['finalized'] is True


def test_the_staging_window_follows_its_schedule(tmp_path):
    from euroflash.campaign import staging_window
    schedule = tmp_path/'schedule.json'
    schedule.write_text(json.dumps({'phases': [
        {'start': '2026-09-24T12:00:00Z', 'max_staging_saps': 16, 'label': 'window-16'},
        {'start': '2026-09-23T12:00:00Z', 'max_staging_saps': 8, 'label': 'window-8'},
        {'start': '2026-09-26T00:00:00Z', 'max_staging_saps': 32, 'label': 'window-32'}]}))
    at = lambda text: time.mktime(time.strptime(text, '%Y-%m-%dT%H:%M:%S')) - time.timezone
    assert staging_window(schedule, 8, now=at('2026-09-23T06:00:00')) == (8, None)
    assert staging_window(schedule, 8, now=at('2026-09-24T11:59:59')) == (8, 'window-8')
    assert staging_window(schedule, 8, now=at('2026-09-25T00:00:00')) == (16, 'window-16')
    assert staging_window(schedule, 8, now=at('2026-09-27T00:00:00')) == (32, 'window-32')
    assert staging_window(tmp_path/'missing.json', 8) == (8, None)


def test_each_gpu_node_takes_its_own_batch_and_is_free_once_its_stages_end(tmp_path, monkeypatch):
    from tests.test_campaign import build, put_online, settle, FULL
    from euroflash import campaign as C
    campaign, api, where = build(tmp_path, monkeypatch, [('1000001', s, FULL) for s in range(3)],
                                 max_staging_saps=3, dispatch_saps=1)
    campaign.tick()
    for s in range(3):
        put_online(api, where, '1000001', s, FULL)
    campaign.tick()
    settle(campaign)
    launched = []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            launched.append(command)
        def poll(self):
            return None
    monkeypatch.setattr(C.subprocess, 'Popen', FakeProcess)
    campaign.o.dispatch_nodes = ['efc-gpu-00', 'efc-gpu-01']
    campaign.o.cpu_nodes = ['efc-cpu-00', 'efc-cpu-01']
    started = campaign.dispatch()
    assert len(started) == 2 and len(launched) == 2
    assert [c[c.index('--nodes') + 1] for c in launched] == ['efc-gpu-00', 'efc-gpu-01']
    assert all(c[c.index('--cpu-nodes') + 1:c.index('--cpu-nodes') + 3] == ['efc-cpu-00', 'efc-cpu-01'] for c in launched)
    assert campaign.dispatch() == 'running', 'both GPU nodes are in their GPU stages'
    # efc-gpu-00 finishes its own stages; its batch goes on on a CPU node, and it takes the third SAP.
    first = next(r for r in started if r.endswith('gpu00'))
    (campaign.root/'results'/first).mkdir(parents=True, exist_ok=True)
    (campaign.root/'results'/first/'efc-gpu-00.gpu-done').write_text('{}')
    third = campaign.dispatch()
    assert len(third) == 1 and 'gpu00' in third[0] and len(campaign.dispatches) == 3


def test_cpu_slots_are_exclusive_across_runs(tmp_path, monkeypatch):
    from euroflash import cluster
    monkeypatch.setattr(cluster, 'cpu_health', lambda node, *args: (node != 'efc-cpu-01', 'probe'))
    notes = []
    first = cluster.CpuSlot(['efc-cpu-00', 'efc-cpu-01', 'efc-cpu-02'], tmp_path, log=notes.append)
    second = cluster.CpuSlot(['efc-cpu-00', 'efc-cpu-01', 'efc-cpu-02'], tmp_path, log=notes.append)
    with first as a, second as b:
        assert (a, b) == ('efc-cpu-00', 'efc-cpu-02'), 'the unusable node is skipped'
    with cluster.CpuSlot(['efc-cpu-00'], tmp_path, log=notes.append) as again:
        assert again == 'efc-cpu-00', 'a slot is released on exit'
    assert any('UNUSABLE efc-cpu-01' in n for n in notes)


def test_the_staging_report_measures_each_phase(tmp_path):
    import sqlite3
    from euroflash.staging_report import report
    from euroflash.campaign import State
    from euroflash.ledger import Ledger
    from db.initialize_db import initialize_database
    root = tmp_path/'campaign'; root.mkdir()
    state = State(root/'campaign-state.sqlite')
    ledger_path = tmp_path/'ledger.sqlite'
    Ledger(ledger_path); initialize_database(str(ledger_path))
    t0 = 1_790_000_000
    with sqlite3.connect(ledger_path) as db:
        for i, (finished, size) in enumerate(((t0 + 3600, 5e9), (t0 + 7200, 5e9), (t0 + 90000, 6e9))):
            name = f'L1_SAP000_B{i:03d}_P000_bf_x'
            db.execute("INSERT INTO inputs(uri,project,discovered,source) VALUES (?,?,?,?)",
                       (f'srm://x/{name}.tar', 'lt5', 0, 'test'))
            db.execute("INSERT INTO archive_receipts VALUES (?,?,?,?,?,?)",
                       (f'srm://x/{name}.tar', 1, 'https://x', 'h', int(size), f'/r/{name}.extracted.json'))
            db.execute("INSERT INTO attempts(item,stage,fingerprint,status,started,finished,seconds,host) "
                       "VALUES (?,?,?,?,?,?,?,?)", (name, 'retrieve', 'f', 'success', finished - 10, finished, 10., 'h'))
    with state.db() as db:
        db.execute("INSERT INTO events VALUES (?,?,?,?)", (t0 + 7300, 'prepared', 'L1_SAP000', ''))
        db.execute("INSERT INTO events VALUES (?,?,?,?)", (t0 + 7400, 'throttled', 'x', 'HTTP Error 429'))
    schedule = tmp_path/'schedule.json'
    schedule.write_text(json.dumps({'phases': [{'start': t0, 'max_staging_saps': 8, 'label': 'w8'},
                                               {'start': t0 + 86400, 'max_staging_saps': 16, 'label': 'w16'}]}))
    rows = report(root, ledger_path, schedule, ramp_hours=6, now=t0 + 2 * 86400)
    assert [r['label'] for r in rows] == ['w8', 'w16']
    assert rows[0]['whole']['bytes'] == 10e9 and rows[0]['whole']['saps_complete'] == 1
    assert rows[0]['whole']['throttled_429_503'] == 1 and rows[1]['whole']['files_retrieved'] == 1
    assert rows[0]['after_ramp']['files_retrieved'] == 0


def test_the_runner_hands_over_without_numpy(tmp_path, monkeypatch):
    """euroflash.run runs on a node's own Python, which has no numpy or scipy."""
    for name in ('numpy', 'scipy'):
        monkeypatch.setitem(sys.modules, name, None)
    for name in [m for m in sys.modules if m.startswith('lotaas_reprocessing.periodicity')]:
        monkeypatch.delitem(sys.modules, name)
    run = runner(tmp_path, stages='gpu')
    run.fp = 'd' * 64
    output = tmp_path/'work'/'processed'/'beam'/'x'; output.mkdir(parents=True)
    (output/'metadata.json').write_text(json.dumps({'filename': '/runs/input/beam.fil'}))
    run.mark_ready('beam', output)
    run.write_done([('beam', output, [])], [])
    assert json.loads((run.handoff()/'beam.ready').read_text())['input'] == '/runs/input/beam.fil'
    assert json.loads((run.handoff()/'.done').read_text())['failed'] == []
    dirs = [beam_dir(tmp_path/'veto', b, [row(0.2, b * 10.)]) for b in (13, 14, 15, 16)]
    from lotaas_reprocessing.periodicity_veto import apply
    assert sum(apply(dirs).values()) == 4


def test_a_failed_single_pulse_branch_still_reaches_periodic_search(tmp_path):
    run = runner(tmp_path, stages='gpu')
    run.fp = 'f' * 64
    output = run.root/'processed/beam/x'
    output.mkdir(parents=True)
    atomic_json(output/'metadata.json', {'filename': '/runs/input/beam.fil', 'periodicity_enabled': True})
    (output/'Periodic_DM_trials').mkdir()
    trial = output/'Periodic_DM_trials/beam_DM10.dat'
    trial.write_bytes(b'periodic data')
    def fail_stage(*args):
        raise RuntimeError('single-pulse candidate overflow')
    run.cpu_stage = fail_stage
    assert run.search('beam', output) == ['single-pulse candidate overflow']
    assert (run.handoff()/'beam.ready').is_file()
    assert trial.read_bytes() == b'periodic data'
    with run.ledger.connect() as db:
        assert db.execute("SELECT status FROM attempts WHERE stage='classify'").fetchone()[0] == 'failed'


def test_a_failed_catalogue_query_is_retried_under_the_node_lock(tmp_path, monkeypatch):
    from lotaas_reprocessing import atnf
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setattr(atnf.time, 'sleep', lambda s: None)
    calls = []

    def flaky(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError('Problem getting catalogue: Compressed file ended before the end-of-stream marker')
        return 'catalogue'
    assert atnf.query_atnf(factory=flaky, radius=1.0) == 'catalogue' and len(calls) == 2
    assert (tmp_path/'.lotaas-atnf.ready').exists(), 'later queries skip the lock'
    assert atnf.query_atnf(factory=lambda **k: 'cached') == 'cached'
