"""Independently resumable CPU stages of one beam's search.

single-pulse     matched filter and clustering over the DM trials
sp-classify      known-pulsar matching and FETCH over the clusters
periodicity      periodic search and sifting over the periodic trials
periodicity-fold folding what the cross-beam veto left (periodicity_veto.json)
finalize         check every completion manifest, then remove the trials

The single-pulse search and the classification are separate so a GPU node can
hand clusters and periodic trials to a CPU node. Folding is separate so the
veto can compare the sifted peaks of every beam in a batch first.
"""
import argparse
from pathlib import Path
import shutil
import time
import yaml
from lotaas_reprocessing.trials import validate_trials, product_outputs
from lotaas_reprocessing.periodicity import atomic_json


def summarise(output, name, started, files, **extra):
    files = [p for p in files if p.is_file()]
    atomic_json(output/name, dict(extra, complete=True, elapsed_seconds=time.perf_counter()-started,
                                  outputs={str(p.relative_to(output)): p.stat().st_size for p in files}))


def single_pulse(output, metadata):
    from lotaas_reprocessing import matched_filter, cluster
    started = time.perf_counter()
    for name in ('single_pulse_summary.json', 'sp_classify_summary.json'):
        (output/name).unlink(missing_ok=True)
    trials = output/'DM_trials'
    validate_trials(metadata, trials)
    sp = metadata.get('single_pulse') or {}
    max_duration = sp.get('max_width_seconds', 1.0)
    matched_filter.run_all_matched_filtering(str(trials), metadata['tsamp'], str(output),
        metadata['observation_info'], metadata['dedispersion_plan'],
        nu_min=metadata.get('nu_min'), nu_max=metadata.get('nu_max'), max_duration=max_duration,
        baseline_seconds=sp.get('baseline_seconds'), baseline_widths=sp.get('baseline_widths', 64),
        merge=bool(sp.get('merge_events', False)))
    raw = output/'all_detected_candidates.cands'; clustered = output/'clustered_candidates.txt'
    cluster.cluster_candidates(str(raw), str(clustered), plan=metadata['dedispersion_plan'])
    from lotaas_reprocessing.single_pulse_quality import measure_clusters
    evidence = output / 'single_pulse_evidence.json'
    atomic_json(evidence, measure_clusters(output, metadata))
    summarise(output, 'single_pulse_summary.json', started,
              [raw, clustered, evidence, output/'all_matched_filter_overview.png', output/'dm_vs_time_clusters.png'],
              max_width_seconds=max_duration, baseline_seconds=sp.get('baseline_seconds'),
              merge_events=bool(sp.get('merge_events', False)))


def sp_classify(output, metadata):
    from lotaas_reprocessing import classify
    started = time.perf_counter()
    summary = output/'sp_classify_summary.json'; summary.unlink(missing_ok=True)
    product_outputs(output, 'single_pulse_summary.json')
    plots = output/'candidate_plots'; plots.mkdir(exist_ok=True)
    import json
    evidence_path = output / 'single_pulse_evidence.json'
    evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else None
    counts = classify.classify_candidates(metadata['filename'], str(output/'clustered_candidates.txt'), str(plots),
                                          metadata['observation_info'], limits=metadata.get('classification'),
                                          tsamp=metadata.get('tsamp'), evidence=evidence,
                                          bad_channels=metadata.get('bad_channels', ()))
    summarise(output, 'sp_classify_summary.json', started,
              [output/'clustered_candidates.txt'] + sorted(plots.glob('*.png')), counts=counts or {})


def periodic_config(metadata):
    from lotaas_reprocessing.periodicity import dm_grid_assessment
    plan = metadata.get('periodicity_dm_plan') or metadata['dedispersion_plan']
    config = dict(metadata.get('periodicity') or {})
    config['dm_grid_assessment'] = dm_grid_assessment(plan, metadata['tsamp'], metadata['nu_min'], metadata['nu_max'])
    return config


def periodicity(output, metadata):
    from lotaas_reprocessing.periodicity import search_periodicity
    search_periodicity(output/'Periodic_DM_trials', output, metadata, periodic_config(metadata))
    prune_trials(output)


def prune_trials(output):
    """Keep only the periodic trials a fold could read: those of the sifted-best peaks.

    A fold reads the trial of its peak and nothing else, so the other ~4 GB
    of a beam can go as soon as it is searched, rather than waiting for the
    cross-beam veto of the whole batch.
    """
    from lotaas_reprocessing.periodicity import read_jsonl
    keep = {row['trial_file'] for row in read_jsonl(output/'periodicity_candidates.jsonl') if row.get('is_sifted_best')}
    for path in (output/'Periodic_DM_trials').glob('*'):
        if path.suffix in ('.dat', '.inf') and path.with_suffix('.dat').name not in keep:
            path.unlink(missing_ok=True)


def periodicity_fold(output, metadata):
    from lotaas_reprocessing.periodicity import fold_periodicity
    fold_periodicity(output/'Periodic_DM_trials', output, metadata, periodic_config(metadata))


def finalize(output, metadata):
    product_outputs(output, 'single_pulse_summary.json')
    if (output/'sp_classify_summary.json').exists() or metadata.get('classification') is not None:
        product_outputs(output, 'sp_classify_summary.json')
    if metadata.get('periodicity_enabled', False):
        product_outputs(output, 'periodicity_summary.json')
    # Every manifest and all its products are complete before any cleanup.
    shutil.rmtree(output/'DM_trials', ignore_errors=True)
    if metadata.get('periodicity_enabled', False):
        shutil.rmtree(output/'Periodic_DM_trials', ignore_errors=True)


STAGES = {'single-pulse': single_pulse, 'sp-classify': sp_classify, 'periodicity': periodicity,
          'periodicity-fold': periodicity_fold, 'finalize': finalize}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output_directory', type=Path)
    parser.add_argument('--search', choices=['both', *STAGES], default='both',
                        help="One stage, or 'both': every stage of a standalone beam in order")
    args = parser.parse_args(); output = args.output_directory
    metadata = yaml.safe_load((output/'metadata.yaml').read_text())
    if args.search != 'both':
        STAGES[args.search](output, metadata)
        return
    # Standalone: each branch is attempted even if the other fails.
    errors = []
    for name, stages in (('single-pulse', [single_pulse, sp_classify]),
                         ('periodicity', [periodicity, periodicity_fold])):
        if name == 'periodicity' and not metadata.get('periodicity_enabled', False):
            continue
        try:
            for stage in stages:
                stage(output, metadata)
        except Exception as error:
            errors.append(f"{name}: {error}")
    if errors:
        raise RuntimeError('\n'.join(errors))
    finalize(output, metadata)


if __name__ == '__main__':
    main()
