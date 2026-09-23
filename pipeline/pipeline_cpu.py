"""Independently resumable single-pulse and periodicity CPU searches."""
import argparse
from pathlib import Path
import shutil
import time
import yaml
from lotaas_reprocessing.trials import validate_trials, product_outputs
from lotaas_reprocessing.periodicity import atomic_json


def single_pulse(output,metadata):
    from lotaas_reprocessing import matched_filter,cluster
    from lotaas_reprocessing import classify
    started=time.perf_counter()
    summary=output/'single_pulse_summary.json';summary.unlink(missing_ok=True)
    trials=output/'DM_trials'
    validate_trials(metadata,trials)
    matched_filter.run_all_matched_filtering(str(trials),metadata['tsamp'],str(output),
        metadata['observation_info'],metadata['dedispersion_plan'],
        nu_min=metadata.get('nu_min'),nu_max=metadata.get('nu_max'))
    raw=output/'all_detected_candidates.cands';clustered=output/'clustered_candidates.txt'
    cluster.cluster_candidates(str(raw),str(clustered),plan=metadata['dedispersion_plan'])
    plots=output/'candidate_plots';plots.mkdir(exist_ok=True)
    classify.classify_candidates(metadata['filename'],str(clustered),str(plots),metadata['observation_info'])
    files=[raw,clustered]+list(plots.glob('*.png'))
    atomic_json(summary,{'complete':True,'elapsed_seconds':time.perf_counter()-started,
        'outputs':{str(p.relative_to(output)):p.stat().st_size for p in files}})


def periodicity(output,metadata):
    from lotaas_reprocessing.periodicity import dm_grid_assessment,run_periodicity_search
    plan=metadata.get('periodicity_dm_plan') or metadata['dedispersion_plan']
    config=dict(metadata.get('periodicity') or {})
    config['dm_grid_assessment']=dm_grid_assessment(plan,metadata['tsamp'],metadata['nu_min'],metadata['nu_max'])
    run_periodicity_search(output/'Periodic_DM_trials',output,metadata,config)


def finalize(output,metadata):
    product_outputs(output,'single_pulse_summary.json')
    if metadata.get('periodicity_enabled',False):
        product_outputs(output,'periodicity_summary.json')
    # Both manifests and all their products are complete before any cleanup.
    shutil.rmtree(output/'DM_trials',ignore_errors=True)
    if metadata.get('periodicity_enabled',False):
        shutil.rmtree(output/'Periodic_DM_trials',ignore_errors=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_directory',type=Path)
    parser.add_argument('--search',choices=['both','single-pulse','periodicity','finalize'],default='both')
    args=parser.parse_args();output=args.output_directory
    metadata=yaml.safe_load((output/'metadata.yaml').read_text())
    errors = []
    if args.search in ('both','single-pulse'):
        try:
            single_pulse(output,metadata)
        except Exception as error:
            if args.search != 'both':
                raise
            errors.append(f"single-pulse: {error}")
    if args.search in ('both','periodicity') and metadata.get('periodicity_enabled',False):
        try:
            periodicity(output,metadata)
        except Exception as error:
            if args.search != 'both':
                raise
            errors.append(f"periodicity: {error}")
    if errors:
        raise RuntimeError('\n'.join(errors))
    if args.search in ('both','finalize'):
        finalize(output,metadata)


if __name__=='__main__':main()
