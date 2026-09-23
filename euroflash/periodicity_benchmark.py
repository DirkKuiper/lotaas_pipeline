"""Measure the complete periodic search, sifting, folding and product writing.

Runs outside the campaign ledger. Bounded samples are spread over the numeric
DM grid, never the lexicographically first filenames. Inputs are hard-linked
into a temporary benchmark directory and are never changed or removed.
"""
import argparse
import json
from pathlib import Path
import os
import tempfile
import numpy as np
import yaml
from lotaas_reprocessing.periodicity import resolved_config,run_periodicity_search
from lotaas_reprocessing.trials import trial_specs,validate_trials


def benchmark(trial_dir,metadata,config,max_trials=None):
    trial_dir=Path(trial_dir)
    plan=metadata.get('periodicity_dm_plan') or metadata['dedispersion_plan']
    validate_trials(metadata,trial_dir,plan)
    specs=sorted(trial_specs(metadata,plan).items(),key=lambda pair:pair[1]['dm'])
    total=len(specs)
    if max_trials is not None:
        if max_trials<1:raise ValueError('max_trials must be positive')
        specs=[specs[i] for i in np.unique(np.linspace(0,total-1,min(max_trials,total)).astype(int))]
    with tempfile.TemporaryDirectory(prefix='periodicity-benchmark-',dir=trial_dir.parent) as temp:
        root=Path(temp);trials=root/'trials';trials.mkdir()
        for name,_ in specs:os.link(trial_dir/name,trials/name)
        subset=dict(metadata,periodicity_dm_plan=[dict(low_dm=s['dm'],high_dm=s['dm']+s['ddm'],
                 ddm=s['ddm'],downsample=s['downsample']) for _,s in specs])
        result=run_periodicity_search(trials,root/'products',subset,config)
        result.update(available_trials=total,benchmark_trials=len(specs),
            benchmark_scope='full_beam' if len(specs)==total else 'sampled_dm_trials',
            input_bytes=sum((trials/name).stat().st_size for name,_ in specs),
            includes=['trial_reads','search','sifting','folding','plots','product_writes'],
            excludes=['preprocessing','GPU_dedispersion','single_pulse_search'])
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial_dir',type=Path)
    parser.add_argument('--metadata',type=Path,help='Default: metadata.yaml next to the trial directory')
    parser.add_argument('--settings',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-trials',type=int)
    args=parser.parse_args()
    metadata=yaml.safe_load((args.metadata or args.trial_dir.parent/'metadata.yaml').read_text())
    cfg=(yaml.safe_load(args.settings.read_text()) or {}).get('periodicity',{}) if args.settings else metadata.get('periodicity',{})
    result=benchmark(args.trial_dir,metadata,resolved_config(cfg),args.max_trials)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
