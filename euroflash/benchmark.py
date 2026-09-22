"""Benchmark representative dedispersion geometry without writing trial files."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import yaml
from lotaas_reprocessing.dedispersion import backend, iter_dedispersed
from lotaas_reprocessing.filterbank import FilterbankFile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('filterbank',type=Path)
    p.add_argument('--settings',type=Path,default=Path('settings.yaml'))
    p.add_argument('--backend',choices=['cpu','gpu'],default='gpu')
    p.add_argument('--trials-per-range',type=int,default=10)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    xp=backend(a.backend)
    fb=FilterbankFile(str(a.filterbank))
    data=fb.get_spectra(0,fb.nspec).T.copy()
    frequencies=fb.frequencies
    plan=yaml.safe_load(a.settings.read_text())['dedispersion_plan']
    results=[]
    for entry in plan:
        dms=np.arange(entry['low_dm'],entry['high_dm'],entry['ddm'])
        selected=dms[np.linspace(0,len(dms)-1,min(a.trials_per_range,len(dms))).astype(int)]
        start=time.monotonic();last=start;times=[]
        for dm,trial in iter_dedispersed(data,fb.tsamp,frequencies,selected,entry['downsample'],xp):
            if xp is not np:xp.cuda.Stream.null.synchronize()
            now=time.monotonic();times.append(now-last);last=now
        median=float(np.median(times[1:] or times))
        r=dict(entry,trials=len(dms),sampled_trials=len(selected),elapsed=time.monotonic()-start,
               first_trial_seconds=times[0],median_trial_seconds=median,
               projected_range_seconds=times[0]+median*(len(dms)-1))
        results.append(r);print(json.dumps(r),flush=True)
    fb.close()
    output={'backend':a.backend,'samples':data.shape[1],'channels':data.shape[0],
            'ranges':results,'projected_dedispersion_seconds':sum(r['projected_range_seconds'] for r in results),
            'includes_preprocessing_classification_io':False}
    a.output.write_text(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
