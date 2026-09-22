"""One-year throughput requirements; forecast only from representative timings."""
import argparse
import json


def capacity(beams, gpus, utilization, gpu_seconds=None, cpu_seconds=None, cpu_workers=None,
             bytes_per_beam=None, download_mbps=None, prepare_seconds=None, prepare_workers=None,
             flatfield_seconds_per_beam=None, archive_beams_per_day=None):
    if beams < 1 or gpus < 1 or not 0 < utilization <= 1:
        raise ValueError('Positive beam/GPU counts and utilization in (0,1] required')
    year=365*86400
    result={'beams':beams,'gpus':gpus,'utilization_assumption':utilization,
            'required_beams_per_day':beams/365,
            'maximum_gpu_seconds_per_beam':gpus*year*utilization/beams,
            'forecast_days':None,'forecast_validated':False}
    if bytes_per_beam:
        result['total_input_tb']=bytes_per_beam*beams/1e12
        result['minimum_download_mbps']=bytes_per_beam*beams/year/utilization/1e6
    stages={}
    if gpu_seconds:
        stages['gpu']=beams*gpu_seconds/gpus/utilization/86400
    if cpu_seconds and cpu_workers:
        stages['cpu']=beams*cpu_seconds/cpu_workers/utilization/86400
    if bytes_per_beam and download_mbps:
        stages['transfer']=beams*bytes_per_beam/(download_mbps*1e6)/utilization/86400
    if prepare_seconds and prepare_workers:
        stages['prepare']=beams*prepare_seconds/prepare_workers/utilization/86400
    if flatfield_seconds_per_beam:
        stages['flatfield']=beams*flatfield_seconds_per_beam/utilization/86400
    if archive_beams_per_day:
        stages['archive']=beams/archive_beams_per_day/utilization
    if stages:
        result['stage_days']=stages
        result['measured_stage_lower_bound_days']=max(stages.values())
        if set(stages)=={'gpu','cpu','transfer','prepare','flatfield','archive'}:
            result['forecast_days']=max(stages.values())
        result['unmeasured_stages']=sorted({'gpu','cpu','transfer','prepare','flatfield','archive'}-set(stages))
        result['scope']='Steady-state estimate at the supplied beam count; pilot extrapolation does not validate full catalogue coverage, sustained availability, or sensitivity.'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--beams',type=int,required=True)
    p.add_argument('--gpus',type=int,default=4)
    p.add_argument('--utilization',type=float,default=.7)
    p.add_argument('--gpu-seconds',type=float)
    p.add_argument('--cpu-seconds',type=float)
    p.add_argument('--cpu-workers',type=int)
    p.add_argument('--bytes-per-beam',type=int)
    p.add_argument('--download-mbps',type=float)
    p.add_argument('--prepare-seconds',type=float)
    p.add_argument('--prepare-workers',type=int)
    p.add_argument('--flatfield-seconds-per-beam',type=float)
    p.add_argument('--archive-beams-per-day',type=float,help='Measured sustained tape staging rate before the utilization allowance')
    a=p.parse_args()
    print(json.dumps(capacity(**vars(a)),indent=2))


if __name__=='__main__':
    main()
