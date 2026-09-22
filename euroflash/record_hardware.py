"""Record hardware, CUDA status and storage as JSON evidence (run on each host)."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time


def command(args):
    try:
        r=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
        return {'exit_code':r.returncode,'output':r.stdout.strip()}
    except (OSError,subprocess.TimeoutExpired) as e:return {'error':str(e)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result={'host':socket.gethostname(),'unix_time':time.time(),'logical_cpus':os.cpu_count(),
            'cpu':command(['lscpu','-J']),'memory':command(['free','-b']),
            'gpus':command(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,driver_version,compute_cap','--format=csv']),
            'storage':command(['df','-hT',str(Path.home()),'/shared/results','/tmp']),
            'apptainer':command(['apptainer','--version'])}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
