"""Transfer prepared beams to non-BFC GPU nodes, run, and collect all results.

The head owns the campaign DB. Each node writes a separate DB; SQLite's backup
API takes a consistent snapshot for import. Shared storage and Slurm are not
required. SSH must already authenticate, optionally via a ControlPath.
"""
import argparse
import concurrent.futures as futures
import getpass
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
import traceback
from euroflash.collect import collect
from euroflash.ledger import Ledger

REPO=Path(__file__).resolve().parents[1]

# Nodes this runner may dispatch to. The allowlist exists to keep work off
# the BFC nodes, so it stays an opt-in: set LOTAAS_ALLOWED_NODES to a
# comma-separated list to run somewhere else. efc-gpu-00 is listed but
# cannot currently run CUDA; see the README on its UVM module.
DEFAULT_ALLOWED={'efc-gpu-00','efc-gpu-01'}


def allowed_nodes():
    configured=os.environ.get('LOTAAS_ALLOWED_NODES','').strip()
    if not configured:
        return set(DEFAULT_ALLOWED)
    return {name.strip() for name in configured.split(',') if name.strip()}


def ssh_args(node,control_dir=None):
    args=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15']
    if control_dir:
        args+=['-S',str(control_dir/('lotaas-'+node.removeprefix('efc-').replace('-','')))]
    return args+[node]


def remote(node,args,control_dir=None):
    return subprocess.run(ssh_args(node,control_dir)+[shlex.join([str(x) for x in args])],check=True)


def node_health(node,control_dir=None):
    """Whether a node can actually run CUDA, before any beam is assigned.

    Beams were split across the nodes named on the command line whether or
    not they could run. On efc-gpu-00 the UVM module is blocked, so cuInit
    returns 999 and its whole share of the batch fails after the transfer.
    The device node is the cheapest reliable symptom and needs no image.
    """
    probe='nvidia-smi -L && test -e /dev/nvidia-uvm && echo UVM_PRESENT'
    try:
        result=subprocess.run(ssh_args(node,control_dir)+[probe],
                              capture_output=True,text=True,timeout=120)
    except subprocess.TimeoutExpired:
        return False,'health probe timed out'
    output=(result.stdout or '')+(result.stderr or '')
    gpus=[line for line in (result.stdout or '').splitlines() if line.startswith('GPU ')]
    if 'UVM_PRESENT' in output:
        if not gpus:
            return False,'nvidia-smi listed no GPUs'
        return True,f'{len(gpus)} GPUs, UVM present'
    if gpus:
        # The probe exits non-zero when the device node is absent, so this
        # case is checked before the return code: nvidia-smi answered, which
        # means the host is up and has a driver, and only UVM is missing.
        return False,'no /dev/nvidia-uvm; CUDA cannot initialise on this host'
    return False,'health probe failed: '+' '.join(output.split())[:200]


def upload(node,files,destination,control_dir=None):
    remote(node,['mkdir','-p',destination],control_dir)
    command=ssh_args(node,control_dir)+[shlex.join(['tar','xf','-','-C',destination])]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdin,mode='w|') as archive:
            for path,name in files:
                archive.add(path,arcname=name,recursive=False)
        process.stdin.close()
        if process.wait():raise RuntimeError(f'Transfer failed to {node}')
    finally:
        if process.poll() is None:process.terminate();process.wait()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='Prepared *_ff.fil files')
    p.add_argument('--work',type=Path,required=True,help='Head-node collection directory')
    p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--nodes',nargs='+',default=['efc-gpu-01'])
    p.add_argument('--remote-root',default=f'/home/{getpass.getuser()}/lotaas-runs',
                   help='Absolute directory on each compute node to stage runs under')
    p.add_argument('--control-dir',type=Path)
    p.add_argument('--image',type=Path,default=REPO/'containers/euroflash-runtime.sif')
    p.add_argument('--settings',type=Path,default=REPO/'settings.yaml')
    p.add_argument('--gpus',default='0,1')
    p.add_argument('--cpu-workers',type=int,default=24)
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--run-name',required=True)
    p.add_argument('--skip-health-check',action='store_true',
                   help='Dispatch without probing CUDA on each node first')
    a=p.parse_args()
    permitted=allowed_nodes()
    if not set(a.nodes)<=permitted or len(set(a.nodes))!=len(a.nodes):
        p.error('Nodes must be unique and among '+', '.join(sorted(permitted))
                +' (set LOTAAS_ALLOWED_NODES to change)')
    if not a.remote_root.startswith('/'):
        p.error('--remote-root must be an absolute path on the compute node')
    if not a.run_name.replace('-','').replace('_','').isalnum():p.error('Use letters, digits, hyphens and underscores in run-name')
    beams=sorted(a.input.resolve().rglob('*_ff.fil'))
    if not beams:p.error('No prepared beams found')
    if len({f.name for f in beams})!=len(beams):p.error('Duplicate beam basenames')
    # A frozen source snapshot avoids edits to a checkout changing a running job.
    source=[(f,str(f.relative_to(REPO))) for f in REPO.rglob('*') if f.is_file()
            and f.suffix in {'.py','.yaml','.lock'} and not any(x in f.parts for x in ['.git','__pycache__','.pytest_cache'])]
    source += [(a.image.resolve(),'containers/runtime.sif'),(a.settings.resolve(),'campaign-settings.yaml')]
    a.work.mkdir(parents=True,exist_ok=True)
    ledger=Ledger(a.ledger)

    # Probe before splitting the batch, so a node that cannot run CUDA does
    # not silently take its share of the beams and fail them all.
    nodes=list(a.nodes)
    if not a.skip_health_check:
        with futures.ThreadPoolExecutor(max_workers=len(nodes)) as pool:
            health=dict(zip(nodes,pool.map(lambda n:node_health(n,a.control_dir),nodes)))
        for node,(ok,detail) in health.items():
            print(('healthy  ' if ok else 'UNUSABLE ')+node+': '+detail,flush=True)
        for node,(ok,detail) in health.items():
            if not ok:
                attempt=ledger.start(a.run_name+'@'+node,'dispatch',a.run_name,
                                     a.work/(node+'.log'),['euroflash.cluster',node,'health'])
                ledger.finish(attempt,error='Node unusable before dispatch: '+detail)
        nodes=[n for n in nodes if health[n][0]]
        if not nodes:
            raise RuntimeError('No usable node among '+', '.join(a.nodes)
                               +'; see the health lines above')
        if len(nodes)!=len(a.nodes):
            print(f'Dispatching to {len(nodes)} of {len(a.nodes)} nodes',flush=True)

    def worker_body(pair):
        index,node=pair
        assigned=beams[index::len(nodes)]
        if not assigned:return None
        root=a.remote_root.rstrip('/')+'/'+a.run_name
        repo=root+'/source';inputs=root+'/input';work=root+'/work'
        upload(node,source,repo,a.control_dir)
        upload(node,[(f,f.name) for f in assigned],inputs,a.control_dir)
        command=['python3','-m','euroflash.run','--prepared','--input',inputs,'--work',work,
                 '--ledger',work+'/ledger.sqlite','--image',repo+'/containers/runtime.sif',
                 '--settings',repo+'/campaign-settings.yaml','--backend','gpu','--gpus',a.gpus,
                 '--cpu-workers',str(a.cpu_workers)]+(['--pilot'] if a.pilot else [])
        expression='cd '+shlex.quote(repo)+' && '+shlex.join(command)
        log=a.work/(node+'.log')
        with log.open('a') as stream:
            result=subprocess.run(ssh_args(node,a.control_dir)+[expression],stdout=stream,stderr=subprocess.STDOUT)
        # Snapshot even failed runs so their errors reach the campaign ledger.
        remote(node,['python3','-c','import sqlite3,sys; s=sqlite3.connect(sys.argv[1]); d=sqlite3.connect(sys.argv[2]); s.backup(d); d.close(); s.close();',work+'/ledger.sqlite',work+'/ledger-snapshot.sqlite'],a.control_dir)
        destination=a.work/node;destination.mkdir(exist_ok=True)
        process=subprocess.Popen(ssh_args(node,a.control_dir)+[shlex.join(['tar','cf','-','-C',work,'.'])],stdout=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=process.stdout,mode='r|') as archive:
                for member in archive:
                    archive.extract(member,destination,filter='data')
            if process.wait():raise RuntimeError(f'Result transfer failed from {node}')
        finally:
            process.stdout.close()
        if result.returncode:
            return node,destination,False
        return node,destination,True
    def worker(pair):
        index,node=pair
        if not beams[index::len(nodes)]:return None
        log=a.work/(node+'.log')
        attempt=ledger.start(a.run_name+'@'+node,'dispatch',a.run_name,log,
                             ['euroflash.cluster',node,a.run_name])
        try:
            _,destination,ok=worker_body(pair)
            collect(destination/'ledger-snapshot.sqlite',a.ledger,node+'/'+a.run_name)
            if not ok:raise RuntimeError('Remote pipeline failed; collected stage errors are in the campaign ledger')
            ledger.finish(attempt,[destination/'ledger-snapshot.sqlite',log])
            return None
        except Exception as error:
            with log.open('a') as stream:traceback.print_exc(file=stream)
            ledger.finish(attempt,error=str(error))
            return node
    with futures.ThreadPoolExecutor(max_workers=len(nodes)) as pool:
        failed=[node for node in pool.map(worker,enumerate(nodes)) if node]
    if failed:raise RuntimeError('Pipeline failed on '+','.join(failed)+'; collected logs and errors are available')


if __name__=='__main__':
    main()
