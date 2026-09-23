"""Transfer prepared beams to non-BFC GPU nodes, run, and collect all results.

The head owns the campaign DB. Each node writes a separate DB; SQLite's backup
API takes a consistent snapshot for import. Shared storage and Slurm are not
required. SSH must already authenticate, optionally via a ControlPath.

With --cpu-nodes the work is split in two tiers. The GPU node dedisperses and
runs the single-pulse search, and marks each beam ready as it finishes. The
head relays every ready beam (its filterbank, clusters and periodic trials) to
a CPU node, which classifies it with FETCH, searches it for periodicity and,
once the batch is complete, applies the cross-beam veto, folds and finishes.
The GPU node is free for the next batch as soon as its own stages end, which
the head records in <work>/<node>.gpu-done. Intermediate files and each
node's ledger stay on that node's local disk; only results come back.
"""
import argparse
import concurrent.futures as futures
import fcntl
import getpass
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
import tempfile
import threading
import time
import traceback
from euroflash.collect import collect
from euroflash.ledger import Ledger

REPO=Path(__file__).resolve().parents[1]

# Nodes this runner may dispatch to. The allowlist exists to keep work off
# the BFC nodes, so it stays an opt-in: set LOTAAS_ALLOWED_NODES to a
# comma-separated list to run somewhere else. efc-gpu-00 is listed but
# cannot currently run CUDA; see the README on its UVM module.
DEFAULT_ALLOWED={'efc-gpu-00','efc-gpu-01'}
# CPU nodes for classification and periodicity; LOTAAS_ALLOWED_CPU_NODES overrides.
DEFAULT_CPU_ALLOWED={f'efc-cpu-{i:02d}' for i in range(8)}


def allowed_nodes(variable='LOTAAS_ALLOWED_NODES',default=DEFAULT_ALLOWED):
    configured=os.environ.get(variable,'').strip()
    if not configured:
        return set(default)
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


def cpu_health(node,control_dir=None,root='/',min_free_gb=300):
    """A CPU node can run the container and has room for a batch's backlog."""
    probe=f'command -v apptainer >/dev/null && echo APPTAINER; df -BG --output=avail {shlex.quote(root)} | tail -1'
    try:
        result=subprocess.run(ssh_args(node,control_dir)+[probe],capture_output=True,text=True,timeout=60)
    except subprocess.TimeoutExpired:
        return False,'health probe timed out'
    output=result.stdout or ''
    if 'APPTAINER' not in output:
        return False,'unreachable or no apptainer: '+' '.join(((result.stderr or '')+output).split())[:200]
    try:
        free=int(output.split()[-1].rstrip('G'))
    except (ValueError,IndexError):
        return False,'cannot read free disk'
    if free<min_free_gb:
        return False,f'{free} GB free under {root}, below {min_free_gb}'
    return True,f'{free} GB free'


class CpuSlot:
    """An exclusive slot on a CPU node, held with flock on the head across cluster runs."""

    def __init__(self,nodes,lock_dir,slots=1,control_dir=None,root='/',wait=30,log=print):
        self.nodes,self.lock_dir,self.slots=list(nodes),Path(lock_dir),slots
        self.control_dir,self.root,self.wait,self.log=control_dir,root,wait,log
        self.node=self.handle=None

    def __enter__(self):
        self.lock_dir.mkdir(parents=True,exist_ok=True)
        unusable={}
        while True:
            for slot in range(self.slots):
                for node in self.nodes:
                    if unusable.get(node,0)>time.time():
                        continue
                    handle=(self.lock_dir/f'{node}.{slot}.lock').open('w')
                    try:
                        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        handle.close();continue
                    ok,detail=cpu_health(node,self.control_dir,self.root)
                    if ok:
                        handle.write(f'{os.getpid()}\n');handle.flush()
                        self.node,self.handle=node,handle
                        self.log(f'CPU tier on {node} (slot {slot}): {detail}')
                        return node
                    handle.close()
                    unusable[node]=time.time()+600
                    self.log(f'UNUSABLE {node}: {detail}')
            time.sleep(self.wait)

    def __exit__(self,*_):
        if self.handle is not None:
            self.handle.close()


def relay(source,destination,root,members,control_dir=None,excludes=()):
    """Stream files between two nodes through the head, without touching its disk."""
    reader=subprocess.Popen(ssh_args(source,control_dir)+[shlex.join(['tar','cf','-',*excludes,'-C',root,*members])],
                            stdout=subprocess.PIPE)
    writer=subprocess.Popen(ssh_args(destination,control_dir)+[shlex.join(['tar','xf','-','-C',root])],
                            stdin=reader.stdout)
    reader.stdout.close()
    if writer.wait() or reader.wait():
        raise RuntimeError(f'Relay of {members[0]} from {source} to {destination} failed')


def listing(node,directory,control_dir=None):
    result=subprocess.run(ssh_args(node,control_dir)+[shlex.join(['ls','-A',directory])],
                          capture_output=True,text=True)
    return set(result.stdout.split()) if result.returncode==0 else set()


def put_text(node,path,text,control_dir=None):
    """Write a small file atomically on a node."""
    partial=path+'.partial'
    subprocess.run(ssh_args(node,control_dir)+[f'cat > {shlex.quote(partial)} && mv {shlex.quote(partial)} {shlex.quote(path)}'],
                   input=text,text=True,check=True)


def fetch(node,directory,destination,control_dir=None,excludes=()):
    destination.mkdir(parents=True,exist_ok=True)
    process=subprocess.Popen(ssh_args(node,control_dir)+[shlex.join(['tar','cf','-',*excludes,'-C',directory,'.'])],
                             stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout,mode='r|') as archive:
            for member in archive:
                archive.extract(member,destination,filter='data')
        if process.wait():raise RuntimeError(f'Result transfer failed from {node}')
    finally:
        process.stdout.close()


def snapshot(node,work,control_dir=None):
    remote(node,['python3','-c','import sqlite3,sys; s=sqlite3.connect(sys.argv[1]); d=sqlite3.connect(sys.argv[2]); s.backup(d); d.close(); s.close();',work+'/ledger.sqlite',work+'/ledger-snapshot.sqlite'],control_dir)


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


def upload_parallel(node,files,destination,control_dir=None,streams=4):
    """upload() over several SSH streams: one stream measured 260-370 MB/s from the head."""
    groups=[[] for _ in range(max(1,min(streams,len(files))))]
    for index,item in enumerate(sorted(files,key=lambda f:-Path(f[0]).stat().st_size)):
        groups[index%len(groups)].append(item)
    remote(node,['mkdir','-p',destination],control_dir)
    with futures.ThreadPoolExecutor(max_workers=len(groups)) as pool:
        list(pool.map(lambda group:upload(node,group,destination,control_dir),groups))


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
    p.add_argument('--workers-per-gpu',type=int,default=3,
                   help='Dedispersion workers sharing each GPU (3 measured 1.77x one worker)')
    p.add_argument('--cpu-workers',type=int,default=24)
    p.add_argument('--cpu-nodes',nargs='+',default=[],
                   help='Classify and search for periodicity on these CPU nodes instead of the GPU node')
    p.add_argument('--cpu-tier-workers',type=int,default=64,help='Concurrent beams on a CPU node')
    p.add_argument('--cpu-slots',type=int,default=1,help='Batches one CPU node may take at once')
    p.add_argument('--cpu-lock-dir',type=Path,help='Head directory of CPU slot locks, shared by concurrent runs')
    p.add_argument('--relay-streams',type=int,default=4,help='Beams relayed to the CPU node at once')
    p.add_argument('--upload-streams',type=int,default=4,help='SSH streams carrying a batch to its GPU node')
    p.add_argument('--relay-backlog',type=int,default=40,
                   help='Beams relayed to the CPU node and not yet searched there, at most')
    p.add_argument('--stage-timeout',action='append',default=[],metavar='STAGE=SECONDS')
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--run-name',required=True)
    p.add_argument('--skip-health-check',action='store_true',
                   help='Dispatch without probing CUDA on each node first')
    p.add_argument('--skip-trials',action='store_true',
                   help='Collect results without retained DM trials; a retry regenerates them on the GPU')
    p.add_argument('--exclude-beams',type=int,nargs='+',default=[12],
                   help='Beams never searched (default: 12, the incoherent beam)')
    p.add_argument('--cleanup-remote',action='store_true',
                   help="Remove this run's input, work and source copy from each node after collection")
    a=p.parse_args()
    if a.workers_per_gpu<1:p.error('--workers-per-gpu must be positive')
    permitted=allowed_nodes()
    if not set(a.nodes)<=permitted or len(set(a.nodes))!=len(a.nodes):
        p.error('Nodes must be unique and among '+', '.join(sorted(permitted))
                +' (set LOTAAS_ALLOWED_NODES to change)')
    cpu_permitted=allowed_nodes('LOTAAS_ALLOWED_CPU_NODES',DEFAULT_CPU_ALLOWED)
    if not set(a.cpu_nodes)<=cpu_permitted or len(set(a.cpu_nodes))!=len(a.cpu_nodes):
        p.error('CPU nodes must be unique and among '+', '.join(sorted(cpu_permitted))
                +' (set LOTAAS_ALLOWED_CPU_NODES to change)')
    if not a.remote_root.startswith('/'):
        p.error('--remote-root must be an absolute path on the compute node')
    if a.cleanup_remote and len(PurePosixPath(a.remote_root).parts)<3:
        p.error('--cleanup-remote needs a --remote-root at least two levels deep')
    if not a.run_name.replace('-','').replace('_','').isalnum():p.error('Use letters, digits, hyphens and underscores in run-name')
    from euroflash.beams import excluded
    beams=[f for f in sorted(a.input.resolve().rglob('*_ff.fil')) if not excluded(f,a.exclude_beams)]
    if not beams:p.error('No prepared beams found')
    if len({f.name for f in beams})!=len(beams):p.error('Duplicate beam basenames')
    # A frozen source snapshot avoids edits to a checkout changing a running job.
    source=[(f,str(f.relative_to(REPO))) for f in REPO.rglob('*') if f.is_file()
            and f.suffix in {'.py','.yaml','.lock'} and not any(x in f.parts for x in ['.git','__pycache__','.pytest_cache'])]
    source += [(a.image.resolve(),'containers/runtime.sif'),(a.settings.resolve(),'campaign-settings.yaml')]
    a.work.mkdir(parents=True,exist_ok=True)
    ledger=Ledger(a.ledger)
    tiered=bool(a.cpu_nodes)
    lock_dir=a.cpu_lock_dir or a.work.parent/'.cpu-slots'

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
                (a.work/f'{node}.gpu-done').touch()
        nodes=[n for n in nodes if health[n][0]]
        if not nodes:
            raise RuntimeError('No usable node among '+', '.join(a.nodes)
                               +'; see the health lines above')
        if len(nodes)!=len(a.nodes):
            print(f'Dispatching to {len(nodes)} of {len(a.nodes)} nodes',flush=True)
    root=a.remote_root.rstrip('/')+'/'+a.run_name
    repo=root+'/source';inputs=root+'/input';work=root+'/work'

    def runner_command(stages,gpu_node=True):
        command=['python3','-m','euroflash.run','--prepared','--input',inputs,'--work',work,
                 '--ledger',work+'/ledger.sqlite','--image',repo+'/containers/runtime.sif',
                 '--settings',repo+'/campaign-settings.yaml','--backend','gpu','--gpus',a.gpus,
                 '--workers-per-gpu',str(a.workers_per_gpu),
                 '--exclude-beams',*map(str,a.exclude_beams),
                 '--cpu-workers',str(a.cpu_workers if gpu_node else a.cpu_tier_workers),
                 '--stages',stages]+(['--pilot'] if a.pilot else [])
        if not gpu_node:
            # Many stages share the node; one full-width thread pool each would oversubscribe it.
            command+=['--threads-per-stage','2']
        for value in a.stage_timeout:
            command+=['--stage-timeout',value]
        return 'cd '+shlex.quote(repo)+' && '+shlex.join(command)

    def run_remote(node,expression,log):
        with log.open('a') as stream:
            return subprocess.run(ssh_args(node,a.control_dir)+[expression],stdout=stream,stderr=subprocess.STDOUT).returncode

    def collect_node(node,excludes):
        snapshot(node,work,a.control_dir)
        destination=a.work/node
        fetch(node,work,destination,a.control_dir,excludes)
        collect(destination/'ledger-snapshot.sqlite',a.ledger,node+'/'+a.run_name)
        return destination

    def cleanup(node):
        if a.cleanup_remote:
            # Only after the results and ledger are safely on the head. A
            # continuous campaign otherwise leaves ~90 GB per SAP on the node.
            remote(node,['rm','-rf','--one-file-system',root],a.control_dir)

    def worker_body(pair):
        index,node=pair
        assigned=beams[index::len(nodes)]
        if not assigned:return True
        upload(node,source,repo,a.control_dir)
        upload_parallel(node,[(f,f.name) for f in assigned],inputs,a.control_dir,a.upload_streams)
        log=a.work/(node+'.log')
        if not tiered:
            code=run_remote(node,runner_command('all'),log)
            # Trials of a failed beam are ~4.7 GB and regenerate in seconds on a GPU.
            collect_node(node,['--exclude=DM_trials','--exclude=Periodic_DM_trials'] if a.skip_trials else [])
            cleanup(node)
            return code==0
        return tiered_body(node,log)

    def tiered_body(node,log):
        """Dedisperse and search on the GPU node while a CPU node takes each finished beam."""
        state={'gpu':None,'collected':False}
        gpu_thread=threading.Thread(target=lambda:state.update(gpu=run_remote(node,runner_command('gpu'),log)))
        gpu_thread.start()
        cpu_log=a.work/'cpu-tier.log'
        def note(text):
            with cpu_log.open('a') as stream:stream.write(time.strftime('%H:%M:%S ')+text+'\n')
        try:
            with CpuSlot(a.cpu_nodes,lock_dir,a.cpu_slots,a.control_dir,'/home',log=note) as cpu:
                upload(cpu,source,repo,a.control_dir)
                remote(cpu,['mkdir','-p',inputs,work+'/handoff'],a.control_dir)
                cpu_state={}
                cpu_thread=threading.Thread(target=lambda:cpu_state.update(
                    code=run_remote(cpu,runner_command('cpu',gpu_node=False),a.work/(cpu+'.log'))))
                cpu_thread.start()
                relayed,failed_relays=set(),{}
                with futures.ThreadPoolExecutor(max_workers=max(1,a.relay_streams)) as pool:
                    while True:
                        if not cpu_thread.is_alive():
                            raise RuntimeError(f'CPU runner on {cpu} exited {cpu_state.get("code")} before the batch was handed over')
                        finished=not gpu_thread.is_alive()
                        ready={name[:-6] for name in listing(node,work+'/handoff',a.control_dir) if name.endswith('.ready')}
                        there=listing(cpu,work+'/handoff',a.control_dir)
                        backlog=len({n[:-6] for n in there if n.endswith('.ready')}-{n[:-9] for n in there if n.endswith('.searched')})
                        todo=sorted(ready-relayed-set(failed_relays))[:max(0,a.relay_backlog-backlog)]
                        def move(item):
                            marker=json.loads(subprocess.run(ssh_args(node,a.control_dir)+['cat',f'{work}/handoff/{item}.ready'],
                                                             capture_output=True,text=True,check=True).stdout)
                            members=[str(PurePosixPath(marker['input']).relative_to(root)),'work/'+marker['output']]
                            relay(node,cpu,root,members,a.control_dir,['--exclude=DM_trials'])
                            put_text(cpu,f'{work}/handoff/{item}.ready',json.dumps(marker),a.control_dir)
                            remote(node,['rm','-rf',marker['input'],work+'/'+marker['output'],
                                         f'{work}/handoff/{item}.ready'],a.control_dir)
                            return item
                        for future,item in [(pool.submit(move,item),item) for item in todo]:
                            try:
                                relayed.add(future.result())
                            except Exception as error:
                                failed_relays[item]=str(error)
                                note(f'relay of {item} failed: {error}')
                        if finished and not (ready-relayed-set(failed_relays)):
                            break
                        if not todo:
                            time.sleep(15)
                note(f'GPU node {node} exited {state["gpu"]}; {len(relayed)} beams relayed, {len(failed_relays)} failed')
                # The GPU node's own stages are over: record them and free it for the next batch.
                collect_node(node,['--exclude=processed','--exclude=DM_trials','--exclude=Periodic_DM_trials'])
                state['collected']=True
                (a.work/f'{node}.gpu-done').write_text(json.dumps({'exit':state['gpu'],'relayed':len(relayed)}))
                done=json.loads(subprocess.run(ssh_args(node,a.control_dir)+['cat',f'{work}/handoff/.done'],
                                               capture_output=True,text=True).stdout or '{}')
                put_text(cpu,f'{work}/handoff/.done',json.dumps(dict(done,relay_failed=sorted(failed_relays))),a.control_dir)
                cleanup(node)
                state['cleaned']=True
                cpu_thread.join()
                collect_node(cpu,['--exclude=DM_trials','--exclude=Periodic_DM_trials'])
                cleanup(cpu)
                return state['gpu']==0 and cpu_state.get('code')==0 and not failed_relays
        finally:
            gpu_thread.join()
            # Whatever happened, leave the GPU node recorded and empty before it takes another batch.
            try:
                if not state['collected']:
                    collect_node(node,['--exclude=processed','--exclude=DM_trials','--exclude=Periodic_DM_trials'])
                if not state.get('cleaned'):
                    cleanup(node)
            except Exception as error:
                note(f'GPU node {node} could not be collected or cleaned: {error}')
            (a.work/f'{node}.gpu-done').touch()

    def worker(pair):
        index,node=pair
        if not beams[index::len(nodes)]:return None
        log=a.work/(node+'.log')
        attempt=ledger.start(a.run_name+'@'+node,'dispatch',a.run_name,log,
                             ['euroflash.cluster',node,a.run_name]+(['cpu:'+','.join(a.cpu_nodes)] if tiered else []))
        try:
            ok=worker_body(pair)
            if not ok:raise RuntimeError('Remote pipeline failed; collected stage errors are in the campaign ledger')
            ledger.finish(attempt,[a.work/node/'ledger-snapshot.sqlite',log])
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
