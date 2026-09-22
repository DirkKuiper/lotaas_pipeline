"""Import live, consistent node-ledger snapshots while searches are running."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from euroflash.cluster import allowed_nodes, ssh_args
from euroflash.collect import collect

SNAPSHOT = '''
import fcntl,json,sqlite3,sys
from pathlib import Path
root=Path(sys.argv[1]);active=True
with (root/'.run.lock').open('a') as lock:
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active=False
    except BlockingIOError:pass
    source=sqlite3.connect(root/'ledger.sqlite',timeout=120)
    destination=sqlite3.connect(':memory:')
    source.backup(destination)
    print(json.dumps({'active':active}),flush=True)
    sys.stdout.buffer.write(destination.serialize())
    destination.close();source.close()
'''


def snapshot(node, run_name, directory, ledger, control_dir=None):
    root=f'/home/dkuiper/lotaas-runs/{run_name}/work'
    command=ssh_args(node,control_dir)+[shlex.join(['python3','-c',SNAPSHOT,root])]
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/(node+'-live.sqlite')
    process=subprocess.Popen(command,stdout=subprocess.PIPE)
    try:
        status=json.loads(process.stdout.readline())
        partial=path.with_suffix('.partial')
        with partial.open('wb') as stream:shutil.copyfileobj(process.stdout,stream)
        if process.wait():raise RuntimeError('Node snapshot failed: '+node)
        partial.replace(path)
    finally:
        process.stdout.close()
        if process.poll() is None:process.terminate();process.wait()
    collect(path,ledger,node+'/'+run_name)
    return status['active']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nodes',nargs='+',default=['efc-gpu-01'])
    p.add_argument('--run-name',required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--control-dir',type=Path)
    p.add_argument('--once',action='store_true')
    a=p.parse_args()
    permitted=allowed_nodes()
    if not set(a.nodes)<=permitted:
        p.error('Nodes must be among '+', '.join(sorted(permitted))
                +' (set LOTAAS_ALLOWED_NODES to change)')
    if not a.run_name.replace('-','').replace('_','').isalnum():p.error('Invalid run name')
    while True:
        states={node:snapshot(node,a.run_name,a.work,a.ledger,a.control_dir) for node in a.nodes}
        print(json.dumps({'active':states}),flush=True)
        if a.once or not any(states.values()):break
        time.sleep(60)


if __name__=='__main__':main()
