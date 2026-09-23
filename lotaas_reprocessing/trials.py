"""Shared validation and atomic reuse of dedispersion products."""
import hashlib
import json
import os
from pathlib import Path
import shutil
from .dm_plan import dm_values, dm_label


def trial_specs(metadata, plan=None):
    base=Path(metadata['filename']).stem
    samples=metadata['samples_processed']
    specs={}
    for entry in plan if plan is not None else metadata['dedispersion_plan']:
        ds=entry['downsample']
        if int(ds)!=ds or ds<1:
            raise ValueError('downsample must be a positive integer')
        for dm in dm_values(entry):
            name=f'{base}_DM{dm_label(dm)}.dat'
            if name in specs:
                raise ValueError('DM plan produces colliding filenames')
            specs[name]={'dm':dm,'downsample':int(ds),'ddm':float(entry['ddm']),
                         'bytes':(samples//int(ds))*4}
    if not specs:
        raise ValueError('Empty DM plan')
    return specs


def file_digest(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def validate_trials(metadata,directory,plan=None):
    expected=trial_specs(metadata,plan)
    actual={p.name:p.stat().st_size for p in Path(directory).glob('*.dat')}
    if actual!={name:spec['bytes'] for name,spec in expected.items()}:
        raise ValueError('DM trials are incomplete, mixed between beams, or have incorrect sample counts')


def link_trial(source,target):
    """Replace even an existing target; never trust stale hard links on retry."""
    source,target=Path(source),Path(target)
    if source==target:
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    partial=target.with_name(target.name+'.partial')
    partial.unlink(missing_ok=True)
    os.link(source,partial)
    os.replace(partial,target)
    inf=target.with_suffix('.inf')
    partial_inf=inf.with_name(inf.name+'.partial')
    shutil.copyfile(source.with_suffix('.inf'),partial_inf)
    os.replace(partial_inf,inf)


def write_manifest(output):
    from .periodicity import atomic_json
    output=Path(output)
    paths=list((output/'DM_trials').glob('*.dat'))+list((output/'Periodic_DM_trials').glob('*.dat'))
    by_inode={}; files={}
    for path in sorted(paths):
        stat=path.stat(); key=(stat.st_dev,stat.st_ino)
        if key not in by_inode:
            by_inode[key]=file_digest(path)
        files[str(path.relative_to(output))]={'bytes':stat.st_size,'sha256':by_inode[key]}
    atomic_json(output/'trial_manifest.json',{'schema':1,'files':files})


def validate_products(output):
    """Resume guard: detect missing, truncated AND same-size corrupt trials."""
    output=Path(output)
    metadata=json.loads((output/'metadata.json').read_text())
    validate_trials(metadata,output/'DM_trials')
    expected={'DM_trials/'+name for name in trial_specs(metadata)}
    if metadata.get('periodicity_enabled'):
        plan=metadata['periodicity_dm_plan']
        validate_trials(metadata,output/'Periodic_DM_trials',plan)
        expected|={'Periodic_DM_trials/'+name for name in trial_specs(metadata,plan)}
    files=json.loads((output/'trial_manifest.json').read_text())['files']
    if set(files)!=expected:
        raise ValueError('Trial manifest does not match the enabled DM plans')
    by_inode={}
    for name,record in files.items():
        path=output/name; stat=path.stat(); key=(stat.st_dev,stat.st_ino)
        if key not in by_inode:
            by_inode[key]=file_digest(path)
        if stat.st_size!=record['bytes'] or by_inode[key]!=record['sha256']:
            raise ValueError(f'Corrupted DM trial: {path}')
    return True


def product_outputs(output,summary_name):
    """Read and verify a completion manifest before cleanup or ledger success."""
    output=Path(output)
    summary_path=output/summary_name
    summary=json.loads(summary_path.read_text())
    if summary.get('complete') is not True or not summary.get('outputs'):
        raise ValueError(f'Incomplete search products: {summary_path}')
    files=[summary_path]
    for name,size in summary['outputs'].items():
        path=output/name
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe product path')
        if not path.is_file() or path.stat().st_size!=size:
            raise ValueError(f'Missing or truncated search product: {path}')
        files.append(path)
    return files
