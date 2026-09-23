"""Run LOTAAS without Slurm, with one search worker per selected GPU."""
import argparse
import concurrent.futures as futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time
import threading
from euroflash.ledger import Ledger

REPO = Path(__file__).resolve().parents[1]


# Every package whose source can change what a search produces. Searched
# recursively: a non-recursive glob left subpackages, and postproc entirely,
# outside the identity of a run.
CODE_FOLDERS = ['pipeline', 'preproc', 'lotaas_reprocessing', 'db', 'euroflash', 'postproc']


def image_digest(image):
    """Content hash of the runtime image, cached beside it.

    The image used to be identified by its path, size and modification time.
    Copying it to a compute node changes all three, so the same image gave
    one identity on the head and another on the node, and no beam could be
    recognised as already searched anywhere but where it ran.
    """
    stat = image.stat()
    marker = f'{stat.st_size}-{stat.st_mtime_ns}'
    cache = image.with_name(image.name + '.sha256')
    try:
        recorded, value = cache.read_text().split()
        if recorded == marker:
            return value
    except (OSError, ValueError):
        pass
    digest = hashlib.sha256()
    with image.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            digest.update(block)
    value = digest.hexdigest()
    try:
        cache.write_text(f'{marker} {value}\n')
    except OSError:
        pass  # A read-only image directory only costs the next run a rehash.
    return value


def fingerprint(settings, image, options, code_files=None):
    """Identity of a search method: the code, settings, image and options.

    Only content and stable names enter this digest. Absolute paths and
    modification times are deliberately excluded, because they made identical
    work on the head and on a compute node carry different identities.

    The batch is excluded too. A stage is keyed on (item, stage, fingerprint)
    and the item already names the beam, so hashing every input in the run
    only meant that adding one beam to a batch invalidated the finished work
    of every other beam in it. Which bytes a beam name stood for is attested
    separately, by the SHA256 archive receipts in the ledger.
    """
    digest = hashlib.sha256()
    if code_files is None:
        code_files = [p for folder in CODE_FOLDERS
                      for p in (REPO/folder).rglob('*.py')
                      if '__pycache__' not in p.parts]
    for path in sorted(code_files):
        digest.update(str(path.relative_to(REPO)).encode()); digest.update(path.read_bytes())
    digest.update(settings.read_bytes())
    digest.update(image_digest(image).encode())
    digest.update(json.dumps(options, sort_keys=True).encode())
    return digest.hexdigest()


class Runner:
    def __init__(self, args):
        self.args = args
        self.root = args.work.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root/'logs').mkdir(exist_ok=True)
        self.ledger = Ledger(args.ledger)
        self.settings = args.settings.resolve()
        self.image = args.image.resolve()
        self.fp = ''

    def command(self, script, arguments, gpu=None):
        paths = {REPO, self.root, self.args.input.resolve(), self.args.ledger.resolve().parent, self.settings.parent}
        command = ['apptainer', 'exec', '--cleanenv']
        if gpu is not None:
            command += ['--nv', '--env', 'CUDA_VISIBLE_DEVICES='+gpu]
        for path in sorted(paths):
            command += ['--bind', f'{path}:{path}']
        command += ['--env', 'PYTHONPATH='+str(REPO), '--env', 'LOTAAS_DB_PATH='+str(self.args.ledger.resolve()),
                    '--env', 'LOTAAS_RUN_FINGERPRINT='+self.fp,
                    str(self.image), 'python', str(REPO/script)]
        return command + [str(a) for a in arguments]

    def step(self, item, stage, command, expected, gpu=None, fingerprint_override=None, validator=None):
        fp = fingerprint_override or self.fp
        if self.ledger.completed(item, stage, fp):
            valid = True
            if validator is not None:
                try:
                    validator()
                except (OSError, ValueError, KeyError):
                    valid = False
            if valid:
                print('Resume:', stage, item, flush=True)
                return
        log = self.root/'logs'/f'{item}-{stage}-{time.time_ns()}.log'
        attempt = self.ledger.start(item, stage, fp, log, command, gpu)
        print('Start:', stage, item, 'device:', gpu, flush=True)
        try:
            with log.open('w') as stream:
                subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)
            if validator is not None:
                validator()
            outputs = expected() if callable(expected) else expected
            if not outputs or any(not Path(p).is_file() for p in outputs):
                raise RuntimeError('Stage exited without expected outputs')
            self.ledger.finish(attempt, outputs)
            print('Success:', stage, item, flush=True)
        except Exception as error:
            self.ledger.finish(attempt, error=str(error))
            raise RuntimeError(f'{stage} failed for {item}; log: {log}') from error

    def prepare(self):
        import re
        from euroflash.provenance import reconcile
        reconcile(self.args.input, self.ledger)
        files = sorted(self.args.input.rglob('*.fits'))
        if not files:
            raise ValueError('No input PSRFITS files found')
        grouped = {}
        ids = set()
        jobs = []
        for raw in files:
            match = re.search(r'(L\d+)_SAP(\d+)_(?:BEAM|B)(\d+)', raw.name)
            if not match:
                raise ValueError(f'Unknown PSRFITS filename: {raw}')
            obs, sap, beam = match[1], int(match[2]), int(match[3])
            item = f'{obs}_SAP{sap:03d}_B{beam:03d}'
            if item in ids:
                raise ValueError(f'Multiple PSRFITS parts/replicas for {item}; resolve before processing')
            ids.add(item)
            directory = self.root/'data'/obs/f'SAP{sap:03d}'/f'B{beam:03d}'
            directory.mkdir(parents=True, exist_ok=True)
            output = directory/f'downsampled_{obs}_SAP{sap:03d}_BEAM{beam:03d}_32bit.fil'
            command = self.command('preproc/downsample_psrfits2fil_32bit.py', ['-o', output, raw])
            # Conversion is keyed on the one file it reads, by name and size,
            # so a re-extracted or truncated archive member is converted again.
            conversion_fp = fingerprint(self.settings, self.image,
                {'stage': 'downsample', 'fscrunch': 4, 'tscrunch': 16,
                 'source': raw.name, 'source_bytes': raw.stat().st_size},
                [REPO/'preproc/downsample_psrfits2fil_32bit.py', REPO/'lotaas_reprocessing/filterbank.py', REPO/'lotaas_reprocessing/sigproc.py'])
            jobs.append((item, 'downsample', command, [output], None, conversion_fp))
            grouped.setdefault(directory.parent, []).append(output)
        with futures.ThreadPoolExecutor(max_workers=self.args.preprocess_workers) as pool:
            for result in pool.map(lambda job: self.step(*job), jobs):
                pass
        if self.args.convert_only:
            return []
        for sap, paths in grouped.items():
            arguments = [sap] + (['--allow-partial'] if self.args.pilot else [])
            self.step(sap.parent.name+'_'+sap.name, 'flatfield',
                      self.command('preproc/flatfield_fil.py', arguments),
                      [p.with_name(p.stem+'_ff.fil') for p in paths])
        return [p.with_name(p.stem+'_ff.fil') for paths in grouped.values() for p in paths]

    def analyze(self, item, output):
        """Separate checkpoints: a periodicity retry does not rerun FETCH."""
        from lotaas_reprocessing.trials import product_outputs
        errors = []
        try:
            self.step(item, 'single_pulse',
                      self.command('pipeline/pipeline_cpu.py', [output, '--search', 'single-pulse']),
                      lambda: product_outputs(output, 'single_pulse_summary.json'))
        except Exception as error:
            errors.append(str(error))
        metadata = json.loads((output/'metadata.json').read_text())
        enabled = metadata.get('periodicity_enabled', False)
        if enabled:
            try:
                self.step(item, 'periodicity',
                          self.command('pipeline/pipeline_cpu.py', [output, '--search', 'periodicity']),
                          lambda: product_outputs(output, 'periodicity_summary.json'))
            except Exception as error:
                errors.append(str(error))
        if errors:
            # Keep an aggregate failure for coverage and the existing reclaim policy.
            attempt = self.ledger.start(item, 'classify', self.fp, self.root/'logs'/f'{item}-searches.log',
                                        ['single_pulse', 'periodicity'])
            self.ledger.finish(attempt, error='\n'.join(errors))
            raise RuntimeError('\n'.join(errors))
        def final_outputs():
            files = [output/'metadata.yaml', output/'metadata.json'] + product_outputs(output, 'single_pulse_summary.json')
            if enabled:
                files += product_outputs(output, 'periodicity_summary.json')
            return files
        self.step(item, 'classify',
                  self.command('pipeline/pipeline_cpu.py', [output, '--search', 'finalize']), final_outputs)

    def process(self, files):
        tasks = queue.Queue()
        for path in files:
            tasks.put(path)
        devices = self.args.gpus.split(',') if self.args.backend == 'gpu' else [None]
        if len(set(devices)) != len(devices):
            raise ValueError('GPU devices must be unique')
        inflight = threading.BoundedSemaphore(self.args.cpu_workers * 2 + len(devices))
        cpu_futures = []
        errors = []
        with futures.ThreadPoolExecutor(max_workers=self.args.cpu_workers) as cpu_pool:
            def worker(gpu):
                while True:
                    try:
                        path = tasks.get_nowait()
                    except queue.Empty:
                        return
                    item = path.stem
                    output = self.root/'processed'/item/self.fp[:16]
                    output.mkdir(parents=True, exist_ok=True)
                    if self.ledger.completed(item, 'classify', self.fp):
                        print('Already analyzed:', item, flush=True)
                        continue
                    args = [path, output, '--settings', self.settings, '--backend', self.args.backend]
                    if self.args.pilot:
                        args += ['--pilot']
                    if self.args.max_samples:
                        args += ['--max-samples', self.args.max_samples]
                    inflight.acquire()
                    try:
                        from lotaas_reprocessing.trials import validate_products
                        self.step(item, 'dedisperse', self.command('pipeline/pipeline_gpu.py', args, gpu),
                                  lambda output=output: [output/'metadata.yaml', output/'metadata.json', output/'trial_manifest.json']
                                  + list((output/'DM_trials').glob('*.dat'))
                                  + list((output/'Periodic_DM_trials').glob('*.dat')), gpu,
                                  validator=lambda output=output: validate_products(output))
                        future = cpu_pool.submit(self.analyze, item, output)
                        future.add_done_callback(lambda _: inflight.release())
                        cpu_futures.append(future)
                    except Exception as e:
                        inflight.release()
                        errors.append(str(e))
            with futures.ThreadPoolExecutor(max_workers=len(devices)) as gpu_pool:
                list(gpu_pool.map(worker, devices))
            for future in cpu_futures:
                try:
                    future.result()
                except Exception as e:
                    errors.append(str(e))
        if errors:
            raise RuntimeError('\n'.join(errors))

    def run(self):
        # Prevent two orchestrators from racing in the same output directory.
        with (self.root/'.run.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            inputs = list(self.args.input.rglob('*_ff.fil' if self.args.prepared else '*.fits'))
            if not inputs:
                raise ValueError('No matching inputs found')
            if not (self.args.prepare_only or self.args.convert_only) and self.args.backend == 'gpu':
                subprocess.run(self.command('euroflash/preflight.py', [], self.args.gpus), check=True, cwd=REPO)
            self.fp = fingerprint(self.settings, self.image,
                                  {'pilot': self.args.pilot, 'max_samples': self.args.max_samples,
                                   'backend': self.args.backend, 'prepared': self.args.prepared})
            run_metadata = {'fingerprint': self.fp, 'settings': str(self.settings),
                'image': str(self.image), 'pilot': self.args.pilot, 'max_samples': self.args.max_samples,
                'backend': self.args.backend, 'input_files': [str(p) for p in inputs]}
            (self.root/'run.json').write_text(json.dumps(run_metadata, indent=2))
            self.ledger.register_run(self.fp, run_metadata)
            prepared = inputs if self.args.prepared else self.prepare()
            if not (self.args.prepare_only or self.args.convert_only):
                self.process(prepared)
            (self.root/'summary.json').write_text(json.dumps(self.ledger.summary(), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='Directory containing extracted PSRFITS')
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--ledger', type=Path, required=True)
    p.add_argument('--image', type=Path, default=REPO/'containers/euroflash-runtime.sif')
    p.add_argument('--settings', type=Path, default=REPO/'settings.yaml')
    p.add_argument('--prepared', action='store_true', help='Input contains filterbanks already flatfielded together on the head node')
    p.add_argument('--backend', choices=['cpu','gpu'], default='gpu')
    p.add_argument('--gpus', default='0', help='Comma-separated visible GPU indices/UUIDs; one worker each')
    p.add_argument('--cpu-workers', type=int, default=4)
    p.add_argument('--preprocess-workers', type=int, default=4)
    p.add_argument('--prepare-only', action='store_true', help='Convert and flatfield, without running the search')
    p.add_argument('--convert-only', action='store_true', help='Convert downloaded beams; defer flatfielding until the full SAP is present')
    p.add_argument('--pilot', action='store_true', help='Permit an incomplete central-beam set')
    p.add_argument('--max-samples', type=int, help='Pilot only: process a prefix')
    a = p.parse_args()
    if a.max_samples and not a.pilot:
        p.error('--max-samples requires --pilot')
    if a.cpu_workers < 1 or a.preprocess_workers < 1:
        p.error('--cpu-workers must be positive')
    a.input = a.input.resolve(); a.ledger = a.ledger.resolve()
    Runner(a).run()


if __name__ == '__main__':
    main()
