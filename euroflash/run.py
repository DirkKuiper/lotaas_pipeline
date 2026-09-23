"""Run LOTAAS without Slurm, with one search worker per selected GPU."""
import argparse
import concurrent.futures as futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
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

# Stage wall-clock limits in seconds. A stage past its limit is killed and the
# beam fails, so one pathological beam cannot hold a batch for hours: two
# single-pulse stages (FETCH over 600 s-wide clusters) held one for five.
TIMEOUTS = {'dedisperse': 1800, 'single_pulse': 3600, 'sp_classify': 2700, 'periodicity': 3600,
            'periodicity_fold': 1800, 'classify': 900}

# Which stages each mode runs. 'gpu' stops after the single-pulse search and
# leaves clusters and periodic trials for a CPU node; 'cpu' takes them from
# there. 'all' is both on one node.
MODES = ('all', 'gpu', 'cpu')


def atomic_json(path, value):
    """Write JSON atomically. The runner uses the node's own Python, without numpy."""
    path = Path(path)
    partial = path.with_name(path.name + '.partial')
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(partial, path)


class StageTimeout(RuntimeError):
    """A stage ran past its limit and was killed."""


def parse_timeouts(values):
    timeouts = dict(TIMEOUTS)
    for value in values or ():
        stage, _, seconds = value.partition('=')
        if stage not in TIMEOUTS or not seconds:
            raise ValueError(f'--stage-timeout needs STAGE=SECONDS with STAGE among {", ".join(TIMEOUTS)}')
        timeouts[stage] = float(seconds)
    return timeouts


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
        from euroflash.beams import INCOHERENT_BEAMS
        self.excluded_beams = tuple(getattr(args, 'exclude_beams', None) or INCOHERENT_BEAMS)
        self.fp = ''
        self.mode = getattr(args, 'stages', None) or 'all'
        self.timeouts = getattr(args, 'timeouts', None) or dict(TIMEOUTS)
        # Stage process groups, so a runner that is stopped takes its children with it.
        self.children = set()
        self.children_lock = threading.Lock()

    def command(self, script, arguments, gpu=None):
        paths = {REPO, self.root, self.args.input.resolve(), self.args.ledger.resolve().parent, self.settings.parent}
        command = ['apptainer', 'exec', '--cleanenv']
        if gpu is not None:
            command += ['--nv', '--env', 'CUDA_VISIBLE_DEVICES='+gpu]
        for path in sorted(paths):
            command += ['--bind', f'{path}:{path}']
        threads = getattr(self.args, 'threads_per_stage', 0)
        if threads and gpu is None:
            for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                         'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
                command += ['--env', f'{name}={threads}']
        command += ['--env', 'PYTHONPATH='+str(REPO), '--env', 'LOTAAS_DB_PATH='+str(self.args.ledger.resolve()),
                    '--env', 'LOTAAS_RUN_FINGERPRINT='+self.fp,
                    str(self.image), 'python', str(REPO/script)]
        return command + [str(a) for a in arguments]

    def run_stage(self, command, log, timeout):
        """Run one stage in its own process group; kill the group if it overruns."""
        with log.open('w') as stream:
            process = subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            with self.children_lock:
                self.children.add(process.pid)
            try:
                try:
                    code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    stop_group(process)
                    raise StageTimeout(f'timed out after {timeout:.0f} s') from None
            finally:
                with self.children_lock:
                    self.children.discard(process.pid)
        if code:
            raise subprocess.CalledProcessError(code, command)

    def stop_children(self, *_):
        with self.children_lock:
            pids = list(self.children)
        for pid in pids:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        raise SystemExit(143)

    def step(self, item, stage, command, expected, gpu=None, fingerprint_override=None, validator=None,
             timeout=None):
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
            self.run_stage(command, log, timeout if timeout is not None else self.timeouts.get(stage))
            if validator is not None:
                validator()
            outputs = expected() if callable(expected) else expected
            if not outputs or any(not Path(p).is_file() for p in outputs):
                raise RuntimeError('Stage exited without expected outputs')
            self.ledger.finish(attempt, outputs)
            print('Success:', stage, item, flush=True)
        except Exception as error:
            self.ledger.finish(attempt, error=str(error))
            reason = f' ({error})' if isinstance(error, StageTimeout) else ''
            raise RuntimeError(f'{stage} failed for {item}{reason}; log: {log}') from error

    def conversion_job(self, raw):
        """The ledger step converting one PSRFITS into its SAP directory."""
        import re
        match = re.search(r'(L\d+)_SAP(\d+)_(?:BEAM|B)(\d+)', raw.name)
        if not match:
            raise ValueError(f'Unknown PSRFITS filename: {raw}')
        obs, sap, beam = match[1], int(match[2]), int(match[3])
        item = f'{obs}_SAP{sap:03d}_B{beam:03d}'
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
        return item, directory.parent, output, (item, 'downsample', command, [output], None, conversion_fp)

    def convert(self, raw, delete_raw=False):
        """Convert one beam; with delete_raw, drop its FITS and tar on success only."""
        from euroflash.rawdata import delete_converted
        _, sap, output, job = self.conversion_job(raw)
        self.step(*job)
        if delete_raw:
            delete_converted(raw)
        return sap, output

    def flatfield(self, sap):
        """Flatfield every converted beam present in one SAP directory."""
        paths = sorted(sap.glob('B*/*_32bit.fil'))
        if not paths:
            raise ValueError(f'No converted beams in {sap}')
        arguments = [sap] + (['--allow-partial'] if self.args.pilot else [])
        self.step(sap.parent.name+'_'+sap.name, 'flatfield',
                  self.command('preproc/flatfield_fil.py', arguments),
                  [p.with_name(p.stem+'_ff.fil') for p in paths])
        return [p.with_name(p.stem+'_ff.fil') for p in paths]

    def prepare(self):
        from euroflash.provenance import reconcile
        reconcile(self.args.input, self.ledger)
        from euroflash.beams import excluded
        files = sorted(self.args.input.rglob('*.fits'))
        if not files:
            raise ValueError('No input PSRFITS files found')
        skipped = [raw for raw in files if excluded(raw, self.excluded_beams)]
        for raw in skipped:
            print('Excluded (incoherent beam):', raw, flush=True)
        files = [raw for raw in files if raw not in skipped]
        if not files:
            raise ValueError('Only excluded beams among the input PSRFITS files')
        ids = set()
        for raw in files:
            item = self.conversion_job(raw)[0]
            if item in ids:
                raise ValueError(f'Multiple PSRFITS parts/replicas for {item}; resolve before processing')
            ids.add(item)
        delete_raw = getattr(self.args, 'delete_raw', False)
        with futures.ThreadPoolExecutor(max_workers=self.args.preprocess_workers) as pool:
            saps = set(sap for sap, _ in pool.map(lambda raw: self.convert(raw, delete_raw), files))
        if self.args.convert_only:
            return []
        # Beams converted by earlier --convert-only runs have no FITS left when
        # raw data is deleted, so the SAP directory, not this batch, is the set.
        return [path for sap in sorted(saps) for path in self.flatfield(sap)]

    def cpu_stage(self, item, output, stage, search, summary):
        from lotaas_reprocessing.trials import product_outputs
        self.step(item, stage, self.command('pipeline/pipeline_cpu.py', [output, '--search', search]),
                  lambda: product_outputs(output, summary))

    @staticmethod
    def periodic(output):
        return json.loads((output/'metadata.json').read_text()).get('periodicity_enabled', False)

    def search(self, item, output):
        """First pass over one beam: the stages that need no other beam. Returns errors.

        The single-pulse branch and the periodic search are attempted even if
        the other fails. Folding waits for the cross-beam veto (fold_and_finish).
        """
        errors = []
        try:
            if self.mode in ('all', 'gpu'):
                self.cpu_stage(item, output, 'single_pulse', 'single-pulse', 'single_pulse_summary.json')
            if self.mode in ('all', 'cpu'):
                self.cpu_stage(item, output, 'sp_classify', 'sp-classify', 'sp_classify_summary.json')
        except Exception as error:
            errors.append(str(error))
        if self.mode in ('all', 'cpu') and self.periodic(output):
            try:
                self.cpu_stage(item, output, 'periodicity', 'periodicity', 'periodicity_search_summary.json')
            except Exception as error:
                errors.append(str(error))
        if self.mode == 'gpu':
            if errors:
                self.fail(item, errors)
            # Dedispersion was validated before search() was called. A failed
            # single-pulse branch must not suppress the independent periodic
            # search. Its missing completion manifest keeps the beam failed.
            # Periodic hard links survive removal of the transient trial tree.
            import shutil
            shutil.rmtree(output/'DM_trials', ignore_errors=True)
            self.mark_ready(item, output)
        return errors

    def handoff(self):
        path = self.root/'handoff'
        path.mkdir(exist_ok=True)
        return path

    def mark_ready(self, item, output):
        """Tell the head this beam can go to a CPU node (euroflash.cluster relays it)."""
        metadata = json.loads((output/'metadata.json').read_text())
        atomic_json(self.handoff()/f'{item}.ready', {
            'item': item, 'fingerprint': self.fp, 'output': str(output.relative_to(self.root)),
            'input': metadata['filename']})

    def fail(self, item, errors):
        # An aggregate failure, for coverage and the reclaim policy.
        attempt = self.ledger.start(item, 'classify', self.fp, self.root/'logs'/f'{item}-searches.log',
                                    ['single_pulse', 'sp_classify', 'periodicity', 'periodicity_fold'])
        self.ledger.finish(attempt, error='\n'.join(errors))

    def veto(self, outputs):
        """Compare the sifted peaks of every searched beam in this batch before any is folded."""
        from lotaas_reprocessing.periodicity_veto import apply
        searched = []
        for item, output in outputs:
            if not self.periodic(output) or not (output/'periodicity_search_summary.json').is_file():
                continue
            searched.append((item, output))
        if not searched:
            return {}
        config = json.loads((searched[0][1]/'metadata.json').read_text()).get('periodicity') or {}
        if not config.get('multibeam_veto', False):
            return {}
        pending = [output for item, output in searched if not self.ledger.completed(item, 'periodicity_fold', self.fp)]
        written = apply([output for _, output in searched], config.get('veto_bins', 1.1),
                        config.get('veto_beams', 4), write=pending)
        print(f'Cross-beam veto over {len(searched)} beams: {sum(written.values())} sifted peaks vetoed',
              flush=True)
        return written

    def fold_and_finish(self, item, output, errors):
        """Second pass: fold what the veto left, then check every product and remove the trials."""
        from lotaas_reprocessing.trials import product_outputs
        enabled = self.periodic(output)
        try:
            searched = enabled and bool(product_outputs(output, 'periodicity_search_summary.json'))
        except (OSError, ValueError):
            searched = False
        if searched:
            try:
                self.cpu_stage(item, output, 'periodicity_fold', 'periodicity-fold', 'periodicity_summary.json')
            except Exception as error:
                errors.append(str(error))
        if errors:
            self.fail(item, errors)
            raise RuntimeError('\n'.join(errors))
        def final_outputs():
            files = [output/'metadata.yaml', output/'metadata.json'] + product_outputs(output, 'single_pulse_summary.json')
            files += product_outputs(output, 'sp_classify_summary.json')
            if enabled:
                files += product_outputs(output, 'periodicity_summary.json')
            return files
        self.step(item, 'classify',
                  self.command('pipeline/pipeline_cpu.py', [output, '--search', 'finalize']), final_outputs)

    def finish_batch(self, searched, cpu_pool):
        """After every first pass: the veto, then folds and finalisation in parallel."""
        errors = []
        if self.mode == 'gpu':
            return [e for _, _, found in searched for e in found]
        self.veto([(item, output) for item, output, _ in searched])
        futures_ = [cpu_pool.submit(self.fold_and_finish, item, output, list(found))
                    for item, output, found in searched]
        for future in futures_:
            try:
                future.result()
            except Exception as error:
                errors.append(str(error))
        return errors

    def process(self, files):
        from euroflash.beams import excluded
        tasks = queue.Queue()
        for path in files:
            if excluded(path, self.excluded_beams):
                print('Excluded (incoherent beam):', path.name, flush=True)
                continue
            tasks.put(path)
        devices = self.args.gpus.split(',') if self.args.backend == 'gpu' else [None]
        if len(set(devices)) != len(devices):
            raise ValueError('GPU devices must be unique')
        # One worker leaves the card idle ~70% of its slot while the same process
        # masks, detrends, plots and writes on the CPU. Three workers sharing a
        # GPU measured 26.8 s per beam against 47.4 s for one (~13 GB each).
        per_gpu = getattr(self.args, 'workers_per_gpu', 1) if self.args.backend == 'gpu' else 1
        workers = [device for device in devices for _ in range(per_gpu)]
        inflight = threading.BoundedSemaphore(self.args.cpu_workers * 2 + len(workers))
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
                        # The manifest lists every trial with its size and digest and
                        # validate_products checks them on success and on resume, so
                        # the ledger records the manifest, not ~4,400 trial paths:
                        # those were 2.4 MB per beam, 1.7 GB of the campaign ledger.
                        self.step(item, 'dedisperse', self.command('pipeline/pipeline_gpu.py', args, gpu),
                                  lambda output=output: [output/'metadata.yaml', output/'metadata.json', output/'trial_manifest.json'],
                                  gpu, validator=lambda output=output: validate_products(output))
                        future = cpu_pool.submit(lambda item=item, output=output: (item, output, self.search(item, output)))
                        future.add_done_callback(lambda _: inflight.release())
                        cpu_futures.append(future)
                    except Exception as e:
                        inflight.release()
                        errors.append(str(e))
            with futures.ThreadPoolExecutor(max_workers=len(workers)) as gpu_pool:
                list(gpu_pool.map(worker, workers))
            searched = []
            for future in cpu_futures:
                try:
                    searched.append(future.result())
                except Exception as e:
                    errors.append(str(e))
            errors += self.finish_batch(searched, cpu_pool)
        self.write_done(searched, errors)
        if errors:
            raise RuntimeError('\n'.join(errors))

    def write_done(self, searched, errors):
        """GPU mode: no more beams will be marked ready in this run."""
        if self.mode != 'gpu':
            return
        atomic_json(self.handoff()/'.done', {'fingerprint': self.fp,
                                             'failed': sorted(item for item, _, found in searched if found),
                                             'errors': len(errors)})

    def process_stream(self, poll=10.0):
        """CPU mode: search beams as the head relays them, then veto, fold and finish.

        The head writes <item>.ready once a beam's files are complete here and
        .done when no more will come. Each beam gets <item>.searched when its
        first pass is over; its periodic trials are pruned by then, which is
        what the head's relay backlog waits for.
        """
        handoff = self.handoff()
        errors, submitted, pending = [], set(), []
        with futures.ThreadPoolExecutor(max_workers=self.args.cpu_workers) as cpu_pool:
            def first_pass(item, output):
                found = self.search(item, output)
                # Without a cross-beam veto there is no scientific dependency
                # on the slowest beam. Finalise and release trials immediately.
                config = json.loads((output/'metadata.json').read_text()).get('periodicity') or {}
                if not config.get('multibeam_veto', False):
                    try:
                        self.fold_and_finish(item, output, list(found))
                    except Exception as error:
                        found.append(str(error))
                    atomic_json(handoff/f'{item}.searched', {'errors': found, 'finalized': True})
                    return item, output, found, True
                atomic_json(handoff/f'{item}.searched', {'errors': found, 'finalized': False})
                return item, output, found, False
            while True:
                done = (handoff/'.done').is_file()   # read before listing, so no beam is missed
                for marker in sorted(handoff.glob('*.ready')):
                    entry = json.loads(marker.read_text())
                    if entry['item'] in submitted:
                        continue
                    if entry['fingerprint'] != self.fp:
                        raise ValueError(f'{entry["item"]} was prepared as {entry["fingerprint"][:16]}, but this '
                                         f'node computes {self.fp[:16]}: the source, settings or image differ')
                    submitted.add(entry['item'])
                    if self.ledger.completed(entry['item'], 'classify', self.fp):
                        continue
                    pending.append(cpu_pool.submit(first_pass, entry['item'], self.root/entry['output']))
                if done:
                    break
                time.sleep(poll)
            searched = []
            for future in pending:
                try:
                    item, output, found, finalized = future.result()
                    if finalized:
                        errors.extend(found)
                    else:
                        searched.append((item, output, found))
                except Exception as e:
                    errors.append(str(e))
            errors += self.finish_batch(searched, cpu_pool)
        if errors:
            raise RuntimeError('\n'.join(errors))

    def run(self):
        # Prevent two orchestrators from racing in the same output directory.
        with (self.root/'.run.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            inputs = list(self.args.input.rglob('*_ff.fil' if self.args.prepared else '*.fits'))
            if not inputs and self.mode != 'cpu':   # a CPU node receives its beams as it runs
                raise ValueError('No matching inputs found')
            if (not (self.args.prepare_only or self.args.convert_only) and self.args.backend == 'gpu'
                    and self.mode != 'cpu'):
                subprocess.run(self.command('euroflash/preflight.py', [], self.args.gpus), check=True, cwd=REPO)
            self.fp = fingerprint(self.settings, self.image,
                                  {'pilot': self.args.pilot, 'max_samples': self.args.max_samples,
                                   'backend': self.args.backend, 'prepared': self.args.prepared})
            run_metadata = {'fingerprint': self.fp, 'settings': str(self.settings),
                'image': str(self.image), 'pilot': self.args.pilot, 'max_samples': self.args.max_samples,
                'backend': self.args.backend, 'input_files': [str(p) for p in inputs]}
            (self.root/'run.json').write_text(json.dumps(run_metadata, indent=2))
            self.ledger.register_run(self.fp, run_metadata)
            signal.signal(signal.SIGTERM, self.stop_children)
            signal.signal(signal.SIGHUP, self.stop_children)
            if self.mode == 'cpu':
                self.process_stream()
            else:
                prepared = inputs if self.args.prepared else self.prepare()
                if not (self.args.prepare_only or self.args.convert_only):
                    self.process(prepared)
            (self.root/'summary.json').write_text(json.dumps(self.ledger.summary(), indent=2))


def stop_group(process, grace=30):
    """SIGTERM a stage's process group, then SIGKILL what is left after `grace` seconds."""
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, None)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='Directory containing extracted PSRFITS')
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--ledger', type=Path, required=True)
    p.add_argument('--image', type=Path, default=REPO/'containers/euroflash-runtime.sif')
    p.add_argument('--settings', type=Path, default=REPO/'settings.yaml')
    p.add_argument('--prepared', action='store_true', help='Input contains filterbanks already flatfielded together on the head node')
    p.add_argument('--backend', choices=['cpu','gpu'], default='gpu')
    p.add_argument('--gpus', default='0', help='Comma-separated visible GPU indices/UUIDs')
    p.add_argument('--workers-per-gpu', type=int, default=1,
                   help='Dedispersion workers sharing each GPU; 3 measured 1.77x the throughput of 1')
    p.add_argument('--cpu-workers', type=int, default=4)
    p.add_argument('--preprocess-workers', type=int, default=4)
    p.add_argument('--prepare-only', action='store_true', help='Convert and flatfield, without running the search')
    p.add_argument('--convert-only', action='store_true', help='Convert downloaded beams; defer flatfielding until the full SAP is present')
    p.add_argument('--delete-raw', action='store_true',
                   help='Delete each PSRFITS and its archive tar once converted; receipts are kept')
    p.add_argument('--pilot', action='store_true', help='Permit an incomplete central-beam set')
    p.add_argument('--max-samples', type=int, help='Pilot only: process a prefix')
    p.add_argument('--exclude-beams', type=int, nargs='+', default=[12],
                   help='Beams never converted or searched (default: 12, the incoherent beam)')
    p.add_argument('--stages', choices=MODES, default='all',
                   help="'gpu': dedispersion and single-pulse search, leaving handoff.json; "
                        "'cpu': classification and periodicity of a handed-over batch; 'all': both")
    p.add_argument('--threads-per-stage', type=int, default=0,
                   help='Cap the thread pools of each CPU stage (0: leave the libraries to decide)')
    p.add_argument('--stage-timeout', action='append', metavar='STAGE=SECONDS',
                   help='Override a stage limit; defaults: ' + ', '.join(f'{k}={v:.0f}' for k, v in TIMEOUTS.items()))
    a = p.parse_args()
    try:
        a.timeouts = parse_timeouts(a.stage_timeout)
    except ValueError as error:
        p.error(str(error))
    if a.stages != 'all' and not a.prepared:
        p.error('--stages gpu|cpu applies to --prepared runs')
    if a.max_samples and not a.pilot:
        p.error('--max-samples requires --pilot')
    if a.cpu_workers < 1 or a.preprocess_workers < 1 or a.workers_per_gpu < 1:
        p.error('--cpu-workers, --preprocess-workers and --workers-per-gpu must be positive')
    a.input = a.input.resolve(); a.ledger = a.ledger.resolve()
    Runner(a).run()


if __name__ == '__main__':
    main()
