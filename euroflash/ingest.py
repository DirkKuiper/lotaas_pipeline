"""Ingest on CPU nodes: an LTA archive is downloaded, unpacked and converted on a compute node, and only its
32-bit filterbank reaches /shared.

On the head an archive wrote about 13 GB per beam into the ZFS pool that holds everyone's /home and /shared
(tar 6.0 GB, extracted FITS 6.0 GB, filterbank 1.2 GB) and read it back twice: the campaign's downloads were most
of the head's 2.9 GB/s of writes, and Cees Bassa asked for the conversion to leave the head (22 October 2026).
Here the archive is streamed from SURF straight through tar into the node's RAM-backed /tmp, its SHA256 taken on
the way, the FITS converted there, and the filterbank copied to its place under /shared and renamed into it.
The download worker on the head only waits on ssh, so more of them cost the head nothing: on the head each one
also extracted and converted, and only 4-7 of 10 were transferring at any moment (23 October 2026).

    head:  Ingest(...).run(node, spec) -> result   (Campaign.process_remote)
    node:  python -m euroflash.ingest node < spec.json   (in the runtime image, from a frozen copy of the code)
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time

CHUNK = 8 * 1024 * 1024
# A scratch directory left by a job that died (node reboot, killed driver) is removed by the next job after this.
STALE_SCRATCH_HOURS = 3
# One archive at 2-3.5 MB/s (25-26 September) took 45 min; past this the job is a failure of its file.
JOB_TIMEOUT = 2 * 3600
# A node whose ssh failed takes no file for this long; the head converts locally while none is left.
NODE_BACKOFF = 600
RESULT_PREFIX = 'INGEST-RESULT '
# Jobs at once on a node's ingest master, ingest and flatfield together: sshd allows 10 sessions a connection.
MAX_JOBS_PER_NODE = 8


class NodeUnavailable(RuntimeError):
    """The node, not the file: ssh could not run the job there."""


# ------------------------------------------------------------------ node side

class HashingReader:
    """A file-like view of the HTTP body that hashes and counts what tar reads."""

    def __init__(self, stream):
        self.stream, self.digest, self.count = stream, hashlib.sha256(), 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        self.count += len(data)
        return data

    def drain(self):
        while self.read(CHUNK):
            pass


def safe_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Unsafe archive member path: {name}')
    return path


def stream_extract(response, length, scratch):
    """Unpack the PSRFITS members of a tar body into scratch as it arrives; returns (fits, sha256, bytes)."""
    reader = HashingReader(response)
    fits = []
    with tarfile.open(fileobj=reader, mode='r|*') as archive:
        for member in archive:
            if not (member.isfile() and member.name.lower().endswith('.fits')):
                continue
            target = Path(scratch)/safe_member(member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(scratch).free < member.size + 2**30:
                raise OSError(f'Insufficient scratch space for {member.name} on {socket.gethostname()}')
            source = archive.extractfile(member)
            with target.open('wb') as output:
                shutil.copyfileobj(source, output, CHUNK)
            if target.stat().st_size != member.size:
                raise IOError(f'Truncated archive member {member.name}')
            fits.append(target)
    reader.drain()           # tar's end-of-archive padding: the receipt's digest covers the whole body
    if reader.count != length:
        raise IOError(f'Incomplete HTTP body: {reader.count} of {length} bytes')
    if not fits:
        raise ValueError('No PSRFITS members in archive')
    return fits, reader.digest.hexdigest(), reader.count


def clean_stale(scratch_root, hours=STALE_SCRATCH_HOURS):
    cutoff = time.time() - hours * 3600
    for path in Path(scratch_root).iterdir():
        try:
            if path.is_dir() and path.stat().st_mtime < cutoff:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


def convert(raw, output, scratch):
    """The repository's conversion script on one FITS, into scratch, then copied beside output and renamed in."""
    from euroflash.run import CONVERSION_SCRIPT, REPO
    local = Path(scratch)/'out'/output.name
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(REPO/CONVERSION_SCRIPT), '-o', str(local), str(raw)], check=True,
                   stdout=sys.stderr, stderr=sys.stderr)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f'{output.name}.partial-{socket.gethostname()}-{os.getpid()}')
    try:
        shutil.copyfile(local, partial)
        os.replace(partial, output)
    finally:
        partial.unlink(missing_ok=True)
    local.unlink()
    return output


def node_job(spec):
    """Download, unpack and convert one archive (spec from the head); returns the result for the head."""
    from euroflash.beams import excluded
    from euroflash.download import open_with_any
    from euroflash.run import conversion_target
    scratch_root = Path(spec['scratch'])
    scratch_root.mkdir(parents=True, exist_ok=True)
    clean_stale(scratch_root)
    scratch = scratch_root/f"{spec['item']}-{os.getpid()}"
    scratch.mkdir()
    try:
        started = time.monotonic()
        with open_with_any(spec['url'], spec['tokens'], {}) as response:
            length = int(response.headers['Content-Length'])
            if length > spec['max_bytes']:
                raise ValueError(f"Archive size {length} exceeds configured cap {spec['max_bytes']}")
            if shutil.disk_usage(scratch).free < length + 2**30:
                raise OSError(f'Insufficient scratch space on {socket.gethostname()}')
            fits, sha256, count = stream_extract(response, length, scratch)
        receipt = {'url': spec['url'], 'bytes': count, 'transferred_bytes': count, 'sha256': sha256,
                   'seconds': time.monotonic() - started}
        members = [{'path': str(p), 'name': p.name, 'bytes': p.stat().st_size} for p in fits]
        if any(excluded(p, spec['exclude_beams']) for p in fits):
            return {'ok': True, 'receipt': receipt, 'fits': members, 'excluded': True, 'outputs': []}
        started, outputs = time.monotonic(), []
        for raw in fits:
            _, _, output = conversion_target(raw.name, spec['data_root'])
            outputs.append(str(convert(raw, output, scratch)))
            raw.unlink()
        return {'ok': True, 'receipt': receipt, 'fits': members, 'excluded': False, 'outputs': outputs,
                'convert_seconds': time.monotonic() - started, 'host': socket.gethostname()}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def node_main():
    spec = json.loads(sys.stdin.read())
    try:
        result = node_job(spec)
    except PermissionError as error:
        result = {'ok': False, 'kind': 'permission', 'error': str(error)}
    except Exception as error:
        result = {'ok': False, 'kind': type(error).__name__, 'error': str(error)}
    print(RESULT_PREFIX + json.dumps(result), flush=True)
    return 0 if result['ok'] else 1


# ------------------------------------------------------------------ head side

class Ingest:
    """The ingest nodes, the frozen code they run, and one job at a time on whichever is least busy."""

    def __init__(self, nodes, repo, settings, image, control_dir=None, remote_root=None,
                 scratch='/tmp/lotaas-ingest', shared=('/shared/results',), log_dir=None, timeout=JOB_TIMEOUT):
        import getpass
        self.nodes = list(nodes)
        # Resolved, as euroflash.cluster does: the checkout's image is a symlink, and a link is what tar would send.
        self.repo, self.settings, self.image = Path(repo), Path(settings).resolve(), Path(image).resolve()
        self.control_dir = Path(control_dir) if control_dir else None
        self.remote_root = remote_root or f'/home/{getpass.getuser()}/lotaas-ingest'
        self.scratch, self.shared, self.timeout = scratch, tuple(shared), timeout
        self.log_dir = Path(log_dir) if log_dir else None
        self.lock = threading.Lock()
        self.max_jobs = MAX_JOBS_PER_NODE
        self.busy = {n: 0 for n in self.nodes}
        self.down_until = {n: 0. for n in self.nodes}
        self.ready = {}
        self.node_locks = {n: threading.Lock() for n in self.nodes}
        self.frozen = None

    def home(self, key):
        """The node a SAP's archives and flatfield go to when it can take them, the same in every driver: its
        flatfield then reads the beams from that node's own cache rather than over NFS."""
        import zlib
        return self.nodes[zlib.crc32(key.encode()) % len(self.nodes)]

    def acquire(self, prefer=None):
        """prefer when it is up and has room, else the least busy node that is, now counted busy; None when none
        is (up to MAX_JOBS_PER_NODE each)."""
        with self.lock:
            now = time.time()
            usable = [n for n in self.nodes if self.down_until[n] <= now and self.busy[n] < self.max_jobs]
            if not usable:
                return None
            node = prefer if prefer in usable else min(usable, key=lambda n: (self.busy[n], self.nodes.index(n)))
            self.busy[node] += 1
            return node

    def release(self, node):
        with self.lock:
            self.busy[node] -= 1

    def back_off(self, node, seconds=NODE_BACKOFF):
        with self.lock:
            self.down_until[node] = time.time() + seconds
            self.ready.pop(node, None)

    def code(self):
        """The code, settings and image as the driver found them first (cluster.freeze_source), and their digest:
        the ingest nodes run what this driver started with, so a change reaches them at its restart."""
        with self.lock:
            if self.frozen is None:
                from euroflash.cluster import freeze_source
                files = freeze_source(self.repo, self.settings, self.image)
                digest = hashlib.sha256()
                for frozen, name in sorted(files, key=lambda f: f[1]):
                    digest.update(name.encode())
                    digest.update(frozen.data if frozen.data is not None else
                                  f'{frozen.stat.st_size}-{frozen.stat.st_mtime_ns}'.encode())
                self.frozen = (files, digest.hexdigest()[:16])
            return self.frozen

    def ssh_args(self, node):
        """ssh through the node's own ingest master (ops/open-masters.sh opens ingest-<node>): the search tier's
        master carries at most sshd's 10 sessions, and a node that ran out of them failed the tier's runs (11
        October 2026). Without that master ssh tries the keys, fails, and the head ingests instead."""
        args = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15']
        if self.control_dir:
            args += ['-S', str(self.control_dir/('ingest-' + node.removeprefix('efc-').replace('-', '')))]
        return args + [node]

    def ssh(self, node, command, **kwargs):
        return subprocess.run(self.ssh_args(node) + [command], **kwargs)

    def upload(self, node, files, destination):
        """The frozen files into destination on the node, as one tar stream over the ingest master."""
        self.ssh(node, shlex.join(['mkdir', '-p', destination]), check=True, capture_output=True)
        process = subprocess.Popen(self.ssh_args(node) + [shlex.join(['tar', 'xf', '-', '-C', destination])],
                                   stdin=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=process.stdin, mode='w|') as archive:
                for frozen, name in files:
                    frozen.add_to(archive, name)
            process.stdin.close()
            if process.wait():
                raise RuntimeError(f'Transfer failed to {node}')
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()

    def prepare(self, node):
        """The frozen code on the node (<remote_root>/<digest>), sent once; older copies removed after a day."""
        files, digest = self.code()
        destination = f'{self.remote_root}/{digest}'
        with self.node_locks[node]:
            if self.ready.get(node) == destination:
                return destination
            probe = self.ssh(node, shlex.join(['test', '-f', destination + '/.complete']) + ' && '
                             + shlex.join(['test', '-f', destination + '/containers/runtime.sif']), capture_output=True)
            if probe.returncode == 255:
                raise NodeUnavailable(f'{node}: ssh failed: {probe.stderr.decode(errors="replace").strip()}')
            if probe.returncode != 0:
                try:
                    self.upload(node, files, destination)
                except (RuntimeError, subprocess.CalledProcessError, OSError) as error:
                    raise NodeUnavailable(f'{node}: code transfer failed: {error}') from error
                self.ssh(node, shlex.join(['touch', destination + '/.complete']), check=True, capture_output=True)
                self.ssh(node, f'find {shlex.quote(self.remote_root)} -mindepth 1 -maxdepth 1 -type d '
                               f'! -name {shlex.quote(digest)} -mmin +1440 -exec rm -rf {{}} +', capture_output=True)
            self.ready[node] = destination
            return destination

    def container(self, destination, *program):
        command = ['apptainer', 'exec', '--cleanenv']
        for path in (self.scratch, *self.shared):
            command += ['--bind', path]
        command += ['--env', 'PYTHONPATH=' + destination, destination + '/containers/runtime.sif', 'python', *program]
        return (f'mkdir -p {shlex.quote(self.scratch)} && cd {shlex.quote(destination)} && '
                + shlex.join([str(p) for p in command]))

    def command(self, destination):
        return self.container(destination, '-m', 'euroflash.ingest', 'node')

    def remote_command(self, node, script, arguments):
        """The command running one of the repository's scripts on the node, in its image, from the frozen code."""
        return self.ssh_args(node) + [self.container(self.prepare(node), script, *arguments)]

    @staticmethod
    def unreachable(error):
        """Whether a failed remote step (euroflash.run.Runner.step) failed for ssh or the container, not the work."""
        cause = error if isinstance(error, (NodeUnavailable, subprocess.CalledProcessError)) else error.__cause__
        return isinstance(cause, NodeUnavailable) or (isinstance(cause, subprocess.CalledProcessError)
                                                     and cause.returncode == 255)

    def run(self, node, spec, item):
        """One archive on one node. Raises NodeUnavailable (the node), PermissionError (the macaroons) or
        RuntimeError (the file); the job's log is kept under log_dir when it failed."""
        destination = self.prepare(node)
        spec = dict(spec, scratch=self.scratch)
        try:
            process = self.ssh(node, self.command(destination), input=json.dumps(spec).encode(),
                               capture_output=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as error:
            self.keep_log(item, node, error.stdout, error.stderr)
            raise RuntimeError(f'ingest on {node} timed out after {self.timeout} s') from None
        out = process.stdout.decode(errors='replace')
        err = process.stderr.decode(errors='replace')
        line = next((l for l in reversed(out.splitlines()) if l.startswith(RESULT_PREFIX)), None)
        if line is None:
            self.keep_log(item, node, out, err)
            if process.returncode == 255 or 'session request failed' in err or 'Connection' in err:
                raise NodeUnavailable(f'{node}: ssh or the container setup exited {process.returncode}: '
                                      f'{err.strip()[-300:]}')
            raise RuntimeError(f'ingest on {node} exited {process.returncode} without a result: {err.strip()[-500:]}')
        result = json.loads(line[len(RESULT_PREFIX):])
        if not result['ok']:
            self.keep_log(item, node, out, err)
            if result.get('kind') == 'permission':
                raise PermissionError(result['error'])
            raise RuntimeError(f"ingest on {node}: {result['error']}")
        return dict(result, node=node)

    def keep_log(self, item, node, out, err):
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        decode = (lambda b: b.decode(errors='replace') if isinstance(b, bytes) else (b or ''))
        (self.log_dir/f'{item}.log').write_text(f'node {node}\n--- stdout\n{decode(out)}\n--- stderr\n{decode(err)}\n')


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['node']:
        return node_main()
    print(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(main())
