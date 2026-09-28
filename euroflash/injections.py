"""The injection lane: FRB-like bursts injected into one beam of every SAP the campaign searches, and searched as
production searches, so the survey's completeness is measured on its own data all along the campaign.

    python3 -m euroflash.injections tick          (cron, every few minutes; one tick at a time)
    python3 -m euroflash.injections status
    python3 -m euroflash.injections report [--source lta|spider] [--fingerprint PREFIX]

A tick:
  stage     one flatfielded beam of each SAP the campaign has prepared or is searching, drawn from the SAP's key,
            is hard-linked into staging/: the link keeps the data after the campaign deletes its own copy;
  make      a staged beam becomes a twin holding BURSTS bursts (lotaas_reprocessing.frb_injection, run in the
            production image on a CPU node, with the channel mask the search will use), and the link goes;
  dispatch  when BATCH twins of one source wait, or the oldest has waited MAX_WAIT, they are searched by
            euroflash.cluster from production's checkout (frozen when the run starts), with production's settings
            for that source and its image: one worker per GPU beside production, the CPU tier on its slot locks;
  collect   when a batch has ended, each burst is followed to its fate (lotaas_reprocessing.injection_fates, on a
            CPU node) into lane.sqlite with the run's code fingerprint, and the twins go.
Production is only read: its state database and configuration, its checkout, settings and image, and the beams
it prepared. Results, ledger and twins live under the lane's own root.
"""
import argparse
import hashlib
import json
import os
import random
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/shared/results/dkuiper/lotaas/injections')
CAMPAIGN_CONFIG = Path('~/.config/lotaas/campaign.json').expanduser()
BURSTS = 10
BATCH = 24
MAX_WAIT = 6 * 3600.0
MAKE_PARALLEL = 4
MAKE_TIMEOUT = 3600.0
MAKE_ATTEMPTS = 3
GPU_NODE = 'efc-gpu-00'
CPU_NODES = ('efc-cpu-03', 'efc-cpu-04', 'efc-cpu-05', 'efc-cpu-06')
EXCLUDE_BEAMS = (12,)
SOURCES = ('lta', 'spider')

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (sap TEXT PRIMARY KEY, source TEXT, beam INTEGER, origin TEXT, link TEXT,
    twin TEXT, truth TEXT, seed INTEGER, state TEXT, node TEXT, batch TEXT, staged REAL, made REAL, detail TEXT,
    attempts INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS batches (name TEXT PRIMARY KEY, source TEXT, node TEXT, checkout TEXT, fingerprint TEXT,
    settings TEXT, started REAL, finished REAL, state TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS bursts (twin TEXT, idx INTEGER, sap TEXT, batch TEXT, source TEXT, fingerprint TEXT,
    dm REAL, tau135 REAL, width REAL, snr_ideal REAL, spectrum TEXT, stage TEXT, search_snr REAL, own_snr REAL,
    dispersion_ratio REAL, fetch REAL, page REAL, queued INTEGER, record TEXT, PRIMARY KEY (twin, idx));
"""


def campaign_options(path=CAMPAIGN_CONFIG):
    """Production's dispatch options, from the supervisor's configuration of the campaign driver."""
    config = json.loads(Path(path).read_text())
    command = config['command']

    def value(flag, default=None, many=False):
        if flag not in command:
            return default
        i = command.index(flag) + 1
        if not many:
            return command[i]
        out = []
        while i < len(command) and not command[i].startswith('--'):
            out.append(command[i])
            i += 1
        return out
    return {'root': Path(config['root']), 'checkout': Path(config['checkout']), 'image': Path(value('--image')),
            'settings': {'lta': Path(value('--settings')), 'spider': Path(value('--spider-settings', value('--settings')))},
            'control_dir': value('--control-dir', str(Path('~/.ssh/control').expanduser())),
            'timeouts': [command[i + 1] for i, flag in enumerate(command) if flag == '--stage-timeout'],
            'cpu_tier_workers': value('--cpu-tier-workers', '12')}


def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


class Lane:
    def __init__(self, root=ROOT, options=None, now=time.time):
        self.root = Path(root)
        self.o = options or campaign_options()
        self.now = now
        for name in ('staging', 'twins', 'truth', 'batches', 'results', 'fates', 'logs', 'code'):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / 'lane.sqlite', timeout=60)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        if 'attempts' not in {r[1] for r in self.db.execute('PRAGMA table_info(samples)')}:
            self.db.execute('ALTER TABLE samples ADD COLUMN attempts INTEGER DEFAULT 0')

    def rows(self, sql, *args):
        return [dict(r) for r in self.db.execute(sql, args)]

    def set(self, table, key, **values):
        column = 'sap' if table == 'samples' else 'name'
        with self.db:
            self.db.execute(f"UPDATE {table} SET {','.join(f'{k}=?' for k in values)} WHERE {column}=?",
                            [*values.values(), key])

    # ------------------------------------------------------------------ code and image the jobs run
    def settings(self, snapshot, source):
        """Production's settings for a source, as the snapshot holds them on the shared disk (nodes lack /home)."""
        return snapshot / 'settings' / f'{source}.yaml'

    def code(self):
        """(snapshot of the production checkout's HEAD on the shared disk, its commit, the image copied beside it).

        The snapshot also holds production's settings for each source, as they were when it was taken."""
        commit = run(['git', '-C', str(self.o['checkout']), 'rev-parse', 'HEAD']).stdout.strip()
        snapshot = self.root / 'code' / commit[:12]
        if not snapshot.is_dir():
            partial = snapshot.with_name(snapshot.name + '.partial')
            shutil.rmtree(partial, ignore_errors=True)
            partial.mkdir()
            archive = subprocess.Popen(['git', '-C', str(self.o['checkout']), 'archive', commit, 'lotaas_reprocessing',
                                        'db', 'euroflash', 'pipeline'], stdout=subprocess.PIPE)
            subprocess.run(['tar', '-x', '-C', str(partial)], stdin=archive.stdout, check=True)
            archive.wait()
            (partial / 'settings').mkdir()
            for source, path in self.o['settings'].items():
                shutil.copy(path, partial / 'settings' / f'{source}.yaml')
            partial.rename(snapshot)
        stat = self.o['image'].stat()
        image = self.root / 'code' / f"runtime-{stat.st_size}-{int(stat.st_mtime)}.sif"
        if not image.is_file():
            shutil.copy(self.o['image'], image.with_name(image.name + '.partial'))
            image.with_name(image.name + '.partial').rename(image)
        return snapshot, commit, image

    def remote(self, node, command, log):
        """Start a container job detached on a CPU node; its exit status lands in `log`.exit."""
        control = Path(self.o['control_dir']) / f"lotaas-{node.removeprefix('efc-').replace('-', '')}"
        ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20']
        if control.exists():
            ssh += ['-S', str(control)]
        script = f"{shlex.join(command)} > {shlex.quote(str(log))} 2>&1; echo $? > {shlex.quote(str(log))}.exit"
        return run(ssh + [node, f"setsid nohup sh -c {shlex.quote(script)} > /dev/null 2>&1 < /dev/null &"],
                   stdin=subprocess.DEVNULL, timeout=60).returncode == 0

    def container(self, snapshot, image, *args):
        return ['apptainer', 'exec', '--cleanenv', '--env', 'OMP_NUM_THREADS=4', '--env', f'PYTHONPATH={snapshot}',
                '--bind', '/shared/results:/shared/results', str(image), 'python', '-m', *map(str, args)]

    # ------------------------------------------------------------------ stage
    def stage(self):
        """Hard-link one beam of each prepared or dispatched SAP not sampled yet."""
        state = sqlite3.connect(f"file:{self.o['root'] / 'campaign-state.sqlite'}?mode=ro", uri=True)
        state.row_factory = sqlite3.Row
        try:
            saps = [dict(r) for r in state.execute("SELECT key, source, sap_dir FROM saps WHERE state IN "
                                                   "('prepared', 'dispatched') AND sap_dir IS NOT NULL")]
        finally:
            state.close()
        sampled = {r['sap'] for r in self.rows('SELECT sap FROM samples')}
        staged = []
        from euroflash.beams import beam_number
        for sap in saps:
            if sap['key'] in sampled:
                continue
            beams = sorted(p for p in Path(sap['sap_dir']).glob('B*/*_ff.fil')
                           if beam_number(p.name) not in EXCLUDE_BEAMS)
            if not beams:
                continue
            origin = random.Random(sap['key']).choice(beams)
            link = self.root / 'staging' / sap['key'] / origin.name
            link.parent.mkdir(exist_ok=True)
            try:
                if not link.exists():
                    os.link(origin, link)
            except FileNotFoundError:
                continue                            # the campaign removed it between glob and link
            seed = int(hashlib.sha1(sap['key'].encode()).hexdigest()[:8], 16)
            source = 'spider' if sap.get('source') == 'spider' else 'lta'
            with self.db:
                self.db.execute("INSERT INTO samples(sap,source,beam,origin,link,seed,state,staged) VALUES "
                                "(?,?,?,?,?,?,'staged',?)", (sap['key'], source, beam_number(origin.name),
                                                             str(origin), str(link), seed, self.now()))
            staged.append(sap['key'])
        return staged

    # ------------------------------------------------------------------ make
    def make(self):
        """Start twins for staged beams on CPU nodes; settle the ones that ended."""
        finished = []
        for s in self.rows("SELECT * FROM samples WHERE state='making'"):
            exit_file = Path(s['twin'] + '.log.exit')
            if exit_file.exists():
                ok = exit_file.read_text().strip() == '0' and Path(s['twin']).is_file() and Path(s['truth']).is_file()
                attempts = (s.get('attempts') or 0) + 1
                if ok or attempts >= MAKE_ATTEMPTS:
                    Path(s['link']).unlink(missing_ok=True)
                log = Path(s['twin'] + '.log')
                self.set('samples', s['sap'], state='ready' if ok else 'failed' if attempts >= MAKE_ATTEMPTS else 'staged',
                         made=self.now(), attempts=attempts,
                         detail=None if ok else (log.read_text()[-2000:] if log.exists() else 'no log'))
                exit_file.unlink()
                finished.append(s['sap'])
            elif self.now() - (s['staged'] or 0) > MAKE_TIMEOUT + 3600:
                Path(s['link']).unlink(missing_ok=True)
                self.set('samples', s['sap'], state='failed', detail='twin not made in time')
        busy = len(self.rows("SELECT sap FROM samples WHERE state='making'"))
        waiting = self.rows("SELECT * FROM samples WHERE state='staged' ORDER BY staged LIMIT ?",
                            max(0, MAKE_PARALLEL - busy))
        if waiting:
            snapshot, _, image = self.code()
        for i, s in enumerate(waiting):
            twin = self.root / 'twins' / s['source'] / Path(s['link']).name
            truth = self.root / 'truth' / f"{s['sap']}.json"
            twin.parent.mkdir(exist_ok=True)
            node = CPU_NODES[(busy + i) % len(CPU_NODES)]
            command = self.container(snapshot, image, 'lotaas_reprocessing.frb_injection', s['link'], twin, truth,
                                     s['seed'], self.settings(snapshot, s['source']), BURSTS)
            Path(str(twin) + '.log.exit').unlink(missing_ok=True)
            if self.remote(node, command, Path(str(twin) + '.log')):
                self.set('samples', s['sap'], state='making', twin=str(twin), truth=str(truth), node=node)
        return finished

    # ------------------------------------------------------------------ dispatch
    def running(self):
        return self.rows("SELECT * FROM batches WHERE state IN ('running', 'analysing')")

    def dispatch(self, popen=subprocess.Popen):
        """Search waiting twins of one source, when enough wait or the oldest has waited long enough."""
        if self.running():
            return None
        for source in SOURCES:
            ready = self.rows("SELECT * FROM samples WHERE state='ready' AND source=? ORDER BY made", source)
            if not ready or (len(ready) < BATCH and self.now() - ready[0]['made'] < MAX_WAIT):
                continue
            ready = ready[:BATCH]
            name = time.strftime('inject-%Y%m%d-%H%M%S-', time.gmtime(self.now())) + source
            batch = self.root / 'batches' / name
            (batch / 'truth').mkdir(parents=True)
            for s in ready:
                os.link(s['twin'], batch / Path(s['twin']).name)
                shutil.copy(s['truth'], batch / 'truth' / f"{Path(s['twin']).stem}.json")
            settings = self.o['settings'][source]
            _, commit, _ = self.code()                 # the checkout the run freezes when it starts
            command = [sys.executable, '-m', 'euroflash.cluster', '--input', str(batch), '--work',
                       str(self.root / 'results' / name), '--ledger', str(self.root / 'ledger.sqlite'),
                       '--nodes', GPU_NODE, '--run-name', name, '--gpus', '0,1', '--workers-per-gpu', '1',
                       '--cpu-workers', '12', '--image', str(self.o['image']), '--settings', str(settings),
                       '--exclude-beams', *map(str, EXCLUDE_BEAMS), '--skip-trials', '--cleanup-remote',
                       '--cpu-nodes', *CPU_NODES, '--cpu-tier-workers', str(self.o['cpu_tier_workers']),
                       '--cpu-lock-dir', str(self.o['root'] / '.cpu-slots'), '--control-dir', str(self.o['control_dir'])]
            for timeout in self.o['timeouts']:
                command += ['--stage-timeout', timeout]
            log = self.root / 'logs' / f'{name}.log'
            script = f"{shlex.join(command)} > {shlex.quote(str(log))} 2>&1; echo $? > {shlex.quote(str(log))}.exit"
            popen(['sh', '-c', script], cwd=str(self.o['checkout']), start_new_session=True,
                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with self.db:
                self.db.execute("INSERT INTO batches(name,source,node,checkout,fingerprint,settings,started,state) "
                                "VALUES (?,?,?,?,?,?,?,'running')",
                                (name, source, GPU_NODE, str(self.o['checkout']), commit, str(settings), self.now()))
                for s in ready:
                    self.db.execute("UPDATE samples SET state='dispatched', batch=? WHERE sap=?", (name, s['sap']))
            return name
        return None

    # ------------------------------------------------------------------ collect
    def collect(self):
        """Follow a finished batch's bursts to their fates, record them, and delete its twins."""
        done = []
        for b in self.running():
            log = self.root / 'logs' / f"{b['name']}.log"
            fates = self.root / 'fates' / f"{b['name']}.json"
            if b['state'] == 'running' and Path(str(log) + '.exit').exists():
                snapshot, _, image = self.code()
                command = self.container(snapshot, image, 'lotaas_reprocessing.injection_fates',
                                         self.root / 'results' / b['name'], self.root / 'batches' / b['name'] / 'truth',
                                         self.settings(snapshot, b['source']), fates, self.root / 'batches' / b['name'])
                analysis_log = self.root / 'logs' / f"{b['name']}.fates.log"
                if self.remote(CPU_NODES[0], command, analysis_log):
                    self.set('batches', b['name'], state='analysing',
                             detail=f"search exit {Path(str(log) + '.exit').read_text().strip()}")
            elif b['state'] == 'analysing':
                exit_file = self.root / 'logs' / f"{b['name']}.fates.log.exit"
                if not exit_file.exists():
                    continue
                if exit_file.read_text().strip() == '0' and fates.is_file():
                    self.record(b, json.loads(fates.read_text()))
                    self.finish(b, 'done')
                else:
                    self.finish(b, 'failed')
                done.append(b['name'])
        return done

    def record(self, batch, result):
        samples = {Path(s['twin']).stem: s for s in self.rows('SELECT * FROM samples WHERE batch=?', batch['name'])}
        with self.db:
            for r in result['bursts']:
                s = samples.get(r['twin'], {})
                c = r.get('cluster') or {}
                page = r.get('page') if isinstance(r.get('page'), (int, float)) else None
                self.db.execute('INSERT OR REPLACE INTO bursts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                (r['twin'], r['index'], s.get('sap'), batch['name'], batch['source'],
                                 batch['fingerprint'], r['dm'], r['tau135'], r['width'], r['snr_ideal'], r['spectrum'],
                                 r['stage'], c.get('snr'), r.get('own_snr'), r.get('dispersion_ratio'), r.get('fetch'),
                                 page, int(bool(r.get('queued'))), json.dumps(r)))

    def finish(self, batch, state):
        for s in self.rows('SELECT * FROM samples WHERE batch=?', batch['name']):
            for path in (s['twin'], s['twin'] + '.log', s['twin'] + '.log.exit'):
                Path(path).unlink(missing_ok=True)
            self.set('samples', s['sap'], state='done' if state == 'done' else 'failed')
        shutil.rmtree(self.root / 'batches' / batch['name'], ignore_errors=True)
        self.set('batches', batch['name'], state=state, finished=self.now())

    def tick(self):
        return {'staged': self.stage(), 'made': self.make(), 'dispatched': self.dispatch(), 'collected': self.collect()}

    # ------------------------------------------------------------------ reading
    def status(self):
        return {'samples': dict(self.db.execute('SELECT state, COUNT(*) FROM samples GROUP BY state').fetchall()),
                'batches': dict(self.db.execute('SELECT state, COUNT(*) FROM batches GROUP BY state').fetchall()),
                'bursts': self.db.execute('SELECT COUNT(*) FROM bursts').fetchone()[0]}


def completeness(bursts, key, edges):
    """[(label, n, fraction queued)] of bursts binned by `key` at `edges`."""
    out = []
    for lo, hi in zip(edges, edges[1:]):
        sel = [b for b in bursts if lo <= b[key] < hi]
        out.append((f'{key} {lo:g}-{hi:g}', len(sel), sum(b['queued'] for b in sel) / len(sel) if sel else None))
    return out


def report(lane, source=None, fingerprint=None):
    sql, args = 'SELECT * FROM bursts WHERE 1=1', []
    if source:
        sql += ' AND source=?'
        args.append(source)
    if fingerprint:
        sql += ' AND fingerprint LIKE ?'
        args.append(fingerprint + '%')
    bursts = lane.rows(sql, *args)
    lines = [f'{len(bursts)} injected bursts, {sum(b["queued"] for b in bursts)} reached the review queue']
    for key, edges in (('snr_ideal', [6, 8, 10, 12, 15, 20, 30, 60]), ('dm', [100, 300, 600, 1000, 2000, 3000]),
                       ('tau135', [0.001, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0])):
        for label, n, frac in completeness(bursts, key, edges):
            lines.append(f'  {label:24s} {n:6d}  ' + (f'{frac:.2f}' if frac is not None else '-'))
    stages = {}
    for b in bursts:
        stages[b['stage']] = stages.get(b['stage'], 0) + 1
    lines.append('  stages: ' + ', '.join(f'{k} {v}' for k, v in sorted(stages.items(), key=lambda kv: -kv[1])))
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['tick', 'status', 'report'])
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--source', choices=SOURCES)
    p.add_argument('--fingerprint')
    a = p.parse_args(argv)
    lane = Lane(a.root)
    if a.command == 'tick':
        print(time.strftime('%Y-%m-%d %H:%M:%S'), json.dumps(lane.tick()), flush=True)
    elif a.command == 'status':
        print(json.dumps(lane.status(), indent=1))
    else:
        print(report(lane, a.source, a.fingerprint))


if __name__ == '__main__':
    main()
