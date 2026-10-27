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
# A trial of a change to production's settings beside production itself: every VARIANT_EVERY-th batch of a source the
# variant lists is searched with production's settings so changed, its batch named after it, and its bursts and real
# candidates are kept apart (variant_bursts, variant_others), where only `report --variant` reads them: the FRB rate and
# limits read production's alone. 'gate7' lowers FETCH's S/N gate from 8 to 7 (26 October 2026): 3,211 of 31,370
# injected LT5 bursts (10%) were found at search S/N 6-8 and never asked about, and a fluence threshold 8/7 lower is
# (8/7)^1.5 = 1.22 times the FRBs; at S/N 7 the first night (27 September) gave ~60 FETCH positives a SAP. The variant
# measures both: what it queues of the injected bursts, and the real candidates per twin it adds. VARIANT_EVERY 0 stops it.
VARIANTS = {'gate7': {'sources': ('lta',), 'set': {('classification', 'min_snr'): 7.0,
                                                   ('classification', 'dispersed', 'min_snr'): 7.0}}}
VARIANT_EVERY = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (sap TEXT PRIMARY KEY, source TEXT, beam INTEGER, origin TEXT, link TEXT,
    twin TEXT, truth TEXT, seed INTEGER, state TEXT, node TEXT, batch TEXT, staged REAL, made REAL, detail TEXT,
    attempts INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS batches (name TEXT PRIMARY KEY, source TEXT, node TEXT, checkout TEXT, fingerprint TEXT,
    settings TEXT, started REAL, finished REAL, state TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS others (twin TEXT, batch TEXT, source TEXT, fingerprint TEXT, type TEXT, dm REAL,
    snr REAL, width INTEGER, time REAL, page REAL, noise INTEGER);
CREATE TABLE IF NOT EXISTS bursts (twin TEXT, idx INTEGER, sap TEXT, batch TEXT, source TEXT, fingerprint TEXT,
    dm REAL, tau135 REAL, width REAL, snr_ideal REAL, spectrum TEXT, stage TEXT, search_snr REAL, own_snr REAL,
    dispersion_ratio REAL, fetch REAL, page REAL, queued INTEGER, record TEXT, PRIMARY KEY (twin, idx));
CREATE TABLE IF NOT EXISTS variant_others (variant TEXT, twin TEXT, batch TEXT, source TEXT, fingerprint TEXT,
    type TEXT, dm REAL, snr REAL, width INTEGER, time REAL, page REAL, noise INTEGER);
CREATE TABLE IF NOT EXISTS variant_bursts (variant TEXT, twin TEXT, idx INTEGER, sap TEXT, batch TEXT, source TEXT,
    fingerprint TEXT, dm REAL, tau135 REAL, width REAL, snr_ideal REAL, spectrum TEXT, stage TEXT, search_snr REAL,
    own_snr REAL, dispersion_ratio REAL, fetch REAL, page REAL, queued INTEGER, record TEXT, PRIMARY KEY (twin, idx));
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


def variant_of(batch_name):
    """The settings variant a batch was searched with (VARIANTS), from its name; None for production's."""
    return next((v for v in VARIANTS if str(batch_name).endswith('-' + v)), None)


def variant_settings(path, variant, out):
    """Production's settings at `path` with VARIANTS[variant]'s changes, written to `out`."""
    import yaml
    settings = yaml.safe_load(Path(path).read_text())
    for keys, value in VARIANTS[variant]['set'].items():
        node = settings
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    Path(out).write_text(yaml.safe_dump(settings, sort_keys=False))
    return Path(out)


def levelled_rows(origin):
    """The 2-bit row length the beam's SAP was levelled with (flatfield_fil.py's row-levelling.json), or 0: the
    twin is levelled again after its bursts go in, as the real beam was (frb_injection)."""
    try:
        return int(json.loads((Path(origin).parent.parent / 'row-levelling.json').read_text()).get('row') or 0)
    except (OSError, ValueError):
        return 0


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
        for table in ('others', 'variant_others'):          # page S/N and the web's verdict, from 27 October 2026
            have = {r[1] for r in self.db.execute(f'PRAGMA table_info({table})')}
            for column, kind in (('page', 'REAL'), ('noise', 'INTEGER')):
                if column not in have:
                    self.db.execute(f'ALTER TABLE {table} ADD COLUMN {column} {kind}')

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
                                     s['seed'], self.settings(snapshot, s['source']), BURSTS, levelled_rows(s['origin']))
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
            variant = self.next_variant(source)
            name = time.strftime('inject-%Y%m%d-%H%M%S-', time.gmtime(self.now())) + source + (f'-{variant}' if variant else '')
            batch = self.root / 'batches' / name
            (batch / 'truth').mkdir(parents=True)
            for s in ready:
                os.link(s['twin'], batch / Path(s['twin']).name)
                shutil.copy(s['truth'], batch / 'truth' / f"{Path(s['twin']).stem}.json")
            settings = self.o['settings'][source]
            if variant:
                settings = variant_settings(settings, variant, batch / 'settings.yaml')
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

    def next_variant(self, source):
        """The variant the next batch of this source is searched with, or None: every VARIANT_EVERY-th batch."""
        if VARIANT_EVERY <= 0:
            return None
        for name, v in VARIANTS.items():
            if source in v['sources']:
                n = self.db.execute('SELECT COUNT(*) FROM batches WHERE source=?', (source,)).fetchone()[0]
                return name if n % VARIANT_EVERY == VARIANT_EVERY - 1 else None
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
                # A variant's fates by its own settings (its S/N gate): they lie beside its twins on the shared disk.
                settings = (Path(b['settings']) if variant_of(b['name']) else self.settings(snapshot, b['source']))
                command = self.container(snapshot, image, 'lotaas_reprocessing.injection_fates',
                                         self.root / 'results' / b['name'], self.root / 'batches' / b['name'] / 'truth',
                                         settings, fates, self.root / 'batches' / b['name'])
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
        variant = variant_of(batch['name'])
        # A variant's results are kept apart: the FRB rate and limits read `bursts` as production's.
        bursts, others, extra = ('variant_bursts', 'variant_others', (variant,)) if variant else ('bursts', 'others', ())
        with self.db:
            for r in result['bursts']:
                s = samples.get(r['twin'], {})
                c = r.get('cluster') or {}
                page = r.get('page') if isinstance(r.get('page'), (int, float)) else None
                row = extra + (r['twin'], r['index'], s.get('sap'), batch['name'], batch['source'],
                               batch['fingerprint'], r['dm'], r['tau135'], r['width'], r['snr_ideal'], r['spectrum'],
                               r['stage'], c.get('snr'), r.get('own_snr'), r.get('dispersion_ratio'), r.get('fetch'),
                               page, int(bool(r.get('queued'))), json.dumps(r))
                self.db.execute(f"INSERT OR REPLACE INTO {bursts} VALUES ({','.join('?' * len(row))})", row)
            # What the twins' real data sent for review: the queue's cost per beam, across the survey.
            for o in result.get('others', []):
                if o['type'] in ('candidate', 'dispersed'):
                    page = o.get('page') if isinstance(o.get('page'), (int, float)) else None
                    noise = int(bool(o['noise'])) if page is not None and 'noise' in o else None
                    row = extra + (o['twin'], batch['name'], batch['source'], batch['fingerprint'], o['type'],
                                   o['dm'], o['snr'], o['width'], o.get('time'), page, noise)
                    self.db.execute(f"INSERT INTO {others} VALUES ({','.join('?' * len(row))})", row)

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
                'bursts': self.db.execute('SELECT COUNT(*) FROM bursts').fetchone()[0],
                'real candidates per twin': dict(self.db.execute(
                    'SELECT type, ROUND(1.0 * COUNT(*) / (SELECT COUNT(DISTINCT twin) FROM bursts), 3) FROM others '
                    'GROUP BY type').fetchall())}


def completeness(bursts, key, edges):
    """[(label, n, fraction queued)] of bursts binned by `key` at `edges`."""
    out = []
    for lo, hi in zip(edges, edges[1:]):
        sel = [b for b in bursts if lo <= b[key] < hi]
        out.append((f'{key} {lo:g}-{hi:g}', len(sel), sum(b['queued'] for b in sel) / len(sel) if sel else None))
    return out


def report(lane, source=None, fingerprint=None, variant=None, since=None):
    """Completeness of the lane's bursts, production's or a variant's (with the real candidates per twin each queued),
    from batches started at or after `since` (unix time) when given, so a variant meets production on the same days."""
    table = 'variant_bursts' if variant else 'bursts'
    sql, args = f'SELECT * FROM {table} WHERE 1=1', []
    if variant:
        sql += ' AND variant=?'
        args.append(variant)
    if since:
        sql += ' AND batch IN (SELECT name FROM batches WHERE started >= ?)'
        args.append(since)
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
    # The web's below-gate rule (injection_fates.gate_rule) would settle these of the queued, were the gate lower.
    settled = sum(1 for b in bursts if b['queued'] and json.loads(b['record'] or '{}').get('gate_noise'))
    if settled:
        lines.append(f'  of the queued, {settled} the web\'s below-gate rule (search S/N < 8, page '
                     f'S/N < 6) would call noise: lower it with the gate')
    twins = {b['twin'] for b in bursts}
    if twins:
        others = 'variant_others' if variant else 'others'
        counts, left, measured = {}, {}, 0
        for kind, twin, page, noise in lane.db.execute(f'SELECT type, twin, page, noise FROM {others}'):
            if twin in twins:
                counts[kind] = counts.get(kind, 0) + 1
                if noise is not None:
                    measured += 1
                    if not noise:
                        left[kind] = left.get(kind, 0) + 1
        lines.append(f'  real candidates per twin ({len(twins)} twins): '
                     + (', '.join(f'{k} {v / len(twins):.2f}' for k, v in sorted(counts.items())) or 'none'))
        if measured:
            lines.append(f'  of them with a page S/N ({measured}), left open by the web\'s noise rules: '
                         + (', '.join(f'{k} {v}' for k, v in sorted(left.items())) or 'none'))
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['tick', 'status', 'report'])
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--source', choices=SOURCES)
    p.add_argument('--fingerprint')
    p.add_argument('--variant', choices=sorted(VARIANTS))
    p.add_argument('--since', help='only batches started at or after this UTC date, YYYY-MM-DD[THH:MM]')
    a = p.parse_args(argv)
    lane = Lane(a.root)
    if a.command == 'tick':
        print(time.strftime('%Y-%m-%d %H:%M:%S'), json.dumps(lane.tick()), flush=True)
    elif a.command == 'status':
        print(json.dumps(lane.status(), indent=1))
    else:
        since = None
        if a.since:
            import calendar
            fmt = '%Y-%m-%dT%H:%M' if 'T' in a.since else '%Y-%m-%d'
            since = calendar.timegm(time.strptime(a.since, fmt))
        print(report(lane, a.source, a.fingerprint, a.variant, since))


if __name__ == '__main__':
    main()
