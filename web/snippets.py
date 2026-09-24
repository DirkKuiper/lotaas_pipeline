"""Keep the data around candidates after the campaign deletes the searched beam.

The campaign removes a SAP's flatfielded filterbanks once they are searched,
and the classifier keeps only a PNG of each candidate. Three steps keep what a
reviewer needs without touching the pipeline or its fingerprint:

hold     Hard-link every prepared or dispatched SAP's filterbanks into held/.
         A link costs nothing while the campaign's copy exists and keeps the
         data when the campaign unlinks its copy after the search.
cut      Once the campaign has deleted its copy (the beam was searched), cut
         the stretch the classifier read around each kept detection, the
         dispersion sweep either side as `your` reads it plus some off-pulse
         margin, at the search's own time resolution for that DM. Write it as
         a SIGPROC snippet with a JSON sidecar.
release  Remove the held link. A beam the campaign still keeps (not searched,
         waiting for a retry) is released without cutting and held again when
         the SAP is dispatched again.

Detections made before this existed are cut from any flatfielded filterbank of
the same beam still found under the source roots.
"""
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
import time

import numpy as np
import yaml

from web import sigproc
from web.dynspec import K_DM
from web.keys import item_of, parse_item, short_id, sp_key

logger = logging.getLogger(__name__)

HOLD_STATES = ('prepared', 'dispatched')
KEEP_STATES = ('prepared', 'dispatched', 'flatfielding', 'staging')
# Not sources of search input: raw data, our own output, and the results tree.
PRUNE = {'processed', 'downloads', 'extracted', 'snippets', 'logs', 'catalogue', 'staging',
         'containers', '.git', '__pycache__'}


def downsample_for(dm, plan):
    """The time resolution the search used at this DM, from its dedispersion plan."""
    for step in plan or []:
        if step['low_dm'] <= dm < step['high_dm']:
            return int(step['downsample'])
    return int(plan[-1]['downsample']) if plan else 1


def default_plan(settings):
    try:
        values = yaml.safe_load(Path(settings).read_text())
        return values.get('dedispersion_plan', []), values.get('bad_channels', [])
    except OSError:
        return [], []


def cut(source, detection, out_dir, plan, bad_channels=(), provenance=None):
    """Cut one detection's snippet from a flatfielded filterbank; returns its path."""
    header, data = sigproc.open_data(source)
    freqs = sigproc.channel_frequencies(header)
    tsamp = float(header['tsamp'])
    dm = float(detection['dm'])
    tcand = float(detection['time_seconds'])
    width = max(1, int(detection['width_samples']))
    search_downsample = downsample_for(dm, plan)
    # Retain at least eight samples across broad events, aligned to the search grid.
    k = search_downsample * max(1, width // (8 * search_downsample))
    delay = K_DM * dm * (1 / freqs.min() ** 2 - 1 / freqs.max() ** 2)
    # Off-pulse room on both sides, and data for trial DMs above the candidate's.
    margin = max(10.0, 64 * width * tsamp)
    # Blocks of k native samples on the search's own grid, so a decimated
    # sample means what it meant to the search.
    start = int(math.floor((tcand - delay - margin) / (tsamp * k))) * k
    stop = int(math.ceil((tcand + delay + margin) / (tsamp * k))) * k
    total = data.shape[0]
    # Do not manufacture constant off-pulse noise beyond the observation.
    start, stop = max(start, 0), min(stop, total // k * k)
    if stop <= start:
        raise ValueError(f'{source} holds no samples near t={tcand:.3f} s')
    block = np.empty(((stop - start) // k, data.shape[1]), dtype=np.float32)
    batch = max(1, 262144 // (k * data.shape[1]))
    for out_start in range(0, len(block), batch):
        out_stop = min(len(block), out_start + batch)
        raw = np.asarray(data[start + out_start * k:start + out_stop * k])
        block[out_start:out_stop] = raw.reshape(-1, k, data.shape[1]).mean(axis=1)

    obs, sap, beam = parse_item(detection['item'])
    key = detection['key']
    name = f'{obs}_SAP{sap:03d}_B{beam:03d}_DM{dm:.3f}_t{tcand:.3f}_{short_id(key)}'
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (name + '.fil')
    partial = path.with_suffix('.fil.partial')
    snippet_header = dict(header, tstart=header['tstart'] + start * tsamp / 86400.0, tsamp=tsamp * k,
                          rawdatafile=Path(source).name)
    sigproc.write(partial, snippet_header, block)
    stat = Path(source).stat()
    meta = {'key': key, 'id': short_id(key), 'type': detection['type'], 'item': detection['item'],
            'dm': dm, 'snr': detection.get('snr'), 'width_samples': width,
            'time_seconds': tcand, 'sample_number': detection.get('sample_number'),
            'probability': detection.get('probability'), 'pulsar': detection.get('pulsar'),
            'downsample': k, 'search_downsample': search_downsample,
            'tsamp_native': tsamp, 'tsamp': tsamp * k,
            'start_sample': start, 't0_relative': start * tsamp - tcand, 'samples': int(block.shape[0]),
            'sweep_seconds': delay, 'margin_seconds': margin,
            'bad_channels': sorted({int(c) for c in bad_channels}),
            'source': str(source), 'source_bytes': stat.st_size, 'source_mtime': stat.st_mtime,
            'created': time.time(), **(provenance or {})}
    path.with_suffix('.json').write_text(json.dumps(meta, indent=1))
    os.replace(partial, path)
    return path


class Snippets:
    """Hold, cut and release against the campaign state and the web index."""

    def __init__(self, cfg):
        self.cfg = cfg.prepare()
        self.plan, self.bad_channels = default_plan(cfg.settings)
        self.sources = {}
        self.sources_scanned = 0.0
        self.hold_enabled = cfg.hold and self._same_filesystem()

    def _same_filesystem(self):
        try:
            same = os.stat(self.cfg.held).st_dev == os.stat(self.cfg.campaign_root).st_dev
        except OSError:
            return False
        if not same:
            logger.warning('held/ is not on the campaign filesystem; hard links impossible, holding disabled')
        return same

    def _state(self):
        db = sqlite3.connect(f'file:{self.cfg.state_db}?mode=ro', uri=True, timeout=60)
        db.row_factory = sqlite3.Row
        try:
            return {r['key']: dict(r) for r in db.execute('SELECT key,state,sap_dir,run_name FROM saps')}
        finally:
            db.close()

    def _index(self):
        db = sqlite3.connect(f'file:{self.cfg.index_db}?mode=ro', uri=True, timeout=60)
        db.row_factory = sqlite3.Row
        return db

    # --------------------------------------------------------------- hold
    def hold(self, saps):
        linked = 0
        if not self.hold_enabled:
            return linked
        for key, sap in saps.items():
            if sap['state'] not in HOLD_STATES or not sap['sap_dir']:
                continue
            target_dir = self.cfg.held / key
            for path in Path(sap['sap_dir']).glob('B*/*_ff.fil'):
                parsed = parse_item(path.name)
                if parsed is None or parsed[2] in self.cfg.exclude_beams:
                    continue
                target = target_dir / path.name
                if target.exists():
                    continue
                target_dir.mkdir(exist_ok=True)
                try:
                    os.link(path, target)
                    linked += 1
                except FileNotFoundError:
                    pass          # the campaign removed it between glob and link
        return linked

    # ---------------------------------------------------- detections to cut
    def _wanted(self, index, where, args):
        types = [t for t in self.cfg.snippet_types]
        clauses = [f"d.detection_type IN ({','.join('?' * len(types))})"]
        params = list(types)
        if self.cfg.rejected_above is not None:
            clauses.append("(d.detection_type='rejected' AND d.classification_probability>?)")
            params.append(self.cfg.rejected_above)
        rows = index.execute(
            f"""SELECT d.id, d.key, d.item, d.detection_type AS type, d.candidate_dm AS dm, d.snr,
                       d.width_samples, d.time_seconds, d.sample_number,
                       d.classification_probability AS probability, d.pulsar_name AS pulsar,
                       b.run_name, b.fp16
                FROM detections d JOIN beam_runs b ON b.id=d.beam_run_id
                WHERE ({' OR '.join(clauses)}) AND d.time_seconds IS NOT NULL AND {where}""",
            params + list(args)).fetchall()
        return [dict(r) for r in rows]

    def _beam_meta(self, index, item, fp16):
        row = index.execute('SELECT dir FROM beams WHERE item=? AND fp16=? ORDER BY mtime DESC LIMIT 1',
                            (item, fp16)).fetchone()
        if row:
            try:
                return json.loads((Path(row['dir']) / 'metadata.json').read_text())
            except (OSError, ValueError):
                pass
        return {}

    def _existing(self):
        return {p.stem.rsplit('_', 1)[-1] for p in self.cfg.snippets.glob('*.fil')}

    def _cut_all(self, index, detections, source, how):
        existing = self._existing()
        made = 0
        for detection in detections:
            if short_id(detection['key']) in existing:
                continue
            meta = self._beam_meta(index, detection['item'], detection['fp16'])
            plan = meta.get('dedispersion_plan') or self.plan
            bad = set(meta.get('bad_channels', self.bad_channels))
            try:
                cut(source, detection, self.cfg.snippets, plan, bad,
                    {'how': how, 'run_name': detection['run_name'], 'fp16': detection['fp16'],
                     'detection_id': detection['id']})
                made += 1
                existing.add(short_id(detection['key']))
            except Exception as error:
                logger.error('Could not cut %s from %s: %s', detection['key'], source, error)
        return made

    # ------------------------------------------------------ cut and release
    def settle(self, saps):
        cut_count = released = 0
        if not self.cfg.held.is_dir():
            return 0, 0
        with self._index() as index:
            for sap_dir in self.cfg.held.iterdir():
                sap = saps.get(sap_dir.name)
                if sap is None or not sap_dir.is_dir():
                    continue
                if sap['state'] in KEEP_STATES:
                    continue
                for held in sorted(sap_dir.glob('*.fil')):
                    item = item_of(held.name)
                    parsed = parse_item(item)
                    original = Path(sap['sap_dir'] or '/nonexistent') / f'B{parsed[2]:03d}' / held.name
                    searched = not original.exists()
                    if not searched:
                        # --keep-prepared: the copy stays even after a search.
                        latest = index.execute(
                            "SELECT run_name,outcome FROM beam_runs WHERE item=? ORDER BY id DESC LIMIT 1",
                            (item,)).fetchone()
                        searched = bool(latest and latest['run_name'] == sap['run_name']
                                        and latest['outcome'] not in ('processing', None))
                    if searched:
                        latest = index.execute('SELECT MAX(id) AS id FROM beam_runs WHERE item=?',
                                               (item,)).fetchone()
                        if latest['id'] is None:
                            continue      # searched, but its records are not merged yet
                        detections = self._wanted(index, 'd.beam_run_id=?', (latest['id'],))
                        cut_count += self._cut_all(index, detections, held, 'held')
                    held.unlink()
                    released += 1
                if not any(sap_dir.iterdir()):
                    sap_dir.rmdir()
        return cut_count, released

    # ----------------------------------------------------- earlier detections
    def scan_sources(self):
        """Every flatfielded filterbank under the source roots, by beam name."""
        found = {}
        for root in [self.cfg.held, *self.cfg.source_roots]:
            for directory, subdirectories, files in os.walk(root):
                subdirectories[:] = [d for d in subdirectories if d not in PRUNE and not d.startswith('.')]
                if Path(directory).resolve() == self.cfg.held.resolve() and root != self.cfg.held:
                    subdirectories[:] = []
                    continue
                for name in files:
                    if name.endswith('_32bit_ff.fil'):
                        found.setdefault(item_of(name), []).append(Path(directory) / name)
        self.sources, self.sources_scanned = found, time.time()
        return found

    def backfill(self, rescan_seconds=3600):
        """Cut detections that have no snippet from any surviving copy of their beam."""
        if time.time() - self.sources_scanned > rescan_seconds:
            self.scan_sources()
        existing = self._existing()
        made = 0
        with self._index() as index:
            detections = [d for d in self._wanted(index, '1=1', ())
                          if short_id(d['key']) not in existing and d['item'] in self.sources]
            for detection in detections:
                # Prefer the campaign's own and held copies, then any other.
                sources = sorted(self.sources[detection['item']],
                                 key=lambda p: (self.cfg.held not in p.parents,
                                                self.cfg.campaign_root not in p.parents, str(p)))
                made += self._cut_all(index, [detection], sources[0], 'found on disk')
        return made

    def run_pass(self):
        saps = self._state()
        linked = self.hold(saps)
        cut_count, released = self.settle(saps)
        backfilled = self.backfill()
        return {'linked': linked, 'cut': cut_count, 'released': released, 'backfilled': backfilled}
