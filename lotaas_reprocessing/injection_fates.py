"""Where each burst injected into a twin beam (frb_injection) ended once the twins were searched (euroflash.injections).

    python -m lotaas_reprocessing.injection_fates RUN_DIR TRUTH_DIR SETTINGS_YAML OUT_JSON

A burst is found by the strongest cluster within 0.5 s + the cluster's width of the burst's time and within
max(10, 10%) of its DM (clustered_candidates.txt), and followed to the classifier's record of that cluster (the
run's ledger snapshots). Stages: 'not clustered'; 'below S/N gate'; 'not classified' (a cap); then the record's
type: unconfirmed, known_pulsar, unclassified, rejected, candidate or dispersed. Candidate and dispersed ones reach
the review queue unless the web's noise rule settles them on the page's S/N (own_data.measure with the beam's
search baseline), as web.triage does: below LOCAL_MIN, or below LOCAL_FRACTION of the search's S/N while under
LOCAL_CLEAR. The run's other detections are counted too, as what the search passes on that was not injected.
"""
import glob
import json
import sqlite3
import sys
from pathlib import Path

import yaml

from lotaas_reprocessing.own_data import measure

# web.triage's noise rule on a queued candidate's page S/N (28 September 2026).
LOCAL_MIN, LOCAL_FRACTION, LOCAL_CLEAR, SEARCH_MIN = 4.0, 0.6, 6.0, 7.0
QUEUED = ('candidate', 'dispersed')
TIME_SLACK, DM_SLACK, DM_FRACTION = 0.5, 10.0, 0.1


def noise_rule(page, search_snr):
    """True when the web's triage would call a queued candidate noise on its page S/N."""
    if not isinstance(page, (int, float)) or search_snr < SEARCH_MIN:
        return False
    return page < LOCAL_MIN or (page < LOCAL_FRACTION * search_snr and page < LOCAL_CLEAR)


def detections(run_dir):
    """{beam item: [detection row]} from every node's ledger snapshot of the run."""
    out = {}
    for snapshot in Path(run_dir).glob('*/ledger-snapshot.sqlite'):
        db = sqlite3.connect(f'file:{snapshot}?mode=ro', uri=True)
        try:
            columns = [r[1] for r in db.execute('PRAGMA table_info(detections)')]
            if not columns:
                continue
            for row in db.execute('SELECT * FROM detections'):
                d = dict(zip(columns, row))
                out.setdefault(Path(d['beam_id']).stem, []).append(d)
        finally:
            db.close()
    return out


def beam_dirs(run_dir):
    """{beam item: its processed directory holding a classification summary}."""
    return {Path(p).parent.parent.name: Path(p).parent
            for p in glob.glob(f'{run_dir}/*/processed/*/*/sp_classify_summary.json')}


def page_snr(source, cluster, meta, settings):
    sp = meta.get('single_pulse') or settings.get('single_pulse') or {}
    try:
        return measure(source, cluster['dm'], cluster['time'], cluster['width'],
                       meta.get('dedispersion_plan') or settings['dedispersion_plan'], meta.get('bad_channels', ()),
                       sp.get('baseline_seconds'), sp.get('baseline_widths'))
    except Exception as error:                     # a stretch past the beam's end, say
        return f'error: {error}'[:200]


def fates(run_dir, truth_dir, settings, sources=None):
    """(bursts, others): each injected burst's fate, and the run's detections that were not injected."""
    run_dir, truth_dir = Path(run_dir), Path(truth_dir)
    tsamp_default = 0.007864
    gate = float((settings.get('classification') or {}).get('min_snr', 7.0))
    records, dirs = detections(run_dir), beam_dirs(run_dir)
    bursts, others = [], []
    for truth_path in sorted(truth_dir.glob('*.json')):
        truth = json.loads(truth_path.read_text())
        item = Path(truth['twin']).stem
        source = Path(sources) / truth['twin'] if sources else None
        d = dirs.get(item)
        if d is None:
            bursts += [dict(b, twin=item, stage='not searched') for b in truth['bursts']]
            continue
        meta = json.loads((d / 'metadata.json').read_text())
        tsamp = float(meta.get('tsamp') or tsamp_default)
        rows = [line.split() for line in (d / 'clustered_candidates.txt').read_text().splitlines()[1:] if line.strip()]
        clusters = [{'dm': float(r[0]), 'snr': float(r[1]), 'time': float(r[2]), 'width': int(float(r[4]))} for r in rows]
        evidence = (json.loads((d / 'single_pulse_evidence.json').read_text())
                    if (d / 'single_pulse_evidence.json').exists() else {})
        dets = records.get(item, [])
        used = set()
        for b in truth['bursts']:
            rec = dict(b, twin=item, stage='not clustered', cluster=None)
            near = [c for c in clusters if abs(c['time'] - b['peak_time']) <= TIME_SLACK + c['width'] * tsamp
                    and abs(c['dm'] - b['dm']) <= max(DM_SLACK, DM_FRACTION * b['dm'])]
            if near:
                c = max(near, key=lambda c: c['snr'])
                rec['cluster'] = c
                rec['local'] = (evidence.get(f"{c['dm']:.3f}|{c['time']:.6f}|{c['width']}") or {}).get('local_snr')
                match = [x for x in dets if abs(x['candidate_dm'] - c['dm']) < 1e-3
                         and abs((x.get('time_seconds') or -1) - c['time']) < 1e-3 and x['width_samples'] == c['width']]
                if match:
                    x = match[0]
                    used.add(x['id'])
                    rec['stage'] = x['detection_type']
                    rec['fetch'] = x.get('classification_probability')
                    rec['models'] = json.loads(x['model_probabilities']) if x.get('model_probabilities') else None
                    rec['own_snr'] = x.get('own_snr')
                    rec['dispersion_ratio'] = x.get('dispersion_ratio')
                    if x['detection_type'] in QUEUED and source is not None:
                        rec['page'] = page_snr(source, c, meta, settings)
                        rec['noise'] = noise_rule(rec['page'], c['snr'])
                else:
                    rec['stage'] = 'below S/N gate' if c['snr'] <= gate else 'not classified'
            rec['queued'] = rec['stage'] in QUEUED and not rec.get('noise', False)
            bursts.append(rec)
        for x in dets:
            if x['id'] not in used and not any(abs((x.get('time_seconds') or -1e9) - b['peak_time']) <= 2.0
                                               for b in truth['bursts']):
                others.append({'twin': item, 'type': x['detection_type'], 'dm': x['candidate_dm'], 'snr': x['snr'],
                               'width': x['width_samples'], 'time': x.get('time_seconds')})
    return bursts, others


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    run_dir, truth_dir, settings_path, out = argv[:4]
    sources = argv[4] if len(argv) > 4 else None
    settings = yaml.safe_load(Path(settings_path).read_text())
    bursts, others = fates(run_dir, truth_dir, settings, sources)
    Path(out).write_text(json.dumps({'run': Path(run_dir).name, 'bursts': bursts, 'others': others}))
    print(len(bursts), 'bursts;', sum(b['queued'] for b in bursts), 'queued;', len(others), 'other detections')


if __name__ == '__main__':
    main()
