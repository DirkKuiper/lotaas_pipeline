"""Verdicts the index reaches from the data alone, recorded so that no one has to give them.

Three kinds of single-pulse candidate are settled by what the rest of the
observation shows, not by looking at the candidate:

- a pulse of a catalogued pulsar seen away from its own beam
  (indexer.derive_known): at the pulsar's DM, within 5 degrees of it, with a
  fold at its period or the pulses themselves keeping its rotation to show the
  pulsar is there. Verdict 'known'.
- an undispersed burst that reached many beams at once: the same moment in
  COINCIDENT_BEAMS or more beams at scattered DMs (indexer.derive_coincidence),
  with a quarter or more of the events there below DM 1, or in beams of all
  three SAPs, which point about 4 degrees apart: no one position on the sky is
  in all of them (a burst in L611408 reached 92 beams at S/N up to 68, bright
  enough to register at every DM, so fewer than a quarter lay below DM 1).
  Verdict 'rfi'.
- an undispersed burst in the candidate's own beam: a burst of events there
  at one moment at scattered DMs, none standing out (indexer.sweeps), where a
  pulse would peak at its DM and fall away. Verdict 'rfi'.

A periodic fold is settled too when it is a catalogued pulsar at its own
period (not a harmonic: those stay for a person) and DM, within 5 degrees: B0823+26
was folded in 92 beams of L611400, 3.3 degrees away in its other SAP, where the
pipeline's own catalogue match does not reach. Verdict 'known'.

Each verdict is recorded by REVIEWER with its evidence in the note. A person's
verdict is never written over, and a person's later verdict replaces this one
(the latest counts everywhere verdicts are read). This triage's own verdict is
replaced when its evidence changes: two classifier 'redetections' first called
'known' were bursts in their own beams with no pulse at the pulsar's DM.
"""
import time

from web import store
from web.indexer import COINCIDENT_BEAMS

REVIEWER = 'auto-triage'
ALL_SAPS = 3                  # a LOTAAS pointing's SAPs
ROUTES = {'fold': 'a fold at its period', 'redetection': 'the classifier redetected it',
          'rotation': 'the pulses keep its rotation'}
LATEST = '(SELECT {0} FROM review_state.reviews v WHERE v.key=c.key ORDER BY created DESC LIMIT 1)'
# No verdict yet, or only this triage's own.
OPEN = f"COALESCE({LATEST.format('reviewer')}, '{REVIEWER}') = '{REVIEWER}'"
QUEUED = "c.kind='sp' AND c.type IN ('candidate', 'known_pulsar') AND COALESCE(c.pilot, 0)=0"
NOT_KNOWN = 'NOT EXISTS (SELECT 1 FROM sp_known k WHERE k.key=c.key)'


def settled(db):
    """[(key, label, note, dm, earlier)] of the open single-pulse candidates the data settle.

    earlier is this triage's latest verdict on the candidate, or None. db is the
    index with reviews.sqlite attached as review_state (web.indexer).
    """
    out = {}
    earlier = LATEST.format('label')
    for r in db.execute(f"""SELECT c.key, c.dm, k.name, k.pulsar, k.separation_deg, k.route, k.z, {earlier} AS earlier
            FROM candidates c JOIN sp_known k ON k.key=c.key WHERE {QUEUED} AND {OPEN}"""):
        seen = ', '.join(ROUTES[route] for route in r['route'].split('+') if route in ROUTES)
        z = f' (Z = {r["z"]:.0f})' if r['z'] else ''
        name = r['name'] if r['name'] == r['pulsar'] else f"{r['name']} ({r['pulsar']})"
        out.setdefault(r['key'], (r['key'], 'known', f"{name}, {r['separation_deg']:.2f} deg from this beam, at its "
                                  f"DM; this observation shows it: {seen}{z}. A known pulsar seen away from its own "
                                  f"beam.", r['dm'], r['earlier']))
    for r in db.execute(f"""SELECT c.key, c.dm, x.beams, x.saps, x.near_zero, x.events, x.dm_min, x.dm_max,
            {earlier} AS earlier FROM candidates c JOIN sp_coincidence x ON x.key=c.key WHERE {QUEUED} AND {OPEN}
            AND x.beams >= ? AND NOT x.consistent AND (4 * x.near_zero >= x.events OR x.saps >= ?)
            AND {NOT_KNOWN}""", (COINCIDENT_BEAMS, ALL_SAPS)):
        where = 'all three SAPs' if r['saps'] >= ALL_SAPS else f"{r['saps']} SAP(s)"
        out.setdefault(r['key'], (r['key'], 'rfi', f"Interference: the same moment in {r['beams']} beams of {where} "
                                  f"at scattered DMs ({r['dm_min']:.1f}-{r['dm_max']:.1f}), {r['near_zero']} of the "
                                  f"{r['events']} events there below DM 1.", r['dm'], r['earlier']))
    for r in db.execute(f"""SELECT c.key, c.dm, s.events, s.expected, s.dm_min, s.dm_max, s.peak_ratio,
            {earlier} AS earlier FROM candidates c JOIN sp_sweep s ON s.key=c.key WHERE {QUEUED} AND {OPEN}
            AND {NOT_KNOWN}"""):
        out.setdefault(r['key'], (r['key'], 'rfi', f"Undispersed burst in this beam: {r['events']} events at one "
                                  f"moment ({r['expected']:.1f} expected from the beam's own rate) at DM "
                                  f"{r['dm_min']:.1f}-{r['dm_max']:.1f}, the strongest {r['peak_ratio']:.2f} times the "
                                  f"median S/N: no DM stands out as a pulse's would.", r['dm'], r['earlier']))
    return list(out.values())


PERIODIC_RADIUS_DEG = 5.0


def periodic_settled(db):
    """[(key, 'known', note, dm, earlier)] of open folds at a catalogued pulsar's own period and DM."""
    from euroflash import psrcat
    pulsars = psrcat.load()
    if not pulsars:
        return []
    out, cones = [], {}
    for r in db.execute(f"""SELECT c.key, c.dm, c.period, b.ra_deg, b.dec_deg, b.tstart_mjd,
            {LATEST.format('label')} AS earlier
            FROM candidates c JOIN periodic p ON p.key=c.key JOIN beams b ON b.dir=p.dir
            WHERE c.kind='periodic' AND c.type='periodic' AND COALESCE(c.pilot, 0)=0 AND c.period > 0
            AND c.dm IS NOT NULL AND b.ra_deg IS NOT NULL AND {OPEN}"""):
        place = (round(r['ra_deg'], 2), round(r['dec_deg'], 2))
        if place not in cones:
            cones[place] = psrcat.cone(pulsars, r['ra_deg'], r['dec_deg'], PERIODIC_RADIUS_DEG)
        for pulsar in cones[place]:
            if abs(r['dm'] - pulsar['dm']) > max(2.0, 0.05 * pulsar['dm']):
                continue
            if abs(r['period'] / psrcat.period_at(pulsar, r['tstart_mjd']) - 1) <= psrcat.PERIOD_TOLERANCE:
                name = pulsar.get('bname') or pulsar['name']
                name = name if name == pulsar['name'] else f"{name} ({pulsar['name']})"
                out.append((r['key'], 'known', f"{name} at its own period and DM, {pulsar['separation_deg']:.2f} deg "
                            f"from this beam.", r['dm'], r['earlier']))
                break
    return out


def record(cfg, db):
    """Record the verdicts settled() finds that differ from the latest; returns how many."""
    rows = [(key, label, note if earlier is None else f"{note} (Replaces this triage's earlier '{earlier}'.)", dm)
            for key, label, note, dm, earlier in settled(db) + periodic_settled(db) if label != earlier]
    if rows:
        reviews = store.reviews(cfg)
        try:
            now = time.time()
            with reviews:
                reviews.executemany('INSERT INTO reviews(key,reviewer,label,note,dm,created) VALUES (?,?,?,?,?,?)',
                                    [(key, REVIEWER, label, note, dm, now) for key, label, note, dm in rows])
        finally:
            reviews.close()
    return len(rows)
