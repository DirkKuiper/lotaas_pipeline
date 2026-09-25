"""Verdicts the index reaches from the data alone, recorded so that no one has to give them.

Two kinds of single-pulse candidate are settled by what the rest of the
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

A periodic fold is settled too when it is a catalogued pulsar at its own
period (not a harmonic: those stay for a person) and DM, within 5 degrees: B0823+26
was folded in 92 beams of L611400, 3.3 degrees away in its other SAP, where the
pipeline's own catalogue match does not reach. Verdict 'known'.

Each is recorded once, by REVIEWER, with its evidence in the note. None is
written over a verdict already given, and a person's later verdict replaces it
(the latest counts everywhere verdicts are read).
"""
import time

from web import store
from web.indexer import COINCIDENT_BEAMS

REVIEWER = 'auto-triage'
ALL_SAPS = 3                  # a LOTAAS pointing's SAPs
ROUTES = {'fold': 'a fold at its period', 'redetection': 'the classifier redetected it',
          'rotation': 'the pulses keep its rotation'}
UNREVIEWED = 'NOT EXISTS (SELECT 1 FROM review_state.reviews v WHERE v.key=c.key)'
QUEUED = "c.kind='sp' AND c.type IN ('candidate', 'known_pulsar') AND COALESCE(c.pilot, 0)=0"


def settled(db):
    """[(key, label, note, dm)] of the unreviewed single-pulse candidates the data settle.

    db is the index with reviews.sqlite attached as review_state (web.indexer).
    """
    out = []
    for r in db.execute(f"""SELECT c.key, c.dm, k.name, k.pulsar, k.separation_deg, k.route, k.z
            FROM candidates c JOIN sp_known k ON k.key=c.key WHERE {QUEUED} AND {UNREVIEWED}"""):
        seen = ', '.join(ROUTES[route] for route in r['route'].split('+') if route in ROUTES)
        z = f' (Z = {r["z"]:.0f})' if r['z'] else ''
        name = r['name'] if r['name'] == r['pulsar'] else f"{r['name']} ({r['pulsar']})"
        out.append((r['key'], 'known', f"{name}, {r['separation_deg']:.2f} deg from this beam, at its DM; "
                    f"this observation shows it: {seen}{z}. A known pulsar seen away from its own beam.", r['dm']))
    for r in db.execute(f"""SELECT c.key, c.dm, x.beams, x.saps, x.near_zero, x.events, x.dm_min, x.dm_max
            FROM candidates c JOIN sp_coincidence x ON x.key=c.key WHERE {QUEUED} AND {UNREVIEWED}
            AND x.beams >= ? AND NOT x.consistent AND (4 * x.near_zero >= x.events OR x.saps >= ?)
            AND NOT EXISTS (SELECT 1 FROM sp_known k WHERE k.key=c.key)""", (COINCIDENT_BEAMS, ALL_SAPS)):
        where = 'all three SAPs' if r['saps'] >= ALL_SAPS else f"{r['saps']} SAP(s)"
        out.append((r['key'], 'rfi', f"Interference: the same moment in {r['beams']} beams of {where} at scattered "
                    f"DMs ({r['dm_min']:.1f}-{r['dm_max']:.1f}), {r['near_zero']} of the {r['events']} events there "
                    f"below DM 1.", r['dm']))
    return out


PERIODIC_RADIUS_DEG = 5.0


def periodic_settled(db):
    """[(key, 'known', note, dm)] of unreviewed folds at a catalogued pulsar's own period and DM."""
    from euroflash import psrcat
    pulsars = psrcat.load()
    if not pulsars:
        return []
    out, cones = [], {}
    for r in db.execute(f"""SELECT c.key, c.dm, c.period, b.ra_deg, b.dec_deg, b.tstart_mjd
            FROM candidates c JOIN periodic p ON p.key=c.key JOIN beams b ON b.dir=p.dir
            WHERE c.kind='periodic' AND c.type='periodic' AND COALESCE(c.pilot, 0)=0 AND c.period > 0
            AND c.dm IS NOT NULL AND b.ra_deg IS NOT NULL AND {UNREVIEWED}"""):
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
                            f"from this beam.", r['dm']))
                break
    return out


def record(cfg, db):
    """Record the verdicts settled() finds; returns how many."""
    rows = settled(db) + periodic_settled(db)
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
