"""Verdicts the index reaches from the data alone, recorded so that no one has to give them.

Two kinds of single-pulse candidate are settled by what the rest of the
observation shows, not by looking at the candidate:

- a pulse of a catalogued pulsar seen away from its own beam
  (indexer.derive_known): at the pulsar's DM, within 5 degrees of it, with a
  fold at its period or the pulses themselves keeping its rotation to show the
  pulsar is there. Verdict 'known'.
- an undispersed burst that reached many beams at once: the same moment in
  COINCIDENT_BEAMS or more beams at scattered DMs, a quarter or more of the
  events there below DM 1 (indexer.derive_coincidence). Verdict 'rfi'.

Each is recorded once, by REVIEWER, with its evidence in the note. None is
written over a verdict already given, and a person's later verdict replaces it
(the latest counts everywhere verdicts are read).
"""
import time

from web import store
from web.indexer import COINCIDENT_BEAMS

REVIEWER = 'auto-triage'
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
            AND x.beams >= ? AND NOT x.consistent AND 4 * x.near_zero >= x.events
            AND NOT EXISTS (SELECT 1 FROM sp_known k WHERE k.key=c.key)""", (COINCIDENT_BEAMS,)):
        out.append((r['key'], 'rfi', f"Undispersed burst: the same moment in {r['beams']} beams of {r['saps']} "
                    f"SAP(s), {r['near_zero']} of the {r['events']} events there below DM 1 "
                    f"(DM {r['dm_min']:.1f}-{r['dm_max']:.1f}).", r['dm']))
    return out


def record(cfg, db):
    """Record the verdicts settled() finds; returns how many."""
    rows = settled(db)
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
