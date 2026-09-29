"""Verdicts the index reaches from the data alone, recorded so that no one has to give them.

Three kinds of single-pulse candidate are settled by what the rest of the
observation shows, not by looking at the candidate:

- a pulse of a catalogued pulsar seen away from its own beam
  (indexer.derive_known): at the pulsar's DM, within 5 degrees of it, with a
  fold at its period or the pulses themselves keeping its rotation to show the
  pulsar is there. Verdict 'known'.
- an undispersed burst that reached many beams at once: the same moment in
  COINCIDENT_BEAMS or more beams at scattered DMs (indexer.derive_coincidence),
  with a quarter or more of the events there below DM 1. Verdict 'rfi'.
- the same moment in COINCIDENT_BEAMS or more beams of all three SAPs, which
  point about 4 degrees apart, whatever the DMs: no one position on the sky
  is in all of them, so it came in through the station beam's sidelobes
  (a burst in L611408 reached 92 beams at S/N up to 68; on 27 September a
  0.75 s band-limited burst reached 59 beams of L556848 at one DM, and a
  train of sparks 171 beams of L603714 at DM ~3). Known pulsars are
  attributed first. Verdict 'rfi'.
- an undispersed burst in the candidate's own beam: a burst of events there
  at one moment at scattered DMs, none standing out (indexer.sweeps), where a
  pulse would peak at its DM and fall away. Verdict 'rfi'.
- a FETCH positive below S/N 8, the gate FETCH has had since 28 September
  2026, whose own data show it under LOCAL_CLEAR. Verdict 'noise'.
- the data dropping out beside it: samples within its sweep where most
  channels fall below a fifth of their level (lost 2-bit rows, recording
  gaps; the flatfield fills them since 29 September 2026). Verdict 'rfi'.
- a candidate as strong at DM 0 as at its own DM, where a real pulse of its
  width would lose half its S/N or more (dynspec.Snippet.dispersion_evidence,
  without the zero-DM filter): an undispersed burst or step. Verdict 'rfi'.
- a step: at its DM, a dip beside it half as deep as its height or more,
  below DM 100 (a flatfielded pulse dips a third at most). Verdict 'rfi'.
- a wide event (32 samples or more) below DM 100 in an LT5_004 beam, within
  a second and its width of a 24.16 s block boundary, where each channel's
  level steps, and no more dispersed than a step. Verdict 'rfi'.
- a dispersed event at the DM of a catalogued pulsar within 5 degrees
  (max(1, 2%)): its pulses through the sidelobes, as the Crab's giant pulses
  in 22 beams of L549917. Verdict 'known'.
- the classifier's redetection of a catalogued pulsar in its own beam, at the
  pulsar's DM. Verdict 'known'.
- a candidate the search put at S/N 7 or more whose own data show under
  LOCAL_MIN at its time, DM and width (the review page's local S/N,
  indexer.measure_local). The search fills each 7.9 s channel block its RFI
  mask flags at a level that can lift the dedispersed baseline for the whole
  block; an event on such a step reached S/N 7 with no pulse there (27
  September: 7.09 for L543457 SAP000 B013 DM 3.0, 1.4 without the mask, and
  as many such events at negative DMs as at positive ones). Verdict 'noise'.

A periodic fold is settled too when it is a catalogued pulsar at its own
period, or a harmonic or multiple of it (psrcat.match), and DM, within 5 degrees: B0823+26
was folded in 92 beams of L611400, 3.3 degrees away in its other SAP, where the
pipeline's own catalogue match does not reach. Verdict 'known'. So is a fold
that repeats a catalogued fold of its own beam at least five times stronger
(euroflash.findings.relative): at a harmonic or subharmonic of it at any DM,
or at an alias of a harmonic above the Nyquist frequency of a downsampled
trial at the DM that alias takes (B2217+47 in L543473 SAP001, 27 September).

Each verdict is recorded by REVIEWER with its evidence in the note. A person's
verdict is never written over, and a person's later verdict replaces this one
(the latest counts everywhere verdicts are read). This triage's own verdict is
replaced when its evidence changes: two classifier 'redetections' first called
'known' were bursts in their own beams with no pulse at the pulsar's DM.
"""
import time

from web import store
from web.indexer import CHANCE_MAX, COINCIDENT_BEAMS

REVIEWER = 'auto-triage'
ALL_SAPS = 3                  # a LOTAAS pointing's SAPs
LOCAL_MIN = 4.0               # local S/N below which a search S/N of SEARCH_MIN or more had nothing there
SEARCH_MIN = 7.0
# Or below this fraction of the search's S/N. Pulses injected into a pilot beam of 27 September
# 2026 read 0.8-1.4 times their search S/N on the review page; interference the search saw
# through read a quarter to a half.
LOCAL_FRACTION = 0.6
# ... unless its own data show it at this S/N or more. None of 103 pilot positives the rule removed
# read above 5.7, while injected FRB-like bursts at DM 300-2500 read 0.4-0.55 times their search S/N
# (10-11 against 20-27) when the review page's snippet, at the search's resolution, splits a
# one-sample pulse (benchmarks/frb-injection-2026-09-28).
LOCAL_CLEAR = 6.0
# Not dispersed: at least this fraction of its S/N at DM 0 where a real pulse of its width would keep at most
# UNDISPERSED_EXPECTED. Of 59 catalogued pulsars' pulses with page S/N 6 or more (DM 4.8-74) none: they keep
# 3-76% at DM 0; it settled 334 of 1,023 open candidates (29 September 2026).
# FETCH positives found before its gate rose to S/N 8 (28 September 2026), below it, whose own data show
# them under LOCAL_CLEAR: none of 24 such at DM 100-365 showed a pulse by eye (29 September).
FETCH_GATE = 8.0
UNDISPERSED_RATIO = 0.8
# A step: a dip at its DM at least this fraction of its height, within three widths. Of 59 catalogued pulsars'
# pulses none dipped below -0.32; of the 477 FETCH positives still open after the rules above, 326 did
# (29 September 2026). Only below BIPOLAR_MAX_DM, where no FRB is: the dispersed route judges those above.
BIPOLAR_RATIO = -0.5
BIPOLAR_MAX_DM = 100.0
UNDISPERSED_EXPECTED = 0.5
ROUTES = {'fold': 'a fold at its period', 'redetection': 'the classifier redetected it',
          'rotation': 'the pulses keep its rotation'}
LATEST = '(SELECT {0} FROM review_state.reviews v WHERE v.key=c.key ORDER BY created DESC LIMIT 1)'
# No verdict yet, or only this triage's own.
OPEN = f"COALESCE({LATEST.format('reviewer')}, '{REVIEWER}') = '{REVIEWER}'"
QUEUED = "c.kind='sp' AND c.type IN ('candidate', 'dispersed', 'known_pulsar') AND COALESCE(c.pilot, 0)=0"
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
            x.consistent, x.expected, {earlier} AS earlier FROM candidates c JOIN sp_coincidence x ON x.key=c.key
            WHERE {QUEUED} AND {OPEN} AND x.beams >= ? AND COALESCE(x.chance, 0) < ?
            AND ((NOT x.consistent AND 4 * x.near_zero >= x.events) OR x.saps >= ?) AND {NOT_KNOWN}""",
            (COINCIDENT_BEAMS, CHANCE_MAX, ALL_SAPS)):
        where = 'all three SAPs' if r['saps'] >= ALL_SAPS else f"{r['saps']} SAP(s)"
        spread = 'at one DM' if r['consistent'] else 'at scattered DMs'
        why = (' No one position on the sky is in beams of all three SAPs.' if r['saps'] >= ALL_SAPS else '')
        chance = f" ({r['expected']:.0f} by chance)" if r['expected'] is not None else ''
        out.setdefault(r['key'], (r['key'], 'rfi', f"Interference: the same moment in {r['beams']} beams{chance} of "
                                  f"{where} {spread} ({r['dm_min']:.1f}-{r['dm_max']:.1f}), {r['near_zero']} of the "
                                  f"{r['events']} events there below DM 1.{why}", r['dm'], r['earlier']))
    for r in db.execute(f"""SELECT c.key, c.dm, s.events, s.expected, s.dm_min, s.dm_max, s.peak_ratio,
            {earlier} AS earlier FROM candidates c JOIN sp_sweep s ON s.key=c.key WHERE {QUEUED} AND {OPEN}
            AND {NOT_KNOWN}"""):
        out.setdefault(r['key'], (r['key'], 'rfi', f"Undispersed burst in this beam: {r['events']} events at one "
                                  f"moment ({r['expected']:.1f} expected from the beam's own rate) at DM "
                                  f"{r['dm_min']:.1f}-{r['dm_max']:.1f}, the strongest {r['peak_ratio']:.2f} times the "
                                  f"median S/N: no DM stands out as a pulse's would.", r['dm'], r['earlier']))
    for r in db.execute(f"""SELECT c.key, c.dm, c.snr, l.local_snr, {earlier} AS earlier FROM candidates c
            JOIN sp_local_snr l ON l.key=c.key WHERE {QUEUED} AND {OPEN} AND l.local_snr IS NOT NULL
            AND (l.local_snr < ? OR (l.local_snr < ? * c.snr AND l.local_snr < ?)) AND c.snr >= ? AND {NOT_KNOWN}""",
            (LOCAL_MIN, LOCAL_FRACTION, LOCAL_CLEAR, SEARCH_MIN)):
        out.setdefault(r['key'], (r['key'], 'noise', f"Search S/N {r['snr']:.1f}, but S/N {r['local_snr']:.1f} on its "
                                  f"own data at its time, DM and width, measured as the search measures (the local "
                                  f"S/N of the review page): no pulse there. A pulse reads 0.8-1.4 times its search "
                                  f"S/N there; what the search found was the rest of interference it saw through.",
                                  r['dm'], r['earlier']))
    for r in db.execute(f"""SELECT c.key, c.dm, c.snr, l.local_snr, {earlier} AS earlier FROM candidates c
            JOIN sp_local_snr l ON l.key=c.key WHERE {QUEUED} AND {OPEN} AND c.type='candidate' AND c.snr < ?
            AND l.local_snr IS NOT NULL AND l.local_snr < ? AND {NOT_KNOWN}""", (FETCH_GATE, LOCAL_CLEAR)):
        out.setdefault(r['key'], (r['key'], 'noise', f"Search S/N {r['snr']:.1f}, below the S/N {FETCH_GATE:g} FETCH "
                                  f"has been asked about since 28 September 2026, and S/N {r['local_snr']:.1f} on its "
                                  f"own data: at this S/N FETCH's positives were noise.", r['dm'], r['earlier']))
    for r in db.execute(f"""SELECT c.key, c.dm, d.dropout, {earlier} AS earlier FROM candidates c
            JOIN sp_dispersion d ON d.key=c.key WHERE {QUEUED} AND {OPEN} AND d.dropout > 0 AND {NOT_KNOWN}"""):
        out.setdefault(r['key'], (r['key'], 'rfi', f"The data drop out beside it: {r['dropout']} samples within its "
                                  f"sweep where most channels fall below a fifth of their level, which no signal "
                                  f"from the sky does (lost 2-bit rows or a recording gap, flatfielded before 29 "
                                  f"September 2026). What the search found is the edge of the gap.", r['dm'],
                                  r['earlier']))
    for r in db.execute(f"""SELECT c.key, c.dm, d.snr_dm, d.snr_zero, d.expected_zero, {earlier} AS earlier
            FROM candidates c JOIN sp_dispersion d ON d.key=c.key WHERE {QUEUED} AND {OPEN} AND d.snr_dm > 0
            AND d.snr_zero >= ? * d.snr_dm AND d.expected_zero <= ? AND {NOT_KNOWN}""",
            (UNDISPERSED_RATIO, UNDISPERSED_EXPECTED)):
        out.setdefault(r['key'], (r['key'], 'rfi', f"Not dispersed: S/N {r['snr_zero']:.1f} at DM 0 against "
                                  f"{r['snr_dm']:.1f} at its DM, where a pulse of its width would keep "
                                  f"{100 * r['expected_zero']:.0f}% at DM 0. An undispersed burst or step the "
                                  f"dedispersion smeared into this DM.", r['dm'], r['earlier']))
    for r in db.execute(f"""SELECT c.key, c.dm, d.snr_dm, d.snr_min, {earlier} AS earlier FROM candidates c
            JOIN sp_dispersion d ON d.key=c.key WHERE {QUEUED} AND {OPEN} AND c.dm < ? AND d.snr_dm > 0
            AND d.snr_min <= ? * d.snr_dm AND {NOT_KNOWN}""", (BIPOLAR_MAX_DM, BIPOLAR_RATIO)):
        out.setdefault(r['key'], (r['key'], 'rfi', f"A step, not a pulse: S/N {r['snr_dm']:.1f} with a dip to "
                                  f"{r['snr_min']:.1f} beside it at its DM, where a pulse dips to a third of its "
                                  f"height at most. A level stepping down and up again (2-bit requantisation).",
                                  r['dm'], r['earlier']))
    for r in db.execute(f"""SELECT c.key, c.dm, c.item, c.time, c.width, d.snr_dm, d.snr_zero, {earlier} AS earlier
            FROM candidates c JOIN sp_dispersion d ON d.key=c.key WHERE {QUEUED} AND {OPEN} AND c.dm < ?
            AND c.width >= ? AND d.snr_dm > 0 AND d.snr_zero >= ? * d.snr_dm AND c.time IS NOT NULL AND {NOT_KNOWN}""",
            (BLOCK_MAX_DM, BLOCK_MIN_WIDTH, UNDISPERSED_RATIO)):
        if not _lt5(r['item']):
            continue
        offset = r['time'] - round(r['time'] / LT5_BLOCK_SECONDS) * LT5_BLOCK_SECONDS
        if abs(offset) <= r['width'] * LOTAAS_TSAMP + 1.0:
            out.setdefault(r['key'], (r['key'], 'rfi', f"At a {LT5_BLOCK_SECONDS:.2f} s block boundary ({offset:+.2f} s), "
                                      f"where each channel's level steps in LT5_004 beams flatfielded before 29 "
                                      f"September 2026, {r['width']} samples wide and no more dispersed than a step "
                                      f"(S/N {r['snr_zero']:.1f} at DM 0, {r['snr_dm']:.1f} at its DM).", r['dm'],
                                      r['earlier']))
    for key, label, note, dm, before in _at_a_nearby_pulsars_dm(db, earlier):
        out.setdefault(key, (key, label, note, dm, before))
    catalogue = _catalogue_dms()
    for r in db.execute(f"""SELECT c.key, c.dm, c.pulsar, {earlier} AS earlier FROM candidates c
            WHERE {QUEUED} AND {OPEN} AND c.type='known_pulsar' AND c.pulsar IS NOT NULL"""):
        dm = catalogue.get(r['pulsar'])
        if dm is not None and abs(r['dm'] - dm) <= max(2.0, 0.05 * dm):
            out.setdefault(r['key'], (r['key'], 'known', f"{r['pulsar']} in its own beam, at its DM ({dm:g}): the "
                                      f"classifier's redetection of a catalogued pulsar.", r['dm'], r['earlier']))
    return list(out.values())


NEARBY_RADIUS_DEG = 5.0
# LT5_004 beams step in each channel every 3072 samples (euroflash.campaign.LT5_BLOCK); the flatfield levels
# them since 29 September 2026. Wide events at such a boundary, no more dispersed than a step, are the step:
# 14 wide FETCH positives at DM 6-100 reviewed that day all lay within a second of one.
LOTAAS_TSAMP = 0.00786432
LT5_BLOCK_SECONDS = 3072 * LOTAAS_TSAMP
BLOCK_MAX_DM = 100.0
BLOCK_MIN_WIDTH = 32


def _lt5(item):
    """LT5_004 observations are numbered from L400000 on; early-cycle ones below."""
    try:
        return int(item.split('_L', 1)[1].split('_', 1)[0]) >= 400000
    except (IndexError, ValueError):
        return False
# A pulsar of NEARBY_BRIGHT_MJY or more at 135 MHz (psrcat.flux_at) reaches further, and its pulses, bright
# over a range of trial DMs, are clustered further from its DM: B0329+54 (26.76) at DM 24.6-27.9 in L528445
# and L528465, up to 5.3 degrees away (29 September 2026).
NEARBY_BRIGHT_RADIUS_DEG = 10.0
# In the pulsar's own beam its pulses are bright too: B1112+50 (DM 9.19) at DM 10.2-10.5 in L169690, 1.66 s
# (its period) apart. Within this separation, max(2, 10%).
OWN_BEAM_DEG = 0.25
NEARBY_BRIGHT_MJY = 1000.0


def _bright(pulsar):
    from euroflash import psrcat
    mjy, _ = psrcat.flux_at(pulsar)
    return mjy is not None and mjy >= NEARBY_BRIGHT_MJY


def _at_a_nearby_pulsars_dm(db, earlier):
    """[(key, 'known', note, dm, earlier)] of open queued candidates at the DM of a catalogued pulsar within
    NEARBY_RADIUS_DEG of their beam, within max(1, 2%), or of a bright one within NEARBY_BRIGHT_RADIUS_DEG,
    within max(3, 10%): its pulses through the station beam's sidelobes. Giant
    pulses of the Crab (DM 56.77) reached 22 beams of L549917, 4-4.8 degrees away, as sporadic single pulses
    that neither a fold nor a rotation sequence attributes (29 September 2026)."""
    from euroflash import psrcat
    pulsars = psrcat.load()
    if not pulsars:
        return []
    out, cones = [], {}
    for r in db.execute(f"""SELECT c.key, c.dm, b.ra_deg, b.dec_deg, {earlier} AS earlier FROM candidates c
            JOIN beams b ON b.dir=c.dir WHERE {QUEUED} AND {OPEN} AND {NOT_KNOWN} AND b.ra_deg IS NOT NULL"""):
        place = (round(r['ra_deg'], 1), round(r['dec_deg'], 1))
        if place not in cones:
            cones[place] = [p for p in psrcat.cone(pulsars, r['ra_deg'], r['dec_deg'], NEARBY_BRIGHT_RADIUS_DEG)
                            if p['separation_deg'] <= NEARBY_RADIUS_DEG or _bright(p)]
        for pulsar in cones[place]:
            wide = _bright(pulsar) or pulsar['separation_deg'] <= OWN_BEAM_DEG
            tolerance = max(3.0 if _bright(pulsar) else 2.0, 0.1 * pulsar['dm']) if wide else max(1.0, 0.02 * pulsar['dm'])
            if pulsar.get('dm') and abs(r['dm'] - pulsar['dm']) <= tolerance:
                name = pulsar.get('bname') or pulsar['name']
                name = name if name == pulsar['name'] else f"{name} ({pulsar['name']})"
                out.append((r['key'], 'known', f"At the DM of {name} ({pulsar['dm']:g}), "
                            f"{pulsar['separation_deg']:.2f} deg from this beam: its pulses through the "
                            f"sidelobes.", r['dm'], r['earlier']))
                break
    return out


def _catalogue_dms():
    """{pulsar name (J and B): catalogue DM}."""
    from euroflash import psrcat
    out = {}
    for p in psrcat.load() or []:
        for name in (p.get('name'), p.get('bname')):
            if name and p.get('dm') is not None:
                out[name] = float(p['dm'])
    return out


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
        found = psrcat.match(r['period'], r['dm'], cones[place], r['tstart_mjd'], dm_tolerance=(2.0, 0.05))
        if found:
            pulsar, relation = found
            name = pulsar.get('bname') or pulsar['name']
            name = name if name == pulsar['name'] else f"{name} ({pulsar['name']})"
            at = 'its own period' if relation == '1/1' else f'{relation} of its period'
            out.append((r['key'], 'known', f"{name} at {at} and its DM, {pulsar['separation_deg']:.2f} deg "
                        f"from this beam.", r['dm'], r['earlier']))
    return out


def periodic_relatives(db):
    """[(key, 'known', note, dm, earlier)] of open folds that repeat a catalogued fold of their beam at least
    RELATIVE_RATIO times stronger (euroflash.findings.relative)."""
    import json
    from euroflash.findings import LOTAAS_TSAMP, RELATIVE_RATIO, relative
    parents = {}
    for r in db.execute("""SELECT key, dir, period, dm, statistic, catalogue FROM periodic
            WHERE catalogue IS NOT NULL AND catalogue != '' AND period > 0 AND statistic > 0"""):
        parents.setdefault(r['dir'], []).append(r)
    out = []
    if not parents:
        return out
    for r in db.execute(f"""SELECT c.key, c.dm, c.period, c.statistic, p.dir, p.row, b.tsamp,
            {LATEST.format('label')} AS earlier
            FROM candidates c JOIN periodic p ON p.key=c.key JOIN beams b ON b.dir=p.dir
            WHERE c.kind='periodic' AND c.type='periodic' AND COALESCE(c.pilot, 0)=0 AND c.period > 0
            AND c.dm IS NOT NULL AND {OPEN}"""):
        for parent in parents.get(r['dir'], ()):
            if parent['key'] == r['key'] or parent['statistic'] < RELATIVE_RATIO * (r['statistic'] or 0):
                continue
            try:
                resolution = json.loads(r['row'] or '{}').get('frequency_resolution_hz') or 1 / 3600.
            except ValueError:
                resolution = 1 / 3600.
            how = relative(1 / r['period'], r['dm'], 1 / parent['period'], parent['dm'],
                           r['tsamp'] or LOTAAS_TSAMP, resolution)
            if how:
                out.append((r['key'], 'known', f"{parent['catalogue']} seen again: {how} of its fold at P "
                            f"{parent['period']:.6f} s, DM {parent['dm']:g} (statistic {parent['statistic']:.0f}) in this "
                            f"beam.", r['dm'], r['earlier']))
                break
    return out


def stale(db, current):
    """Keys of queued single-pulse candidates whose latest verdict is this triage's own and
    which no rule gives this pass (current); none when the pass derived no candidates.

    Chance coincidences took 3,459 'rfi' verdicts in the first night of the search of 27
    September 2026, whose far busier beams put some 24 others at any moment by chance.
    """
    if db.execute("SELECT 1 FROM candidates WHERE kind='sp' LIMIT 1").fetchone() is None:
        return []
    latest = LATEST.format('reviewer')
    return [r['key'] for r in db.execute(f'SELECT c.key FROM candidates c WHERE {QUEUED} AND {latest} = ?',
                                         (REVIEWER,)) if r['key'] not in current]


def record(cfg, db):
    """Record the verdicts settled() finds that differ from the latest, and take back this
    triage's own where no rule gives them any more; returns how many changed."""
    first = {}
    for key, label, note, dm, earlier in settled(db) + periodic_settled(db) + periodic_relatives(db):
        first.setdefault(key, (key, label, note, dm, earlier))    # one verdict per candidate and pass
    rows = [(key, label, note if earlier is None else f"{note} (Replaces this triage's earlier '{earlier}'.)", dm)
            for key, label, note, dm, earlier in first.values() if label != earlier]
    withdrawn = stale(db, first)
    if rows or withdrawn:
        reviews = store.reviews(cfg)
        try:
            now = time.time()
            with reviews:
                reviews.executemany('INSERT INTO reviews(key,reviewer,label,note,dm,created) VALUES (?,?,?,?,?,?)',
                                    [(key, REVIEWER, label, note, dm, now) for key, label, note, dm in rows])
                # Taken back, not overwritten: the candidate is open again, and the record is kept.
                for key in withdrawn:
                    reviews.execute("""INSERT INTO triage_withdrawals(key,label,note,created,withdrawn,why)
                        SELECT key, label, note, created, ?, 'no rule gives it any more' FROM reviews
                        WHERE key=? AND reviewer=?""", (now, key, REVIEWER))
                    reviews.execute('DELETE FROM reviews WHERE key=? AND reviewer=?', (key, REVIEWER))
        finally:
            reviews.close()
    return len(rows) + len(withdrawn)
