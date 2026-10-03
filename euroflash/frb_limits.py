"""The survey's upper limit on the FRB rate at 135 MHz so far, kept day by day: what the campaign can publish if
it finds nothing.

    python -m euroflash.frb_limits [--out DIR] [--epochs ops/limits-epochs.yaml] [--haslam FITS]

For every SAP the campaign searched since the FRB configuration of 28 September 2026 (web index: beams of
non-pilot runs, complete single-pulse search):
  epoch        when its run was made (ops/limits-epochs.yaml); the injection lane's bursts of that epoch's
               releases and of the SAP's source (LT5 from the LTA, early-cycle from SPIDER) give the chance that a
               burst of a given ideal S/N and scattering reaches the review queue (frb_rate.completeness_model);
  sensitivity  Jy per channel-sample sigma at the beam centre, K = 14.8 Jy (T_rec + T_sky) / (400 K + 350 K)
               sin(elevation)^-1.39: the pulsar calibration of 2 October 2026 (SEFD 410 Jy at the zenith under
               a 350 K sky, +-25%), with the field's sky temperature from the Haslam 408 MHz map (index -2.55)
               and its elevation at mid-observation;
  field        its tied-array beams at their positions inside the station beam (frb_rate.gain_samples), for the
               length of the observation.
With N(>F) = R (F / 1 Jy ms)^alpha per sky per day, the expected detections are R times the summed exposure;
none found gives R < 3 / exposure at 95%. Limits are given for the lane's population (DM log-uniform 100-3000,
scattering log-uniform 1 ms - 3 s at 135 MHz) and for its bursts scattered less than 50 ms, at alpha -1.4 and -1.5,
with the SEFD's +-25%; for the whole campaign as well (each source's remaining SAPs exposed as those searched); and
against published rates near 135 MHz (REFERENCES) and CHIME's rate carried down in frequency. Writes latest.json,
history.jsonl (one line a run) and limits.png into --out.
"""
import argparse
import datetime as dt
import json
import math
import sqlite3
from pathlib import Path

import numpy as np

from euroflash import frb_rate

REPO = Path(__file__).resolve().parents[1]
OUT = Path('/shared/results/dkuiper/lotaas/limits')
HASLAM = Path('/shared/results/dkuiper/lotaas/catalogue/haslam408_dsds_Remazeilles2014.fits')
K_ZENITH_COLD, T_REC, T_REF, ELEVATION_INDEX = 14.8, 400.0, 350.0, 1.39
SEFD_ERROR = 0.25
RADIOMETER = math.sqrt(2 * 48828.125 * 0.007864319719374176)        # SEFD = K x this
LOFAR = (52.9153, 6.8698)                                             # latitude, longitude of the core
FLUENCES = (50.0, 100.0, 200.0, 500.0, 1000.0)                        # Jy ms
ALPHAS = (-1.4, -1.5)
# CHIME/FRB Catalog 1: 525 per sky per day above 5 Jy ms at 600 MHz, N(>F) ~ F^-1.4.
CHIME = (525.0, 5.0, 600.0, -1.4)
# Published rates near 135 MHz: (label, rate per sky per day, 95% low, 95% high, above fluence Jy ms).
REFERENCES = (
    ('MWA 139-170 MHz, one FRB (Di Pietrantonio et al. 2026, arXiv:2609.17275)', 54.0, 1.0, 302.0, 57.0),
    ('ARTEMIS 145 MHz, none, 5 ms pulses (Karastergiou et al. 2015)', None, None, 29.0, 62.0),
)
CAMPAIGN_DB = Path('/shared/results/dkuiper/lotaas/campaign/campaign-state.sqlite')
MIN_LANE_BURSTS = 200


def parse_time(text):
    return dt.datetime.fromisoformat(text.replace('Z', '+00:00')).timestamp()


def load_epochs(path):
    import yaml
    epochs = yaml.safe_load(Path(path).read_text())['epochs']
    for e in epochs:
        e['start'] = parse_time(str(e['from']))
        bad = [r for releases in (e.get('lane') or {}).values() for r in releases if not isinstance(r, str)]
        if bad:
            raise ValueError(f"epoch {e['name']}: quote the release shas {bad} (YAML reads digits as numbers)")
    return sorted(epochs, key=lambda e: e['start'])


def epoch_at(epochs, when):
    found = None
    for e in epochs:
        if e['start'] <= when:
            found = e
    return found


def sky_temperature(haslam):
    """f(ra, dec) -> sky temperature at 135 MHz (K), averaged over 2.5 degrees; None without the map."""
    if not haslam or not Path(haslam).is_file():
        return None
    from astropy.io import fits
    hdu = fits.open(haslam)[1]
    t408 = np.asarray(hdu.data.field(0)).ravel().astype(float)
    nside = int(hdu.header['NSIDE'])
    vectors = pixel_vectors(nside)
    # J2000 -> Galactic rotation.
    rot = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
                    [0.4941094279, -0.4448296300, 0.7469822445],
                    [-0.8676661490, -0.1980763734, 0.4559837762]])

    def at(ra, dec):
        a, d = math.radians(ra), math.radians(dec)
        g = rot @ np.array([math.cos(d) * math.cos(a), math.cos(d) * math.sin(a), math.sin(d)])
        disc = vectors @ g > math.cos(math.radians(2.5))
        return float(t408[disc].mean() * (135.0 / 408.0) ** -2.55)
    return at


def pixel_vectors(nside):
    """Unit vectors of every RING-ordered HEALPix pixel centre."""
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)
    p = np.arange(npix, dtype=np.int64)
    z, phi = np.empty(npix), np.empty(npix)
    north = p < ncap
    ring = (1 + np.sqrt(1 + 2 * p[north]).astype(np.int64)) // 2
    ring = np.where(2 * ring * (ring - 1) > p[north], ring - 1, ring)
    ring = np.where(2 * ring * (ring + 1) <= p[north], ring + 1, ring)
    k = p[north] + 1 - 2 * ring * (ring - 1)
    z[north] = 1 - ring ** 2 / (3.0 * nside ** 2)
    phi[north] = (k - 0.5) * math.pi / (2 * ring)
    eq = (p >= ncap) & (p < npix - ncap)
    ip = p[eq] - ncap
    ring = ip // (4 * nside) + nside
    k = ip % (4 * nside) + 1
    odd = 0.5 * (1 + ((ring + nside) & 1))
    z[eq] = (2 * nside - ring) * 2.0 / (3 * nside)
    phi[eq] = (k - odd) * math.pi / (2 * nside)
    south = p >= npix - ncap
    ip = npix - p[south]
    ring = (1 + np.sqrt(2 * ip - 1).astype(np.int64)) // 2
    ring = np.where(2 * ring * (ring - 1) >= ip, ring - 1, ring)
    ring = np.where(2 * ring * (ring + 1) < ip, ring + 1, ring)
    k = 4 * ring + 1 - (ip - 2 * ring * (ring - 1))
    z[south] = -1 + ring ** 2 / (3.0 * nside ** 2)
    phi[south] = (k - 0.5) * math.pi / (2 * ring)
    s = np.sqrt(1 - z * z)
    return np.stack([s * np.cos(phi), s * np.sin(phi), z], axis=1)


def elevation(ra, dec, mjd):
    t = (mjd - 51544.5) / 36525.0
    gmst = (280.46061837 + 360.98564736629 * (mjd - 51544.5) + 0.000387933 * t * t) % 360
    ha = math.radians((gmst + LOFAR[1] - ra) % 360)
    d, la = math.radians(dec), math.radians(LOFAR[0])
    return math.degrees(math.asin(math.sin(d) * math.sin(la) + math.cos(d) * math.cos(la) * math.cos(ha)))


def jy_per_sigma(t_sky, elevation_deg):
    t_sky = T_REF if t_sky is None else t_sky
    return K_ZENITH_COLD * (T_REC + t_sky) / (T_REC + T_REF) / math.sin(math.radians(max(elevation_deg, 20.0))) ** ELEVATION_INDEX


def searched_saps(web_db, since):
    """[{observation, sap, source, made, beams: {beam: (ra, dec)}, mjd, hours}] of the non-pilot SAPs whose single-pulse
    search completed in a run made since `since` (a SAP searched twice counts once, by its latest run)."""
    db = sqlite3.connect(f'file:{web_db}?mode=ro', uri=True)
    sources = {}
    try:
        for obs, sap, source in db.execute('SELECT o.observation, o.sap, s.source FROM obs_sap o JOIN saps s ON s.key=o.key'):
            sources[(obs, sap)] = source
    except sqlite3.Error:
        pass
    saps = {}
    for obs, sap, beam, ra, dec, mjd, made, path in db.execute(
            """SELECT b.observation, b.sap, b.beam, b.ra_deg, b.dec_deg, b.tstart_mjd, r.created, b.dir FROM beams b
               JOIN runs r ON r.fingerprint=b.fingerprint WHERE COALESCE(b.pilot, 0)=0 AND b.sp_complete=1
               AND b.ra_deg IS NOT NULL AND r.created >= ?""", (since,)):
        s = saps.setdefault((obs, sap), {'observation': obs, 'sap': sap, 'made': made, 'beams': {}, 'mjd': mjd, 'dir': path,
                                         'source': sources.get((obs, sap), 'lta')})
        if made >= s['made']:
            s['made'], s['dir'], s['mjd'] = made, path, mjd or s['mjd']
        s['beams'][beam] = (ra, dec)
    db.close()
    for s in saps.values():
        try:
            meta = json.loads((Path(s['dir']) / 'metadata.json').read_text())
            s['hours'] = meta['samples_processed'] * meta['tsamp'] / 3600.0
        except (OSError, ValueError, KeyError, TypeError):
            s['hours'] = 1.0
    return list(saps.values())


def kappa_units(bursts):
    """(ideal S/N per (channel-sample sigma x s) of fluence, scattering time) of each burst recording fluence."""
    k, tau = [], []
    for b in bursts:
        if b.get('fluence_units') and b.get('sigma_mean'):
            k.append(b['snr_ideal'] * b['sigma_mean'] / b['fluence_units'])
            tau.append(b['tau135'])
    return np.array(k), np.array(tau)


def lane_bursts(lane_db):
    """{(release sha, source): bursts recording their fluence} of the injection lane."""
    db = sqlite3.connect(f'file:{lane_db}?mode=ro', uri=True)
    out = {}
    for fp, source, queued, record in db.execute('SELECT fingerprint, source, queued, record FROM bursts WHERE record IS NOT NULL'):
        b = json.loads(record)
        if b.get('fluence_units') and b.get('sigma_mean'):
            out.setdefault((fp, source), []).append(dict(b, queued=bool(queued)))
    db.close()
    return out


def replay(bursts, source, min_search_snr=None, min_votes=None):
    """The bursts as a later epoch's rules would have queued them: nothing below `min_search_snr`, and for the
    sources in `min_votes` none that FETCH judged with fewer votes (one model's probability > 0.5 is a vote)."""
    out = []
    for b in bursts:
        queued = b['queued']
        snr = (b.get('cluster') or {}).get('snr')
        if queued and min_search_snr is not None and snr is not None and snr < min_search_snr:
            queued = False
        need = (min_votes or {}).get(source)
        if queued and need and b.get('models') and sum(p > 0.5 for p in b['models'].values()) < need:
            queued = False
        out.append(dict(b, queued=queued))
    return out


def epoch_lane(epochs, lane, epoch, source):
    """(bursts, where they came from) standing for `source` in `epoch`."""
    by_name = {e['name']: e for e in epochs}
    e = by_name[epoch]
    releases = (e.get('lane') or {}).get(source, [])
    own = [b for (fp, src), bs in lane.items() if src == source and any(fp.startswith(r) for r in releases) for b in bs]
    stand_in = e.get('stand_in')
    if len(own) >= MIN_LANE_BURSTS or not stand_in:
        return own, f"{epoch} ({', '.join(releases)})"
    bursts, came = epoch_lane(epochs, lane, stand_in['epoch'], source)
    return replay(bursts, source, stand_in.get('min_search_snr'), stand_in.get('min_votes')), f'{came} replayed as {epoch}'


def unit_exposure(bursts, alpha, scattered_below=None):
    """Mean over the bursts of the detections per unit N(>1 Jy ms) at unit gain and K = 1 Jy per sigma, and the
    completeness model it used."""
    if scattered_below is not None:
        bursts = [b for b in bursts if b['tau135'] < scattered_below]
    model = frb_rate.completeness_model(bursts)
    k, tau = kappa_units(bursts)
    if not len(k) or not model:
        return None, model
    return float(frb_rate.detected_per_unit(model, alpha, tau, k).mean()), model


def campaign_saps(path):
    """{source: SAPs the campaign means to search} (every state but 'excluded'), {} without its database."""
    try:
        db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        rows = db.execute("SELECT source, count(*) FROM saps WHERE state != 'excluded' GROUP BY source").fetchall()
        db.close()
        return dict(rows)
    except sqlite3.Error:
        return {}


def compute(web_db, lane_db, epochs, haslam, campaign_db=None, tab_fwhm=0.40, station_fwhm=4.3, seed=0):
    rng = np.random.default_rng(seed)
    since = epochs[0]['start']
    saps = searched_saps(web_db, since)
    lane = lane_bursts(lane_db)
    tsky = sky_temperature(haslam)
    fields = []
    for s in saps:
        e = epoch_at(epochs, s['made'])
        coherent = {b: v for b, v in s['beams'].items() if b != 12}
        if e is None or not coherent:
            continue
        positions = np.array(list(coherent.values()))
        core = np.array([v for b, v in coherent.items() if b >= 13] or list(coherent.values())).mean(axis=0)
        t = tsky(*core) if tsky else None
        el = elevation(core[0], core[1], (s['mjd'] or 57500.0) + s['hours'] / 48.0)
        gains, area = frb_rate.gain_samples(positions, core, tab_fwhm, station_fwhm, rng)
        fields.append(dict(s, epoch=e['name'], t_sky=t, elevation=el, k=jy_per_sigma(t, el), gains=gains, area=area,
                           tied_beams=len(coherent)))
    result = {'time': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), 'saps': len(fields),
              'beams': int(sum(f['tied_beams'] for f in fields)),
              'hours': round(sum(f['hours'] for f in fields), 1),
              'by_source': {}, 'by_epoch': {}, 'sefd_jy': {}, 'limits': [], 'expected': [], 'references': [],
              'campaign_saps': {}}
    for f in fields:
        result['by_source'][f['source']] = result['by_source'].get(f['source'], 0) + 1
        result['by_epoch'][f['epoch']] = result['by_epoch'].get(f['epoch'], 0) + 1
    planned = campaign_saps(campaign_db) if campaign_db else {}
    result['campaign_saps'] = planned
    ks = np.array([f['k'] for f in fields]) if fields else np.array([K_ZENITH_COLD])
    result['sefd_jy'] = {q: round(float(np.percentile(ks, p)) * RADIOMETER) for q, p in (('p10', 10), ('median', 50), ('p90', 90))}
    sky_deg2 = frb_rate.SKY_DEG2
    completeness_used = {}
    for alpha in ALPHAS:
        for label, below in (('lane mix of scattering (1 ms - 3 s)', None), ('scattered < 50 ms', 0.05)):
            units, total, by_source = {}, 0.0, {}
            for f in fields:
                key = (f['epoch'], f['source'])
                if key not in units:
                    bursts, came = epoch_lane(epochs, lane, *key)
                    units[key] = unit_exposure(bursts, alpha, below)[0]
                    completeness_used[f'{f["epoch"]}/{f["source"]}'] = {
                        'from': came, 'bursts': len(bursts), 'queued': round(sum(b['queued'] for b in bursts) / max(len(bursts), 1), 3)}
                if units[key] is None:
                    continue
                part = (f['area'] / sky_deg2 * f['hours'] / 24.0 * units[key] * f['k'] ** alpha
                        * float((f['gains'] ** -alpha).mean()))
                total += part
                by_source[f['source']] = by_source.get(f['source'], 0.0) + part
            # The whole campaign, each source's SAPs as exposed on average as those searched so far.
            projected = sum(by_source[s] / result['by_source'][s] * planned.get(s, result['by_source'][s])
                            for s in by_source)
            for scale, which in ((1.0, 'nominal'), (1 + SEFD_ERROR, 'sefd_high'), (1 - SEFD_ERROR, 'sefd_low')):
                exposure, whole = total * scale ** alpha, projected * scale ** alpha
                result['limits'].append({'alpha': alpha, 'population': label, 'sefd': which,
                                         'exposure_sky_days': exposure, 'projected_exposure_sky_days': whole,
                                         'r95': {f'{fl:g}': 3.0 / exposure * fl ** alpha if exposure > 0 else None
                                                 for fl in FLUENCES},
                                         'r95_projected': {f'{fl:g}': 3.0 / whole * fl ** alpha if whole > 0 else None
                                                           for fl in FLUENCES}})
                if which == 'nominal' and below is None:
                    for name, rate, low, high, fluence in REFERENCES:
                        n1 = (lambda r: r * fluence ** -alpha if r is not None else None)
                        result['references'].append({
                            'alpha': alpha, 'reference': name, 'fluence': fluence, 'rate': rate, 'rate_95': [low, high],
                            'ours_r95': 3.0 / exposure * fluence ** alpha if exposure > 0 else None,
                            'ours_r95_projected': 3.0 / whole * fluence ** alpha if whole > 0 else None,
                            'expected_now': n1(rate) and n1(rate) * exposure, 'expected_projected': n1(rate) and n1(rate) * whole,
                            'expected_projected_at_95_high': n1(high) * whole})
            if alpha == CHIME[3] and below is None:
                rate, f_ref, nu_ref, index = CHIME
                for beta in (0.0, -1.0, -1.5):
                    # N(>1 Jy ms) at 135 MHz for a CHIME-like population of spectral index beta.
                    r1 = rate * (1.0 / (f_ref * (135.0 / nu_ref) ** beta)) ** alpha
                    n, whole = r1 * total, r1 * projected
                    result['expected'].append({'spectral_index': beta, 'expected': n, 'p_at_least_one': 1 - math.exp(-n),
                                               'expected_projected': whole, 'p_projected': 1 - math.exp(-whole)})
    result['completeness_from'] = completeness_used
    return result, fields


def plot(history, latest, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    colours = {'lane': 'C0', 'scattered': 'C1'}
    for entry in latest['limits']:
        if entry['sefd'] != 'nominal' or entry['alpha'] != -1.4:
            continue
        colour = colours['lane' if 'mix' in entry['population'] else 'scattered']
        fl = [float(k) for k in entry['r95']]
        ax[0].loglog(fl, list(entry['r95'].values()), '-o', ms=3, color=colour, label=f"so far, {entry['population']}")
        ax[0].loglog(fl, list(entry['r95_projected'].values()), ':', color=colour, label='whole campaign, projected')
        band = [x for x in latest['limits'] if x['alpha'] == -1.4 and x['population'] == entry['population']]
        lo = min(band, key=lambda x: x['r95']['100'])['r95'].values()
        hi = max(band, key=lambda x: x['r95']['100'])['r95'].values()
        ax[0].fill_between(fl, list(lo), list(hi), color=colour, alpha=0.12, lw=0)
    for name, rate, low, high, fluence in REFERENCES:
        short = name.split(' (')[0]
        if rate is not None:
            ax[0].errorbar([fluence], [rate], yerr=[[rate - low], [high - rate]], fmt='s', color='k', ms=5, capsize=3,
                           label=short)
            span = np.array([fluence, max(FLUENCES)])
            ax[0].fill_between(span, low * (span / fluence) ** -1.4, high * (span / fluence) ** -1.4, color='0.5',
                               alpha=0.15, lw=0)
            ax[0].plot(span, rate * (span / fluence) ** -1.4, '--', color='0.4', lw=1)
        else:
            ax[0].plot([fluence], [high], 'v', color='0.4', ms=7, label=short)
    ax[0].set_xlim(40, 1300)
    ax[0].set_xlabel('fluence at 135 MHz (Jy ms)'); ax[0].set_ylabel('R(>F) per sky per day, 95% upper limit')
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3, which='both')
    ax[0].set_title(f"{latest['saps']} SAPs, {latest['hours']:.0f} h to {latest['time'][:10]}; alpha -1.4, "
                    f"SEFD +-{SEFD_ERROR:.0%} shaded", fontsize=9)
    times, values = [], []
    for h in history:
        e = next((x for x in h['limits'] if x['alpha'] == -1.4 and x['sefd'] == 'nominal' and 'mix' in x['population']), None)
        if e and e['r95'].get('100'):
            times.append(dt.datetime.fromisoformat(h['time'])); values.append(e['r95']['100'])
    ax[1].semilogy(times, values, 'o-', ms=3)
    ax[1].set_ylabel('R(>100 Jy ms), 95% upper limit'); ax[1].set_title('as the survey grows (alpha -1.4, lane mix)', fontsize=9)
    if times:
        ax[1].set_xlim(min(times) - dt.timedelta(days=1), max(times) + dt.timedelta(days=1))
    ax[1].tick_params(axis='x', labelrotation=30)
    ax[1].grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--web', type=Path, default=frb_rate.WEB_DB)
    p.add_argument('--lane', type=Path, default=frb_rate.LANE_DB)
    p.add_argument('--epochs', type=Path, default=REPO / 'ops' / 'limits-epochs.yaml')
    p.add_argument('--haslam', type=Path, default=HASLAM)
    p.add_argument('--campaign', type=Path, default=CAMPAIGN_DB)
    p.add_argument('--out', type=Path, default=OUT)
    a = p.parse_args(argv)
    result, _ = compute(a.web, a.lane, load_epochs(a.epochs), a.haslam, a.campaign)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'latest.json').write_text(json.dumps(result, indent=1) + '\n')
    with (a.out / 'history.jsonl').open('a') as stream:
        stream.write(json.dumps(result) + '\n')
    history = [json.loads(line) for line in (a.out / 'history.jsonl').read_text().splitlines() if line.strip()]
    plot(history, result, a.out / 'limits.png')
    nominal = next(x for x in result['limits'] if x['alpha'] == -1.4 and x['sefd'] == 'nominal' and 'mix' in x['population'])
    print(f"{result['saps']} SAPs, {result['hours']} h, SEFD median {result['sefd_jy']['median']} Jy: "
          f"R(>100 Jy ms) < {nominal['r95']['100']:.3g} per sky per day (95%, alpha -1.4, lane mix); "
          f"whole campaign {nominal['r95_projected']['100']:.3g}")


if __name__ == '__main__':
    main()
