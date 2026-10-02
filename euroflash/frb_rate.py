"""What the survey's searched beams say about the FRB rate at 135 MHz, from the injection lane's completeness.

    python -m euroflash.frb_rate [--lane LANE_DB] [--web WEB_DB] [--fingerprint PREFIX] [--sefd 440 580 890]
                                 [--alpha -1.4] [--tab-fwhm 0.40] [--station-fwhm 4.3] [--hours 1.0]

Ingredients, each stated so that it can be replaced:
  completeness  C(S/N) of an injected burst reaching the review queue (euroflash.injections), a logistic in
                log S/N fitted per class of scattering time at 135 MHz, where S/N is the burst's ideal
                (radiometer) S/N in the beam it lands in;
  S/N per Jy s  each injected burst's ideal S/N per unit fluence. Its fluence in Jy s is fluence_units /
                sigma_mean (channel-sample sigma x seconds) times the Jy one sigma stands for, SEFD /
                sqrt(2 x channel width x sample time). The noise carries the scale, not the data's level,
                which holds the requantiser's offset (sigma / level is 0.025 in LT5 beams and 0.10 in the
                early-cycle ones; radiometer noise alone would give 0.036). Bursts of the population are
                drawn from the injected ones, so its scattering, widths, DMs and spectra are theirs;
  SEFD          measured on 18 pulsars folded in their tied-array beams against literature fluxes at 135 MHz
                (2 Oct 2026, benchmarks/sefd-calibration-2026-10-02): 410 Jy at the zenith under a 350 K sky,
                times (400 K + sky) / 750 K and sin(elevation)^-1.39. Over the searched SAPs that is 440, 580
                and 890 Jy at the 10th, 50th and 90th percentile, the defaults here;
  beams         the tied-array beams at their recorded positions (web.sqlite), Gaussian of TAB_FWHM, inside
                a Gaussian station beam of STATION_FWHM centred on the SAP's 61-beam core: an FRB anywhere
                in a SAP's field is seen by its best beam at that beam's relative gain;
  exposure      searched beams of the chosen code fingerprint, grouped into SAPs, HOURS each;
  population    N(>F) = R (F / 1 Jy ms)^alpha per sky per day at 135 MHz, fluence F in Jy ms.
The expected number of detections is R times the survey's effective exposure in sky x days; with none
found, R < 3.0 / exposure at 95% confidence.
"""
import argparse
import json
import math
import sqlite3
from pathlib import Path

import numpy as np

LANE_DB = Path('/shared/results/dkuiper/lotaas/injections/lane.sqlite')
WEB_DB = Path('/shared/results/dkuiper/lotaas/web/web.sqlite')
SKY_DEG2 = 4 * math.pi * (180 / math.pi) ** 2
TAU_CLASSES = (0.0, 0.05, 0.2, 0.5, 1.0, 1e9)


def logistic(x, x50, sigma, top):
    return top / (1 + np.exp(np.clip(-(x - x50) / sigma, -700, 700)))


def fit_completeness(snr, found):
    """(log10 S/N at half the plateau, width in log10 S/N, plateau) maximising the Bernoulli likelihood."""
    x, y = np.log10(np.asarray(snr, dtype=float)), np.asarray(found, dtype=float)
    best = (-np.inf, None)
    for x50 in np.linspace(0.6, 1.8, 61):
        for sigma in np.linspace(0.02, 0.4, 20):
            p = np.clip(logistic(x, x50, sigma, 1.0), 1e-9, 1 - 1e-9)
            for top in np.linspace(0.5, 1.0, 26):
                q = np.clip(top * p, 1e-9, 1 - 1e-9)
                ll = float((y * np.log(q) + (1 - y) * np.log(1 - q)).sum())
                if ll > best[0]:
                    best = (ll, (float(x50), float(sigma), float(top)))
    return best[1]


def load_bursts(lane_db, fingerprint=None):
    db = sqlite3.connect(f'file:{lane_db}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    sql, args = 'SELECT * FROM bursts', []
    if fingerprint:
        sql += ' WHERE fingerprint LIKE ?'
        args.append(fingerprint + '%')
    rows = [dict(json.loads(r['record']), queued=int(r['queued'])) for r in db.execute(sql, args)]
    db.close()
    return rows


def completeness_model(bursts, minimum=20):
    """{(tau low, tau high): (x50, sigma, plateau, n)} fitted on the injected bursts of that class."""
    model = {}
    for lo, hi in zip(TAU_CLASSES, TAU_CLASSES[1:]):
        sel = [b for b in bursts if lo <= b['tau135'] < hi]
        if len(sel) >= minimum:
            model[(lo, hi)] = fit_completeness([b['snr_ideal'] for b in sel], [b['queued'] for b in sel]) + (len(sel),)
    return model


def completeness(model, tau, snr):
    """C for bursts of scattering times `tau` at ideal S/N `snr` (arrays); 0 in classes without a fit."""
    tau, snr = np.asarray(tau, dtype=float), np.asarray(snr, dtype=float)
    out = np.zeros_like(snr)
    for (lo, hi), (x50, sigma, top, _) in model.items():
        sel = (tau >= lo) & (tau < hi)
        out[sel] = logistic(np.log10(np.maximum(snr[sel], 1e-3)), x50, sigma, top)
    return out


def snr_per_jy_s(bursts, sefd):
    """(ideal S/N per Jy s of fluence at this SEFD, scattering time) of the bursts whose truth records fluence."""
    kappa, tau = [], []
    for b in bursts:
        if b.get('fluence_units') and b.get('sigma_mean'):
            jy_per_sigma = sefd / math.sqrt(2 * b['channel_mhz'] * 1e6 * b['tsamp'])
            kappa.append(b['snr_ideal'] * b['sigma_mean'] / b['fluence_units'] / jy_per_sigma)
            tau.append(b['tau135'])
    return np.array(kappa), np.array(tau)


def sap_fields(web_db, fingerprint=None):
    """{(observation, sap): {beam: (ra, dec)}} of the SAPs searched (by that code), from the web's index."""
    db = sqlite3.connect(f'file:{web_db}?mode=ro', uri=True)
    fields = {}
    for obs, sap, beam, ra, dec, fp in db.execute('SELECT observation, sap, beam, ra_deg, dec_deg, fingerprint FROM '
                                                   'beams WHERE ra_deg IS NOT NULL AND sp_complete = 1'):
        if fingerprint and not (fp or '').startswith(fingerprint):
            continue
        fields.setdefault((obs, sap), {}).setdefault(beam, (ra, dec))
    db.close()
    return fields


def gain_samples(positions, core, tab_fwhm, station_fwhm, rng, n=2000, radius=2.5):
    """(relative gains of the best beam at n random points within `radius` degrees of the core, that area).

    positions: (beams, 2) [ra, dec] in degrees; core: the station beam's centre."""
    ra0, dec0 = core
    x = (positions[:, 0] - ra0) * math.cos(math.radians(dec0))
    y = positions[:, 1] - dec0
    r = radius * np.sqrt(rng.random(n))
    phi = 2 * math.pi * rng.random(n)
    px, py = r * np.cos(phi), r * np.sin(phi)
    k = 4 * math.log(2)
    station = np.exp(-k * (px ** 2 + py ** 2) / station_fwhm ** 2)
    tab = np.exp(-k * ((px[:, None] - x[None, :]) ** 2 + (py[:, None] - y[None, :]) ** 2) / tab_fwhm ** 2).max(axis=1)
    return station * tab, math.pi * radius ** 2


def detected_per_unit(model, alpha, taus, kappa):
    """Per burst, the detections per unit N(>1 Jy ms) arriving at unit gain: the integral of its class's
    completeness over the fluence distribution, integral of C(s) (-alpha) s^alpha dln s with s = kappa g F the
    ideal S/N, which leaves (kappa g)^-alpha outside. Exact, where drawing fluences from the power law leaves
    the sum to the few brightest draws."""
    lns = np.linspace(0.0, math.log(1e5), 4000)
    out = np.zeros_like(kappa)
    for (lo, hi), (x50, sigma, top, _) in model.items():
        c = logistic(lns / math.log(10), x50, sigma, top)
        integral = float((c * -alpha * np.exp(alpha * lns)).sum() * (lns[1] - lns[0]))
        sel = (taus >= lo) & (taus < hi)
        out[sel] = integral * (kappa[sel] * 1e-3) ** -alpha            # kappa is per Jy s, fluences are in Jy ms
    return out


def exposure(bursts, model, fields, sefd, alpha, tab_fwhm, station_fwhm, hours, seed=0):
    """(effective exposure in sky x days, SAPs, beams): the expected detections per unit R of N(>1 Jy ms)."""
    rng = np.random.default_rng(seed)
    kappa, taus = snr_per_jy_s(bursts, sefd)
    if not len(kappa):
        raise ValueError('no injected burst records its fluence yet (the lane truth has it from 94e7a4f on)')
    per_burst = detected_per_unit(model, alpha, taus, kappa).mean()
    total, saps, beams = 0.0, 0, 0
    for beams_of in fields.values():
        coherent = {b: v for b, v in beams_of.items() if b != 12}
        if not coherent:
            continue
        positions = np.array(list(coherent.values()))
        core = np.array([v for b, v in coherent.items() if b >= 13] or list(coherent.values())).mean(axis=0)
        gains, area = gain_samples(positions, core, tab_fwhm, station_fwhm, rng)
        total += area / SKY_DEG2 * hours / 24.0 * per_burst * float((gains ** -alpha).mean())
        saps += 1
        beams += len(coherent)
    return total, saps, beams


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lane', type=Path, default=LANE_DB)
    p.add_argument('--web', type=Path, default=WEB_DB)
    p.add_argument('--fingerprint', help='code fingerprint prefix of the searches (and injections) to count')
    p.add_argument('--sefd', type=float, nargs='+', default=[440.0, 580.0, 890.0],
                   help='Jy at a beam centre: the searched fields at their 10th, 50th and 90th percentile')
    p.add_argument('--alpha', type=float, default=-1.4)
    p.add_argument('--tab-fwhm', type=float, default=0.40, help='degrees at 135 MHz')
    p.add_argument('--station-fwhm', type=float, default=4.3, help='degrees at 135 MHz')
    p.add_argument('--hours', type=float, default=1.0, help='per SAP')
    p.add_argument('--bursts', type=Path, nargs='+', help='injected bursts as JSON lists instead of the lane')
    p.add_argument('--project-saps', type=int, help='the exposure of this many SAPs laid out as a searched one')
    a = p.parse_args(argv)
    if a.bursts:
        bursts = [b for path in a.bursts for b in json.loads(Path(path).read_text())]
    else:
        bursts = load_bursts(a.lane, a.fingerprint)
    model = completeness_model(bursts)
    print(f'{len(bursts)} injected bursts; completeness by scattering at 135 MHz (S/N at half the plateau):')
    for (lo, hi), (x50, sigma, top, n) in model.items():
        print(f'  tau {lo:g}-{hi:g} s: n {n}, S/N50 {10 ** x50:.1f}, plateau {top:.2f}, width {sigma:.2f} dex')
    fields = sap_fields(a.web, None if a.project_saps else a.fingerprint)
    if a.project_saps:                                     # one complete SAP's layout, as many times as asked
        layout = next(v for v in fields.values() if len(v) >= 70)
        fields = {('projected', 0): layout}
    for sefd in a.sefd:
        exp, saps, beams = exposure(bursts, model, fields, sefd, a.alpha, a.tab_fwhm, a.station_fwhm, a.hours)
        if a.project_saps:
            exp, saps, beams = exp * a.project_saps, a.project_saps, beams * a.project_saps
        limit = 3.0 / exp if exp > 0 else float('inf')
        print(f'SEFD {sefd:.0f} Jy: {saps} SAPs ({beams} beams), effective exposure {exp:.3g} sky-days for '
              f'N(>1 Jy ms); none found gives R(>1 Jy ms) < {limit:.3g} per sky per day (95%), '
              f'R(>100 Jy ms) < {limit * 100 ** a.alpha:.3g}')


if __name__ == '__main__':
    main()
