"""The rate a survey without detections allows (euroflash.frb_rate)."""
import json
import math
import sqlite3

import numpy as np

from euroflash import frb_rate


def test_the_completeness_fit_finds_the_threshold():
    rng = np.random.default_rng(1)
    snr = np.exp(rng.uniform(math.log(5), math.log(60), 800))
    found = rng.random(800) < 0.9 / (1 + np.exp(-(np.log10(snr) - 1.1) / 0.05))
    x50, sigma, top = frb_rate.fit_completeness(snr, found)
    assert abs(10 ** x50 - 12.6) < 2.0 and abs(top - 0.9) < 0.08


def fake_lane(path, n=400, seed=2):
    rng = np.random.default_rng(seed)
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE bursts (twin TEXT, idx INTEGER, fingerprint TEXT, queued INTEGER, record TEXT)')
        for i in range(n):
            snr = float(np.exp(rng.uniform(math.log(6), math.log(60))))
            record = {'tau135': 0.01, 'snr_ideal': snr, 'fluence_units': snr * 1e-3, 'level': 1.0}
            db.execute('INSERT INTO bursts VALUES (?,?,?,?,?)', ('t', i, 'abc', int(snr > 12), json.dumps(record)))


def fake_web(path):
    """One SAP: the 61-beam core on a 0.255-degree grid and the ring of 12 at 2 degrees."""
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE beams (observation TEXT, sap INTEGER, beam INTEGER, ra_deg REAL, dec_deg REAL, '
                   'fingerprint TEXT, sp_complete INTEGER)')
        beam, rows = 13, []
        for q in range(-4, 5):
            for r in range(-4, 5):
                if abs(q + r) <= 4 and beam <= 73:
                    rows.append((beam, 0.255 * (q + r / 2), 0.255 * r * math.sqrt(3) / 2))
                    beam += 1
        rows += [(b, 2.0 * math.cos(b * math.pi / 6), 2.0 * math.sin(b * math.pi / 6)) for b in range(12)]
        db.executemany("INSERT INTO beams VALUES ('L1', 0, ?, ?, ?, 'abc', 1)",
                       [(b, 180 + x, 45 + y) for b, x, y in rows])


def test_no_detection_bounds_the_rate_and_a_better_system_bounds_it_harder(tmp_path):
    fake_lane(tmp_path / 'lane.sqlite')
    fake_web(tmp_path / 'web.sqlite')
    bursts = frb_rate.load_bursts(tmp_path / 'lane.sqlite', 'abc')
    model = frb_rate.completeness_model(bursts)
    fields = frb_rate.sap_fields(tmp_path / 'web.sqlite', 'abc')
    assert list(fields) == [('L1', 0)] and len(fields[('L1', 0)]) == 73
    low, saps, beams = frb_rate.exposure(bursts, model, fields, 400.0, -1.4, 0.40, 4.3, 1.0)
    high, _, _ = frb_rate.exposure(bursts, model, fields, 200.0, -1.4, 0.40, 4.3, 1.0)
    assert saps == 1 and beams == 73 and 0 < low < high
    # One SAP-hour, about 4 square degrees: at most (4 / 41253) / 24 sky-days, less what is too faint.
    assert high < 4.5 / frb_rate.SKY_DEG2 / 24


def test_a_survey_can_be_projected_from_injected_bursts_and_one_sap_s_layout(tmp_path, capsys):
    fake_web(tmp_path / 'web.sqlite')
    rng = np.random.default_rng(3)
    bursts = []
    for _ in range(300):
        snr = float(np.exp(rng.uniform(math.log(6), math.log(60))))
        bursts.append({'tau135': 0.01, 'snr_ideal': snr, 'fluence_units': snr * 1e-3, 'level': 1.0, 'queued': int(snr > 12)})
    (tmp_path / 'bursts.json').write_text(json.dumps(bursts))
    frb_rate.main(['--web', str(tmp_path / 'web.sqlite'), '--bursts', str(tmp_path / 'bursts.json'),
                   '--project-saps', '100', '--sefd', '400'])
    out = capsys.readouterr().out
    assert '100 SAPs (7300 beams)' in out and 'R(>100 Jy ms) <' in out
