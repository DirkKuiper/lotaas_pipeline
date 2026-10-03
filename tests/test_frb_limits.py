"""The running upper limit on the FRB rate (euroflash.frb_limits)."""
import json
import math
import sqlite3

import numpy as np
import pytest

from euroflash import frb_limits

UNITS = {'sigma_mean': 0.025, 'tsamp': 0.007864, 'channel_mhz': 0.048828125}
EPOCHS = """
epochs:
  - name: one
    from: '2026-09-28T00:00:00Z'
    lane: {lta: ['aaa111']}
  - name: two
    from: '2026-10-01T00:00:00Z'
    lane: {lta: ['bbb222']}
    stand_in: {epoch: one, min_search_snr: 8.0, min_votes: {lta: 1}}
"""


def epochs(tmp_path, text=EPOCHS):
    (tmp_path / 'epochs.yaml').write_text(text)
    return frb_limits.load_epochs(tmp_path / 'epochs.yaml')


def test_epochs_are_ordered_and_a_sap_belongs_to_the_last_that_began(tmp_path):
    e = epochs(tmp_path)
    assert [x['name'] for x in e] == ['one', 'two']
    assert frb_limits.epoch_at(e, frb_limits.parse_time('2026-09-30T00:00:00Z'))['name'] == 'one'
    assert frb_limits.epoch_at(e, frb_limits.parse_time('2026-10-02T00:00:00Z'))['name'] == 'two'
    assert frb_limits.epoch_at(e, frb_limits.parse_time('2026-09-01T00:00:00Z')) is None


def test_an_unquoted_all_digit_sha_is_refused(tmp_path):
    with pytest.raises(ValueError, match='quote'):
        epochs(tmp_path, "epochs:\n  - {name: x, from: '2026-01-01T00:00:00Z', lane: {lta: [3796107]}}\n")


def burst(snr, queued, votes=None, search=None, tau=0.01):
    b = dict(UNITS, tau135=tau, snr_ideal=snr, fluence_units=snr * 2.5e-5, queued=queued,
             cluster={'snr': snr if search is None else search})
    if votes is not None:
        b['models'] = {m: (0.9 if i < votes else 0.1) for i, m in enumerate('abcdef')}
    return b


def test_replay_drops_what_later_rules_would_not_queue():
    bursts = [burst(20, True, votes=0), burst(20, True, votes=1), burst(7.5, True, search=7.5), burst(20, True),
              burst(20, False, votes=6)]
    lta = [b['queued'] for b in frb_limits.replay(bursts, 'lta', 8.0, {'lta': 1})]
    spider = [b['queued'] for b in frb_limits.replay(bursts, 'spider', 8.0, {'lta': 1})]
    assert lta == [False, True, False, True, False]          # FETCH asked, no vote; below S/N 8
    assert spider == [True, True, False, True, False]        # the vote rule is LT5's only


def test_a_new_epoch_borrows_its_stand_in_until_the_lane_has_enough_bursts(tmp_path):
    e = epochs(tmp_path)
    lane = {('aaa111' + 'f' * 34, 'lta'): [burst(20, True, votes=0)] * 300}
    got, came = frb_limits.epoch_lane(e, lane, 'two', 'lta')
    assert len(got) == 300 and not any(b['queued'] for b in got) and 'replayed as two' in came
    lane[('bbb222' + 'f' * 34, 'lta')] = [burst(20, True)] * frb_limits.MIN_LANE_BURSTS
    got, came = frb_limits.epoch_lane(e, lane, 'two', 'lta')
    assert len(got) == frb_limits.MIN_LANE_BURSTS and all(b['queued'] for b in got) and came.startswith('two')


def test_healpix_ring_pixels_are_unit_vectors_spread_evenly():
    v = frb_limits.pixel_vectors(1)
    assert np.allclose(v[:4, 2], 2 / 3) and np.allclose(v[4:8, 2], 0) and np.allclose(v[8:, 2], -2 / 3)
    v = frb_limits.pixel_vectors(16)
    assert len(v) == 12 * 16 ** 2 and np.allclose(np.linalg.norm(v, axis=1), 1)
    assert np.allclose(v.mean(axis=0), 0, atol=1e-3)
    assert np.allclose(v[0], [math.sqrt(1 - (1 - 1 / 768) ** 2) * math.cos(math.pi / 4),
                              math.sqrt(1 - (1 - 1 / 768) ** 2) * math.sin(math.pi / 4), 1 - 1 / 768])
    # A 20-degree cap holds its share of the pixels.
    direction = np.array([0.3, -0.5, 0.81]) / np.linalg.norm([0.3, -0.5, 0.81])
    share = (v @ direction > math.cos(math.radians(20))).mean()
    assert abs(share - (1 - math.cos(math.radians(20))) / 2) < 0.004


def test_a_field_on_the_meridian_at_the_latitude_of_lofar_is_at_the_zenith():
    mjd = 61000.25
    t = (mjd - 51544.5) / 36525.0
    lst = (280.46061837 + 360.98564736629 * (mjd - 51544.5) + 0.000387933 * t * t + frb_limits.LOFAR[1]) % 360
    assert frb_limits.elevation(lst, frb_limits.LOFAR[0], mjd) == pytest.approx(90, abs=1e-6)
    assert frb_limits.elevation((lst + 180) % 360, frb_limits.LOFAR[0], mjd) == pytest.approx(2 * 52.9153 - 90, abs=1e-6)


def test_sensitivity_follows_the_calibration():
    assert frb_limits.jy_per_sigma(350.0, 90.0) == pytest.approx(14.8)
    assert frb_limits.jy_per_sigma(1100.0, 90.0) == pytest.approx(14.8 * 2)
    assert frb_limits.jy_per_sigma(350.0, 30.0) == pytest.approx(14.8 * 2 ** 1.39)


def fake_survey(tmp_path, saps=1, hours_each=1.0):
    rng = np.random.default_rng(3)
    with sqlite3.connect(tmp_path / 'lane.sqlite') as db:
        db.execute('CREATE TABLE bursts (fingerprint TEXT, source TEXT, queued INTEGER, record TEXT)')
        for _ in range(400):
            snr = float(np.exp(rng.uniform(math.log(6), math.log(60))))
            db.execute('INSERT INTO bursts VALUES (?,?,?,?)', ('aaa111' + 'f' * 34, 'lta', int(snr > 12),
                                                                json.dumps(burst(snr, snr > 12))))
    with sqlite3.connect(tmp_path / 'web.sqlite') as db:
        db.execute('CREATE TABLE beams (observation TEXT, sap INTEGER, beam INTEGER, ra_deg REAL, dec_deg REAL, '
                   'tstart_mjd REAL, fingerprint TEXT, pilot INTEGER, sp_complete INTEGER, dir TEXT)')
        db.execute('CREATE TABLE runs (fingerprint TEXT, created REAL)')
        db.execute("INSERT INTO runs VALUES ('run', ?)", (frb_limits.parse_time('2026-09-29T00:00:00Z'),))
        for s in range(saps):
            d = tmp_path / f'L{s}'
            d.mkdir()
            (d / 'metadata.json').write_text(json.dumps({'samples_processed': int(hours_each * 3600 / 0.5), 'tsamp': 0.5}))
            beam = 13
            for q in range(-4, 5):
                for r in range(-4, 5):
                    if abs(q + r) <= 4 and beam <= 73:
                        db.execute("INSERT INTO beams VALUES (?, 0, ?, ?, ?, 60000.0, 'run', 0, 1, ?)",
                                   (f'L{s}', beam, 180 + 0.255 * (q + r / 2), 52.9 + 0.255 * r * math.sqrt(3) / 2, str(d)))
                        beam += 1
    with sqlite3.connect(tmp_path / 'campaign.sqlite') as db:
        db.execute('CREATE TABLE saps (key TEXT, source TEXT, state TEXT)')
        db.executemany('INSERT INTO saps VALUES (?, ?, ?)', [(f'k{i}', 'lta', 'pending') for i in range(10 * saps)]
                       + [('x', 'lta', 'excluded')])


def test_the_limit_tightens_with_time_on_sky_and_projects_to_the_whole_campaign(tmp_path):
    e = epochs(tmp_path)
    fake_survey(tmp_path)
    one, _ = frb_limits.compute(tmp_path / 'web.sqlite', tmp_path / 'lane.sqlite', e, None, tmp_path / 'campaign.sqlite')
    assert one['saps'] == 1 and one['hours'] == 1.0 and one['campaign_saps'] == {'lta': 10}
    assert one['by_epoch_source'] == {'one/lta': 1}
    (tmp_path / 'more').mkdir()
    fake_survey(tmp_path / 'more', hours_each=2.0)
    two, _ = frb_limits.compute(tmp_path / 'more' / 'web.sqlite', tmp_path / 'more' / 'lane.sqlite', e, None,
                                tmp_path / 'more' / 'campaign.sqlite')

    def r100(r, key='r95', sefd='nominal'):
        return next(x for x in r['limits'] if x['alpha'] == -1.4 and 'mix' in x['population'] and x['sefd'] == sefd)[key]['100']
    assert r100(two) == pytest.approx(r100(one) / 2, rel=0.05)
    assert r100(one, 'r95_projected') == pytest.approx(r100(one) / 10, rel=1e-6)
    assert r100(one, sefd='sefd_low') < r100(one) < r100(one, sefd='sefd_high')
    # Unscattered bursts above 12 sigma always queued: one SAP-hour of ~4 deg2 bounds R(>100 Jy ms) at
    # no less than 3 / (4 / 41253 / 24) x 100^-1.4 per sky per day.
    assert r100(one) > 3 / (4.5 / frb_limits.frb_rate.SKY_DEG2 / 24) * 100 ** -1.4
    assert 0 < one['expected'][0]['expected'] < one['expected'][-1]['expected']
