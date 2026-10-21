"""Same-DM repeats (web.repeats): an RRAT's few pulses an hour at one DM in one beam."""
import json
import sqlite3

import numpy as np

from web import repeats
from web.indexer import Indexer
from web.tests.test_app import client_for

# J2033+0042's pulses in L565971 SAP001 B007 (2017), from its clustered_candidates.txt.
RRAT = [(37.8, 8.31, 3054.333, 2), (37.1, 6.52, 3084.508, 34), (37.8, 7.12, 2141.926, 9),
        (37.9, 7.89, 2197.043, 6), (37.6, 6.30, 1214.463, 5), (38.0, 6.52, 3475.471, 3)]


# The other 30 clusters of that beam at S/N >= 6.
BACKGROUND = [(3.1, 7.82, 2735.899, 66), (6.3, 7.39, 2737.896, 34), (9.6, 8.34, 840.346, 48), (5.6, 6.93, 836.166, 91),
              (9.6, 7.61, 842.52, 91), (13.3, 6.32, 2245.145, 91), (106.2, 6.78, 2425.679, 9), (1892.8, 6.12, 2210.755, 32),
              (1902.8, 6.15, 3073.124, 112), (24.5, 6.41, 839.127, 34), (2669.8, 6.09, 3127.105, 32),
              (329.8, 6.08, 1210.099, 60), (35.6, 6.16, 1861.41, 18), (349.3, 6.31, 3064.049, 16), (4.1, 6.22, 2373.538, 5),
              (73.7, 6.15, 1021.831, 6), (740.8, 6.18, 285.412, 32), (82.4, 6.16, 3485.632, 5), (959.8, 6.04, 986.469, 32),
              (963.8, 6.24, 1318.029, 16), (1146.8, 6.24, 1678.623, 32), (1670.8, 6.55, 200.697, 16),
              (2134.8, 6.22, 2728.479, 32), (2194.8, 6.53, 1690.892, 32), (2269.8, 6.1, 1771.422, 32),
              (2309.8, 6.24, 588.88, 32), (2394.8, 6.02, 2681.167, 32), (2409.8, 6.22, 990.778, 32),
              (2749.8, 6.16, 2167.784, 32), (2849.8, 6.17, 1206.198, 32)]


def beam_events(seed, extra=()):
    """A beam's clusters away from any source, shaped as J2033+0042's beam's: interference at DM 3-12, the
    wide high-DM features at 200-3000, a few between; S/N 6-8.5, anywhere in the hour."""
    rng = np.random.default_rng(seed)
    dm = np.concatenate([np.exp(rng.uniform(np.log(3), np.log(12), 12)), np.exp(rng.uniform(np.log(200), np.log(3000), 18)),
                         np.exp(rng.uniform(np.log(3), np.log(3000), 6))])
    n = len(dm)
    rows = zip(dm, rng.uniform(6, 8.5, n), rng.uniform(0, 3600, n), rng.integers(1, 60, n))
    return [tuple(map(float, r)) for r in rows] + list(extra)


def test_an_rrats_pulses_at_one_dm_stand_out_and_noise_does_not():
    found = [g for g in repeats.groups(BACKGROUND + RRAT) if abs(g['dm'] - 37.8) < 1]
    assert found and found[0]['k'] == 6 and found[0]['log10fap'] < -14
    assert not repeats.groups(BACKGROUND)
    # The same pulses in a busier beam weigh less, but stand out still.
    found = [g for g in repeats.groups(beam_events(4, RRAT), keep=0) if abs(g['dm'] - 37.8) < 1]
    assert found and found[0]['k'] == 6 and found[0]['log10fap'] < -4
    for seed in range(40):
        assert all(g['log10fap'] > repeats.QUEUE_LOG10FAP for g in repeats.groups(beam_events(seed)))
    # Two pulses at one DM are what chance gives often enough: J1931+4229's two an hour stay below.
    two = [(51.1, 10.0, 3354.5, 5), (51.0, 9.8, 221.5, 9)]
    assert all(g['log10fap'] > repeats.QUEUE_LOG10FAP for g in repeats.groups(beam_events(3, two), keep=0))


def test_interference_spread_over_dms_is_not_a_group():
    rng = np.random.default_rng(5)
    spread = [(float(d), 7.5, float(t), 20) for d, t in zip(rng.uniform(4, 9, 30), rng.uniform(0, 3600, 30))]
    assert not [g for g in repeats.groups(beam_events(1, spread), keep=0) if g['dm'] < 10]


def test_a_train_on_the_requantisers_blocks_is_instrumental():
    block = repeats.LT5_BLOCKS[0]
    rng = np.random.default_rng(2)
    assert repeats.instrumental([100.0 + n * block + rng.normal(0, 0.03) for n in (0, 7, 30, 101)])
    assert not repeats.instrumental([e[2] for e in RRAT])
    rows = repeats.EC_ROWS[0]
    assert repeats.instrumental([12.0 + n * rows for n in (3, 40, 400)], early_cycle=True)


def write_clusters(beam_dir, extra):
    rows = [(30.0, 12.0, 15.0, 3)] + BACKGROUND + list(extra)
    beam_dir.joinpath('clustered_candidates.txt').write_text(
        'DM\tS/N\tTime\tSample\tFilter_Width\tDM_scaled\tCluster\n'
        + ''.join(f'{d}\t{s}\t{t}\t{int(t / 0.00786)}\t{int(w)}\t1\t{i}\n' for i, (d, s, t, w) in enumerate(rows)))
    beam_dir.joinpath('dm_vs_time_clusters.png').write_bytes(b'\x89PNG\r\n\x1a\n')


def test_an_unexplained_repeat_reaches_the_queue_with_its_pulses(cfg, campaign):
    write_clusters(campaign['beam_dir'], RRAT)
    Indexer(cfg).run_pass()
    db = sqlite3.connect(cfg.index_db)
    rows = db.execute("SELECT class, k, log10fap FROM sp_repeats WHERE abs(dm - 37.8) < 1").fetchall()
    assert rows and rows[0][0] == 'one beam' and rows[0][1] >= 5
    client = client_for(cfg)
    listing = client.get('/single-pulse?type=repeat').text
    assert 'same-DM repeat' in listing
    cid = listing.split('href="/verify/', 1)[1].split('?')[0]
    page = client.get(f'/verify/{cid}').text
    assert 'pulses at DM 37.' in page and '3054.333' in page and 'dm_vs_time_clusters' in page


def test_a_catalogued_pulsars_repeat_counts_as_found_and_is_not_queued(cfg, campaign, monkeypatch, tmp_path):
    catalogue = tmp_path / 'psrcat.db'
    catalogue.write_text('PSRJ     J0847+6825\nRAJ      08:47:20.0\nDECJ     +68:25:00\nDM       37.84\n'
                         'P0       5.0134\n@----\n')
    monkeypatch.setenv('LOTAAS_PSRCAT', str(catalogue))
    write_clusters(campaign['beam_dir'], RRAT)
    Indexer(cfg).run_pass()
    db = sqlite3.connect(cfg.index_db)
    assert db.execute("SELECT class, pulsar FROM sp_repeats WHERE abs(dm - 37.8) < 1").fetchone() == ('known', 'J0847+6825')
    assert not db.execute("SELECT 1 FROM candidates WHERE type='repeat'").fetchone()
    sp_count, best = db.execute("SELECT sp_count, sp_best_snr FROM catalogue_pulsars WHERE psrj='J0847+6825'").fetchone()
    assert sp_count >= 5 and abs(best - 8.31) < 0.01
