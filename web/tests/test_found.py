"""What counts as finding a catalogued source (indexer.found), after the audit of 6 October 2026."""
import sqlite3

import numpy as np

from web import store
from web.indexer import Indexer, FUNDAMENTAL_TOLERANCE, HARMONIC_MIN_STATISTIC, WIDE_MIN_STATISTIC

PSR = {'name': 'J0700+6418', 'bname': 'B0655+64', 'ra': 105.0, 'dec': 64.3, 'dm': 8.77,
       'f0': 1 / 0.1956709, 'period': 0.1956709}


def fold(period, dm, statistic, item='downsampled_L1_SAP000_BEAM006_32bit_ff'):
    return {'period': period, 'dm': dm, 'statistic': statistic, 'item': item}


def evidence(folds=(), singles=None, known=None):
    return ({}, {'L1': list(folds)}, singles or {}, known or {}, {})


def test_a_binary_off_the_catalogue_period_by_its_orbit_is_found():
    # B0655+64 folded at statistic 8,527, its period 3.6e-4 below the catalogue's.
    sp, top, n = Indexer.found(PSR, {'L1'}, evidence([fold(0.1956007, 8.8, 8527.5)]))
    assert n == 1 and top[2] == '1/1'
    assert Indexer.fold_relation(fold(PSR['period'] * (1 + 1.5 * FUNDAMENTAL_TOLERANCE), 8.8, 9000), PSR) is None
    # So far out, a fold near the search's threshold is what chance gives (J1852+0056_P at statistic 12).
    assert Indexer.fold_relation(fold(0.1956007, 8.8, WIDE_MIN_STATISTIC / 4), PSR) is None
    assert Indexer.fold_relation(fold(PSR['period'] * (1 + 2e-4), 8.8, 13.0), PSR) == '1/1'


def test_a_harmonic_counts_only_far_above_what_chance_gives():
    weak = fold(PSR['period'] / 2, 8.8, HARMONIC_MIN_STATISTIC / 2)
    strong = fold(PSR['period'] / 2, 8.8, HARMONIC_MIN_STATISTIC * 2)
    assert Indexer.found(PSR, {'L1'}, evidence([weak]))[2] == 0
    assert Indexer.found(PSR, {'L1'}, evidence([strong]))[1][2] == '1/2'
    # Its own period at another DM is something else.
    assert Indexer.fold_relation(fold(PSR['period'], 40.0, 9000), PSR) is None


def test_pulses_count_for_the_pulsar_they_were_shown_to_be():
    # B2217+47's pulses near J2236+4929's DM made it 'found'; only pulses attributed to a source count for it.
    known = {('L1', 'J2219+4754'): [(25.0, 'item-a'), (20.0, 'item-b')]}
    other = {'name': 'J2236+4929', 'ra': 339.0, 'dec': 49.5, 'dm': 42.9, 'f0': 1 / 0.9317, 'period': 0.9317}
    assert Indexer.found(other, {'L1'}, evidence(known=known))[0] == []
    mine = {'name': 'J2219+4754', 'bname': 'B2217+47', 'ra': 335.0, 'dec': 47.9, 'dm': 43.5,
            'f0': 1 / 0.5385, 'period': 0.5385}
    assert len(Indexer.found(mine, {'L1'}, evidence(known=known))[0]) == 2
    # A person's verdict may name it by its B name.
    assert len(Indexer.found(mine, {'L1'}, evidence(known={('L1', 'B2217+47'): [(9.0, 'item-c')]}))[0]) == 1


def test_classifier_redetections_need_the_dm_to_stand_out_and_a_reachable_period():
    singles = {('L1', PSR['name']): [(14.0, 'item-a')]}
    stands_out = lambda o, s: (22.0, 'item-b')
    assert Indexer.found(PSR, {'L1'}, evidence(singles=singles))[0] == []          # no excess check: nothing
    assert Indexer.found(PSR, {'L1'}, evidence(singles=singles), lambda o, s: None)[0] == []
    assert Indexer.found(PSR, {'L1'}, evidence(singles=singles), stands_out)[0] == [(22.0, 'item-b'), (14.0, 'item-a')]
    # The Crab's giant pulses: no redetection under its name, but its DM stands out.
    assert Indexer.found(PSR, {'L1'}, evidence(), stands_out)[0] == [(22.0, 'item-b')]
    msp = dict(PSR, name='J1048+2339', bname=None, f0=1 / 0.0047, period=0.0047)
    single = {('L1', 'J1048+2339'): [(19.9, 'item-a')]}
    known = {('L1', 'J1048+2339'): [(19.9, 'item-b')]}
    assert Indexer.found(msp, {'L1'}, evidence(singles=single, known=known), stands_out)[0] == []


def test_single_pulses_say_nothing_beside_a_brighter_pulsar_at_the_same_dm():
    # B1919+21 (1.5 Jy) 5.8 degrees from J1929+16 at DM 12.4 against 12.0.
    weak = {'name': 'J1929+16', 'ra': 292.3, 'dec': 16.0, 'dm': 12.0, 'f0': 1 / 0.5297, 'period': 0.5297}
    bright = {'name': 'J1921+2153', 'bname': 'B1919+21', 'ra': 290.44, 'dec': 21.88, 'dm': 12.44,
              'f0': 1 / 1.3373, 'S150': 1500.0}
    far = dict(bright, name='J1100+2153', ra=165.0)
    known = {('L1', 'J1929+16'): [(16.0, 'item-a')]}
    rival = Indexer.rival([weak, bright])
    assert rival(weak) == 'J1921+2153' and rival(bright) is None
    assert Indexer.found(weak, {'L1'}, evidence(known=known), None, rival)[0] == []
    assert Indexer.found(weak, {'L1'}, evidence(known=known), None, Indexer.rival([weak, far]))[0] == [(16.0, 'item-a')]
    # Unless they keep its own rotation.
    assert Indexer.found(weak, {'L1'}, evidence(known=known), None, rival, lambda o, s: False)[0] == []
    assert Indexer.found(weak, {'L1'}, evidence(known=known), None, rival, lambda o, s: True)[0] == [(16.0, 'item-a')]


def test_rotation_tells_a_pulsars_own_pulses_from_a_brighter_ones(tmp_path):
    rng = np.random.default_rng(3)
    slow = {'name': 'J1847-0308_P', 'ra': 281.8, 'dec': -3.1, 'dm': 149.8, 'f0': 1 / 29.7693, 'period': 29.7693}
    own, other = tmp_path/'own'/'fp', tmp_path/'other'/'fp'
    # 25 pulses on its rotation (phase jitter 0.02), in a beam 4 degrees away: sidelobes count.
    turns = rng.choice(120, 25, replace=False)
    clusters(own, [(149.6, 8.0, 29.7693 * (k + 0.3 + rng.normal(0, 0.02))) for k in turns])
    # 2,000 pulses of another pulsar at the same DM, at its own period.
    clusters(other, [(149.9, 9.0, 0.5580 * k) for k in rng.choice(6000, 2000, replace=False)])
    index = Index()
    dirs = {'L1': [(str(own), 285.0, -1.0)], 'L2': [(str(other), 281.8, -3.1)]}
    _, rotation = Indexer.cluster_tests(index, dirs, {'L1': 57000.0, 'L2': 57000.0})
    assert rotation('L1', slow) and not rotation('L2', slow)
    row = index.db.execute("SELECT z_on, z_control FROM sp_dm_excess WHERE observation='L1'").fetchone()
    assert row['z_on'] > 15 and row['z_control'] < 5


class Index:
    def __init__(self):
        self.db = sqlite3.connect(':memory:')
        self.db.row_factory = sqlite3.Row
        self.db.executescript(store.INDEX)


def clusters(path, rows):
    """rows: (dm, snr) or (dm, snr, time)."""
    path.mkdir(parents=True)
    (path/'clustered_candidates.txt').write_text(
        'DM\tS/N\tTime\tSample\tFilter_Width\tDM_scaled\tCluster\n'
        + ''.join(f'{r[0]}\t{r[1]}\t{r[2] if len(r) > 2 else 1.0}\t1\t1\t0\t{i}\n' for i, r in enumerate(rows)))


def test_dm_excess_tells_a_pulsar_from_noise_at_its_dm(tmp_path):
    pulsar = dict(PSR, dm=30.0)
    # Beam directories as the campaign's: <item>/<fingerprint>.
    a, b, c, far = (tmp_path/name/'709267540b919881' for name in ('a', 'b', 'c', 'far'))
    clusters(a, [(30.0, 9.0)] * 12 + [(34.0, 8.0)])                       # its DM crowded
    clusters(b, [(30.2, 8.0), (26.0, 8.0), (33.0, 8.5), (36.0, 9.0), (24.0, 8.0), (38.0, 7.5)])
    clusters(far, [(30.0, 20.0)] * 50)                                    # 3 degrees away: not looked at
    dirs = {'L1': [(str(a), 105.0, 64.3), (str(b), 105.5, 64.4), (str(far), 105.0, 67.3)],
            'L2': [(str(b), 105.5, 64.4)]}
    index = Index()
    excess, _ = Indexer.cluster_tests(index, dirs, {})
    assert excess('L1', pulsar) == (9.0, 'a') and excess('L2', pulsar) is None
    # Every beam of the observation is counted for keeping; those within 1 degree for the counts.
    assert index.db.execute('SELECT beams, on_count FROM sp_dm_excess WHERE observation=?', ('L1',)).fetchone()[:] == (3, 13)
    # Kept until the observation gains beams: then counted again.
    (a/'clustered_candidates.txt').unlink()
    assert Indexer.cluster_tests(index, dirs, {})[0]('L1', pulsar) == (9.0, 'a')
    clusters(c, [])
    dirs['L1'].append((str(c), 105.2, 64.3))
    assert Indexer.cluster_tests(index, dirs, {})[0]('L1', pulsar) is None


def test_a_duplicate_catalogue_entry_shares_what_lofar_published():
    a = {'name': 'J1647+6609', 'ra': 251.8853, 'dec': 66.1395, 'dm': 22.55, 'f0': 0.625078358705}
    b = {'name': 'J1647+6608', 'ra': 251.8855, 'dec': 66.1395, 'dm': 22.55, 'f0': 0.62507876955}
    c = {'name': 'J1650+6600', 'ra': 252.5, 'dec': 66.0, 'dm': 22.6, 'f0': 0.71}          # another period
    seen = {'J1647+6608': {'seen': ['LOTAAS', 'Sanidas et al. 2019']}}
    assert Indexer.duplicate_seen(a, [a, b, c], seen) == ('J1647+6608', seen['J1647+6608'])
    assert Indexer.duplicate_seen(c, [a, b, c], seen) is None


def test_a_source_is_found_by_what_another_entry_for_it_folds_at():
    # LOTAAS lists J2352+65 at P 1.164 s; the fold matches the catalogue's J2351+6500 (1.1648832 s).
    rough = {'name': 'J2352+65', 'ra': 358.0, 'dec': 65.0, 'dm': 152.7, 'f0': 1 / 1.164, 'period': 1.164}
    precise = {'name': 'J2351+6500', 'ra': 357.85, 'dec': 65.01, 'dm': 154.294, 'f0': 1 / 1.1648832}
    folds = evidence([fold(1.16485, 154.4, 15.5)])
    assert Indexer.found(rough, {'L1'}, folds)[2] == 0
    sp, top, n = Indexer.found_as_any(Indexer, rough, {'L1'}, folds, (None, None, None), [rough, precise])
    assert n == 1 and top[2] == '1/1'
