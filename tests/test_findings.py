"""Which searched beams keep their filterbank: the rule across an observation and the survey."""
import json
import sqlite3
import time
from argparse import Namespace
from pathlib import Path

import pytest

from euroflash import findings, psrcat

RES = 1 / 3597.5
PSRCAT = """#CATALOGUE 2.8.1
PSRJ     J0323+3944                    tm93
RAJ      03:23:26.6               2    cn95
DECJ     +39:44:52                4    cn95
DM       26.1898                  6    bkk+16
F0       0.329807                 10   hlk+04
F1       -6.9E-17                 5    hlk+04
PEPOCH   48719                         hlk+04
@-----------------------------------------------------------------
PSRJ     J0152+0948                    lsc+19
RAJ      01:52:23.7               2    lsc+19
DECJ     +09:48:10                4    lsc+19
DM       22.881                   6    lsc+19
P0       2.7466468                14   lsc+19
@-----------------------------------------------------------------
PSRJ     J9999+0000
ELONG    12.3
@-----------------------------------------------------------------
"""


@pytest.fixture(autouse=True)
def catalogue(tmp_path, monkeypatch):
    path = tmp_path/'psrcat.db'
    path.write_text(PSRCAT)
    monkeypatch.setenv('LOTAAS_PSRCAT', str(path))
    return path


def fold(f, dm, statistic=15.0, **extra):
    return dict({'frequency_hz': f, 'refined_frequency_hz': f, 'period_seconds': 1 / f,
                 'refined_period_seconds': 1 / f, 'frequency_resolution_hz': RES, 'dm': dm,
                 'statistic': statistic, 'harmonic_count': 4, 'rfi_like': False, 'multibeam_rfi': False,
                 'catalogue_matches': [], 'plot': f'periodicity_plots/periodic_001_DM{dm:.3f}_P{1 / f:.9f}.png'},
                **extra)


def beam(results, run, obs, sap, number, folds=(), vetoed=(), ra='03:20:00', dec='+41:00:00', node='efc-cpu-00'):
    item = f'downsampled_{obs}_SAP{sap:03d}_BEAM{number:03d}_32bit_ff'
    output = results/run/node/'processed'/item/'ec416ee22a75922c'
    output.mkdir(parents=True)
    (output/'metadata.json').write_text(json.dumps({'observation_info': {'RA (J2000)': ra, 'DEC (J2000)': dec},
                                                    'tstart_mjd': 60900.0}))
    (output/'periodicity_search_summary.json').write_text('{}')
    (output/'periodicity_folded_candidates.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in folds))
    (output/'periodicity_veto.json').write_text(json.dumps(
        {'vetoed': [{'frequency_hz': f, 'dm': dm, 'index': i} for i, (f, dm) in enumerate(vetoed)]}))
    return item


def judge_run(root, run):
    index = findings.Index(root)
    index.backfill(root/'results')
    judge = findings.Judge(index)
    return {item: judge.keep(b) for item, b in index.beams.items() if item and b['run'] == run}


def test_the_catalogue_is_read_without_numpy_and_matches_harmonics_at_their_dm():
    pulsars = psrcat.load()
    assert [p['name'] for p in pulsars] == ['J0323+3944', 'J0152+0948']
    near = psrcat.cone(pulsars, 50.8, 41.2, 5.0)
    assert [p['name'] for p in near] == ['J0323+3944'] and 1.4 < near[0]['separation_deg'] < 1.6
    p0 = psrcat.period_at(near[0], 60900.)
    assert psrcat.match(p0 * (1 + 5e-5), 26.2, near)[1] == '1/1'
    assert psrcat.match(p0 / 17, 25.0, near)[1] == '1/17'
    assert psrcat.match(p0 / 2, 26.9, near)[1] == '1/2'
    assert psrcat.match(p0, 60.0, near) is None                 # the right period at the wrong DM
    assert psrcat.match(p0 * 1.1, 26.2, near) is None


def test_a_fold_is_judged_against_its_whole_observation(tmp_path):
    results = tmp_path/'results'
    run = 'campaign-20260924-010000-gpu00'
    line = 0.29575
    # An interference line: this beam's fold, and the veto's peaks in six others.
    beam(results, run, 'L603686', 0, 13, [fold(line, 8.1)])
    for number, dm in zip(range(14, 20), (0.0, 0.7, 1.3, 3.0, 7.1, 0.3)):
        beam(results, run, 'L603686', 0, number, vetoed=[(line + 0.2 * RES, dm)])
    lone = beam(results, run, 'L603686', 0, 30, [fold(1 / 0.7, 57.0)])
    low = beam(results, run, 'L603686', 0, 31, [fold(1 / 0.9, 1.0)])
    high = beam(results, run, 'L603686', 0, 32, [fold(1 / 294.9, 1500.0)])
    j0323 = beam(results, run, 'L603686', 0, 33, [fold(17 * 0.329807, 25.1)])
    verdicts = judge_run(tmp_path, run)
    assert verdicts[lone][0] and not verdicts['downsampled_L603686_SAP000_BEAM013_32bit_ff'][0]
    assert 'family of 7 beams' in verdicts['downsampled_L603686_SAP000_BEAM013_32bit_ff'][1]
    assert verdicts[low] == (False, 'dm<2') and verdicts[high] == (False, 'dm>1000')
    assert verdicts[j0323] == (False, 'catalogue J0323+3944 1/17')


def test_a_bright_pulsar_in_many_beams_keeps_its_filterbanks(tmp_path):
    results = tmp_path/'results'
    run = 'campaign-20260924-020000-gpu00'
    f = 1 / 0.73
    kept = [beam(results, run, 'L700000', 1, number, [fold(f, 61.0 + 0.1 * (number % 5))]) for number in range(13, 25)]
    verdicts = judge_run(tmp_path, run)
    assert all(verdicts[item][0] for item in kept)


def test_an_interference_line_of_distant_observations_keeps_nothing(tmp_path):
    results = tmp_path/'results'
    f = 1 / 5.9314
    for n, (obs, ra, dec) in enumerate((('L500001', '10:00:00', '+20:00:00'), ('L500002', '14:00:00', '+60:00:00'))):
        run = f'campaign-20260924-0{n}0000-gpu00'
        for number, dm in zip(range(13, 20), (0.0, 0.4, 2.2, 5.5, 9.0, 0.1, 13.0)):
            beam(results, run, obs, 0, number, vetoed=[(f, dm)], ra=ra, dec=dec)
    run = 'campaign-20260924-050000-gpu00'
    here = beam(results, run, 'L500003', 2, 40, [fold(f, 3.4)], ra='22:00:00', dec='+30:00:00')
    other = beam(results, run, 'L500003', 2, 41, [fold(1 / 1.234567, 30.0)], ra='22:00:00', dec='+30:00:00')
    verdicts = judge_run(tmp_path, run)
    assert verdicts[here] == (False, 'recurs in 2 other observations') and verdicts[other][0]
    # The same line in pointings that overlap this one proves nothing.
    near = tmp_path/'near'
    for n, obs in enumerate(('L500001', 'L500002')):
        run = f'campaign-20260924-0{n}0000-gpu00'
        for number, dm in zip(range(13, 20), (0.0, 0.4, 2.2, 5.5, 9.0, 0.1, 13.0)):
            beam(near/'results', run, obs, 0, number, vetoed=[(f, dm)], ra='22:10:00', dec='+31:00:00')
    run = 'campaign-20260924-050000-gpu00'
    here = beam(near/'results', run, 'L500003', 2, 40, [fold(f, 3.4)], ra='22:00:00', dec='+30:00:00')
    assert judge_run(near, run)[here][0]


def campaign_with_kept(tmp_path, reviews=None):
    from euroflash import campaign as C
    root = tmp_path/'campaign'
    (root/'results').mkdir(parents=True)
    state = C.State(root/'campaign-state.sqlite')
    prepared = root/'prepared'/'data'/'L603686'/'SAP000'/'B040'
    prepared.mkdir(parents=True)
    fil = prepared/'downsampled_L603686_SAP000_BEAM040_32bit.fil'
    C.flattened(fil).write_bytes(b'ff')
    with state.db() as db:
        db.execute("INSERT INTO files(surl,name,sap_key,beam,state,fil,updated) VALUES (?,?,?,?,?,?,?)",
                   ('srm://x/B040', 'L1263256_SAP000_B040_P000_bf.tar', 'L1263256_SAP000', 40, 'kept',
                    json.dumps([str(fil)]), time.time()))
    campaign = C.Campaign.__new__(C.Campaign)
    campaign.root, campaign.state = root, state
    campaign.o = Namespace(reviews=reviews, keep_prepared=False)
    return campaign, C.flattened(fil)


def test_a_later_sap_releases_a_beam_kept_for_a_fold_it_explains(tmp_path):
    campaign, ff = campaign_with_kept(tmp_path)
    results = campaign.root/'results'
    f = 1 / 3.3812
    beam(results, 'campaign-20260924-010000-gpu00', 'L603686', 0, 40, [fold(f, 8.1)])
    assert campaign.release_kept({'L603686'}, 'campaign-20260924-010000-gpu00') == []
    assert ff.exists()
    # The next SAP of the observation holds the same frequency in five beams at scattered DMs.
    for number, dm in zip(range(13, 18), (0.0, 1.1, 2.7, 0.3, 5.9)):
        beam(results, 'campaign-20260924-020000-gpu01', 'L603686', 1, number, vetoed=[(f, dm)])
    released = campaign.release_kept({'L603686'}, 'campaign-20260924-020000-gpu01')
    assert [name for name, _ in released] == ['L1263256_SAP000_B040_P000_bf.tar'] and not ff.exists()
    row = campaign.state.rows('SELECT state, detail FROM files')[0]
    assert row['state'] == 'searched' and row['detail'].startswith('released after campaign-20260924-020000-gpu01: family')


def test_a_reviewer_verdict_holds_a_beam(tmp_path):
    reviews = tmp_path/'reviews.sqlite'
    campaign, ff = campaign_with_kept(tmp_path, reviews)
    results = campaign.root/'results'
    f = 1 / 3.3812
    item = beam(results, 'campaign-20260924-010000-gpu00', 'L603686', 0, 40, [fold(f, 8.1)])
    for number, dm in zip(range(13, 18), (0.0, 1.1, 2.7, 0.3, 5.9)):
        beam(results, 'campaign-20260924-020000-gpu01', 'L603686', 1, number, vetoed=[(f, dm)])
    key = findings.periodic_key({'item': item, 'fp': 'ec416ee22a75922c'}, fold(f, 8.1))
    with sqlite3.connect(reviews) as db:
        db.execute('CREATE TABLE reviews (id INTEGER PRIMARY KEY, key TEXT, reviewer TEXT, label TEXT, '
                   'note TEXT, dm REAL, created REAL)')
        db.execute("INSERT INTO reviews(key,reviewer,label,created) VALUES (?,?,?,?)", (key, 'Dirk', 'rfi', 1.))
        db.execute("INSERT INTO reviews(key,reviewer,label,created) VALUES (?,?,?,?)", (key, 'Dirk', 'unsure', 2.))
    assert campaign.release_kept({'L603686'}) == [] and ff.exists()
    with sqlite3.connect(reviews) as db:
        db.execute("INSERT INTO reviews(key,reviewer,label,created) VALUES (?,?,?,?)", (key, 'Dirk', 'noise', 3.))
    assert len(campaign.release_kept({'L603686'})) == 1 and not ff.exists()


def test_the_keep_rule_runs_without_numpy(monkeypatch):
    """The campaign driver judges beams on the head's own Python, which has no numpy."""
    import importlib
    import sys
    for name in ('numpy', 'scipy'):
        monkeypatch.setitem(sys.modules, name, None)
    for name in ('euroflash.findings', 'euroflash.psrcat', 'lotaas_reprocessing.periodicity_veto'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module('euroflash.findings')
    assert module.Judge and importlib.import_module('lotaas_reprocessing.periodicity_veto').dm_consistent([30.0, 30.2])


def test_ecliptic_positions_are_read_too():
    # B0138+59 (J0141+6009) has an ecliptic timing solution; RA 01:41:39.9, Dec +60:09:32.
    ra, dec = psrcat.ecliptic_to_equatorial(50.28081, 45.30668)
    assert abs(ra - 25.416) < 0.01 and abs(dec - 60.159) < 0.01
    text = 'PSRJ     J0141+6009\nELONG    50.28081\nELAT     45.30668\nDM       34.8\nF0       0.8130\n@---\n'
    pulsar, = psrcat.parse(text)
    assert abs(pulsar['ra'] - 25.416) < 0.01 and abs(pulsar['dec'] - 60.159) < 0.01
