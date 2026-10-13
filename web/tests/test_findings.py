"""Findings: catalogue uncertainties, the leads the index finds, and recording what a check found."""
import json
import math

import pytest

from web import findings
from web.indexer import Indexer
from web.tests.test_app import client_for, fold

LOOSE = ('PSRJ     J2347+02                      mss+20\nRAJ      23:47                         mss+20\n'
         'DECJ     +02                           mss+20\nDM       15.0                     4    mss+20\n'
         'P0       1.38347                       ar_drift21\nSURVEY   gbncc,ar327\n@----\n')
TIMED = ('PSRJ     J0332+5434\nPSRB     B0329+54\nRAJ      03:32:59.4096            1    dgb+19\n'
         'DECJ     +54:34:43.329            1    dgb+19\nDM       26.76410                 10   hsh+12\n'
         'F0       1.399541538720           6    hlk+04\nPEPOCH   46473.0                       hlk+04\n'
         'F1       -4.011970E-15            14   hlk+04\nSURVEY   lotaas\n@----\n'
         'PSRJ     J2029+5459                    msk+24\nELONG    337.5                    1    msk+24\n'
         'ELAT     71.0                     1    msk+24\nF0       1.727386681              2    msk+24\n'
         'PEPOCH   59000                         msk+24\nF1       -1.0E-15                 1    msk+24\n'
         'DM       59.75                    5    msk+24\n@----\n')


def test_catalogue_uncertainties_are_one_unit_of_the_last_quoted_digit():
    loose, timed, ecliptic = findings.parse(LOOSE + TIMED)
    # '23:47 +02': a minute of RA and a degree of Dec when no error is quoted.
    assert loose['pos_err'] == pytest.approx(math.hypot(0.25 * math.cos(math.radians(2)), 1.0))
    assert loose['dm_err'] == pytest.approx(0.4) and loose['p_err'] == pytest.approx(1e-5)
    assert not loose['timed'] and findings.loose(loose) and loose['year'] == 2020  # mss+20, before ar_drift21
    assert timed['timed'] and not findings.loose(timed) and timed['bname'] == 'B0329+54'
    assert timed['p_err'] == pytest.approx(6e-12 / 1.39954153872 ** 2)
    assert ecliptic['timed'] and 290 < ecliptic['ra'] < 320 and 50 < ecliptic['dec'] < 60


def hit(ra, dec, period, dm=16.2, statistic=57.0, item='B062', key='k1'):
    return {'key': key, 'item': item, 'dm': dm, 'period': period, 'statistic': statistic, 'seconds': 3600.0,
            'ra': ra, 'dec': dec, 'observation': 'L591573', 'sap': 2, 'observed': '2017-05-11 09:54:00'}


def test_a_fold_sharpening_a_loose_entry_is_a_lead_until_the_literature_holds_it(tmp_path):
    catalogue = findings.parse(LOOSE + TIMED)
    folds = [hit(356.95, 2.84, 1.385944), hit(356.80, 2.95, 1.385944, statistic=20.0, item='B063', key='k2')]
    leads = findings.fold_leads(folds, catalogue, [], set())
    lead = leads['J2347+02']
    kinds = [kind for kind, _ in lead['claims']]
    # Our period is 18 sigma off the drift scan's; our DM (+-0.3) is no better than its +-0.4.
    assert kinds == ['position', 'conflict', 'first_lofar', 'precovery']
    assert lead['evidence'] == ['k1', 'k2'] and lead['ours']['beams'] == 2
    # Olszanski et al. 2025 timed it: nothing is left to add, so there is no lead.
    table = tmp_path / 'literature.tsv'
    table.write_text('name\taliases\treference\turl\tra\tdec\tpos_err_arcsec\tperiod_s\tperiod_err_s\tdm\tdm_err\t'
                     'lofar\tnote\nJ2347+0300\tJ2347+02\tOlszanski et al. 2025\t\t23:47:44.471\t+03:00:12.1\t5\t'
                     '1.38605589766\t1e-9\t16.074\t0.006\t1\t\n')
    assert findings.fold_leads(folds, catalogue, findings.literature(table), set()) == {}
    # A fold a percent off a pulsar whose position is known to an arcminute, a degree away, is chance.
    precise = findings.parse(LOOSE.replace('RAJ      23:47                         mss+20',
                                           'RAJ      23:44:00   1    xyz+25').replace(
        'DECJ     +02                           mss+20', 'DECJ     +02:00:00   1    xyz+25'))
    assert findings.fold_leads([hit(356.95, 2.84, 1.40)], precise, [], set()) == {}


def test_leads_are_listed_checked_and_recorded(cfg, campaign, monkeypatch, tmp_path):
    # A loose entry 1 degree wide over the campaign's beam (08:47:08 +68:24:59), at a rough period.
    catalogue = tmp_path / 'compare.db'
    catalogue.write_text('PSRJ     J0847+68                      abc+20\nRAJ      08:47                    abc+20\n'
                         'DECJ     +68                           abc+20\nDM       30                       2    abc+20\n'
                         'P0       0.750                         abc+20\nSURVEY   gbncc\n@----\n')
    cfg.compare_catalogue = catalogue
    monkeypatch.setattr(findings, 'LITERATURE', tmp_path / 'none.tsv')
    fold(campaign['beam_dir'], 30.1, 0.7512345, 120.0, 'loose')
    Indexer(cfg).run_pass()
    client = client_for(cfg)
    page = client.get('/findings').text
    leads = page.split('<h2>Leads to check</h2>')[1]
    assert 'J0847+68' in leads
    for claim in ('Better position', 'Better period', 'Better DM', 'First LOFAR detection'):
        assert claim in leads
    assert 'P 0.75123' in leads and 'ATNF J0847+68, abc+20' in leads and 'Detected before its discovery' in leads
    cid = leads.split('href="/verify/')[1].split('"')[0]
    assert 'Findings<span class="badge-count">1</span>' in client.get('/').text

    for bad in ({'source': 'J0847+68', 'status': 'finding', 'reviewer': 'Dirk'},                    # no claim
                {'source': 'J0847+68', 'status': 'finding', 'claims': ['wow'], 'reviewer': 'Dirk'},
                {'source': 'J0847+68', 'status': 'finding', 'claims': ['period']},                  # no reviewer
                {'source': 'J0847+68', 'status': 'finding', 'claims': ['period'], 'reviewer': 'Dirk',
                 'evidence': ['nope']}):
        assert client.post('/api/finding', json=bad).status_code == 400
    # Checked and published already: off the leads, on record.
    answer = client.post('/api/finding', json={'source': 'J0847+68', 'status': 'not_new', 'reviewer': 'Claude',
                                               'summary': 'timed by someone in 2025', 'checked': 'ADS'})
    assert answer.status_code == 200 and answer.json()['findings'][0]['status'] == 'not_new'
    page = client.get('/findings').text
    assert 'J0847+68' not in page.split('<h2>Leads to check</h2>')[1].split('</table>')[0]
    assert 'timed by someone in 2025' in page.split('Checked: not new (')[1]
    # A finding, from the candidate's own page.
    verify = client.get(f'/verify/{cid}').text
    assert 'Record as a finding' in verify and 'class="finding-form"' in verify
    answer = client.post('/api/finding', json={
        'source': 'J0847+68', 'status': 'finding', 'reviewer': 'Dirk', 'evidence': [cid],
        'claims': [['period', 'P 0.751234 s against 0.750'], 'first_lofar'], 'summary': 'refined',
        'names': ['J0847+6830', 'J0847+68']})
    assert answer.status_code == 200 and len(answer.json()['findings']) == 2
    page = client.get('/findings').text
    found = page.split('<h2>Findings</h2>')[1].split('<h2>')[0]
    assert 'J0847+68' in found and 'P 0.751234 s against 0.750' in found and f'/verify/{cid}' in found
    assert 'J0847+6830' in found and '1 earlier' in found
    assert 'Finding</span> <a href="/findings#J0847+68">J0847+68</a>' in client.get(f'/verify/{cid}').text
    stored = json.loads(answer.json()['findings'][0]['evidence'])
    assert stored and stored[0].startswith('periodicity|')
    # Once a paper the catalogue lacks holds it, the index finds nothing to add.
    monkeypatch.setattr(findings, 'LITERATURE', tmp_path / 'literature.tsv')
    (tmp_path / 'literature.tsv').write_text(
        'name\taliases\treference\turl\tra\tdec\tpos_err_arcsec\tperiod_s\tperiod_err_s\tdm\tdm_err\tlofar\tnote\n'
        'J0847+6825\tJ0847+68\tSomeone et al. 2025\t\t08:47:10\t+68:25:00\t1\t0.75123456\t1e-11\t30.12\t0.01\t1\t\n')
    assert Indexer(cfg).derive_findings(force=True) == 0
