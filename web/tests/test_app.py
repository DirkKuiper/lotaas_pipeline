import json

from fastapi.testclient import TestClient

from web.app import create_app, token
from web.indexer import Indexer
from web.snippets import Snippets
from web.tests.conftest import ITEM, synthetic_filterbank


def client_for(cfg, authenticated=True):
    client = TestClient(create_app(cfg, run_background=False))
    if authenticated:
        client.cookies.set('lotaas_web', token(cfg))
    return client


def test_every_request_needs_the_token(cfg, campaign):
    Indexer(cfg).run_pass()
    anonymous = client_for(cfg, authenticated=False)
    assert anonymous.get('/').status_code == 401
    assert anonymous.get('/api/health').status_code == 401
    assert anonymous.get('/static/style.css').status_code == 200
    login = anonymous.get('/?token=' + token(cfg), follow_redirects=False)
    assert login.status_code == 303 and 'lotaas_web' in login.headers['set-cookie']
    assert (cfg.token_file.stat().st_mode & 0o077) == 0


def test_pages_render(cfg, campaign):
    synthetic_filterbank(cfg.source_roots[0] / f'{ITEM}.fil')
    indexer = Indexer(cfg)
    indexer.run_pass()
    Snippets(cfg).run_pass()
    indexer.run_pass()
    client = client_for(cfg)
    db = indexer.db
    cid = db.execute("SELECT id FROM candidates WHERE type='candidate'").fetchone()[0]
    reject = db.execute("SELECT id FROM candidates WHERE type='rejected'").fetchone()[0]
    for path in ['/', '/staging', '/coverage', '/sap/L1163405_SAP000', f'/beam/{ITEM}', '/single-pulse',
                 '/single-pulse?type=all&review=unreviewed', '/periodic', '/periodic?type=all&max_dm=x',
                 f'/verify/{cid}', f'/verify/{reject}',
                 f'/api/sp/{cid}/view?dm=0&nsub=16', f'/api/sp/{cid}/dm', '/api/health', f'/snippet/{cid}.fil']:
        assert client.get(path).status_code == 200, path
    assert client.get('/verify', follow_redirects=False).status_code == 303
    assert client.get('/sap/nope').status_code == 404


def fold(beam_dir, dm, period, statistic, plot, rfi_like=False):
    row = {'dm': dm, 'period_seconds': period, 'refined_period_seconds': period, 'statistic': statistic,
           'harmonic_count': 16, 'fold_chi2': 630.0, 'fold_bins': 64, 'rfi_like': rfi_like,
           'catalogue_matches': [], 'plot': f'periodicity_plots/{plot}.png', 'fold_data': f'periodicity_plots/{plot}.npz'}
    with (beam_dir / 'periodicity_folded_candidates.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')


def test_single_pulse_and_periodic_are_listed_and_queued_apart(cfg, campaign):
    beam_dir = campaign['beam_dir']
    fold(beam_dir, 26.3, 3.0317926, 560.3, 'psr')
    fold(beam_dir, 4000.0, 21.9, 12.1, 'drift')
    fold(beam_dir, 0.0, 0.02, 40.0, 'mains', rfi_like=True)
    incoherent = beam_dir.parents[1] / ITEM.replace('BEAM025', 'BEAM012') / beam_dir.name
    incoherent.mkdir(parents=True)
    (incoherent / 'metadata.json').write_text((beam_dir / 'metadata.json').read_text())
    fold(incoherent, 0.0, 0.288, 993.5, 'incoherent')
    Indexer(cfg).run_pass()
    client = client_for(cfg)

    sp = client.get('/single-pulse').text
    assert 'FETCH positive' in sp and 'P = ' not in sp and 'B012' not in sp
    periodic = client.get('/periodic').text
    assert '3.031793' in periodic and '21.900000' in periodic
    assert '0.020000' not in periodic            # RFI-like folds are listed only when asked for
    assert 'B012' not in periodic                # the incoherent beam is out of the campaign
    assert '0.020000' in client.get('/periodic?type=periodic_rfi').text
    assert 'B012' in client.get('/periodic?incoherent=include').text
    assert '21.900000' not in client.get('/periodic?max_dm=1000').text
    # At DM 4000 a pulsar would be scattered far beyond a 22 s period; at DM 26 it would not.
    assert periodic.count('τ &gt; P') == 1

    first = client.get('/verify?kind=periodic', follow_redirects=False)
    assert first.status_code == 303 and 'kind=periodic' in first.headers['location']
    page = client.get(first.headers['location']).text
    assert 'Periodic candidate' in page and '1 of 2' in page
    queued = client.get('/verify?kind=sp').text
    assert 'FETCH positive · DM 30.00' in queued and 'Periodic candidate' not in queued
    assert client.get('/candidates?type=periodic', follow_redirects=False).headers['location'] == '/periodic?type=queue'
    assert client.get('/candidates', follow_redirects=False).headers['location'] == '/single-pulse'


def test_reviews_are_kept_apart_and_validated(cfg, campaign):
    Indexer(cfg).run_pass()
    client = client_for(cfg)
    cid = Indexer(cfg).db.execute("SELECT id FROM candidates WHERE type='candidate'").fetchone()[0]
    assert client.post('/api/review', json={'id': cid, 'label': 'rfi'}).status_code == 400
    assert client.post('/api/review', json={'id': cid, 'label': 'maybe', 'reviewer': 'a'}).status_code == 400
    saved = client.post('/api/review', json={'id': cid, 'label': 'astro', 'reviewer': 'dk', 'note': 'sweep ok',
                                             'dm': 30.1, 'slack': True})
    assert saved.status_code == 200
    reviews = saved.json()['reviews']
    assert reviews[0]['label'] == 'astro' and reviews[0]['slack_ts'] is None   # threads are off by default
    assert cfg.reviews_db.is_file()
    # The index can be thrown away; the verdict survives it.
    cfg.index_db.unlink()
    Indexer(cfg).run_pass()
    page = client_for(cfg).get(f'/verify/{cid}')
    assert 'sweep ok' in page.text


def test_slack_reply_goes_to_the_posts_thread(cfg, campaign):
    from web.slackthread import post_verdict
    Indexer(cfg).run_pass()

    class FakeSlack:
        def __init__(self):
            self.calls = []

        def call(self, method, **params):
            self.calls.append((method, params))
            if method == 'files.info':
                return {'file': {'shares': {'public': {'C1': [{'ts': '123.456'}]}}}}
            return {'ts': '999.1'}

    slack = FakeSlack()
    key = f'candidate|{ITEM}|DM30.000|W3|SN12.000'
    assert post_verdict(cfg, key, 'rfi', 'narrowband', 'dk', 30.0, slack=slack) == '999.1'
    method, params = slack.calls[-1]
    assert method == 'chat.postMessage' and params['thread_ts'] == '123.456' and params['channel'] == 'C1'
    assert 'RFI' in params['text'] and 'narrowband' in params['text']


def test_a_period_in_many_beams_is_rfi_unless_it_keeps_one_dm(cfg, campaign):
    beam_dir = campaign['beam_dir']

    def beam(number):
        path = beam_dir.parents[1] / ITEM.replace('BEAM025', f'BEAM{number:03d}') / beam_dir.name
        path.mkdir(parents=True, exist_ok=True)
        (path / 'metadata.json').write_text((beam_dir / 'metadata.json').read_text())
        return path

    # Narrowband RFI: the same 5.93 s in five beams, at whatever DM each trial happened to have.
    for number, dm in ((25, 1.0), (30, 40.7), (31, 700.0), (40, 3100.0), (41, 0.0)):
        fold(beam(number), dm, 5.9323 * (1 + 1e-4 * (number % 3)), 20.0, f'rfi{number}')
    # A bright pulsar in four neighbouring beams, at one DM.
    for number in (50, 51, 52, 53):
        fold(beam(number), 26.2 + 0.1 * (number % 2), 3.0317926, 50.0, f'psr{number}')
    # A lone fold at an unrelated period.
    fold(beam(60), 55.0, 1.2345, 14.0, 'lone')
    Indexer(cfg).run_pass()
    client = client_for(cfg)

    shown = client.get('/periodic').text
    assert '5.932' not in shown                                  # hidden as multi-beam RFI
    assert shown.count('3.031793') == 4 and '1.234500' in shown  # the pulsar and the lone fold stay
    everything = client.get('/periodic?multibeam=include').text
    assert everything.count('multi-beam</span>') == 5
    rfi = client.get('/periodic?multibeam=include&max_dm=1').text
    cid = rfi.split('href="/verify/')[1].split('?')[0]
    page = client.get(f'/verify/{cid}').text
    assert '5 beams in 1 SAP(s)' in page and 'multi-beam: RFI' in page and 'Same period in other beams' in page

