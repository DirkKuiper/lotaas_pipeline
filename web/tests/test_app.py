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
    for path in ['/', '/staging', '/coverage', '/sap/L1163405_SAP000', f'/beam/{ITEM}', '/candidates',
                 '/candidates?type=all&review=unreviewed', f'/verify/{cid}', f'/verify/{reject}',
                 f'/api/sp/{cid}/view?dm=0&nsub=16', f'/api/sp/{cid}/dm', '/api/health', f'/snippet/{cid}.fil']:
        assert client.get(path).status_code == 200, path
    assert client.get('/verify', follow_redirects=False).status_code == 303
    assert client.get('/sap/nope').status_code == 404


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
