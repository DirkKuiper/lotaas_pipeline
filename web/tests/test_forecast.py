import csv
import json

import pytest

from web import store
from web.forecast import catalogue, projection, update
from web.indexer import Indexer
from web.tests.test_app import client_for
from web.tests.conftest import ITEM


def write_catalogue(cfg, groups):
    """Project, observation, source label, [(name, bytes, uri), ...]."""
    root = cfg.observation_catalogue
    (root / 'csv').mkdir(parents=True, exist_ok=True)
    projects = {}
    with (root / 'observations.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['project_id', 'obsid', 'nfiles', 'totsize_gb', 'obstype'])
        for project, obs, kind, files in groups:
            writer.writerow([project, obs, len(files), f'{sum(f[1] for f in files)/1e9:.3f}', kind])
            projects.setdefault(project, []).extend((name, size, '2025-01-01', uri, obs.lstrip('L'))
                                                    for name, size, uri in files)
    for project, files in projects.items():
        with (root / 'csv' / (project.lower() + '.csv')).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['FILENAME', 'FILESIZE', 'CREATION_DATE', 'URI', 'OBSERVATIONID'])
            writer.writerows(files)


def archive(name, size):
    return name, size, 'srm://x/' + name


def test_catalogue_scopes_bytes_and_old_gzip_layout(cfg):
    write_catalogue(cfg, [
        ('LT5_004', 'L559289', 'survey', [
            archive('L1163405_SAP000_B025_P000_bf_aa.tar', 5_000_000_000),
            archive('L1163405_SAP000_B012_P000_bf_ab.tar', 6_000_000_000),
            archive('L1163405_summaryCS_ac.tar', 7_000_000_000)]),
        ('LC3_014', 'L253404', 'survey', [
            archive('L253404_SAP000_B025_P000_bf.tar_aa.gz', 20_000_000_000)]),
        ('LT5_004', 'L10', 'confirmation', [archive('L11_SAP000_B025_P000_bf_aa.tar', 2_000_000_000)]),
        ('LT5_004', 'L20', 'unknown', [archive('L21_SAP000_B025_P000_bf_aa.tar', 3_000_000_000)]),
    ])
    db = store.index(cfg.prepare())
    result = update(db, cfg, now=1000)
    assert result['catalogue']['available']
    assert result['catalogue']['kinds'] == {'survey': 2, 'confirmation': 1, 'unknown': 1}
    assert result['catalogue']['all_archive_bytes'] == 43_000_000_000
    scopes = {s['key']: s for s in result['scopes']}
    assert scopes['lt5']['bytes'] == 5_000_000_000
    assert scopes['survey']['bytes'] == 25_000_000_000
    assert scopes['survey']['files'] == 2 and scopes['survey']['gzip_files'] == 1
    assert scopes['survey']['remaining_beams'] == 2
    assert all(p['pace_days'] is None for p in scopes['survey']['projections'])


def test_progress_uses_receipts_and_first_production_success_not_current_state(cfg, campaign):
    known = 'L1163405_SAP000_B025_P000_bf_aa.tar'
    pending = 'L1163405_SAP001_B000_P000_bf_bb.tar'
    write_catalogue(cfg, [('LT5_004', 'L559289', 'survey', [archive(known, 5_000_000_000),
                                                                    archive(pending, 9_000_000_000)])])
    indexer = Indexer(cfg)
    indexer.sync_state()
    indexer.sync_ledger()
    db = indexer.db
    pending_item = ITEM.replace('SAP000_BEAM025', 'SAP001_BEAM000')
    with db:
        db.execute('INSERT INTO archive_receipts VALUES (?,?,?)', ('srm://x/' + known, known[:-4], 5_000_000_000))
        db.execute('INSERT INTO runs VALUES (?,?,?,?)', ('pilot', 1, '{}', 0))
        db.execute('INSERT INTO runs VALUES (?,?,?,?)', ('new', 0, '{}', 0))
        db.execute('INSERT INTO archive_beams VALUES (?,?,?,?,?,?)',
                   ('srm://x/' + pending, 'pending.fits', pending_item, 'L559289', 1, 0))
        # Original success at t=85 is still in the fixture. Later retries must not
        # inflate rate or shift that first success into a more recent window.
        for item, stage, fp, status, finished in [
                (known[:-4], 'retrieve', 'download', 'success', 80),
                (known[:-4], 'retrieve', 'download', 'success', 90000),
                (ITEM, 'classify', 'new', 'success', 90000),
                (pending_item, 'classify', 'pilot', 'success', 95000),
                (pending_item, 'classify', 'new', 'failed', 95000)]:
            db.execute('INSERT INTO attempts(item,stage,fingerprint,status,started,finished) VALUES (?,?,?,?,?,?)',
                       (item, stage, fp, status, finished - 1, finished))
    result = update(db, cfg, now=100000)
    assert all(r['bytes_per_day'] == r['beams_per_day'] == 0 for r in result['rates'])
    current = result['scopes'][0]
    assert current['files'] == 2 and current['bytes'] == 14_000_000_000
    assert current['remaining_bytes'] == 9_000_000_000
    assert current['remaining_beams'] == 1 and current['searched_files'] == 1
    assert all(p['pace_days'] is None for p in current['projections'])
    result = update(db, cfg, now=1000)
    day = next(r for r in result['rates'] if r['hours'] == 24)
    assert day['bytes_per_day'] == 5_000_000_000
    assert day['beams_per_day'] == 1
    current = result['scopes'][0]
    projection_day = next(p for p in current['projections'] if p['hours'] == 24)
    assert projection_day['download_days'] == pytest.approx(1.8)
    assert projection_day['search_days'] == 1
    assert projection_day['pace_days'] == pytest.approx(1.8)
    # A conversion becoming searched never removes its historical arrival.
    with db:
        db.execute("UPDATE files SET state='kept',updated=990 WHERE beam=25")
    assert update(db, cfg, now=1000)['rates'] == result['rates']


def test_receipt_bytes_are_mirrored_from_ledger(cfg, campaign):
    uri = 'srm://x/L1163405_SAP000_B025_P000_bf_aa.tar'
    with campaign['ledger'].connect() as db:
        db.execute('INSERT INTO inputs VALUES (?,?,?,?,?)', (uri, 'LT5_004', 0, 'test', 'online'))
        db.execute('INSERT INTO archive_receipts VALUES (?,?,?,?,?,?)',
                   (uri, 7, 'https://archive/file.tar', 'abc', 5_000_000_000, '/test/receipt.json'))
    indexer = Indexer(cfg)
    indexer.sync_ledger()
    row = indexer.db.execute('SELECT * FROM archive_receipts').fetchone()
    assert dict(row) == {'uri': uri, 'item': 'L1163405_SAP000_B025_P000_bf_aa', 'bytes': 5_000_000_000}


def test_missing_or_invalid_catalogue_does_not_claim_zero_remaining_bytes(cfg, campaign):
    indexer = Indexer(cfg)
    indexer.sync_state()
    indexer.sync_ledger()
    result = update(indexer.db, cfg, now=1000)
    assert not result['catalogue']['available']
    assert len(result['scopes']) == 1
    assert result['scopes'][0]['unknown_sizes'] == 2
    assert all(p['pace_days'] is None for p in result['scopes'][0]['projections'])
    write_catalogue(cfg, [('LT5_004', 'L1', 'survey', [archive('L2_SAP000_B025_P000_bf_aa.tar', 5_000_000_000)])])
    assert catalogue(indexer.db, cfg.observation_catalogue)['available']
    summary = cfg.observation_catalogue / 'observations.csv'
    summary.write_text(summary.read_text().replace(',1,5.000,', ',2,5.000,'))
    assert not catalogue(indexer.db, cfg.observation_catalogue)['available']
    assert update(indexer.db, cfg, now=1000)['scopes'][0]['unknown_sizes'] == 2


def test_complete_stage_needs_no_rate_and_incomplete_saps_still_count(cfg, campaign):
    assert projection({'unknown_sizes': 0, 'remaining_bytes': 0, 'remaining_beams': 0},
                      {'hours': 24, 'bytes_per_day': 0, 'beams_per_day': 0})['pace_days'] == 0
    indexer = Indexer(cfg)
    indexer.sync_state()
    indexer.sync_ledger()
    with indexer.db:
        indexer.db.execute("UPDATE saps SET state='incomplete' WHERE key='L1163405_SAP001'")
    current = update(indexer.db, cfg)['scopes'][0]
    assert current['files'] == 2 and current['blocked_files'] == current['incomplete_saps'] == 1


def test_dashboard_exposes_catalogue_scope_and_authenticated_forecast(cfg, campaign):
    write_catalogue(cfg, [('LT5_004', 'L559289', 'survey', [
        archive('L1163405_SAP000_B025_P000_bf_aa.tar', 5_000_000_000),
        archive('L1163405_SAP001_B000_P000_bf_bb.tar', 9_000_000_000)])])
    indexer = Indexer(cfg)
    indexer.sync_state()
    indexer.sync_ledger()
    result = update(indexer.db, cfg)
    client = client_for(cfg)
    response = client.get('/')
    assert response.status_code == 200
    assert 'All LOTAAS projects' in response.text
    assert '14.0 GB' in response.text and 'conditional projections' in response.text
    assert client.get('/api/forecast').json()['scopes'][2]['bytes'] == 14_000_000_000
    assert client_for(cfg, authenticated=False).get('/api/forecast').status_code == 401
    # The index persisted the same forecast the page consumes.
    assert json.loads(indexer.db.execute("SELECT value FROM meta WHERE name='forecast'").fetchone()[0]) == result


def test_all_projects_saps_include_unqueued_non_survey_and_unmapped_observations(cfg, campaign):
    known = archive('L1163405_SAP000_B025_P000_bf_aa.tar', 5_000_000_000)
    write_catalogue(cfg, [
        ('LT5_004', 'L559289', 'survey', [known,
            archive('L999_SAP000_B026_P000_bf_new.tar', 3_000_000_000),
            archive('L1163405_SAP001_B000_P000_bf_bb.tar', 9_000_000_000)]),
        # Same observation/SAP identifier in another project must stay distinct.
        ('LC3_014', 'L559289', 'confirmation', [
            archive('L559289_SAP000_B001_S0_P000_bf.h5_old.tar', 7_000_000_000)]),
        ('LC3_014', 'L22', 'unknown', [archive('L22_SAP002_B000_P000_bf.tar_old.gz', 2_000_000_000)]),
        ('LC0_034', 'L23', 'unknown', [archive('legacy_unknown.tar', 4_000_000_000)]),
        ('LT5_004', 'L24', 'unknown', [archive('L24_summaryCS.tar', 1_000_000_000)]),
    ])
    indexer = Indexer(cfg)
    indexer.sync_state()
    indexer.sync_ledger()
    result = update(indexer.db, cfg, now=1000)
    all_scope = next(s for s in result['scopes'] if s['key'] == 'all')
    assert all_scope['observations'] == 5
    assert all_scope['saps'] == 4
    assert all_scope['queued_saps'] == 2
    assert all_scope['sap_states']['not_queued'] == 2
    assert all_scope['unmapped_observations'] == 2
    # A queue SAP marked searched is still incomplete against the full catalogue.
    assert all_scope['searched_files'] == 1 and all_scope['searched_saps'] == 0
    assert all_scope['files'] == 5 and all_scope['bytes'] == 26_000_000_000
    assert all_scope['all_archive_bytes'] == 31_000_000_000
    for key in ('observations', 'saps', 'files', 'bytes', 'queued_saps', 'searched_files', 'unmapped_observations'):
        assert sum(p[key] for p in result['projects']) == all_scope[key]
    client = client_for(cfg)
    overview = client.get('/').text
    assert '<h1>All LOTAAS projects</h1>' in overview and '/ 4</span>' in overview
    assert '2 additional observations need SAP mapping' in overview
    filtered = client.get('/?project=LC3_014').text
    assert '<h1>LC3_014</h1>' in filtered and '/ 2</span>' in filtered
    assert '9.0 GB' in filtered
    coverage = client.get('/coverage').text
    assert coverage.count('<tr data-state=') == 4 and 'L23' in coverage and 'L24' in coverage
    filtered_coverage = client.get('/coverage?project=LC3_014').text
    assert filtered_coverage.count('<tr data-state=') == 2 and 'L23' not in filtered_coverage
    sap = client.get('/catalogue/LC3_014/L559289/0')
    assert sap.status_code == 200 and 'h5_old.tar' in sap.text and 'not queued' in sap.text
    assert client.get('/?project=nonexistent').status_code == 404
    assert client.get('/catalogue/LC3_014/L559289/9').status_code == 404
    assert client_for(cfg, authenticated=False).get('/catalogue/LC3_014/L559289/0').status_code == 401


def test_archive_needs_all_mapped_beams_completed(cfg, campaign):
    known = archive('L1163405_SAP000_B025_P000_bf_aa.tar', 5_000_000_000)
    write_catalogue(cfg, [('LT5_004', 'L559289', 'survey', [known])])
    indexer = Indexer(cfg)
    indexer.sync_state()
    indexer.sync_ledger()
    result = update(indexer.db, cfg, now=1000)
    assert result['scopes'][-1]['searched_saps'] == 1
    with indexer.db:
        indexer.db.execute('INSERT INTO archive_beams VALUES (?,?,?,?,?,?)',
                          (known[2], 'second.fits', 'not_yet_searched', 'L559289', 0, 26))
    result = update(indexer.db, cfg, now=1000)
    assert result['scopes'][-1]['searched_saps'] == result['scopes'][-1]['searched_files'] == 0
