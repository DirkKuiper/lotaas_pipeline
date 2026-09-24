"""Early-cycle SAPs from SPIDER: their own scope and pace, kept out of the LT5_004 figures."""
import json

from euroflash.campaign import SPIDER_POSITION, State
from web.forecast import update
from web.indexer import Indexer
from web.tests.test_app import client_for

EC = 'L261129_SAP000'
TAR = 'ssh://spider/project/euflash/Data/EC_LOTAAS/261129/SAP000/L261129_SAP000_BEAM0{:02d}_beam_data.tar'


def add_spider_sap(cfg, tmp_path):
    sap_dir = tmp_path / 'prepared' / 'data' / 'L261129' / 'SAP000'
    sap_dir.mkdir(parents=True)
    (sap_dir / 'row-levelling.json').write_text(json.dumps({'row': 512, 'edge_excess': {'32': 1.08, '512': 15.3}}))
    state = State(cfg.state_db)
    with state.db() as db:
        db.execute('INSERT INTO saps(key,position,files,state,detail,sap_dir,run_name,updated,source) '
                   "VALUES (?,?,?,?,NULL,?,'run-ec',100.0,'spider')", (EC, SPIDER_POSITION, 2, 'dispatched', str(sap_dir)))
        for beam, st in ((30, 'searched'), (31, 'online')):
            db.execute('INSERT INTO files(surl,name,sap_key,beam,state,updated) VALUES (?,?,?,?,?,?)',
                       (TAR.format(beam), TAR.format(beam).rsplit('/', 1)[1], EC, beam, st, 100.0))


def test_early_cycle_saps_have_their_own_scope_and_pages(cfg, campaign, tmp_path):
    indexer = Indexer(cfg)
    indexer.sync_state()
    indexer.sync_ledger()
    before = update(indexer.db, cfg, now=1000)['scopes'][0]
    add_spider_sap(cfg, tmp_path)
    indexer.sync_state()
    db = indexer.db
    item = 'downsampled_L261129_SAP000_BEAM030_32bit_ff'
    with db:
        db.execute('INSERT INTO archive_receipts VALUES (?,?,?)',
                   (TAR.format(30), 'L261129_SAP000_BEAM030_beam_data', 1_190_430_720))
        db.execute('INSERT INTO archive_beams VALUES (?,?,?,?,?,?)', (TAR.format(30), 'x.fil', item, 'L261129', 0, 30))
        db.execute("INSERT OR IGNORE INTO runs VALUES ('ec', 0, '{}', 0)")
        for stage, fp in (('retrieve', 'ssh-tar-v1'), ('classify', 'ec')):
            db.execute('INSERT INTO attempts(item,stage,fingerprint,status,started,finished) VALUES (?,?,?,?,?,?)',
                       (item if stage == 'classify' else 'L261129_SAP000_BEAM030_beam_data', stage, fp, 'success', 900, 950))
    assert dict(db.execute('SELECT key, source FROM saps').fetchall())[EC] == 'spider'
    result = update(db, cfg, now=1000)
    inventory = result['scopes'][0]
    assert inventory['files'] == before['files'], 'LT5_004 figures leave SPIDER beams out'
    early = next(s for s in result['scopes'] if s['key'] == 'ec')
    assert (early['files'], early['searched_files'], early['retrieved_files']) == (2, 1, 1)
    assert early['sap_states'] == {'dispatched': 1} and early['observations'] == 1
    assert early['bytes'] == 2 * 1_190_430_720, 'the unfetched beam is taken at the fetched size'
    assert next(p for p in early['projections'] if p['hours'] == 24)['search_days'] == 1
    assert any(p['project'] == 'EC_LOTAAS' for p in result['projects'])
    client = client_for(cfg)
    assert 'Early cycles from SPIDER' in client.get('/').text
    assert 'Early-cycle LOTAAS from SPIDER' in client.get('/?project=EC_LOTAAS').text
    coverage = client.get('/coverage?project=EC_LOTAAS').text
    assert EC in coverage and 'L1163405_SAP000' not in coverage
    sap = client.get(f'/sap/{EC}').text
    assert 'SPIDER, early cycle' in sap and '512 samples (4.03 s), levelled' in sap


def test_saps_indexed_before_the_source_column_are_corrected(cfg, campaign, tmp_path):
    indexer = Indexer(cfg)
    add_spider_sap(cfg, tmp_path)
    indexer.sync_state()
    with indexer.db:
        indexer.db.execute("UPDATE saps SET source='lta'")        # as an index made before the column left it
    indexer.sync_state()
    assert dict(indexer.db.execute('SELECT key, source FROM saps').fetchall())[EC] == 'spider'
