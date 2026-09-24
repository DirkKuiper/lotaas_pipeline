"""SPIDER beams: streamed, checked and corrected on the way in, levelled per 2-bit row, queued behind LT5_004."""
import hashlib
import importlib.util
import io
import json
import os
import sqlite3
import struct
import tarfile
from pathlib import Path

import pytest

from euroflash import campaign as C
from euroflash import spider
from euroflash.ledger import Ledger
from tests.test_campaign import FULL, build, put_online, settle

EDGE = 151.068115234375
FOFF = -0.048828125


def header(fch1=EDGE, nchans=4, tsamp=1.0, nbits=32, source='LOTAAS-P0442A-SAP0'):
    def string(s):
        return struct.pack('<i', len(s)) + s.encode()
    out = string('HEADER_START') + string('source_name') + string(source)
    for key, value in (('telescope_id', 11), ('nchans', nchans), ('nbits', nbits), ('nifs', 1)):
        out += string(key) + struct.pack('<i', value)
    for key, value in (('fch1', fch1), ('foff', FOFF), ('tsamp', tsamp), ('tstart', 57085.3)):
        out += string(key) + struct.pack('<d', value)
    return out + string('HEADER_END')


def beam_tar(path, obs='261129', sap='000', beam='030', samples=700, member=None, keep=0):
    """A beam tar as the early-cycle pipeline wrote it: filterbank plus RFI plot."""
    data = header() + bytes(range(256)) * (samples * 16 // 256) + bytes(samples * 16 % 256)
    name = member or f'downsampled_L{obs}_SAP{sap}_BEAM{beam}_32bit.fil'
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, 'w') as archive:
        for member_name, payload in ((name, data), (name.replace('.fil', '_rfi_diagnostic_plot.png'), b'png')):
            info = tarfile.TarInfo(member_name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    if keep:
        # Cut inside the filterbank, not just the tar's zero padding.
        with path.open('r+b') as f:
            f.truncate(keep)
    return data


@pytest.fixture
def fake_ssh(tmp_path, monkeypatch):
    """`ssh [-o opt]... destination command` runs the command here; exit 255 while `down` exists."""
    bin_dir = tmp_path/'bin'
    bin_dir.mkdir()
    script = bin_dir/'ssh'
    script.write_text('#!/bin/sh\n'
                      '[ -e "$FAKE_SSH_DOWN" ] && { echo "Permission denied (publickey)." >&2; exit 255; }\n'
                      'while [ "$1" = "-o" ]; do shift 2; done\n'
                      'shift\n'
                      'exec sh -c "$1"\n')
    script.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}:{os.environ["PATH"]}')
    down = tmp_path/'ssh-down'
    monkeypatch.setenv('FAKE_SSH_DOWN', str(down))
    return down


def url(path):
    return 'ssh://spider' + str(path)


def test_a_beam_arrives_converted_with_fch1_at_the_first_channel_centre(tmp_path, fake_ssh):
    tar = tmp_path/'EC'/'261129'/'SAP000'/'L261129_SAP000_BEAM030_beam_data.tar'
    data = beam_tar(tar)
    out = tmp_path/'out'/'downsampled_L261129_SAP000_BEAM030_32bit.fil'
    receipt = spider.fetch(url(tar), out, tmp_path/'r.json')
    written = out.read_bytes()
    found, length, _ = spider.sigproc_header(written)
    assert found['fch1'] == EDGE + FOFF / 2 == 151.043701171875, 'the value conversion from PSRFITS gives'
    assert written[length:] == data[length:], 'the data are untouched'
    assert receipt['sha256'] == hashlib.sha256(tar.read_bytes()).hexdigest()
    assert receipt['bytes'] == tar.stat().st_size and receipt['samples'] == 700
    assert receipt['fch1_archive'] == EDGE
    assert json.loads((tmp_path/'r.json').read_text()) == receipt
    assert not list(out.parent.glob('*.partial'))


def test_unpadded_early_names_are_the_same_beam(tmp_path, fake_ssh):
    tar = tmp_path/'L169690'/'SAP0'/'L169690_SAP0_BEAM13_beam_data.tar'
    beam_tar(tar, obs='169690', sap='0', beam='13')
    assert spider.parse(url(tar))['beam'] == 13
    out = tmp_path/'downsampled_L169690_SAP000_BEAM013_32bit.fil'
    spider.fetch(url(tar), out, tmp_path/'r.json')
    assert out.is_file()


@pytest.mark.parametrize('damage, error', [
    ({'keep': 512 + 5000}, 'ended|truncated|bytes|unexpected end'),
    ({'member': 'downsampled_L261129_SAP000_BEAM031_32bit.fil'}, 'is not beam'),
    ({'samples': 100}, 'about an hour'),
])
def test_a_damaged_or_wrong_beam_leaves_nothing_behind(tmp_path, fake_ssh, damage, error):
    tar = tmp_path/'261129'/'SAP000'/'L261129_SAP000_BEAM030_beam_data.tar'
    beam_tar(tar, **damage)
    out = tmp_path/'out'/'b.fil'
    with pytest.raises(Exception, match=error):
        spider.fetch(url(tar), out, tmp_path/'r.json')
    assert not out.exists() and not (tmp_path/'r.json').exists()
    assert not list((tmp_path/'out').glob('*'))


def test_a_missing_file_is_not_mistaken_for_a_broken_connection(tmp_path, fake_ssh):
    with pytest.raises(FileNotFoundError):
        spider.fetch(url(tmp_path/'L1_SAP000_BEAM013_beam_data.tar'), tmp_path/'b.fil', tmp_path/'r.json')
    fake_ssh.touch()
    with pytest.raises(spider.Unreachable):
        spider.fetch(url(tmp_path/'L1_SAP000_BEAM013_beam_data.tar'), tmp_path/'b.fil', tmp_path/'r.json')


def test_the_inventory_lists_survey_saps_and_skips_confirmations_and_work(tmp_path, fake_ssh):
    root = tmp_path/'EC_LOTAAS'
    for beam in (13, 14):
        (root/'261129'/'SAP000').mkdir(parents=True, exist_ok=True)
        (root/'261129'/'SAP000'/f'L261129_SAP000_BEAM{beam:03d}_beam_data.tar').write_bytes(b'x')
    (root/'254378'/'SAP000').mkdir(parents=True)
    for beam in (0, 127):
        (root/'254378'/'SAP000'/f'L254378_SAP000_BEAM{beam:03d}_beam_data.tar').write_bytes(b'x')
    (root/'L169690'/'_work'/'SAP0').mkdir(parents=True)
    (root/'L169690'/'_work'/'SAP0'/'L169690_SAP0_BEAM13_beam_data.tar').write_bytes(b'x')
    urls, skipped = spider.inventory('spider', str(root))
    assert [spider.parse(u)['beam'] for u in urls] == [13, 14]
    assert all(u.startswith('ssh://spider' + str(root)) for u in urls)
    assert skipped == ['L254378_SAP000: 2 beams, a confirmation observation']
    assert len(spider.inventory('spider', str(root), confirmation=True)[0]) == 4


def spider_campaign(tmp_path, monkeypatch, saps, lta=(), spider_saps=1, **overrides):
    """A campaign with LTA SAPs (fake StageIT) and SPIDER SAPs (fake ssh)."""
    urls = []
    for obs, sap, beams in saps:
        for beam in beams:
            tar = tmp_path/'EC'/obs/f'SAP{sap:03d}'/f'L{obs}_SAP{sap:03d}_BEAM{beam:03d}_beam_data.tar'
            beam_tar(tar, obs=obs, sap=f'{sap:03d}', beam=f'{beam:03d}')
            urls.append(url(tar))
    inventory = tmp_path/'spider.txt'
    inventory.write_text('# test\n' + '\n'.join(urls) + '\n')
    campaign, api, where = build(tmp_path, monkeypatch, list(lta), **overrides)
    campaign.o.spider_saps = spider_saps
    campaign.o.spider_workers = 2
    campaign.o.spider_backoff_seconds = 600
    campaign.state.load(C.parse_inventory(inventory), C.missing_central_beams, tuple(campaign.o.exclude_beams),
                        3, source='spider')
    return campaign, api, where


def test_spider_saps_need_no_staging_and_arrive_ready_to_flatfield(tmp_path, monkeypatch, fake_ssh):
    campaign, api, _ = spider_campaign(tmp_path, monkeypatch, [('261129', 0, FULL), ('261129', 1, FULL)])
    campaign.tick()                      # admitted: every beam online at once
    campaign.tick()                      # fetched, flatfield started
    assert api.requests == {}, 'nothing is asked of StageIT'
    settle(campaign)
    saps = {r['key']: r for r in campaign.state.rows('SELECT * FROM saps')}
    assert saps['L261129_SAP000']['state'] == 'prepared' and saps['L261129_SAP000']['source'] == 'spider'
    sap_dir = Path(saps['L261129_SAP000']['sap_dir'])
    assert sap_dir == tmp_path/'campaign'/'prepared'/'data'/'L261129'/'SAP000'
    assert len(list(sap_dir.glob('B*/*_ff.fil'))) == 73, 'beam 12 is excluded as for LT5_004'
    assert campaign.runner.levelled == [(sap_dir, 'auto')], 'early-cycle beams are levelled per 2-bit row'
    assert saps['L261129_SAP001']['state'] == 'pending', '--spider-saps bounds the SAPs in flight'
    with Ledger(tmp_path/'ledger.sqlite').connect() as db:
        assert db.execute("SELECT COUNT(*) FROM attempts WHERE stage='retrieve' AND fingerprint='ssh-tar-v1' "
                          "AND status='success'").fetchone()[0] == 73
        beams = {r['item'] for r in db.execute('SELECT item FROM archive_beams')}
        assert 'downsampled_L261129_SAP000_BEAM030_32bit_ff' in beams
    assert campaign.status()['spider']['saps'] == {'prepared': 1, 'pending': 1}


def test_lt5_keeps_first_claim_and_spider_saps_do_not_block_its_staging(tmp_path, monkeypatch, fake_ssh):
    campaign, api, where = spider_campaign(tmp_path, monkeypatch, [('261129', 0, FULL)],
                                           lta=[('1000001', 0, FULL), ('1000001', 1, FULL)],
                                           max_staging_saps=1, max_prepared_saps=1)
    campaign.tick()
    campaign.tick()
    settle(campaign)
    positions = {r['key']: r['position'] for r in campaign.state.rows('SELECT key,position FROM saps')}
    assert positions['L261129_SAP000'] > max(positions['L1000001_SAP000'], positions['L1000001_SAP001'])
    assert campaign.state.rows("SELECT state FROM saps WHERE key='L261129_SAP000'")[0]['state'] == 'prepared'
    assert len(api.requests) == 1, 'the prepared SPIDER SAP does not use up the LTA prepared limit'
    put_online(api, where, '1000001', 0, FULL)
    campaign.tick()
    settle(campaign)
    ready = campaign.state.rows("SELECT key FROM saps WHERE state='prepared' ORDER BY position")
    assert [r['key'] for r in ready] == ['L1000001_SAP000', 'L261129_SAP000'], 'LT5_004 is dispatched first'
    levelled = {'spider' if 'L261129' in str(sap) else 'lta': rows for sap, rows in campaign.runner.levelled}
    assert levelled == {'spider': 'auto', 'lta': None}, 'only early-cycle beams are levelled'


def test_a_broken_connection_pauses_spider_without_striking_its_files(tmp_path, monkeypatch, fake_ssh):
    campaign, _, _ = spider_campaign(tmp_path, monkeypatch, [('261129', 0, FULL)])
    fake_ssh.touch()
    campaign.tick()
    campaign.tick()
    files = campaign.state.rows("SELECT state,failures FROM files WHERE beam!=12")
    assert {f['state'] for f in files} == {'online'} and not any(f['failures'] for f in files)
    assert campaign.spider_throttle_until > 0
    assert campaign.retrieve() == 0, 'no fetch during the back-off'
    fake_ssh.unlink()
    campaign.spider_throttle_until = 0
    campaign.tick()
    campaign.tick()
    settle(campaign)
    assert campaign.state.rows("SELECT state FROM saps")[0]['state'] == 'prepared'


def test_a_state_written_before_spider_gets_the_source_column(tmp_path):
    path = tmp_path/'campaign-state.sqlite'
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE saps (key TEXT PRIMARY KEY, position INTEGER NOT NULL, files INTEGER NOT NULL, '
               'state TEXT NOT NULL, detail TEXT, sap_dir TEXT, run_name TEXT, updated REAL)')
    db.execute("INSERT INTO saps VALUES ('L1_SAP000', 0, 74, 'searched', NULL, NULL, NULL, 0)")
    db.commit()
    db.close()
    assert C.State(path).rows('SELECT source FROM saps') == [{'source': 'lta'}]


# ------------------------------------------------------------ row levelling (preproc/flatfield_fil.py)
np = pytest.importorskip('numpy')


def flatfield_module():
    spec = importlib.util.spec_from_file_location('flatfield_fil', Path(__file__).parents[1]/'preproc'/'flatfield_fil.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stepped(rows_of, nsamp=32 * 512 * 4, nchan=16, step=3.0, seed=1):
    """Channels x samples of white noise around 100 with a random level per row, as the conversion left it."""
    rng = np.random.default_rng(seed)
    data = 100 + rng.normal(size=(nchan, nsamp))
    if rows_of:
        # Mostly shared by the channels, as the real steps are (they stand out at DM 0).
        rows = nsamp // rows_of
        levels = rng.normal(scale=step, size=(1, rows)) + rng.normal(scale=step / 4, size=(nchan, rows))
        data += np.repeat(levels, rows_of, axis=1)
    return data.astype(np.float32)


@pytest.mark.parametrize('row', [32, 512, 0])
def test_the_row_length_is_found_from_the_steps(row):
    F = flatfield_module()
    beams = [stepped(row, step=0.5, seed=seed).sum(axis=0, dtype=np.float64) for seed in range(5)]
    found, excess = F.detect_row_length(beams)
    assert found == row, excess


def test_levelling_removes_the_row_steps_and_keeps_a_pulse_within_a_row():
    F = flatfield_module()
    data = stepped(512)
    data[:, 1000:1004] += 5.0                      # a 4-sample pulse inside one row
    F.level_rows(data, 512)
    means = data[:, :(data.shape[1] // 512) * 512].reshape(data.shape[0], -1, 512).mean(axis=2)
    assert np.ptp(means, axis=1).max() < 1e-3, 'every row now sits at its channel level'
    assert np.isclose(np.median(means), 100, atol=0.5), 'the level is kept for the flatfield'
    pulse = (data[:, 1000:1004].mean() - np.median(data)) / 5.0
    assert 0.95 < pulse < 1.01, 'a pulse loses only its share of the row mean (4/512)'
