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
    periodic = client.get('/periodic?triage=all').text
    assert '3.031793' in periodic and '21.900000' in periodic
    assert '0.020000' not in periodic            # RFI-like folds are listed only when asked for
    assert 'B012' not in periodic                # the incoherent beam is out of the campaign
    assert '0.020000' in client.get('/periodic?type=periodic_rfi').text
    assert 'B012' in client.get('/periodic?incoherent=include&triage=all').text
    assert '21.900000' not in client.get('/periodic?max_dm=1000&triage=all').text
    # At DM 4000 a pulsar would be scattered far beyond a 22 s period; at DM 26 it would not.
    assert periodic.count('τ &gt; P') == 1

    first = client.get('/verify?kind=periodic&triage=all', follow_redirects=False)
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
                                             'dm': 30.1})
    assert saved.status_code == 200
    reviews = saved.json()['reviews']
    assert reviews[0]['label'] == 'astro' and reviews[0]['dm'] == 30.1
    assert cfg.reviews_db.is_file()
    # The index can be thrown away; the verdict survives it.
    cfg.index_db.unlink()
    Indexer(cfg).run_pass()
    page = client_for(cfg).get(f'/verify/{cid}')
    assert 'sweep ok' in page.text


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

    shown = client.get('/periodic?triage=all').text
    assert '5.932' not in shown                                  # hidden as multi-beam RFI
    assert shown.count('3.031793') == 4 and '1.234500' in shown  # the pulsar and the lone fold stay
    everything = client.get('/periodic?multibeam=include&triage=all').text
    assert everything.count('multi-beam</span>') == 5
    rfi = client.get('/periodic?multibeam=include&max_dm=1&triage=all').text
    cid = rfi.split('href="/verify/')[1].split('?')[0]
    page = client.get(f'/verify/{cid}').text
    assert '5 beams in 1 SAP(s)' in page and 'multi-beam: RFI' in page and 'Same period in other beams' in page


def add_detections(campaign, rows):
    """(beam number, dm, snr, width, type, pulsar, time) detections of ITEM's observation in the ledger."""
    with campaign['ledger'].connect() as db:
        run = db.execute('SELECT MAX(id) FROM beam_runs').fetchone()[0]
        for number, dm, snr, width, kind, pulsar, time in rows:
            beam = ITEM.replace('BEAM025', f'BEAM{number:03d}') + '.fil'
            db.execute("INSERT INTO detections(beam_id,candidate_dm,snr,width_samples,detection_type,pulsar_name,"
                       "classification_probability,beam_run_id,time_seconds,sample_number) VALUES "
                       "(?,?,?,?,?,?,NULL,?,?,1)", (beam, dm, snr, width, kind, pulsar, run, time))


def test_an_event_in_many_beams_at_scattered_dms_leaves_the_queue(cfg, campaign):
    # A level step: one moment in six beams, each at whatever DM suited it. Aligned
    # for DM (t + 0.0489 DM), the six events fall within half a second.
    step = [(40 + i, dm, 35.0, 6, 'unconfirmed', None, 40.0 - 0.0489 * dm) for i, dm in
            enumerate((3.7, 22.4, 85.9, 139.9, 5.8))]
    add_detections(campaign, step + [(45, 22.6, 36.0, 6, 'candidate', None, 40.0 - 0.0489 * 22.6)])
    # A bright pulsar's pulse in five neighbouring beams, all at its DM.
    add_detections(campaign, [(50 + i, 26.2, 20.0 - i, 1, 'known_pulsar', 'J0323+3944', 2465.645) for i in range(5)])
    Indexer(cfg).run_pass()
    client = client_for(cfg)
    shown = client.get('/single-pulse').text
    assert 'B045' not in shown and '(1 hidden)' in shown
    assert shown.count('5 beams</span>') == 5                    # the pulsar's pulses stay, marked
    everything = client.get('/single-pulse?coincident=include').text
    assert 'B045' in everything and '6 beams</span>' in everything
    cid = everything.split('B045')[0].rsplit('href="/verify/', 1)[1].split('?')[0]
    page = client.get(f'/verify/{cid}').text
    assert 'Same moment in 6 beams of 1 SAP' in page and 'scattered DMs: interference' in page


def test_known_pulsars_no_lofar_paper_reports_are_listed_with_what_was_found(cfg, campaign, monkeypatch, tmp_path):
    from web import lofar
    catalogue = tmp_path/'psrcat.db'
    catalogue.write_text(
        'PSRJ     J0847+6830\nRAJ      08:47:30.0\nDECJ     +68:30:00\nDM       30.0\nP0       0.5\n'
        'S400     12.0\nSURVEY   gbncc\n@----\n'
        'PSRJ     J0850+6800\nRAJ      08:50:00.0\nDECJ     +68:00:00\nDM       55.0   0.1  bkk+16\n'
        'P0       1.2\nS150     45.0\n@----\n'
        'PSRJ     J0849+6810\nRAJ      08:49:00.0\nDECJ     +68:10:00\nDM       20.0\nP0       0.8\nSURVEY   lotaas\n@----\n'
        'PSRJ     J0846+6820\nRAJ      08:46:00.0\nDECJ     +68:20:00\nDM       40.0\nP0       0.0031\n@----\n'
        'PSRJ     J0848+6815\nRAJ      08:48:00.0\nDECJ     +68:15:00\nDM       400.0\nP0       0.5\n@----\n'
        'PSRJ     J0845+6900\nPSRB     B0840+69\nRAJ      08:45:00.0\nDECJ     +69:00:00\nDM       10.0\n'
        'P0       0.7\nS400     12.0\n@----\n'
        'PSRJ     J1200+4500\nRAJ      12:00:00.0\nDECJ     +45:00:00\nDM       20.0\nP0       0.9\n@----\n')
    (tmp_path/'psrcat_ref').write_text('***bkk+16  bkk+16:  Bilous, A. V., Kondratiev, V. I. & et al., 2016. '
                                       'A LOFAR census of non-recycled pulsars. aap, 591, A134.\n\n'
                                       '***xyz+20  xyz+20:  Other, A., 2020. Pulsars at 1.4 GHz. apj, 1, 1.\n')
    census = tmp_path/'lofar-pulsars.tsv'
    census.write_text('# test\nname\treference\tdetected\tlimit_mjy\nB0840+69\tBilous et al. 2016\t0\t5.0\n')
    monkeypatch.setenv('LOTAAS_PSRCAT', str(catalogue))
    monkeypatch.setattr(lofar, 'TABLE', census)
    add_detections(campaign, [(25, 30.0, 14.0, 2, 'known_pulsar', 'J0847+6830', 812.0),
                              (25, 10.1, 9.0, 2, 'known_pulsar', 'J0845+6900', 1200.0)])
    fold(campaign['beam_dir'], 30.1, 0.25 * (1 + 4e-5), 80.0, 'half')     # its second harmonic
    Indexer(cfg).run_pass()
    client = client_for(cfg)
    # A reviewer found the J0845+6900 'redetection' to be noise: it no longer counts.
    cid = client.get('/single-pulse').text.split('J0845+6900')[0].rsplit('href="/verify/', 1)[1].split('?')[0]
    assert client.post('/api/review', json={'id': cid, 'label': 'noise', 'reviewer': 'Dirk'}).status_code == 200
    Indexer(cfg).run_pass()
    page = client.get('/pulsars').text
    table = page.split('<tbody>')[1]
    assert 'J0850+6800' not in table and 'J0849+6810' not in table and 'J1200+4500' not in table
    row = table.split('J0847+6830')[1].split('</tr>')[0]
    assert 'gbncc' in row and 'best S/N 14.0' in row and '(1/2)' in row and 'both · first LOFAR?' in row
    row = table.split('J0845+6900')[1].split('</tr>')[0]
    assert 'not detected: Bilous et al. 2016: &lt; 5 mJy' in row and 'flag">not found' in row
    assert 'scattered beyond P' in table.split('J0848+6815')[1].split('</tr>')[0]
    assert 'P below the search' in table.split('J0846+6820')[1].split('</tr>')[0]
    # Found, then within reach and not found, then scattered, then too fast.
    assert table.index('J0847+6830') < table.index('J0845+6900') < table.index('J0848+6815') < table.index('J0846+6820')
    flat = page.replace('\n', '')
    assert '>6</div>' in flat and '1 published by LOTAAS' in flat and '1 by other LOFAR work' in flat
    everything = client.get('/pulsars?show=all').text.split('<tbody>')[1]
    assert 'Bilous et al. 2016' in everything.split('J0850+6800')[1].split('</tr>')[0]
    assert 'LOTAAS' in everything.split('J0849+6810')[1].split('</tr>')[0]
    overview = client.get('/').text.replace('\n', '')
    assert 'Known pulsars no LOFAR paper reports, found here' in overview
    assert '1 <span class="muted small">/ 4 near searched beams' in overview


def test_lofar_references_names_and_scattering():
    from web import lofar
    ref = ('***bkk+16  bkk+16:  Bilous, A. V. & Kondratiev, V. I., 2016. A LOFAR census of non-recycled '
           'pulsars. aap, 591, A134.\n***bgt+21  bgt+21:  Bondonneau, L., 2021. Pulsars with NenuFAR: Backend '
           'and pipelines. aap, 652, A34.\n')
    assert lofar.citations(ref) == {'bkk+16': 'Bilous et al. 2016'}              # NenuFAR is not LOFAR
    resolve = lofar.resolver([('J0636+5128', None), ('J0033+5700', None), ('J0014+4746', 'B0011+47')])
    assert resolve('J0636+5129') == 'J0636+5128'                                  # a refined position
    assert resolve('J0033+57') == 'J0033+5700' and resolve('B0011+47') == 'J0014+4746'
    assert resolve('J2000+0000') is None
    assert 20 < lofar.scattering_ms(100.0) < 40 and lofar.scattering_ms(0) is None


def test_unreviewed_candidates_come_first_and_saving_moves_to_the_next_unreviewed(cfg, campaign):
    # Four FETCH positives in other beams; the strongest two get verdicts.
    add_detections(campaign, [(30 + i, 40.0 + i, 20.0 - i, 2, 'candidate', None, 500.0 + 300 * i) for i in range(4)])
    Indexer(cfg).run_pass()
    client = client_for(cfg)
    listing = client.get('/single-pulse?sort=snr').text
    ids = [chunk.split('?')[0] for chunk in listing.split('href="/verify/')[1:] if 'B03' in chunk.split('</a>')[0]]
    for cid in ids[:2]:
        assert client.post('/api/review', json={'id': cid, 'label': 'noise', 'reviewer': 'Dirk'}).status_code == 200
    Indexer(cfg).run_pass()
    listing = client.get('/single-pulse?sort=snr').text
    rows = listing.split('<tbody>')[1].split('</tr>')
    labels = ['to review' in row for row in rows if 'href="/verify/' in row]
    assert labels == sorted(labels, reverse=True) and labels.count(True) >= 3     # unreviewed first
    # On a reviewed candidate the page offers the next one without a verdict.
    page = client.get(f'/verify/{ids[0]}?kind=sp&sort=snr').text
    target = page.split('id="next-unreviewed" href="/verify/')[1].split('?')[0]
    assert target not in ids[:2]
    assert 'left</a>' in page
    # With every candidate reviewed, nothing is offered.
    for chunk in listing.split('href="/verify/')[1:]:
        client.post('/api/review', json={'id': chunk.split('?')[0], 'label': 'noise', 'reviewer': 'Dirk'})
    page = client.get(f'/verify/{ids[0]}?kind=sp').text
    assert 'id="next-unreviewed"' not in page and 'nothing else to review here' in page


def test_published_lotaas_sources_are_set_against_what_the_campaign_found(cfg, campaign, monkeypatch, tmp_path):
    catalogue = tmp_path/'psrcat.db'
    catalogue.write_text(
        'PSRJ     J0847+6830                    sbc+19\nRAJ      08:47:30.0\nDECJ     +68:30:00\nDM       30.0\n'
        'P0       0.5\nS150     40\nSURVEY   lotaas\n@----\n'
        'PSRJ     J0850+6800                    kkl+15\nRAJ      08:50:00.0\nDECJ     +68:00:00\nDM       55.0\n'
        'P0       1.2\nSURVEY   gbncc,lotaas\nTYPE     RRAT\n@----\n'
        'PSRJ     J1200+4500                    sbc+19\nRAJ      12:00:00.0\nDECJ     +45:00:00\nDM       20.0\n'
        'P0       0.9\nSURVEY   lotaas\n@----\n'
        'PSRJ     J0846+6820\nRAJ      08:46:00.0\nDECJ     +68:20:00\nDM       40.0\nP0       0.0031\nSURVEY   lotaas\n@----\n'
        'PSRJ     J0845+6900\nRAJ      08:45:00.0\nDECJ     +69:00:00\nDM       10.0\nP0       0.7\nSURVEY   gbncc\n@----\n')
    monkeypatch.setenv('LOTAAS_PSRCAT', str(catalogue))
    add_detections(campaign, [(25, 30.0, 14.0, 2, 'known_pulsar', 'J0847+6830', 812.0)])
    fold(campaign['beam_dir'], 30.1, 0.25 * (1 + 4e-5), 80.0, 'half')
    Indexer(cfg).run_pass()
    client = client_for(cfg)
    page = client.get('/lotaas').text
    assert 'J0845+6900' not in page and 'J1200+4500' not in page         # not LOTAAS; not searched yet
    row = page.split('J0847+6830')[1].split('</tr>')[0]
    assert 'discovery' in row and 'flag ok">both' in row and '(1/2)' in row and 'best S/N 14.0' in row
    # The link opens the redetection, not whichever event of the beam has the lowest id.
    redetection = Indexer(cfg).db.execute("SELECT id FROM candidates WHERE type='known_pulsar'").fetchone()[0]
    assert f'href="/verify/{redetection}">1, best S/N 14.0' in row
    row = page.split('J0850+6800')[1].split('</tr>')[0]
    assert 'RRAT' in row and 'single pulse' in row and 'not found' in row
    everything = client.get('/lotaas?show=all').text
    assert 'J1200+4500' in everything.split('<tbody>')[1] and 'not searched yet' in everything
    assert '>3 <span class="muted small">/ 4</span>' in page.replace('\n', '')   # searched of published
    assert 'P below the search' in page.split('J0846+6820')[1].split('</tr>')[0]
    assert page.index('J0847+6830') < page.index('J0850+6800') < page.index('J0846+6820')   # found, missed, out of reach
    overview = client.get('/').text
    assert 'Published LOTAAS sources redetected' in overview and 'LOTAAS sources →' in overview


def test_the_viewer_opens_in_detail_and_offers_the_other_views(cfg, campaign):
    synthetic_filterbank(cfg.source_roots[0] / f'{ITEM}.fil')
    indexer = Indexer(cfg)
    indexer.run_pass()
    Snippets(cfg).run_pass()
    indexer.run_pass()
    client = client_for(cfg)
    cid = indexer.db.execute("SELECT id FROM candidates WHERE type='candidate'").fetchone()[0]
    page = client.get(f'/verify/{cid}').text
    assert 'id="preset"' in page and 'Matched to the pulse' in page and 'id="smooth"' in page
    candidate = json.loads(page.split('<script id="candidate" type="application/json">')[1].split('</script>')[0])
    assert candidate['smooth'] == 1.0 and set(candidate['presets']) == {'detail', 'matched', 'full'}
    initial = json.loads(page.split('<script id="initial" type="application/json">')[1].split('</script>')[0])
    assert initial['view']['smooth'] == 1.0 and initial['view']['nsub'] == candidate['presets']['detail'][0]
    plain = client.get(f'/api/sp/{cid}/view?smooth=0&nsub=8&tscrunch=3').json()
    assert plain['smooth'] == 0 and plain['nsub'] == 8 and plain['smoothed_pixel_snr'] is None


def beam_results(cfg, number, clustered=(), dec='+68:24:59.00'):
    """A searched beam of ITEM's observation on disk, with these (dm, snr, time, width) cluster centres."""
    item = ITEM.replace('BEAM025', f'BEAM{number:03d}')
    beam_dir = cfg.result_roots[0] / 'run-a' / 'efc-gpu-01' / 'processed' / item / 'ffffffffffffffff'
    beam_dir.mkdir(parents=True, exist_ok=True)
    (beam_dir / 'metadata.json').write_text(json.dumps({
        'pilot': False, 'tsamp': 0.007864319719374176, 'nu_min': 119.45, 'nu_max': 151.04, 'tstart_mjd': 57713.16,
        'observation_info': {'RA (J2000)': '08:47:08.00', 'DEC (J2000)': dec, 'Object': 'LOTAAS-P1254B-SAP0'}}))
    (beam_dir / 'clustered_candidates.txt').write_text(
        'DM\tS/N\tTime\tSample\tFilter_Width\tDM_scaled\tCluster\n'
        + ''.join(f'{dm}\t{snr}\t{time}\t1\t{width}\t0\t{i}\n' for i, (dm, snr, time, width) in enumerate(clustered)))
    return item


def verdicts(cfg):
    from web import store
    db = store.reviews(cfg)
    try:
        return [dict(r) for r in db.execute('SELECT key, reviewer, label, note FROM reviews')]
    finally:
        db.close()


def test_an_undispersed_burst_across_beams_is_interference_even_at_one_low_dm(cfg, campaign):
    # L605714 at sunset: in every beam an undispersed burst peaked below DM 2, which the
    # classifier never records, and only its DM 2-4 tail reached FETCH. The tails agree
    # within 1.5 of one DM, like a pulsar in neighbouring beams; the beams' own cluster
    # centres below DM 2 show what the burst was.
    per_dm = 4148.808 * (1 / (119.45 * 151.04) - 1 / 151.04 ** 2)
    for number in range(60, 65):
        beam_results(cfg, number, [(0.4, 9.0, 300.0 - 0.4 * per_dm, 2)])
    add_detections(campaign, [(60 + i, dm, 8.5, 2, 'candidate', None, 300.0 - dm * per_dm)
                              for i, dm in enumerate((2.4, 2.9, 3.3, 3.8, 2.6))])
    # A bright pulsar's pulse in five neighbouring beams, all at its DM, no burst below DM 2.
    for number in range(70, 75):
        beam_results(cfg, number)
    add_detections(campaign, [(70 + i, 26.2, 20.0 - i, 1, 'known_pulsar', 'J0323+3944', 2465.645) for i in range(5)])
    indexer = Indexer(cfg)
    indexer.run_pass()
    client = client_for(cfg)
    shown = client.get('/single-pulse').text
    assert 'B060' not in shown and 'B064' not in shown and '(5 hidden)' in shown
    assert shown.count('5 beams</span>') == 5                    # the pulsar's pulses stay
    page = client.get('/verify/' + client.get('/single-pulse?coincident=include').text
                      .split('B062')[0].rsplit('href="/verify/', 1)[1].split('?')[0]).text
    assert 'scattered DMs: interference' in page
    recorded = verdicts(cfg)
    assert sorted(v['label'] for v in recorded) == ['rfi'] * 5
    assert all(v['reviewer'] == 'auto-triage' and '4 of the 9 events there below DM 1' in v['note'] for v in recorded)
    indexer.run_pass()
    assert len(verdicts(cfg)) == 5                               # recorded once
    listing = client.get('/single-pulse?coincident=include&review=rfi').text
    assert listing.count('RFI</span>') == 5


def test_a_known_pulsar_seen_away_from_its_beam_is_recognised(cfg, campaign, monkeypatch, tmp_path):
    # B0823+26 in L611400: FETCH positives in 33 beams up to 3.6 degrees away, on its rotation.
    catalogue = tmp_path / 'psrcat.db'
    catalogue.write_text(
        'PSRJ     J0850+6625\nPSRB     B0845+66\nRAJ      08:50:00.0\nDECJ     +66:25:00\nDM       19.5\n'
        'P0       0.5306\n@----\n'
        'PSRJ     J0851+6630\nRAJ      08:51:00.0\nDECJ     +66:30:00\nDM       40.0\nP0       1.0\n@----\n')
    monkeypatch.setenv('LOTAAS_PSRCAT', str(catalogue))
    for number in range(30, 40):
        beam_results(cfg, number)
    period = 0.5306
    on = [(30 + k % 10, 19.4 + 0.02 * (k % 10), 9.0 + k % 4, 2, 'candidate', None, 100.0 + 7 * k * period + 0.004 * (k % 3))
          for k in range(20)]
    off = [(33, 19.5, 8.0, 2, 'candidate', None, 100.0 + 51.5 * period)]           # half a turn out
    other = [(35, 40.0, 8.0, 2, 'candidate', None, 500.0), (36, 40.1, 8.0, 2, 'candidate', None, 900.0)]
    add_detections(campaign, on + off + other)
    indexer = Indexer(cfg)
    indexer.run_pass()
    known = {r['key']: dict(r) for r in indexer.db.execute('SELECT * FROM sp_known')}
    assert len(known) == 20 and all(k['name'] == 'B0845+66' and 'rotation' in k['route'] and k['z'] > 15
                                    for k in known.values())
    assert 1.9 < min(k['separation_deg'] for k in known.values()) < 2.1
    client = client_for(cfg)
    shown = client.get('/single-pulse').text
    assert '(20 hidden)' in shown and shown.count('B0845+66 2.0°') == 0
    body = shown.split('<tbody>')[1]
    # The pulse off the rotation, and the other pulsar's two (no fold shows it is there), stay.
    assert '<td class="num">19.50</td>' in body and '<td class="num">40.00</td>' in body and '40.10' in body
    everything = client.get('/single-pulse?known=include').text
    assert everything.count('B0845+66 2.0°</span>') == 20
    cid = everything.split('B0845+66 2.0°')[0].rsplit('href="/verify/', 1)[1].split('?')[0]
    page = client.get(f'/verify/{cid}').text
    assert 'these pulses keep its rotation' in page and 'a known pulsar seen away from its own beam' in page
    recorded = verdicts(cfg)
    assert sorted(v['label'] for v in recorded) == ['known'] * 20
    assert all(v['key'] in known and 'B0845+66 (J0850+6625), 2.0' in v['note'] for v in recorded)


def test_a_burst_in_all_three_saps_at_scattered_dms_is_interference(cfg, campaign):
    # L611408: one burst in 92 beams of all three SAPs, bright at every DM, so fewer than
    # a quarter of the events lay below DM 1. No one position is in all three SAPs.
    per_dm = 4148.808 * (1 / (119.45 * 151.04) - 1 / 151.04 ** 2)
    rows = [(number, dm, 30.0, 9, 'candidate', None, 99.0 - dm * per_dm)
            for number, dm in zip(range(20, 26), (5.1, 7.9, 12.4, 18.8, 26.0, 33.5))]
    add_detections(campaign, rows)
    with campaign['ledger'].connect() as db:
        run = db.execute('SELECT MAX(id) FROM beam_runs').fetchone()[0]
        for sap, dm in ((1, 9.3), (2, 14.1)):
            db.execute("INSERT INTO detections(beam_id,candidate_dm,snr,width_samples,detection_type,pulsar_name,"
                       "classification_probability,beam_run_id,time_seconds,sample_number) VALUES "
                       "(?,?,20.0,9,'candidate',NULL,0.9,?,?,1)",
                       (ITEM.replace('SAP000', f'SAP{sap:03d}').replace('BEAM025', 'BEAM030') + '.fil', dm, run,
                        99.0 - dm * per_dm))
    Indexer(cfg).run_pass()
    recorded = verdicts(cfg)
    assert len(recorded) == 8 and {v['label'] for v in recorded} == {'rfi'}
    assert all('all three SAPs at scattered DMs' in v['note'] for v in recorded)


def test_a_fold_at_a_catalogued_pulsars_own_period_is_settled(cfg, campaign, monkeypatch, tmp_path):
    catalogue = tmp_path / 'psrcat.db'
    catalogue.write_text('PSRJ     J0826+2637\nPSRB     B0823+26\nRAJ      08:40:00.0\nDECJ     +66:00:00\n'
                         'DM       19.476\nP0       0.5306603\n@----\n')
    monkeypatch.setenv('LOTAAS_PSRCAT', str(catalogue))
    beam_dir = campaign['beam_dir']
    fold(beam_dir, 19.5, 0.5306603 * (1 - 9e-5), 381.9, 'psr')          # its own period, Doppler shifted
    fold(beam_dir, 19.4, 0.5306603 / 2, 60.0, 'second')                  # a harmonic: for a person
    fold(beam_dir, 55.0, 0.5306603, 40.0, 'other')                       # its period at another DM
    Indexer(cfg).run_pass()
    recorded = verdicts(cfg)
    assert [v['label'] for v in recorded] == ['known']
    assert 'B0823+26 (J0826+2637) at its own period and DM, 2.51 deg' in recorded[0]['note']


PER_DM = 4148.808 * (1 / (119.45 * 151.04) - 1 / 151.04 ** 2)   # s of band-averaged delay per unit DM


def test_a_burst_at_scattered_dms_in_one_beam_is_interference(cfg, campaign):
    # L606808 SAP002 B073: seven events at one undispersed moment, DM 3-148, all at S/N 7.0-7.3.
    burst = [(40, dm, snr, 3, kind, None, 1041.0 - dm * PER_DM) for dm, snr, kind in
             ((3.0, 7.2, 'rejected'), (38.1, 7.2, 'candidate'), (62.0, 7.3, 'candidate'), (66.6, 7.3, 'unconfirmed'),
              (103.7, 7.3, 'candidate'), (127.9, 7.1, 'unconfirmed'), (147.5, 7.0, 'unconfirmed'))]
    # A bright pulse and the tails of its DM curve: one DM stands out.
    pulse = [(41, dm, snr, 2, kind, None, 2000.0 - dm * PER_DM) for dm, snr, kind in
             ((26.2, 40.0, 'candidate'), (5.0, 7.5, 'rejected'), (15.0, 7.6, 'rejected'), (40.0, 7.5, 'rejected'),
              (60.0, 7.4, 'rejected'))]
    # A busy beam: four events at one moment are its ordinary rate, not a burst.
    busy = [(42, 10.0 + (i * 37) % 900, 7.1, 2, 'rejected', None, t) for i, t in
            enumerate(x for x in range(20, 3600, 3) if abs(x - 2600) > 20)]
    busy += [(42, dm, 7.2, 2, kind, None, 2600.0 - dm * PER_DM) for dm, kind in
             ((12.0, 'candidate'), (55.0, 'rejected'), (90.0, 'rejected'), (140.0, 'rejected'))]
    add_detections(campaign, burst + pulse + busy)
    indexer = Indexer(cfg)
    indexer.run_pass()
    from web.keys import parse_item
    swept = {(parse_item(r['key'].split('|')[1])[2], r['key'].split('|')[2])
             for r in indexer.db.execute('SELECT key FROM sp_sweep')}
    assert swept == {(40, f'DM{dm:.3f}') for dm in (3.0, 38.1, 62.0, 66.6, 103.7, 127.9, 147.5)}
    recorded = verdicts(cfg)
    assert sorted(v['key'].split('|')[2] for v in recorded) == ['DM103.700', 'DM38.100', 'DM62.000']
    assert all(v['label'] == 'rfi' and 'Undispersed burst in this beam: 7 events' in v['note'] for v in recorded)
    cid = indexer.db.execute("SELECT id FROM candidates WHERE item LIKE '%BEAM040%' AND dm=62.0").fetchone()[0]
    assert 'A burst in this beam: 7 events' in client_for(cfg).get(f'/verify/{cid}').text


def test_a_redetection_inside_a_burst_is_not_the_pulsar(cfg, campaign, monkeypatch, tmp_path):
    # L605714 SAP001 B003: B1737+13 'redetected' by one of twelve events of a burst, no pulse at its DM.
    from web import store
    from web.keys import sp_key
    catalogue = tmp_path / 'psrcat.db'
    catalogue.write_text('PSRJ     J0847+6825\nPSRB     B0842+68\nRAJ      08:47:08.0\nDECJ     +68:24:59\n'
                         'DM       48.668\nP0       0.8031\n@----\n')
    monkeypatch.setenv('LOTAAS_PSRCAT', str(catalogue))
    fold(campaign['beam_dir'], 48.7, 0.8031 * (1 - 7e-5), 389.0, 'psr')        # the pulsar is in the beam
    burst = [(25, dm, snr, 127, kind, None, 1458.5 - dm * PER_DM) for dm, snr, kind in
             ((0.7, 8.2, 'rejected'), (9.3, 8.1, 'rejected'), (21.0, 8.3, 'unconfirmed'), (35.2, 8.0, 'rejected'),
              (61.5, 8.1, 'candidate'), (74.0, 8.2, 'rejected'), (96.7, 8.3, 'rejected'))]
    add_detections(campaign, burst + [(25, 48.6, 8.4, 127, 'known_pulsar', 'J0847+6825', 1458.5 - 48.6 * PER_DM)])
    item = ITEM
    redetection = sp_key('known_pulsar', item, 48.6, 127, 8.4)
    positive = sp_key('candidate', item, 61.5, 127, 8.1)
    indexer = Indexer(cfg)
    db = store.reviews(cfg)
    with db:   # the triage's earlier verdict, and a person's
        db.execute("INSERT INTO reviews(key,reviewer,label,note,created) VALUES (?,?,?,?,?)",
                   (redetection, 'auto-triage', 'known', 'earlier', 1.0))
        db.execute("INSERT INTO reviews(key,reviewer,label,note,created) VALUES (?,?,?,?,?)",
                   (positive, 'Dirk', 'unsure', 'look again', 1.0))
    db.close()
    indexer.run_pass()
    assert indexer.db.execute('SELECT COUNT(*) FROM sp_known').fetchone()[0] == 0
    recorded = verdicts(cfg)
    latest = [v for v in recorded if v['key'] == redetection][-1]
    assert latest['label'] == 'rfi' and "Replaces this triage's earlier 'known'" in latest['note']
    assert [v['reviewer'] for v in recorded if v['key'] == positive] == ['Dirk']      # a person's stands
    indexer.run_pass()
    assert len(verdicts(cfg)) == len(recorded)                                          # nothing repeated
