"""A candidate's S/N on its own data (own_data), and the classifier's check with it before FETCH."""
import json
import sqlite3
import struct

import numpy as np

from lotaas_reprocessing import own_data

TSAMP = 0.007864319719374176
FCH1, FOFF, NCHANS = 151.04, -0.4936, 64
PLAN = [{'low_dm': 0.0, 'high_dm': 1000.0, 'ddm': 0.1, 'downsample': 1}]


def write_filterbank(path, pulses=(), nsamp=20000, seed=1, burst=None):
    """Gaussian noise with dispersed boxcars (time at the top of the band, dm, width, amplitude)."""
    rng = np.random.default_rng(seed)
    data = rng.normal(10.0, 1.0, (nsamp, NCHANS)).astype(np.float32)
    freqs = FCH1 + np.arange(NCHANS) * FOFF
    for t, dm, width, amplitude in pulses:
        for channel, start in enumerate(np.round((t + own_data.sweep_seconds(dm, freqs)) / TSAMP).astype(int)):
            data[start:start + width, channel] += amplitude
    if burst is not None:
        channel, start, stop, level = burst
        data[int(start / TSAMP):int(stop / TSAMP), channel] += level

    def field(key, value=None, kind=None):
        out = struct.pack('<i', len(key)) + key.encode()
        if kind == 'i':
            out += struct.pack('<i', value)
        elif kind == 'd':
            out += struct.pack('<d', value)
        return out
    header = (field('HEADER_START') + field('nchans', NCHANS, 'i') + field('nbits', 32, 'i') + field('nifs', 1, 'i')
              + field('tsamp', TSAMP, 'd') + field('fch1', FCH1, 'd') + field('foff', FOFF, 'd')
              + field('tstart', 57713.0, 'd') + field('HEADER_END'))
    path.write_bytes(header + data.tobytes())
    return path


def test_a_pulse_reads_on_its_own_data_and_nothing_reads_nothing(tmp_path):
    fil = write_filterbank(tmp_path / 'beam.fil', [(60.0, 30.0, 3, 1.2)])
    centre = 60.0 + TSAMP                          # the search reports a boxcar's centre
    pulse = own_data.measure(fil, 30.0, centre, 3, PLAN, (), 2.0)
    nothing = own_data.measure(fil, 30.0, 100.0, 3, PLAN, (), 2.0)
    assert pulse > 12 and abs(nothing) < 4
    # A quiet channel's strong burst beside the pulse changes nothing (the search's RFI mask).
    burst = write_filterbank(tmp_path / 'burst.fil', [(60.0, 30.0, 3, 1.2)], burst=(7, 52.0, 58.0, 2000.0))
    assert abs(own_data.measure(burst, 30.0, centre, 3, PLAN, (), 2.0) - pulse) < 0.15 * pulse


def test_the_classifier_does_not_ask_fetch_about_what_its_own_data_do_not_show(tmp_path, monkeypatch):
    from test_tiers import classifier
    classify = classifier(tmp_path, monkeypatch)
    fil = write_filterbank(tmp_path / 'beam.fil', [(60.0, 30.0, 3, 1.2)])
    asked = []

    class Model:
        def __init__(self, p):
            self.p = p

        def predict(self, inputs, batch_size=1, verbose=0):
            asked.append(self.p)
            return np.array([[1 - self.p, self.p]])

    from lotaas_reprocessing import fetch_models
    monkeypatch.setattr(fetch_models, 'load_models', lambda names, factory=None: {'a': Model(0.2), 'd': Model(0.4)})
    monkeypatch.setattr(classify, 'fetch_inputs', lambda *args, **kwargs: (None, np.zeros((1, 256, 256, 1)),
                                                                          np.zeros((1, 256, 256, 1)), 1))
    monkeypatch.setattr(classify, 'FilterbankFile', lambda *args: type('F', (), {
        'fch1': FCH1, 'foff': FOFF, 'nchans': NCHANS, 'close': lambda self: None})())
    candidates = tmp_path / 'cands.tsv'
    candidates.write_text('DM\tS/N\tTime\tSample\tFilter_Width\n'
                          f'30.0\t15.0\t{60.0 + TSAMP}\t7631\t3\n30.1\t9.0\t100.0\t12716\t3\n')
    limits = {'min_dm': 2.0, 'min_snr': 8.0, 'min_own_snr': 4.0, 'min_own_fraction': 0.5}
    counts = classify.classify_candidates(str(fil), candidates, str(tmp_path / 'plots'),
                                          {'RA (J2000)': '12:00:00', 'DEC (J2000)': '+45:00:00'},
                                          limits=limits, tsamp=TSAMP, plan=PLAN, baseline_seconds=2.0)
    assert counts['own_data'] == 1 and counts['fetch'] == 1 and len(asked) == 2      # FETCH asked once, two models
    with sqlite3.connect(tmp_path / 'classifier.sqlite') as db:
        rows = dict(db.execute('SELECT round(candidate_dm, 1), detection_type || "|" || COALESCE(model_probabilities, "") '
                               'FROM detections').fetchall())
    assert rows[30.1] == 'unconfirmed|'
    kind, probabilities = rows[30.0].split('|', 1)
    assert kind == 'rejected' and json.loads(probabilities) == {'a': 0.2, 'd': 0.4}


def test_a_dispersed_pulse_fades_at_too_low_a_dm_and_what_is_not_dispersed_does_not(tmp_path):
    width = 13                                     # 0.1 s, as FRBs are at 135 MHz and DM 300
    fil = write_filterbank(tmp_path / 'beam.fil', [(60.0, 300.0, width, 1.0)], burst=(22, 80.0, 80.2, 15.0))
    centre = 60.0 + (width - 1) / 2 * TSAMP
    ratio, snr = own_data.dispersion_ratio(own_data.load(fil, 300.0, centre, width, PLAN), baseline_seconds=2.0)
    assert snr > 15 and ratio < 0.3
    # Noise, and one channel's burst where DM 300 puts it: nothing a dispersed pulse would do.
    delay = float(own_data.sweep_seconds(300.0, FCH1 + np.arange(NCHANS) * FOFF)[22])
    for t in (120.0, 80.1 - delay):
        ratio, snr = own_data.dispersion_ratio(own_data.load(fil, 300.0, t, width, PLAN), baseline_seconds=2.0)
        assert ratio is None or ratio > 0.7 or snr < 5


def test_the_classifier_keeps_what_fetch_rejects_but_its_own_data_show_dispersed(tmp_path, monkeypatch):
    from test_tiers import classifier
    classify = classifier(tmp_path, monkeypatch)
    width = 13
    fil = write_filterbank(tmp_path / 'beam.fil', [(60.0, 300.0, width, 1.0), (140.0, 30.0, width, 1.0)])

    class Model:
        def predict(self, inputs, batch_size=1, verbose=0):
            return np.array([[0.8, 0.2]])

    from lotaas_reprocessing import fetch_models
    monkeypatch.setattr(fetch_models, 'load_models', lambda names, factory=None: {'a': Model()})
    # What the review plot of a kept candidate reads from FETCH's inputs.
    planes = type('Candidate', (), {'tsamp': TSAMP, 'dmt': np.random.default_rng(0).normal(size=(256, 256)),
                                    'dedispersed': np.random.default_rng(1).normal(size=(256, NCHANS))})()
    monkeypatch.setattr(classify, 'fetch_inputs', lambda *args, **kwargs: (planes, np.zeros((1, 256, 256, 1)),
                                                                          np.zeros((1, 256, 256, 1)), 1))
    monkeypatch.setattr(classify, 'FilterbankFile', lambda *args: type('F', (), {
        'fch1': FCH1, 'foff': FOFF, 'nchans': NCHANS, 'close': lambda self: None})())
    centre = (width - 1) / 2 * TSAMP
    candidates = tmp_path / 'cands.tsv'
    candidates.write_text('DM\tS/N\tTime\tSample\tFilter_Width\n'
                          f'300.0\t20.0\t{60.0 + centre}\t7630\t{width}\n30.0\t20.0\t{140.0 + centre}\t17802\t{width}\n')
    limits = {'min_dm': 2.0, 'min_snr': 8.0, 'min_own_snr': 4.0, 'min_own_fraction': 0.5}
    route = {'min_dm': 100.0, 'min_own_snr': 8.0, 'max_ratio': 0.5, 'max_width_seconds': 0.5}
    info = {'RA (J2000)': '12:00:00', 'DEC (J2000)': '+45:00:00'}
    counts = classify.classify_candidates(str(fil), candidates, str(tmp_path / 'plots'), info,
                                          limits=dict(limits, dispersed=route), tsamp=TSAMP, plan=PLAN,
                                          baseline_seconds=2.0)
    assert counts['fetch'] == 2 and counts['dispersed'] == 1
    with sqlite3.connect(tmp_path / 'classifier.sqlite') as db:
        rows = {round(dm): (kind, own, ratio) for dm, kind, own, ratio in db.execute(
            'SELECT candidate_dm, detection_type, own_snr, dispersion_ratio FROM detections')}
    assert rows[300][0] == 'dispersed' and rows[300][1] > 8 and rows[300][2] < 0.3
    assert rows[30] == ('rejected', rows[30][1], None)        # below the route's DM: FETCH decides
    assert len(list((tmp_path / 'plots').glob('DM300.0_*.png'))) == 1   # plotted for review, as FETCH positives are
    # Wider than the route judges: FETCH's verdict stands.
    counts = classify.classify_candidates(str(fil), candidates, str(tmp_path / 'plots3'), info,
                                          limits=dict(limits, dispersed=dict(route, max_width_seconds=0.05)),
                                          tsamp=TSAMP, plan=PLAN, baseline_seconds=2.0)
    assert counts['dispersed'] == 0
    # Without the route FETCH's verdict stands.
    counts = classify.classify_candidates(str(fil), candidates, str(tmp_path / 'plots2'), info,
                                          limits=limits, tsamp=TSAMP, plan=PLAN, baseline_seconds=2.0)
    assert counts['dispersed'] == 0



def test_fetch_is_not_asked_about_what_is_wider_than_it_judges_and_tiers_set_the_cuts(tmp_path, monkeypatch):
    from test_tiers import classifier
    from lotaas_reprocessing import fetch_models
    classify = classifier(tmp_path, monkeypatch)
    width = 26                                     # 0.2 s at DM 300
    fil = write_filterbank(tmp_path / 'beam.fil', [(60.0, 300.0, width, 0.6)])
    asked = []
    monkeypatch.setattr(fetch_models, 'load_models', lambda names, factory=None: asked.append(names) or {})
    planes = type('Candidate', (), {'tsamp': TSAMP, 'dmt': np.zeros((256, 256)), 'dedispersed': np.zeros((256, NCHANS))})()
    monkeypatch.setattr(classify, 'fetch_inputs', lambda *args, **kwargs: (planes, None, None, 1))
    monkeypatch.setattr(classify, 'FilterbankFile', lambda *args: type('F', (), {
        'fch1': FCH1, 'foff': FOFF, 'nchans': NCHANS, 'close': lambda self: None})())
    candidates = tmp_path / 'cands.tsv'
    candidates.write_text('DM\tS/N\tTime\tSample\tFilter_Width\n'
                          f'300.0\t20.0\t{60.0 + (width - 1) / 2 * TSAMP}\t7640\t{width}\n')
    info = {'RA (J2000)': '12:00:00', 'DEC (J2000)': '+45:00:00'}
    limits = {'min_dm': 2.0, 'min_snr': 8.0, 'min_own_snr': 4.0, 'min_own_fraction': 0.5}
    route = {'min_dm': 100.0, 'fetch_max_width_seconds': 0.15,
             'tiers': [{'max_width_seconds': 0.13, 'min_own_snr': 5.0, 'max_ratio': 0.8},
                       {'max_width_seconds': 0.5, 'min_own_snr': 8.0, 'max_ratio': 0.5}]}
    counts = classify.classify_candidates(str(fil), candidates, str(tmp_path / 'plots'), info,
                                          limits=dict(limits, dispersed=route), tsamp=TSAMP, plan=PLAN,
                                          baseline_seconds=2.0)
    assert counts['unjudged'] == 1 and counts['fetch'] == 0 and counts['dispersed'] == 1 and not asked
    with sqlite3.connect(tmp_path / 'classifier.sqlite') as db:
        kind, probability, galactic = db.execute('SELECT detection_type, classification_probability, dm_galactic '
                                                 'FROM detections').fetchone()
    assert kind == 'dispersed' and probability is None and 10 < galactic < 300     # l 140, b 71: high latitude
    assert classify.dispersed_tier(route, 0.1)['min_own_snr'] == 5.0 and classify.dispersed_tier(route, 0.6) is None
    assert classify.dispersed_tier({'min_dm': 100, 'min_own_snr': 8, 'max_ratio': 0.5}, 2.0)['max_ratio'] == 0.5


def test_the_route_s_own_gate_lets_fainter_clusters_reach_it_but_not_fetch(tmp_path, monkeypatch):
    from test_tiers import classifier
    import pandas as pd
    classify = classifier(tmp_path, monkeypatch)
    limits = dict(classify.DEFAULT_LIMITS, min_snr=8.0, dispersed={'min_dm': 100.0, 'min_snr': 7.0})
    assert classify.snr_gate(limits, 300.0) == 7.0 and classify.snr_gate(limits, 50.0) == 8.0
    assert list(classify.snr_gate(limits, pd.Series([50.0, 300.0]))) == [8.0, 7.0]
    assert classify.snr_gate(dict(limits, dispersed={'min_dm': 100.0}), 300.0) == 8.0



def test_clusters_let_in_by_the_route_s_lower_gate_meet_its_faint_cuts(tmp_path, monkeypatch):
    from test_tiers import classifier
    from lotaas_reprocessing import fetch_models
    classify = classifier(tmp_path, monkeypatch)
    width = 13
    fil = write_filterbank(tmp_path / 'beam.fil', [(60.0, 300.0, width, 1.0)])
    monkeypatch.setattr(fetch_models, 'load_models', lambda names, factory=None: {})
    planes = type('Candidate', (), {'tsamp': TSAMP, 'dmt': np.zeros((256, 256)), 'dedispersed': np.zeros((256, NCHANS))})()
    monkeypatch.setattr(classify, 'fetch_inputs', lambda *args, **kwargs: (planes, None, None, 1))
    monkeypatch.setattr(classify, 'FilterbankFile', lambda *args: type('F', (), {
        'fch1': FCH1, 'foff': FOFF, 'nchans': NCHANS, 'close': lambda self: None})())
    candidates = tmp_path / 'cands.tsv'
    candidates.write_text('DM\tS/N\tTime\tSample\tFilter_Width\n'
                          f'300.0\t7.5\t{60.0 + (width - 1) / 2 * TSAMP}\t7630\t{width}\n')
    info = {'RA (J2000)': '12:00:00', 'DEC (J2000)': '+45:00:00'}
    limits = {'min_dm': 2.0, 'min_snr': 8.0, 'min_own_snr': 4.0, 'min_own_fraction': 0.5}
    route = {'min_dm': 100.0, 'min_snr': 7.0, 'tiers': [{'max_width_seconds': 0.5, 'min_own_snr': 5.0, 'max_ratio': 0.8}]}
    kept = classify.classify_candidates(str(fil), candidates, str(tmp_path / 'a'), info, limits=dict(limits, dispersed=route),
                                        tsamp=TSAMP, plan=PLAN, baseline_seconds=2.0)
    strict = dict(route, faint={'min_own_snr': 1000.0, 'max_ratio': 0.5})
    dropped = classify.classify_candidates(str(fil), candidates, str(tmp_path / 'b'), info, limits=dict(limits, dispersed=strict),
                                           tsamp=TSAMP, plan=PLAN, baseline_seconds=2.0)
    assert kept['dispersed'] == 1 and kept['unjudged'] == 1 and kept['fetch'] == 0 and dropped['dispersed'] == 0
