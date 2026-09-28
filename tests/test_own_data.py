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
