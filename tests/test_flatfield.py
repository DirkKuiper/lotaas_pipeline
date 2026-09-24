"""Flatfielding a SAP before its last beams arrive, and those beams once they do."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lotaas_reprocessing.filterbank import FilterbankFile, create_filterbank_file

REPO = Path(__file__).resolve().parents[1]
HEADER = {'telescope_id': 11, 'machine_id': 11, 'data_type': 1, 'source_name': 'SYNTHETIC_FLATFIELD',
          'nchans': 8, 'nifs': 1, 'tsamp': 0.00786, 'tstart': 60900.0, 'fch1': 150.0, 'foff': -4.0,
          'src_raj': 0.0, 'src_dej': 0.0, 'az_start': 0.0, 'za_start': 0.0}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fb = create_filterbank_file(str(path), dict(HEADER), nbits=32)
    try:
        fb.append_spectra(data.astype(np.float32))
    finally:
        fb.close()


def read(path):
    fb = FilterbankFile(str(path))
    try:
        return np.array(fb.get_spectra(0, fb.nspec), dtype=np.float64)
    finally:
        fb.close()


def beams(directory, numbers, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(256)[:, None]
    common = (1 + np.arange(8)[None, :]) * (2 + np.sin(t / 20.))      # bandpass and slow gain, every beam
    written = {}
    for number in numbers:
        data = common * rng.uniform(0.9, 1.1, (256, 8))
        path = directory/f'B{number:03d}'/f'downsampled_L1_SAP000_BEAM{number:03d}_32bit.fil'
        write(path, data)
        written[number] = data
    return written


def flatfield(directory, *arguments):
    return subprocess.run([sys.executable, str(REPO/'preproc/flatfield_fil.py'), str(directory), *arguments],
                          env=dict(os.environ, PYTHONPATH=str(REPO)), capture_output=True, text=True)


def test_a_sap_is_flatfielded_without_its_last_beams_and_they_join_later(tmp_path):
    sap = tmp_path/'SAP000'
    early = beams(sap, [n for n in range(13, 74) if n not in (38, 58)], seed=1)
    refused = flatfield(sap)
    assert refused.returncode != 0 and 'Missing 2 central beams' in refused.stderr
    mean_path = tmp_path/'flatfield-mean.npy'
    done = flatfield(sap, '--allow-partial', '--save-mean', str(mean_path))
    assert done.returncode == 0, done.stderr
    mean = np.load(mean_path)
    assert mean.dtype == np.float32 and mean.shape == (8, 256)
    first = read(sap/'B013'/'downsampled_L1_SAP000_BEAM013_32bit_ff.fil')
    assert np.allclose(first.mean(), 1.0, atol=0.05) and first.std() < 0.1       # the common structure is gone
    # The early beams' unflattened files are removed after flatfielding (the
    # campaign does it); the late ones arrive and use the saved mean.
    for path in sap.glob('B*/*_32bit.fil'):
        path.unlink()
    late = beams(sap, [38, 58], seed=2)
    joined = flatfield(sap, '--mean', str(mean_path))
    assert joined.returncode == 0, joined.stderr
    for number, data in late.items():
        flat = read(sap/f'B{number:03d}'/f'downsampled_L1_SAP000_BEAM{number:03d}_32bit_ff.fil')
        # Channel order in the file is the reverse of the flatfield's (it flips on read and write).
        expected = data / np.flipud(mean).T.astype(np.float64)
        assert np.allclose(flat, expected, rtol=1e-5)
    assert len(list(sap.glob('B*/*_ff.fil'))) == 61
