"""Make a separate validation filterbank with two known dispersed pulses.

This is a test fixture, never a survey observation or a campaign detection.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from lotaas_reprocessing.filterbank import FilterbankFile, create_filterbank_file


def inject(source, destination):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError('Use a new output path for the injected test fixture')
    destination.parent.mkdir(parents=True, exist_ok=True)
    fb = FilterbankFile(str(source))
    if fb.nspec * fb.tsamp < 3500:
        raise ValueError('This fixture requires a full one-hour beam')
    scale = float(np.median(np.std(fb.get_spectra(0, 8192), axis=0)))
    pulses = [dict(dm=83.2, time=800., sigma=.12, amplitude=.7*scale),
              dict(dm=2204.8, time=2300., sigma=1., amplitude=.3*scale)]
    header = fb.header.copy(); header['source_name'] = 'SYNTHETIC_INJECTION'
    out = create_filterbank_file(str(destination), header, nbits=32)
    try:
        for start in range(0, fb.nspec, 8192):
            data = fb.get_spectra(start, min(start+8192, fb.nspec)).copy()
            if len(data) != min(8192, fb.nspec-start):
                raise ValueError('Incorrect number of input spectra')
            t = (start + np.arange(len(data))) * fb.tsamp
            for pulse in pulses:
                delays = pulse['dm'] * (fb.frequencies**-2 - fb.frequencies.max()**-2) / 2.41e-4
                data += pulse['amplitude'] * np.exp(-.5 * ((t[:, None]-pulse['time']-delays[None, :])/pulse['sigma'])**2)
            out.append_spectra(data)
    finally:
        out.close(); fb.close()
    check = FilterbankFile(str(destination))
    try:
        if check.nspec != fb.nspec:
            raise ValueError('Injected fixture does not preserve the input length')
    finally:
        check.close()
    destination.with_suffix('.injections.json').write_text(json.dumps(
        {'synthetic': True, 'source': str(source), 'pulses': pulses}, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source'); parser.add_argument('destination')
    args=parser.parse_args(); inject(args.source,args.destination)
