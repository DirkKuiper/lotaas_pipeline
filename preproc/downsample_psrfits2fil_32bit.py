#!/usr/bin/env python3
"""Stream PSRFITS subintegrations into a 32-bit SIGPROC filterbank."""
import argparse
import os
import re
import numpy as np
from astropy.io import fits
from astropy.time import Time
from lotaas_reprocessing import filterbank


def parse_ids(filename):
    match = re.search(r'(L\d+)_SAP(\d+)_(?:BEAM|B)(\d+)', os.path.basename(filename))
    if not match:
        raise ValueError(f'Cannot parse observation/SAP/beam: {filename}')
    return match[1], int(match[2]), int(match[3])


def unpack(data, nbits, nsamp, nchan):
    if nbits == 2:
        packed = np.asarray(data, dtype=np.uint8).reshape(-1)
        # PSRFITS packs four unsigned values, most significant bits first.
        values = ((packed[:, None] >> np.array([6, 4, 2, 0], dtype=np.uint8)) & 3).reshape(-1)
    elif nbits == 8:
        values = np.asarray(data).reshape(-1)
    else:
        raise ValueError(f'Unsupported PSRFITS bit depth: {nbits}')
    return values.reshape(nsamp, nchan)


def convert(input_path, output_path, fscrunch=4, tscrunch=16, dc=False, max_subints=None):
    if fscrunch <= 0 or tscrunch <= 0:
        raise ValueError('Scrunch factors must be positive')
    with fits.open(input_path, memmap=True) as hdus:
        hdr, sub = hdus[0].header, hdus['SUBINT']
        nchan, nsamp = sub.header['NCHAN'], sub.header['NSBLK']
        if sub.header.get('NPOL', 1) != 1:
            raise ValueError('Only Stokes-I PSRFITS (NPOL=1) is supported')
        if nchan % fscrunch or nsamp % tscrunch:
            raise ValueError('Channel/subintegration lengths must be divisible by scrunch factors')
        freqs = np.asarray(sub.data['DAT_FREQ'][0]).reshape(-1, fscrunch).mean(axis=1)
        if freqs.size < 2 or not np.allclose(np.diff(freqs), np.median(np.diff(freqs)), rtol=1e-3, atol=1e-5):
            raise ValueError('Expected uniformly spaced frequency channels')
        reverse = freqs[0] < freqs[-1]
        if reverse:
            freqs = freqs[::-1]
        if 'STT_IMJD' in hdr:
            start = hdr['STT_IMJD'] + (hdr.get('STT_SMJD', 0) + hdr.get('STT_OFFS', 0)) / 86400
        else:
            start = float(Time(hdr['DATE-OBS'], format='isot', scale='utc').mjd)
        header = dict(telescope_id=11, machine_id=-1, data_type=1,
                      source_name=hdr['SRC_NAME'], barycentric=0, pulsarcentric=0,
                      src_raj=float(hdr['RA'].replace(':', '')),
                      src_dej=float(hdr['DEC'].replace(':', '')), tstart=start,
                      tsamp=float(sub.header['TBIN']) * tscrunch,
                      foff=float(freqs[1] - freqs[0]), fch1=float(freqs[0]),
                      nchans=len(freqs), nifs=1, nbits=32)
        partial = str(output_path) + '.partial'
        output = filterbank.create_filterbank_file(partial, header, nbits=32)
        count = min(len(sub.data), max_subints) if max_subints else len(sub.data)
        try:
            for index in range(count):
                row = sub.data[index]
                data = unpack(row['DATA'], sub.header['NBITS'], nsamp, nchan).astype(np.float32)
                data = (data * row['DAT_SCL'].reshape(1, nchan) + row['DAT_OFFS'].reshape(1, nchan)) * row['DAT_WTS'].reshape(1, nchan)
                if dc:
                    data[:, ::16] = np.nan
                data = np.nanmean(data.reshape(nsamp, -1, fscrunch), axis=2)
                data = np.nanmean(data.reshape(-1, tscrunch, len(freqs)), axis=1)
                if reverse:
                    data = data[:, ::-1]
                if not np.isfinite(data).all():
                    raise ValueError(f'Nonfinite output in subintegration {index}')
                output.append_spectra(data)
        finally:
            output.close()
        os.replace(partial, output_path)
    print(f'Converted {count} subintegrations: {output_path}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-f', '--fscrunch', type=int, default=4)
    p.add_argument('-t', '--tscrunch', type=int, default=16)
    p.add_argument('-o', '--output')
    p.add_argument('-d', '--dc', action='store_true')
    p.add_argument('--max-subints', type=int, help='Pilot only: convert a prefix')
    p.add_argument('input')
    a = p.parse_args()
    obs, sap, beam = parse_ids(a.input)
    output = a.output or f'{obs}_SAP{sap:03d}_B{beam:03d}_32bit.fil'
    convert(a.input, output, a.fscrunch, a.tscrunch, a.dc, a.max_subints)


if __name__ == '__main__':
    main()
