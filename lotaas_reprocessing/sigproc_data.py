"""Read SIGPROC filterbank files, with the samples mapped by numpy.

The pipeline's FilterbankFile decodes header strings as text and fails on the
flatfielded beams, so this reads the header as bytes. Data are returned as
(time, channel) in file order; for LOFAR's negative foff channel 0 is fch1,
the highest frequency. Shared by the review page (web.sigproc) and the
classifier's own-data check (own_data); lotaas_reprocessing.sigproc is
PRESTO's header writer, used by the conversion.
"""
import struct

import numpy as np

INTEGERS = {'telescope_id', 'machine_id', 'data_type', 'nchans', 'nbits', 'nifs', 'nbeams',
            'ibeam', 'barycentric', 'pulsarcentric', 'nbins', 'nsamples'}
STRINGS = {'rawdatafile', 'source_name'}
DTYPES = {32: np.float32, 8: np.uint8}


def _string(stream):
    size = struct.unpack('<i', stream.read(4))[0]
    if not 0 < size < 4096:
        raise ValueError('Not a SIGPROC header')
    return stream.read(size).decode('latin-1')


def read_header(path):
    """The header as a dict, and its length in bytes."""
    with open(path, 'rb') as stream:
        if _string(stream) != 'HEADER_START':
            raise ValueError(f'{path} has no SIGPROC header')
        header = {}
        while True:
            key = _string(stream)
            if key == 'HEADER_END':
                return header, stream.tell()
            if key in INTEGERS:
                header[key] = struct.unpack('<i', stream.read(4))[0]
            elif key in STRINGS:
                header[key] = _string(stream)
            else:
                header[key] = struct.unpack('<d', stream.read(8))[0]


def open_data(path):
    """The header and a read-only (time, channel) memory map of the samples."""
    header, offset = read_header(path)
    dtype = DTYPES.get(header['nbits'])
    if dtype is None or header.get('nifs', 1) != 1:
        raise ValueError(f'{path}: only single-IF 8- and 32-bit data are supported')
    data = np.memmap(path, dtype=dtype, mode='r', offset=offset)
    nchans = header['nchans']
    return header, data[:data.size // nchans * nchans].reshape(-1, nchans)


def channel_frequencies(header):
    return header['fch1'] + np.arange(header['nchans']) * header['foff']
