"""Read and write SIGPROC filterbank files, with the samples mapped by numpy.

The pipeline's own reader decodes header strings as text and fails on the
flatfielded beams, so this reads the header as bytes. Data are returned as
(time, channel) in file order; for LOFAR's negative foff channel 0 is fch1,
the highest frequency.
"""
import struct

import numpy as np

INTEGERS = {'telescope_id', 'machine_id', 'data_type', 'nchans', 'nbits', 'nifs', 'nbeams',
            'ibeam', 'barycentric', 'pulsarcentric', 'nbins', 'nsamples'}
STRINGS = {'rawdatafile', 'source_name'}
DTYPES = {32: np.float32, 8: np.uint8}
ORDER = ['telescope_id', 'machine_id', 'data_type', 'rawdatafile', 'source_name', 'barycentric',
         'pulsarcentric', 'src_raj', 'src_dej', 'az_start', 'za_start', 'tstart', 'tsamp',
         'fch1', 'foff', 'nchans', 'nbeams', 'ibeam', 'nbits', 'nifs']


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


def _encode(key):
    return struct.pack('<i', len(key)) + key.encode('latin-1')


def write(path, header, data):
    """Write 32-bit samples under a header copied from the source, keys in SIGPROC order."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    header = dict(header, nbits=32, nchans=data.shape[1], nifs=1)
    header.pop('nsamples', None)
    keys = [k for k in ORDER if k in header] + sorted(set(header) - set(ORDER))
    with open(path, 'wb') as stream:
        stream.write(_encode('HEADER_START'))
        for key in keys:
            value = header[key]
            stream.write(_encode(key))
            if key in INTEGERS:
                stream.write(struct.pack('<i', int(value)))
            elif key in STRINGS:
                stream.write(_encode(str(value)))
            else:
                stream.write(struct.pack('<d', float(value)))
        stream.write(_encode('HEADER_END'))
        data.tofile(stream)
