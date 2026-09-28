"""Read and write SIGPROC filterbank files, with the samples mapped by numpy.

The pipeline's own reader decodes header strings as text and fails on the
flatfielded beams, so this reads the header as bytes. Data are returned as
(time, channel) in file order; for LOFAR's negative foff channel 0 is fch1,
the highest frequency.
"""
import struct

import numpy as np

# The reading side is shared with the classifier's own-data check (lotaas_reprocessing.sigproc_data).
from lotaas_reprocessing.sigproc_data import (DTYPES, INTEGERS, STRINGS, _string, channel_frequencies,  # noqa: F401
                                              open_data, read_header)

ORDER = ['telescope_id', 'machine_id', 'data_type', 'rawdatafile', 'source_name', 'barycentric',
         'pulsarcentric', 'src_raj', 'src_dej', 'az_start', 'za_start', 'tstart', 'tsamp',
         'fch1', 'foff', 'nchans', 'nbeams', 'ibeam', 'nbits', 'nifs']



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
