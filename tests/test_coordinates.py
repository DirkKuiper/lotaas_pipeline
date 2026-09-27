"""Right ascension outside 0-24 h, as LOFAR and the early-cycle converter wrote some beams, is wrapped wherever read or written."""
import importlib.util
from pathlib import Path
import struct

import pytest

from euroflash.spider import sigproc_header
from lotaas_reprocessing.coordinates import packed_ra, ra_hours


@pytest.mark.parametrize('given, packed', [
    ('-00:07:35.0', 235225.0),     # L1276887 SAP000 B010 (L606840), 7 min 35 s west of 0 h
    (-735.0, 235225.0),            # the same beam as the converter used to write it
    (-2929.0, 233031.0),           # L1272213 SAP001 B010, Dec +88.9
    (324108.0, 84108.0),           # L167142 SAP001 B001 (early cycle), Dec +89.6
    ('03:32:59.0', 33259.0),       # an ordinary beam is left as it was
    (33259.0, 33259.0),
    ('+12:00:00', 120000.0),
    ('23:59:59.99999', 0.0),       # rounds onto 0 h, not 24 h
])
def test_right_ascension_is_packed_within_a_day(given, packed):
    assert packed_ra(given) == pytest.approx(packed, abs=1e-6)
    assert 0 <= ra_hours(given) < 24


def test_the_header_reader_places_a_beam_west_of_0h_where_the_sky_has_it():
    pytest.importorskip('matplotlib')
    from lotaas_reprocessing.plotting import parse_ra_dec
    assert parse_ra_dec(-735.0, 640318.0) == ('23:52:25.00', '+64:03:18.00')
    assert parse_ra_dec(33259.0, -24608.0) == ('03:32:59.00', '-02:46:08.00')


def filterbank(path, src_raj):
    def string(s):
        return struct.pack('<i', len(s)) + s.encode()
    head = string('HEADER_START')
    for key, value in (('nchans', 4), ('nbits', 32), ('nifs', 1)):
        head += string(key) + struct.pack('<i', value)
    for key, value in (('src_raj', src_raj), ('src_dej', 640318.0), ('fch1', 151.04), ('foff', -0.05), ('tsamp', 1.0)):
        head += string(key) + struct.pack('<d', value)
    path.write_bytes(head + string('HEADER_END') + bytes(range(64)))
    return path


def test_beams_converted_before_the_fix_are_repaired_in_place(tmp_path):
    spec = importlib.util.spec_from_file_location('wrap_ra', Path(__file__).parents[1]/'ops'/'wrap_ra.py')
    wrap_ra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrap_ra)
    west = filterbank(tmp_path/'west.fil', -735.0)
    before = west.read_bytes()
    assert wrap_ra.wrap(west) == (-735.0, 235225.0) and west.read_bytes() == before, 'a dry run changes nothing'
    assert wrap_ra.wrap(west, apply=True) == (-735.0, 235225.0)
    header, length, _ = sigproc_header(west.read_bytes())
    assert header['src_raj'] == 235225.0 and header['src_dej'] == 640318.0
    assert west.read_bytes()[length:] == before[length:] and len(west.read_bytes()) == len(before)
    assert wrap_ra.wrap(filterbank(tmp_path/'fine.fil', 33259.0), apply=True) is None
