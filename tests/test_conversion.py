import numpy as np
import pytest
from astropy.io import fits
from preproc.downsample_psrfits2fil_32bit import (convert, reduce_integer_first, reduce_reference,
                                                  reduce_subint)
from lotaas_reprocessing.filterbank import FilterbankFile


def two_bit_row(rng, nsamp, nchan, weights=None):
    return {'DATA': rng.integers(0, 256, nsamp * nchan // 4, dtype=np.uint8),
            'DAT_SCL': rng.uniform(0.2, 3.0, nchan).astype(np.float32),
            'DAT_OFFS': rng.uniform(-2.0, 5.0, nchan).astype(np.float32),
            'DAT_WTS': (np.ones(nchan) if weights is None else weights).astype(np.float32)}


@pytest.mark.parametrize('fscrunch,tscrunch', [(4, 16), (8, 4), (1, 1), (2, 32)])
def test_integer_first_matches_the_reference_reduction(fscrunch, tscrunch):
    rng = np.random.default_rng(3)
    nsamp, nchan = 256, 64
    weights = np.ones(nchan); weights[[5, 17, 40]] = 0
    row = two_bit_row(rng, nsamp, nchan, weights)
    fast = reduce_integer_first(row, 2, nsamp, nchan, fscrunch, tscrunch)
    reference = reduce_reference(row, 2, nsamp, nchan, fscrunch, tscrunch)
    assert fast.dtype == np.float32 and fast.shape == reference.shape
    np.testing.assert_allclose(fast, reference, rtol=2e-6, atol=1e-6)


def test_eight_bit_rows_take_the_integer_path_too():
    rng = np.random.default_rng(4)
    nsamp, nchan = 128, 32
    row = {'DATA': rng.integers(0, 256, nsamp * nchan, dtype=np.uint8),
           'DAT_SCL': rng.uniform(.5, 2, nchan), 'DAT_OFFS': rng.uniform(-1, 1, nchan),
           'DAT_WTS': np.ones(nchan)}
    np.testing.assert_allclose(reduce_integer_first(row, 8, nsamp, nchan, 4, 16),
                               reduce_reference(row, 8, nsamp, nchan, 4, 16), rtol=2e-6, atol=1e-5)


def test_a_nonfinite_scale_keeps_the_reference_nan_semantics():
    rng = np.random.default_rng(5)
    nsamp, nchan = 64, 16
    row = two_bit_row(rng, nsamp, nchan)
    row['DAT_SCL'][3] = np.nan
    result = reduce_subint(row, 2, nsamp, nchan, 4, 16)
    # nanmean drops the bad channel from its group rather than poisoning it.
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result, reduce_reference(row, 2, nsamp, nchan, 4, 16))


def test_unsupported_bit_depth_is_refused():
    with pytest.raises(ValueError, match='bit depth'):
        reduce_subint({'DATA': np.zeros(4), 'DAT_SCL': np.ones(4), 'DAT_OFFS': np.zeros(4),
                       'DAT_WTS': np.ones(4)}, 4, 4, 4, 1, 1)


def test_whole_file_conversion_matches_the_reference(tmp_path):
    rng = np.random.default_rng(6)
    nchan, nsblk, nsub = 64, 512, 3
    rows = [two_bit_row(rng, nsblk, nchan) for _ in range(nsub)]
    columns = [fits.Column(name='DAT_FREQ', format=f'{nchan}E', array=np.tile(np.linspace(120., 150., nchan), (nsub, 1))),
               fits.Column(name='DAT_SCL', format=f'{nchan}E', array=np.array([r['DAT_SCL'] for r in rows])),
               fits.Column(name='DAT_OFFS', format=f'{nchan}E', array=np.array([r['DAT_OFFS'] for r in rows])),
               fits.Column(name='DAT_WTS', format=f'{nchan}E', array=np.array([r['DAT_WTS'] for r in rows])),
               fits.Column(name='DATA', format=f'{nsblk * nchan // 4}B', array=np.array([r['DATA'] for r in rows]))]
    table = fits.BinTableHDU.from_columns(columns, name='SUBINT')
    for key, value in dict(NCHAN=nchan, NSBLK=nsblk, NBITS=2, NPOL=1, TBIN=.0005).items():
        table.header[key] = value
    primary = fits.PrimaryHDU()
    for key, value in {'SRC_NAME': 'SYNTHETIC', 'RA': '03:32:59.0', 'DEC': '+54:34:43.0',
                       'DATE-OBS': '2026-01-01T00:00:00'}.items():
        primary.header[key] = value
    source = tmp_path / 'L000001_SAP000_BEAM013.fits'
    fits.HDUList([primary, table]).writeto(source)
    output = tmp_path / 'out.fil'
    convert(source, output)
    fil = FilterbankFile(str(output))
    converted = fil.get_spectra(0, fil.nspec)
    fil.close()
    expected = np.concatenate([reduce_reference(r, 2, nsblk, nchan, 4, 16) for r in rows])[:, ::-1]
    assert converted.shape == expected.shape
    np.testing.assert_allclose(converted, expected, rtol=2e-6, atol=1e-6)
