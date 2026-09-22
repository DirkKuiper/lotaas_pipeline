"""Create clearly labelled synthetic PSRFITS for container integration tests."""
from pathlib import Path
import sys
import numpy as np
from astropy.io import fits


def make(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    nchan, nsblk, nsub = 1024, 4096, 4
    rng = np.random.default_rng(55)
    for beam in [13, 14, 15]:
        data = np.clip(rng.normal(100, 8, (nsub, nsblk*nchan)), 0, 255).astype('uint8')
        freq = np.tile(np.linspace(120., 170., nchan, dtype='float32'), (nsub, 1))
        columns = [fits.Column(name='DAT_FREQ', format=f'{nchan}E', array=freq),
                   fits.Column(name='DAT_SCL', format=f'{nchan}E', array=np.ones((nsub, nchan))),
                   fits.Column(name='DAT_OFFS', format=f'{nchan}E', array=np.zeros((nsub, nchan))),
                   fits.Column(name='DAT_WTS', format=f'{nchan}E', array=np.ones((nsub, nchan))),
                   fits.Column(name='DATA', format=f'{nsblk*nchan}B', array=data)]
        table = fits.BinTableHDU.from_columns(columns, name='SUBINT')
        for k,v in dict(NCHAN=nchan, NSBLK=nsblk, NBITS=8, NPOL=1, TBIN=.005, CHAN_BW=50/1023).items():
            table.header[k]=v
        primary = fits.PrimaryHDU()
        for k,v in {'SRC_NAME':'SYNTHETIC_VALIDATION','RA':'03:32:59.0','DEC':'+54:34:43.0','DATE-OBS':'2026-01-01T00:00:00','OBSFREQ':145.}.items():
            primary.header[k]=v
        fits.HDUList([primary,table]).writeto(directory/f'L000000_SAP000_BEAM{beam:03d}.fits',overwrite=True)


if __name__ == '__main__':
    make(sys.argv[1])
