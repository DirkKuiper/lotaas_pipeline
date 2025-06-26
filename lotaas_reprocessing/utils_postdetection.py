# src/utils_postdetection.py

import numpy as np
import os
from lotaas_reprocessing import filterbank
from lotaas_reprocessing.numpy_utils import fourier_domain_dedispersion

def get_sn_from_beam(fil_path, dm, tcand, block_size=256):
    fil = filterbank.FilterbankFile(fil_path, "read")
    tsamp = fil.tsamp
    nu = np.flipud(fil.frequencies)
    data = np.flipud(fil.get_spectra(0, fil.nspec).T)

    # Optional: basic normalization
    data = data / np.median(data) - 1

    # Dedisperse only at this DM
    dedispersed = fourier_domain_dedispersion(data, tsamp, nu, [dm])
    dedispersed = np.real(np.fft.irfft(dedispersed, axis=1))[0]

    # Get closest time index
    time_axis = np.arange(dedispersed.shape[0]) * tsamp
    closest_idx = (np.abs(time_axis - tcand)).argmin()

    # Calculate S/N
    snr = dedispersed[closest_idx] / np.std(dedispersed)

    ra = fil.header.get("src_raj")
    dec = fil.header.get("src_dej")
    fil.close()

    return snr, ra, dec