#!/usr/bin/env python3
import tqdm
import numpy as np
from scipy import stats

def fourier_domain_dedispersion(I_t_nu, tsamp, nu, dm):
    # Data shape
    nchan, nsamp = I_t_nu.shape

    # Time axis
    t = np.arange(nsamp) * tsamp

    # FFT data (real to complex) along time axis
    I_f_nu = np.fft.rfft(I_t_nu, axis=1)

    # Compute spin frequencies (FFT'ed axis of time)
    f = np.fft.rfftfreq(nsamp, tsamp)
    nfreq = len(f)

    # DMs to dedisperse over
    ndm = len(dm)

    # Output array
    I_f_dm = np.zeros((ndm, nfreq), dtype="complex64")

    # Loop over DMs
    numax = np.max(nu)
    for i, dm in enumerate(tqdm.tqdm(dm)):
        # Compute DM delays
        tdm = dm * (nu**(-2) - numax**(-2)) / 2.41e-4

        # Compute phasor
        phasor = np.exp(2j * np.pi * f * tdm[:, np.newaxis]).astype("complex64")

        # Multiply with phasor and sum
        I_f_dm[i] = np.sum(I_f_nu * phasor, axis=0).astype("complex64")

    return I_f_dm

# Sigma clipping, iteratively remove outliers (based on scipy.stats.sigmaclip)
def sigmaclip_2d(z, smin=4.0, smax=4.0):
    # Copy data
    zc = z.copy()
    
    # Data mask
    c = np.ones_like(zc, dtype="bool")

    delta = 1
    while delta:
        cchan = np.sum(c, axis=0)
        zc[~c] = np.nan
        zstd = np.nanstd(zc, axis=0)
        zmean = np.nanmean(zc, axis=0)
        
        zmin = zmean - smin * zstd
        zmax = zmean + smax * zstd

        c = (zc >= zmin) & (zc <= zmax)
        delta = np.sum(cchan - np.sum(c, axis=0))

    return c

# Compute RFI mask
def compute_rfi_mask(data, block_size, sigma_low=3.0, sigma_high=3.0):
    # data shape
    data_copy = data.copy()
    nchan, nsamp = data_copy.shape
    nsub = nsamp // block_size
          
    # Pad array
    if nsub * block_size < nsamp:
        print(f"Padding array by {nsamp - nsub * block_size} samples")
        data_copy = np.pad(data_copy, ((0, 0), (0, (nsub + 1) * block_size - nsamp)), mode="mean")
        nchan, nsamp_padded = data_copy.shape
        nsub = nsamp_padded // block_size

    # Reshape array and normalize by block mean
    data = data_copy.reshape(nchan, nsub, block_size)
    data_mean = np.mean(data, axis=2)
    data_scaled = data / data_mean[:, :, np.newaxis]

    # Find outliers in standard deviation
    data_std = np.std(data_scaled, axis=2)
    cstd = sigmaclip_2d(data_std, sigma_low, sigma_high)

    # Find outliers in skewness
    data_skew = stats.skew(data_scaled, axis=2)
    cskew = sigmaclip_2d(data_skew, sigma_low, sigma_high)

    # Find outliers in kurtosis    
    data_kurt = stats.kurtosis(data_scaled, axis=2)
    ckurt = sigmaclip_2d(data_kurt, sigma_low, sigma_high)

    # Compute combined mask
    c = cstd & cskew & ckurt

    # Create mask
    mask = np.zeros_like(data_std, dtype="bool")
    mask[~c] = True

    return (np.ones((nchan, nsub, block_size), dtype="bool") * mask[:, :, np.newaxis]).reshape(nchan, -1)[:, :nsamp]