#!/usr/bin/env python3
import tqdm
import numpy as np
import cupy as cp

def fourier_domain_dedispersion(I_t_nu, tsamp, nu, dm):
    # Data shape
    nchan, nsamp = I_t_nu.shape

    # Time axis
    t = cp.arange(nsamp) * tsamp

    # FFT data (real to complex) along time axis
    I_f_nu = cp.fft.rfft(cp.array(I_t_nu), axis=1)

    # Compute spin frequencies (FFT'ed axis of time)
    f = cp.fft.rfftfreq(nsamp, tsamp).astype("float32")
    nfreq = len(f)

    # DMs to dedisperse over
    ndm = len(dm)

    # Output array
    I_f_dm = cp.zeros((ndm, nfreq), dtype="complex64")

    # Loop over DMs
    numax = cp.max(nu)
    for i, dm in enumerate(tqdm.tqdm(dm)):
        # Compute DM delays
        tdm = cp.array(dm * (nu**(-2) - numax**(-2)) / 2.41e-4).astype("float32")

        # Compute phasor
        phasor = cp.exp(2j * cp.pi * f * tdm[:, cp.newaxis]).astype("complex64")

        # Multiply with phasor and sum
        I_f_dm[i] = cp.sum(I_f_nu * phasor, axis=0).astype("complex64")

    # Free unused memory
    I_f_nu = None
    phasor = None

    return cp.asnumpy(I_f_dm)

# Custom kurtosis and skew functions for cupy as scipy is not compatible
def compute_skew(data):
    mean = cp.mean(data, axis=2, keepdims=True)
    std_dev = cp.std(data, axis=2, keepdims=True)
    skewness = cp.mean(((data - mean) / std_dev) ** 3, axis=2)
    return skewness

def compute_kurtosis(data):
    mean = cp.mean(data, axis=2, keepdims=True)
    std_dev = cp.std(data, axis=2, keepdims=True)
    kurtosis = cp.mean(((data - mean) / std_dev) ** 4, axis=2) - 3
    return kurtosis

# Sigma clipping, iteratively remove outliers
def sigmaclip_2d(z, smin=4.0, smax=4.0):
    zc = z.copy()
    c = cp.ones_like(zc, dtype=bool)

    delta = 1
    while delta:
        cchan = cp.sum(c, axis=0)
        zc[~c] = cp.nan
        zstd = cp.nanstd(zc, axis=0)
        zmean = cp.nanmean(zc, axis=0)

        zmin = zmean - smin * zstd
        zmax = zmean + smax * zstd

        c = (zc >= zmin) & (zc <= zmax)
        delta = cp.sum(cchan - cp.sum(c, axis=0))

    return c

# Compute RFI mask, but using cupy and custom stats functions
def compute_rfi_mask(data, block_size, sigma_low=3.0, sigma_high=3.0):
    # Copy data to cupy array
    data_copy = cp.array(data)
    nchan, nsamp = data_copy.shape
    nsub = nsamp // block_size

    # Pad array
    if nsub * block_size < nsamp:
        print(f"Padding array by {nsamp - nsub * block_size} samples")
        data_copy = cp.pad(data_copy, ((0, 0), (0, (nsub + 1) * block_size - nsamp)), mode="mean")
        nchan, nsamp_padded = data_copy.shape
        nsub = nsamp_padded // block_size

    # Reshape array and normalize by block mean
    data = data_copy.reshape(nchan, nsub, block_size)
    data_mean = cp.mean(data, axis=2)
    data_scaled = data / data_mean[:, :, cp.newaxis]

    # Find outliers in standard deviation
    data_std = cp.std(data_scaled, axis=2)
    cstd = sigmaclip_2d(data_std, sigma_low, sigma_high)

    # Find outliers in skewness
    data_skew = compute_skew(data_scaled)
    cskew = sigmaclip_2d(data_skew, sigma_low, sigma_high)

    # Find outliers in kurtosis
    data_kurt = compute_kurtosis(data_scaled)
    ckurt = sigmaclip_2d(data_kurt, sigma_low, sigma_high)

    # Compute combined mask
    c = cstd & cskew & ckurt

    # Create mask
    mask = cp.zeros_like(data_std, dtype=bool)
    mask[~c] = True

    return cp.asnumpy((cp.ones((nchan, nsub, block_size), dtype=bool) * mask[:, :, cp.newaxis]).reshape(nchan, -1)[:, :nsamp])