import numpy as np
import matplotlib.pyplot as plt
from lotaas_reprocessing import filterbank

# Path to the masked filterbank file
masked_fil_file = "/home/euflash-dkuiper/L600877_ff/L600877_SAP001_B027_ff_masked.fil"

# Open the masked filterbank file
fil = filterbank.FilterbankFile(masked_fil_file, "read")

# Read data
data = fil.get_spectra(0, fil.nspec).T

# Get axes
nchan, nsamp = data.shape
print(f"Original Samples: {nsamp}, Channels: {nchan}")

tsamp = fil.tsamp
print(f"Original Sampling Time: {tsamp} s")

t = np.arange(nsamp) * tsamp  # Time array
frequencies = np.flipud(fil.frequencies)  # Flip frequencies to match orientation

# **Downsample by averaging over 128 time samples**
downsample_factor = 128
new_nsamp = nsamp // downsample_factor  # Number of time bins after downsampling

if new_nsamp > 0:
    # Reshape and average over the downsampling axis
    data_downsampled = data[:, :new_nsamp * downsample_factor].reshape(nchan, new_nsamp, downsample_factor).mean(axis=2)

    # Update time axis
    t_downsampled = np.arange(new_nsamp) * tsamp * downsample_factor  # Adjusted time bins

    print(f"Downsampled Samples: {new_nsamp}, Time Resolution: {tsamp * downsample_factor:.6f} s")

    # Compute vmin and vmax for better contrast
    vmin, vmax = np.nanpercentile(data_downsampled, [5, 99])

    # Plot downsampled dynamic spectrum
    plt.figure(figsize=(12, 6))
    plt.imshow(
        data_downsampled, aspect="auto", cmap="viridis", origin="lower",
        extent=[t_downsampled.min(), t_downsampled.max(), frequencies.min(), frequencies.max()],
        vmin=vmin, vmax=vmax
    )
    plt.colorbar(label="Power")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (MHz)")
    plt.title("Masked Filterbank Data (Downsampled by 128)")

    # Save the plot
    plot_filename = "masked_filterbank_downsampled.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Masked filterbank plot saved as {plot_filename}")
else:
    print("Downsampling factor is too large for the dataset!")