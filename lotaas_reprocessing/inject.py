# inject.py
import numpy as np
import matplotlib.pyplot as plt
from will import create

def inject_synthetic_pulse(masked_data, nu, tsamp, output_dir, base_fname, dm=45.5, sigma_time=10, sigma_freq=10):
    """
    Injects a synthetic Gaussian pulse into the dynamic spectrum.

    Parameters:
        masked_data (ndarray): 2D array of the dynamic spectrum with shape (nchan, nsamp).
        nu (ndarray): Array of frequency channels.
        tsamp (float): Sampling time in seconds.
        output_dir (str): Directory to save the output plot.
        base_fname (str): Base filename for saving the output.
        dm (float): Dispersion measure in pc/cm^3. Default is 45.5.
        sigma_time (float): Pulse width in seconds. Default is 10.
        sigma_freq (float): Frequency spread in MHz. Default is 10.

    Returns:
        ndarray: Modified dynamic spectrum with the synthetic pulse injected.
    """
    # Create synthetic Gaussian pulse
    pulse_obj = create.SimpleGaussPulse(
        sigma_time=sigma_time,  # Pulse width in seconds
        sigma_freq=sigma_freq,  # Frequency spread in MHz
        center_freq=np.median(nu),  # Center frequency
        dm=dm,  # Dispersion measure
        tau=0,  # Intrinsic pulse broadening in microseconds
        phi=0,  # Phase offset
        spectral_index_alpha=0,  # Spectral index
        chan_freqs=nu,  # Frequency channels
        tsamp=tsamp,  # Sampling time in seconds
        nscint=0,  # Scintillation parameter
        bandpass=np.ones_like(nu),  # Flat bandpass
    )

    # Generate the pulse
    nsamp = masked_data.shape[1]
    pulse = pulse_obj.sample_pulse(nsamp=nsamp).T / 50

    # Verify dimensions
    if pulse.shape[0] != masked_data.shape[0]:
        raise ValueError(
            f"Pulse frequency channels ({pulse.shape[0]}) do not match the data's frequency channels ({masked_data.shape[0]})"
        )

    # Inject the pulse into the dynamic spectrum
    injection_start = nsamp // 2  # Start time sample for injection
    injection_end = injection_start + pulse.shape[1]

    if injection_end > masked_data.shape[1]:
        raise ValueError(f"Pulse injection exceeds the time samples of the data: {injection_end} > {masked_data.shape[1]}")

    masked_data[:, injection_start:injection_end] += pulse

    # Save visualization
    plt.figure(figsize=(10, 6))
    plt.imshow(masked_data, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Flux Density")
    plt.title(f"Dynamic Spectrum with Injected Pulse (DM={dm})")
    plt.xlabel("Time Samples")
    plt.ylabel("Frequency Channels")
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{base_fname}_with_injected_pulse.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Injected pulse plot saved to {save_path}")

    return masked_data
