import numpy as np
import matplotlib.pyplot as plt

def plot_sap_beam_layout(ax, beam_snrs, positions=None, cmap="viridis"):
    """
    Plots the SAP beam layout on a given axis.

    Parameters:
        ax: matplotlib axis to draw into
        beam_snrs: dict of {beam_id (int or str): snr (float)}
        positions: optional dict of {beam_id: (ra_deg, dec_deg)} in degrees
        cmap: matplotlib colormap name
    """
    beam_radius = 3.5  # smaller beams
    sap_boundary_radius = 18  # arcmin

    def beam_label_to_int(label):
        if isinstance(label, str) and label.upper().startswith("B"):
            return int(label[1:])
        return int(label)

    numeric_snrs = {beam_label_to_int(k): v for k, v in beam_snrs.items()}
    max_snr = max(numeric_snrs.values(), default=1)
    norm = plt.Normalize(vmin=0, vmax=max_snr)
    colormap = plt.get_cmap(cmap)

    if positions:
        ra_vals = [p[0] for p in positions.values()]
        dec_vals = [p[1] for p in positions.values()]
        ra0 = np.mean(ra_vals)
        dec0 = np.mean(dec_vals)
        cos_dec = np.cos(np.deg2rad(dec0))

        all_x = []
        all_y = []

        for beam_id, snr in numeric_snrs.items():
            ra_deg, dec_deg = positions.get(beam_id, (np.nan, np.nan))
            if np.isnan(ra_deg) or np.isnan(dec_deg):
                continue

            # Offset from center in arcminutes
            x = (ra_deg - ra0) * 60 * cos_dec
            y = (dec_deg - dec0) * 60
            all_x.append(x)
            all_y.append(y)

            color = colormap(norm(snr))
            ax.add_patch(plt.Circle((x, y), beam_radius, color=color, ec='black', lw=0.4))
            ax.text(x, y, str(beam_id), fontsize=6, ha='center', va='center',
                    color='white' if norm(snr) > 0.4 else 'black')

        # Set axis limits with padding
        if all_x and all_y:
            margin = 5  # arcmin
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

        # SAP boundary circle
        ax.add_patch(plt.Circle((0, 0), sap_boundary_radius, color='gray', fill=False, linestyle='--', lw=1))

    else:
        # fallback mode
        ax.set_xlim(-sap_boundary_radius - 5, sap_boundary_radius + 5)
        ax.set_ylim(-sap_boundary_radius - 5, sap_boundary_radius + 5)
        ax.add_patch(plt.Circle((0, 0), sap_boundary_radius, color='gray', fill=False, linestyle='--', lw=1))

    ax.set_aspect('equal')
    ax.set_xlabel("RA offset (arcmin)")
    ax.set_ylabel("DEC offset (arcmin)")
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(True, linestyle=':', alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    return sm

def matched_filter_sn_at_idx(signal, tsamp, width_samples, idx, slope=1000):
    """
    Apply matched filtering using a smoothed boxcar kernel and return S/N near a given index.

    Parameters:
        signal (np.ndarray): Dedispersed 1D signal array.
        tsamp (float): Sampling time in seconds.
        width_samples (int): Boxcar width in samples.
        idx (int): Index around which to compute S/N.
        slope (float): Slope for smoothing the kernel edges.

    Returns:
        sn (float): Signal-to-noise ratio at or near the index.
    """
    # Create time array
    t = np.arange(len(signal)) * tsamp
    width_sec = width_samples * tsamp

    def heaviside_step(t, step_time, slope):
        return 0.5 + 0.5 * np.tanh(slope * (t - step_time))

    def generate_boxcar_kernel(t, width, slope=1000):
        tmax = np.max(t)
        kernel = (1 - heaviside_step(t, 0.5 * width, slope) +
                  heaviside_step(t, tmax - 0.5 * width, slope))
        kernel = kernel / np.sqrt(width)
        return kernel

    kernel = generate_boxcar_kernel(t, width_sec, slope=slope)
    signal_fft = np.fft.rfft(signal)
    kernel_fft = np.fft.rfft(kernel)
    response_fft = signal_fft * np.conj(kernel_fft)
    filtered_response = np.fft.irfft(response_fft)

    window = 10
    start = max(0, idx - window)
    end = min(len(filtered_response), idx + window + 1)
    peak_idx = np.argmax(filtered_response[start:end]) + start

    sn = filtered_response[peak_idx] / np.std(filtered_response)

    return sn, peak_idx