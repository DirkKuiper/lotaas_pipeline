import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

def average_in_time(data, factor):
    """Average the data along the time axis by a given factor."""
    nsamp = data.shape[1]
    reshaped_data = data[:, :nsamp // factor * factor].reshape(data.shape[0], -1, factor)
    return np.nanmean(reshaped_data, axis=2)

def parse_ra_dec(ra_raw, dec_raw):
    """Convert RA and DEC from HHMMSS and DDMMSS formats to HMS and DMS strings."""
    ra_h = int(ra_raw // 10000)
    ra_m = int((ra_raw % 10000) // 100)
    ra_s = ra_raw % 100
    ra_str = f"{ra_h:02}:{ra_m:02}:{ra_s:05.2f}"

    dec_sign = "+" if dec_raw >= 0 else "-"
    dec_d = int(abs(dec_raw) // 10000)
    dec_m = int((abs(dec_raw) % 10000) // 100)
    dec_s = abs(dec_raw) % 100
    dec_str = f"{dec_sign}{dec_d:02}:{dec_m:02}:{dec_s:05.2f}"

    return ra_str, dec_str

def get_observation_info(fil, block_size, masked_frac, factor):
    """Retrieve observation metadata from filterbank header and add additional info in a formatted way."""
    header = fil.header
    
    # RA and DEC in HHMMSS/DDMMSS format
    ra_raw = header.get("src_raj", 0.0)
    dec_raw = header.get("src_dej", 0.0)
    ra_str, dec_str = parse_ra_dec(ra_raw, dec_raw)
    
    # Observation date
    mjd_obs = header.get("tstart", 0.0)
    observation_date = Time(mjd_obs, format="mjd").iso if mjd_obs else "Unknown"
    
    # Frequency range
    freq_min = np.min(fil.frequencies)
    freq_max = np.max(fil.frequencies)
    
    # Time resolution and total observation time
    tsamp = header.get("tsamp", 0.0)
    nsamp = fil.nspec  # Total number of samples (assuming `nspec` is the number of time samples)
    total_time = nsamp * tsamp  # Adjusted for the time-averaging factor
    num_intervals = nsamp // block_size  # Number of intervals based on block size
    
    # Compile formatted observation info
    observation_info = {
        "Object": header.get("source_name", "Unknown"),
        "Telescope": "LOFAR",
        "Instrument": "Unknown",
        "RA (J2000)": ra_str,
        "DEC (J2000)": dec_str,
        "Observation Date": observation_date,
        "T_sample (s)": f"{tsamp:.6f}",
        "T_total (s)": f"{total_time:.2f}",
        "Frequency Range (MHz)": f"{freq_min:.2f} - {freq_max:.2f}",
        "Block Size": block_size,
        "RFI Fraction": f"{masked_frac:.2%}",
        "Time Averaging Factor": factor,
        "Num Channels": header.get("nchans", "Unknown"),
        "Num Intervals": num_intervals,
        "Barycentric": header.get("barycentric", "Unknown"),
        "Pulsarcentric": header.get("pulsarcentric", "Unknown"),
        "Sampling Time (tsamp)": f"{tsamp:.6f}",
        "Number of Bits": header.get("nbits", "Unknown"),
        "Channel Width (MHz)": f"{header.get('foff', 0.0):.6f}"
    }

    return observation_info


# plotting.py
def rfi_diagnostic_plot(masked_data, original_data, mask, t, nu, factor, filename, save_path="rfi_diagnostic_plot.png", observation_info=None):
    """
    Plot the unflagged dynamic spectrum with flagged data vmin/vmax, and subpanels for 
    average spectrum, time series, and flagging fractions, including time averaging factor.

    Parameters:
    - masked_data: 2D array of masked dynamic spectrum data
    - original_data: 2D array of original dynamic spectrum data
    - mask: 2D array of the computed RFI mask
    - t: Time axis in seconds
    - nu: Frequency axis in MHz
    - factor: Time averaging factor
    - filename: Name of the file being processed (used in the title)
    - save_path: Path to save the output plot image
    - observation_info: Dictionary containing additional observation details (e.g., RA, DEC, Source Name)
    """
    
    # Pre-compute averaged data for the dynamic spectrum
    averaged_data = average_in_time(original_data, factor)
    averaged_masked_data = average_in_time(masked_data, factor)
    averaged_t = t[:averaged_data.shape[1]] * factor

    # Define time averaging factors for the time series plot
    time_factors = [16, 128, 512, 1024]
    
    # Define color scheme for averaged time series plots
    colors = ['blue', 'orange', 'green', 'purple']

    # Calculate vmin and vmax for consistent dynamic spectrum display
    vmin, vmax = np.nanpercentile(averaged_masked_data, [5, 99])

    # Start plotting
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(5, 4, width_ratios=[2, 2, 1, 1], height_ratios=[1, 0.5, 1, 1, 4], wspace=0.05, hspace=0.05)

    # Helper function to plot time-averaged time series
    def plot_time_series(ax, data, label, color, alpha=0.5):
        """Plot the time series with given averaging factor."""
        for i, tfactor in enumerate(time_factors):
            averaged_data = np.nanmean(average_in_time(data, tfactor), axis=0)
            averaged_t_factor = t[:averaged_data.shape[0]] * tfactor
            ax.plot(averaged_t_factor, averaged_data, color=colors[i], alpha=alpha, label=f"{label} Avg by {tfactor}")

    # Plot flagging fraction per time bin with multiple averaging factors and original data
    ax_flag_frac_time = fig.add_subplot(gs[1, 0:2])
    flagging_fraction_orig = np.sum(mask, axis=0) / mask.shape[0]
    ax_flag_frac_time.plot(t[:len(flagging_fraction_orig)], flagging_fraction_orig, color="black", alpha=1, label="Original")
    ax_flag_frac_time.set_ylabel("Flagging Fraction")
    ax_flag_frac_time.legend(loc="upper right", fontsize="small")
    ax_flag_frac_time.tick_params(labelbottom=False)

    # Plot mean intensity time series with original and averaged data
    ax_time_series = fig.add_subplot(gs[2, 0:2], sharex=ax_flag_frac_time)
    time_series_orig = np.nanmean(original_data, axis=0)
    ax_time_series.plot(t[:len(time_series_orig)], time_series_orig, color="black", alpha=0.7, label="Original")
    plot_time_series(ax_time_series, original_data, "Intensity", "black")
    ax_time_series.set_ylabel("Mean Intensity")
    ax_time_series.set_ylim(vmin * 2, vmax * 2)
    ax_time_series.legend(loc="upper right", fontsize="small")
    ax_time_series.tick_params(labelbottom=False)

    # Plot flagged mean intensity time series
    ax_flagged_time_series = fig.add_subplot(gs[3, 0:2], sharex=ax_flag_frac_time)
    time_series_masked_orig = np.nanmean(masked_data, axis=0)
    ax_flagged_time_series.plot(t[:len(time_series_masked_orig)], time_series_masked_orig, color="black", alpha=0.7, label="Original Flagged")
    plot_time_series(ax_flagged_time_series, masked_data, "Flagged Intensity", "black")
    ax_flagged_time_series.set_ylabel("Flagged Intensity")
    ax_flagged_time_series.set_ylim(vmin * 2, vmax * 2)
    ax_flagged_time_series.legend(loc="upper right", fontsize="small")
    ax_flagged_time_series.tick_params(labelbottom=False)

    # Dynamic Spectrum (averaged)
    ax_dyn_spec = fig.add_subplot(gs[4, 0:2], sharex=ax_time_series)
    cmap = "viridis"
    ax_dyn_spec.imshow(averaged_data, origin="lower", aspect="auto", interpolation="none", cmap=cmap,
                       vmin=vmin, vmax=vmax, extent=[np.min(averaged_t), np.max(averaged_t), np.min(nu), np.max(nu)])
    ax_dyn_spec.set_xlabel("Time (s)")
    ax_dyn_spec.set_ylabel("Frequency (MHz)")

    # Average Spectrum on the right (averaged)
    ax_spectrum = fig.add_subplot(gs[4, 2], sharey=ax_dyn_spec)
    avg_spectrum_orig = np.nanmean(averaged_data, axis=1)
    avg_spectrum_masked = np.nanmean(averaged_masked_data, axis=1)
    ax_spectrum.plot(avg_spectrum_orig, nu, color="blue", alpha=0.5, label="Original")
    ax_spectrum.plot(avg_spectrum_masked, nu, color="red", alpha=0.5, label="Flagged")
    ax_spectrum.set_xlabel("Mean Intensity")
    ax_spectrum.legend(loc="upper right", fontsize="small")
    ax_spectrum.tick_params(labelleft=False)

    # Flagging Fraction per Frequency Channel
    ax_flag_frac_freq = fig.add_subplot(gs[4, 3], sharey=ax_dyn_spec)
    flagging_fraction_freq = np.sum(mask, axis=1) / mask.shape[1]
    ax_flag_frac_freq.plot(flagging_fraction_freq, nu, color="black")
    ax_flag_frac_freq.set_xlabel("Flagging Fraction")
    ax_flag_frac_freq.set_xlim(0, max(flagging_fraction_freq) + 0.05)
    ax_flag_frac_freq.tick_params(labelleft=False)

    # Display RFI Mask in top right
    ax_rfi_mask = fig.add_subplot(gs[1:4, 2:4])
    ax_rfi_mask.imshow(mask, origin="lower", aspect="auto", cmap="Purples", interpolation="none")
    ax_rfi_mask.set_title("RFI Mask")
    ax_rfi_mask.axis("off")

    # Left and right columns of observation information as separate tables
    if observation_info:
        # Left column data with math text formatting
        left_col_data = [
            ["Object:", observation_info["Object"]],
            ["Telescope:", observation_info["Telescope"]],
            ["Instrument:", observation_info["Instrument"]],
            ["$RA_{J2000}$:", observation_info["RA (J2000)"]],
            ["$DEC_{J2000}$:", observation_info["DEC (J2000)"]],
            ["Observation Date:", observation_info["Observation Date"]],
            ["$T_{sample}$ (s):", observation_info["T_sample (s)"]]
        ]

        # Right column data with math text formatting
        right_col_data = [
            ["$T_{total}$ (s):", observation_info["T_total (s)"]],
            ["Frequency Range (MHz):", observation_info["Frequency Range (MHz)"]],
            ["Block Size:", observation_info["Block Size"]],
            ["RFI Fraction:", observation_info["RFI Fraction"]],
            ["Time Averaging Factor:", observation_info["Time Averaging Factor"]],
            ["Num Channels:", observation_info["Num Channels"]],
            ["Num Intervals:", observation_info["Num Intervals"]]
        ]

        # Extra column data with math text formatting
        extra_col_data = [
            ["Data Type:", f"{observation_info['Number of Bits']} bit"],
            ["Barycentric:", observation_info["Barycentric"]],
            ["Pulsarcentric:", observation_info["Pulsarcentric"]],
            ["Channel Width (MHz):", observation_info["Channel Width (MHz)"][1:]]
        ]


        # Left column table
        ax_left_info = fig.add_subplot(gs[0, 0])
        ax_left_info.axis("off")
        left_table = ax_left_info.table(
            cellText=left_col_data,
            cellLoc="left",
            loc="center"
        )
        left_table.auto_set_font_size(False)
        left_table.set_fontsize(12)
        left_table.scale(1, 1.2)
        for _, cell in left_table.get_celld().items():
            cell.set_edgecolor(None)  # Remove borders

        # Right column table
        ax_right_info = fig.add_subplot(gs[0, 1])
        ax_right_info.axis("off")
        right_table = ax_right_info.table(
            cellText=right_col_data,
            cellLoc="left",
            loc="center"
        )
        right_table.auto_set_font_size(False)
        right_table.set_fontsize(12)
        right_table.scale(1, 1.2)
        for _, cell in right_table.get_celld().items():
            cell.set_edgecolor(None)  # Remove borders

         # Extra column table
        ax_extra_info = fig.add_subplot(gs[0, 2:4])
        ax_extra_info.axis("off")
        extra_table = ax_extra_info.table(
            cellText=extra_col_data,
            cellLoc="left",
            loc="center"
        )
        extra_table.auto_set_font_size(False)
        extra_table.set_fontsize(12)
        extra_table.scale(1, 1.2)
        for _, cell in extra_table.get_celld().items():
            cell.set_edgecolor(None)  # Remove borders

    # Add a title at the top of the figure
    fig.suptitle(f"RFI Diagnostic Plot for {filename}", fontsize=16, fontweight='bold', y=0.9)
    # Save the plot
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.close(fig)

def plot_dedispersed_intensity_sn_focus(I_t_dm, dms, tsamp, output_dir, center_time, time_window=50):
        """
        Plots intensity (in S/N units) as a function of time and DM, focused around a specific time.

        Parameters:
            I_t_dm (2D array): Dedispersed intensity array (DM x Time).
            dms (array-like): Array of dispersion measures.
            tsamp (float): Sampling time in seconds.
            output_dir (str): Directory to save the plot.
            center_time (float): The center time in seconds to focus the plot around.
            time_window (float): Time window in seconds around the center time.
        """
        # Normalize the data to S/N units
        mean_intensity = np.mean(I_t_dm)
        std_intensity = np.std(I_t_dm)
        I_t_dm_sn = (I_t_dm - mean_intensity) / std_intensity

        # Print normalization details for debugging
        print(f"Mean intensity (background): {mean_intensity}")
        print(f"Standard deviation (noise): {std_intensity}")

        # Calculate percentiles for plotting
        vmin = np.percentile(I_t_dm_sn, 1)  # 1st percentile
        vmax = np.percentile(I_t_dm_sn, 99)  # 99th percentile
        print(f"Plotting range (S/N): vmin={vmin}, vmax={vmax}")

        # Generate time axis
        nsamp = I_t_dm.shape[1]
        time_axis = np.arange(nsamp) * tsamp  # Time in seconds

        # Identify the indices for the focus range
        start_time = center_time - time_window / 2
        end_time = center_time + time_window / 2
        start_idx = max(0, int(start_time / tsamp))
        end_idx = min(nsamp, int(end_time / tsamp))
        
        # Slice the data for the focused time range
        I_t_dm_sn_focus = I_t_dm_sn[:, start_idx:end_idx]
        time_axis_focus = time_axis[start_idx:end_idx]

        # Create the figure
        plt.figure(figsize=(12, 6))

        # Plot normalized intensity (S/N)
        plt.imshow(
            I_t_dm_sn_focus,
            aspect='auto',
            origin='lower',
            extent=[time_axis_focus[0], time_axis_focus[-1], dms[0], dms[-1]],
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(label="Signal-to-Noise (S/N)")
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("DM (pc cm$^{-3}$)", fontsize=12)
        plt.title(f"Dedispersed Intensity (S/N) Around {center_time}s", fontsize=14)

        # Save the plot
        plot_path = os.path.join(output_dir, f"dedispersed_intensity_sn_focus_{center_time}s.png")
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Focused dedispersed intensity (S/N) plot saved to {plot_path}")