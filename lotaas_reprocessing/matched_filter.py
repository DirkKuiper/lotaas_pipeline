import numpy as np
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from functools import lru_cache

def heaviside_step(t, step_time, slope):
    """Smoothed Heaviside step function."""
    return 0.5 + 0.5 * np.tanh(slope * (t - step_time))

def generate_boxcar_kernel(t, width, slope):
    """Generates a smoothed, zero-mean boxcar kernel over the time array."""
    tmax = np.max(t)
    kernel = (1 - heaviside_step(t, 0.5 * width, slope) + 
              heaviside_step(t, tmax - 0.5 * width, slope))
    kernel = kernel / np.sqrt(width)  # Normalize to unit sum
    return kernel   # Adjust to zero mean

def compute_filter_widths(tsamp, downsample, max_duration=600):
        """
        Dynamically generate an array of filter widths based on downsampling.
        
        Args:
            tsamp (float): Base time sampling resolution in seconds.
            downsample (int): Current downsampling factor.
            max_duration (float): Maximum duration to search for in seconds.
            
        Returns:
            np.array: Optimized filter widths in samples.
        """
        min_width = 1  # Smallest width to test in samples
        max_width = max(1, int(max_duration / (tsamp * downsample)))  # Convert max duration to samples
        
        # Generate exponentially spaced filter widths
        filter_widths = np.unique(np.geomspace(min_width, max_width, num=16).astype(int))
        
        return np.array(filter_widths)

@lru_cache(maxsize=8)
def _kernel_spectra(nsamp, tsamp, downsample):
    """Reuse identical filters across DM trials of the same length/sampling."""
    # Windows as long as a short pilot observation wrap onto themselves and
    # become constant. Restrict the search to half of the available duration.
    widths = compute_filter_widths(tsamp, downsample,
                                  max_duration=min(600, (nsamp-1)*tsamp*downsample/2))
    t = np.arange(nsamp) * tsamp * downsample
    spectra = tuple(np.fft.rfft(generate_boxcar_kernel(t, int(w) * tsamp * downsample, slope=1000))
                    for w in widths)
    return widths, spectra


def run_matched_filtering(data_file, tsamp, dm, downsample=1, detection_threshold=5):
    signal_data = np.fromfile(data_file, dtype="float32")
    nsamp = signal_data.size
    if nsamp < 2 or not np.isfinite(signal_data).all():
        raise ValueError(f"Invalid DM trial: {data_file}")
    if np.ptp(signal_data) == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, np.array([], dtype=int)
    widths, kernels = _kernel_spectra(nsamp, tsamp, downsample)
    signal_fft = np.fft.rfft(signal_data)
    indices, strengths, detected_widths = [], [], []
    # Keep one response at a time, retaining the original FFT/kernel statistic.
    for width, kernel_fft in zip(widths, kernels):
        response = np.fft.irfft(signal_fft * np.conj(kernel_fft), n=nsamp)
        std = np.std(response)
        if std <= 0:
            continue
        response /= std
        selected = np.flatnonzero(response >= detection_threshold)
        indices.append(selected)
        strengths.append(response[selected])
        detected_widths.append(np.full(len(selected), width, dtype=int))
    if not indices:
        empty = np.array([], dtype=float)
        return empty, empty, empty, np.array([], dtype=int)
    samples = np.concatenate(indices)
    strengths = np.concatenate(strengths)
    return (samples * tsamp * downsample, np.full(len(samples), dm), strengths,
            np.concatenate(detected_widths))


def run_all_matched_filtering(dm_trials_dir, tsamp, output_dir, observation_info, dedispersion_plan, detection_threshold=5):
    """Runs CPU-based matched filtering across all DM trials."""
    
    all_candidates = []
    candidate_count = 0

    # Find all .dat files in the DM trials directory
    dm_files = sorted(f for f in os.listdir(dm_trials_dir) if f.endswith(".dat"))
    if not dm_files:
        raise ValueError("No dedispersed trials to search")

    # Progress bar for matched filtering
    for dm_file in tqdm(dm_files, desc="Matched Filtering Progress", unit="file"):
        dm_filepath = os.path.join(dm_trials_dir, dm_file)
        match = re.search(r"_DM([0-9.]+)\.dat", dm_file)

        if match:
            dm = float(match.group(1))

            # Get downsampling factor directly from dedispersion plan
            downsample = 1  # Default
            for plan in dedispersion_plan:
                if plan["low_dm"] <= dm < plan["high_dm"]:
                    downsample = plan["downsample"]
                    break
            detection_times, detection_dms, detection_strengths, detection_widths_samples = run_matched_filtering(
            dm_filepath, tsamp, dm, downsample, detection_threshold
            )

            # Calculate sample indices
            detection_samples = np.rint(detection_times / tsamp).astype(int)
            detection_widths_samples = detection_widths_samples * downsample  # base filterbank samples
            from lotaas_reprocessing.cluster import MAX_CANDIDATES
            candidate_count += len(detection_times)
            if candidate_count > MAX_CANDIDATES:
                raise RuntimeError(f"Too many candidates (>{MAX_CANDIDATES}) at {dm_file}; inspect RFI before retrying")

            # Collect all candidates
            if len(detection_times):
                all_candidates.append(np.column_stack((detection_dms,detection_strengths,
                    detection_times,detection_samples,detection_widths_samples)))

    # Compact numeric arrays avoid millions of Python tuples and scalar objects.
    candidates = np.concatenate(all_candidates) if all_candidates else np.empty((0,5))
    del all_candidates

    # Write all candidates to a .cands file
    cands_filepath = os.path.join(output_dir, "all_detected_candidates.cands")
    with open(cands_filepath, "w") as cands_file:
        cands_file.write("# DM(pc/cm^3)  Detection Strength  Time(s)  Sample  Filter Width(samples)\n")
        np.savetxt(cands_file,candidates,fmt=['%.3f','%.3f','%.6f','%d','%d'],delimiter='  ')
    print(f"All candidates written to {cands_filepath}")

    # Separate candidates into lists for plotting
    dms,strengths,times = candidates[:,:3].T

    # Calculate summed S/N for each DM
    sorted_dms,dm_inverse = np.unique(dms,return_inverse=True)
    summed_sn = np.bincount(dm_inverse,weights=strengths)
    del dm_inverse

    # Multi-panel plot setup
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(3, 3, height_ratios=[1, 3, 1], width_ratios=[1, 1, 1], wspace=0.3, hspace=0.5)

    # Top-left: Histogram of Signal-to-Noise
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(strengths, bins=20, color='black', edgecolor='black')
    ax1.set_yscale('log')
    ax1.set_xlabel("Signal-to-Noise")
    ax1.set_ylabel("Number of Pulses")

    # Top-center: Histogram of DM
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(dms, bins=100, color='black', edgecolor='black')
    ax2.set_xlabel("DM (pc cm$^{-3}$)")
    ax2.set_xscale('log')
    ax2.set_ylabel("Number of Pulses")

    # Top-right: Signal-to-Noise vs. DM
    ax3 = fig.add_subplot(gs[0, 2])
    display_stride=max(1,int(np.ceil(len(times)/100000)))
    ax3.scatter(dms[::display_stride], strengths[::display_stride], color='black', s=1)
    ax3.set_xlabel("DM (pc cm$^{-3}$)")
    ax3.set_xscale('log')
    ax3.set_ylabel("Signal-to-Noise")

    # Middle panel: Time vs. DM scatter plot
    ax4 = fig.add_subplot(gs[1, :])
    scatter = ax4.scatter(times[::display_stride], dms[::display_stride], c=strengths[::display_stride], cmap='viridis', s=5)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("DM (pc cm$^{-3}$)")
    ax4.set_yscale('log')
    fig.colorbar(scatter, ax=ax4, label="Detection Strength")

    # Bottom panel: Summed S/N by DM
    ax5 = fig.add_subplot(gs[2, :])
    ax5.bar(sorted_dms, summed_sn, width=(sorted_dms[1] - sorted_dms[0]) if len(sorted_dms) > 1 else 0.1, color='blue', alpha=0.7, edgecolor='black')
    ax5.set_xlabel("DM (pc cm$^{-3}$)")
    ax5.set_xscale('log')
    ax5.set_ylabel("Summed Signal-to-Noise")
    ax5.set_title("Summed Signal-to-Noise as a Function of DM")

    # Title and metadata using observation_info
    fig.text(0.5, 0.99, f"Source: {observation_info['Object']}", ha='center', va='top', fontsize=10)
    fig.text(0.5, 0.97, f"Telescope: {observation_info['Telescope']}   Instrument: {observation_info['Instrument']}", ha='center', va='top', fontsize=10)
    fig.text(0.5, 0.95, f"Observation Date: {observation_info['Observation Date']}", ha='center', va='top', fontsize=10)
    fig.text(0.5, 0.93, f"N positives: {len(times)}   Sampling time: {np.round(tsamp * 1e3, 2)} ms   Frequency Range: {observation_info['Frequency Range (MHz)']}", ha='center', va='top', fontsize=10)
    fig.suptitle(f"Single Pulse Results for '{output_dir}'", fontsize=14, fontweight='bold', y=1.02)

    # Save the figure
    overview_path = os.path.join(output_dir, "all_matched_filter_overview.png")
    plt.savefig(overview_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Overview plot saved as {overview_path}")
