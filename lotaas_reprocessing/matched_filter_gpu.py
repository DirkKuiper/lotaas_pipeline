import os
import re
import cupy as cp  # CuPy for GPU acceleration
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.gridspec import GridSpec

def heaviside_step(t, step_time, slope):
    """Smoothed Heaviside step function."""
    return 0.5 + 0.5 * cp.tanh(slope * (t - step_time))

def generate_boxcar_kernel(t, width, slope):
    """Generates a smoothed, zero-mean boxcar kernel over the time array."""
    tmax = cp.max(t)
    kernel = (1 - heaviside_step(t, 0.5 * width, slope) + 
              heaviside_step(t, tmax - 0.5 * width, slope))
    kernel = kernel / cp.sqrt(width)  # Normalize to unit sum
    return kernel  # Zero mean

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
        max_width = int(max_duration / (tsamp * downsample))  # Convert max duration to samples
        
        # Generate exponentially spaced filter widths
        filter_widths = np.unique(np.geomspace(min_width, max_width, num=16).astype(int))
        
        return cp.array(filter_widths)

def run_matched_filtering(data_file, tsamp, dm, downsample=1, detection_threshold=5):
    """Runs GPU-accelerated matched filtering, dynamically adjusting filter widths."""

    # Compute dynamic filter widths based on downsampling
    filter_widths_samples = compute_filter_widths(tsamp, downsample)

    # Load dedispersed data into GPU memory
    signal_data = cp.asarray(np.fromfile(data_file, dtype="float32"))
    nsamp = signal_data.size  # Number of time samples
    t = cp.arange(nsamp) * tsamp * downsample  # GPU time array (accounting for downsampling)

    # GPU Fourier Transform
    signal_fft = cp.fft.rfft(signal_data)

    # Allocate memory for filtered responses (GPU)
    filtered_responses = cp.zeros((len(filter_widths_samples), nsamp))

    # GPU-accelerated matched filtering
    for i, width_samples in enumerate(filter_widths_samples):
        width_seconds = width_samples * tsamp * downsample  # Adjusted width
        kernel = generate_boxcar_kernel(t, width_seconds, slope=1000)
        kernel_fft = cp.fft.rfft(kernel)
        response_fft = signal_fft * cp.conj(kernel_fft)
        filtered_response = cp.fft.irfft(response_fft)
        filtered_responses[i] = filtered_response / cp.std(filtered_response)

    # Identify significant responses (on GPU)
    significant_indices = cp.where(filtered_responses >= detection_threshold)
    detection_times = t[significant_indices[1]].get()  # Move to CPU
    detection_strengths = filtered_responses[significant_indices].get()  # Move to CPU
    detection_dms = np.full(len(detection_strengths), dm)  # Keep this on CPU
    detection_widths_samples = filter_widths_samples[significant_indices[0]].get()  # Move to CPU

    return detection_times, detection_dms, detection_strengths, detection_widths_samples

def run_all_matched_filtering(dm_trials_dir, tsamp, output_dir, observation_info, dedispersion_plan, detection_threshold=5):
    """Runs GPU-based matched filtering across all DM trials."""
    
    all_candidates = []

    # Find all .dat files in the DM trials directory
    dm_files = [f for f in os.listdir(dm_trials_dir) if f.endswith(".dat")]

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
            detection_samples = (detection_times / tsamp).astype(int)

            # Collect all candidates
            for time, dm, strength, sample, width_samples in zip(detection_times, detection_dms, detection_strengths, detection_samples, detection_widths_samples):
                all_candidates.append((dm, strength, time, sample, width_samples))

    # Write all candidates to a .cands file
    cands_filepath = os.path.join(output_dir, "all_detected_candidates.cands")
    with open(cands_filepath, "w") as cands_file:
        cands_file.write("# DM(pc/cm^3)  Detection Strength  Time(s)  Sample  Filter Width(samples)\n")
        for dm, strength, time, sample, width_samples in all_candidates:
            cands_file.write(f"{dm:.3f}  {strength:.3f}  {time:.6f}  {sample}  {width_samples}\n")
    print(f"All candidates written to {cands_filepath}")

    # Separate candidates into lists for plotting
    dms = [c[0] for c in all_candidates]
    strengths = [c[1] for c in all_candidates]
    times = [c[2] for c in all_candidates]

    # Calculate summed S/N for each DM
    from collections import defaultdict
    summed_sn_by_dm = defaultdict(float)
    for dm, strength in zip(dms, strengths):
        summed_sn_by_dm[dm] += strength

    sorted_dms = sorted(summed_sn_by_dm.keys())
    summed_sn = [summed_sn_by_dm[dm] for dm in sorted_dms]

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
    ax3.scatter(dms, strengths, color='black', s=1)
    ax3.set_xlabel("DM (pc cm$^{-3}$)")
    ax3.set_xscale('log')
    ax3.set_ylabel("Signal-to-Noise")

    # Middle panel: Time vs. DM scatter plot
    ax4 = fig.add_subplot(gs[1, :])
    scatter = ax4.scatter(times, dms, c=strengths, cmap='viridis', s=5)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("DM (pc cm$^{-3}$)")
    ax4.set_yscale('log')
    fig.colorbar(scatter, ax=ax4, label="Detection Strength")

    # Bottom panel: Summed S/N by DM
    ax5 = fig.add_subplot(gs[2, :])
    ax5.bar(sorted_dms, summed_sn, width=sorted_dms[1] - sorted_dms[0], color='blue', alpha=0.7, edgecolor='black')
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