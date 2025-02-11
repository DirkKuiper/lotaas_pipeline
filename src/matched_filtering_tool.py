import numpy as np
import matplotlib.pyplot as plt
import os
import re
import argparse
from matplotlib.gridspec import GridSpec
from scipy.signal import detrend, butter, filtfilt

def high_pass_filter(data, cutoff_freq, sampling_rate, order=3):
    """
    Apply a high-pass filter to the data.
    :param data: Input signal (1D array).
    :param cutoff_freq: Cutoff frequency of the filter (Hz).
    :param sampling_rate: Sampling rate of the data (Hz).
    :param order: Order of the Butterworth filter.
    :return: Filtered signal.
    """
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normalized_cutoff, btype="high", analog=False)
    return filtfilt(b, a, data)

def heaviside_step(t, step_time, slope):
    """Smoothed Heaviside step function."""
    return 0.5 + 0.5 * np.tanh(slope * (t - step_time))

def generate_boxcar_kernel(t, width, slope):
    """Generates a smoothed, zero-mean boxcar kernel over the time array."""
    tmax = np.max(t)
    kernel = (1 - heaviside_step(t, 0.5 * width, slope) + 
              heaviside_step(t, tmax - 0.5 * width, slope))
    kernel = kernel / np.sqrt(width)  # Normalize to unit sum
    return kernel - np.mean(kernel)   # Adjust to zero mean

def parse_inf_file(inf_file):
    """Parses .inf file for observation metadata."""
    metadata = {}
    try:
        with open(inf_file, 'r') as f:
            for line in f:
                if "Width of each time series bin" in line:
                    metadata['tsamp'] = float(line.split("=")[-1].strip())
                elif "Object name" in line:
                    metadata['Object'] = line.split("=")[-1].strip()
                elif "Frequency range" in line:
                    metadata['Frequency Range (MHz)'] = line.split("=")[-1].strip()
                elif "Telescope" in line:
                    metadata['Telescope'] = line.split("=")[-1].strip()
                elif "Instrument" in line:
                    metadata['Instrument'] = line.split("=")[-1].strip()
                elif "Observation Date" in line:
                    metadata['Observation Date'] = line.split("=")[-1].strip()
    except FileNotFoundError:
        print(f"Could not find .inf file: {inf_file}")
    return metadata

def plot_time_series(data_file, tsamp, output_dir):
    """Plots the original time series from the .dat file."""
    signal_data = np.fromfile(data_file, dtype="float32")
    signal_data = detrend(signal_data)  # Remove a linear trend
    nsamp = len(signal_data)
    t = np.arange(nsamp) * tsamp  # Time array

    # Plot the time series
    plt.figure(figsize=(12, 4))
    plt.plot(t, signal_data, color='black', lw=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Original Time Series")
    time_series_path = os.path.join(output_dir, "original_time_series.png")
    plt.savefig(time_series_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Time series plot saved as {time_series_path}")

def run_matched_filtering(data_file, tsamp, dm, detection_threshold=5):
    # Filter widths in seconds
    filter_widths = np.array([2, 3, 4, 6, 9, 14, 20, 30, 45, 70, 100, 150, 220, 300]) * tsamp

    # Load dedispersed data and set up time axis
    signal_data = np.fromfile(data_file, dtype="float32")
    # signal_data = detrend(signal_data)  # Remove a linear trend
    nsamp = len(signal_data)
    t = np.arange(nsamp) * tsamp  # Time array

    # Apply high-pass filter to remove long-period variations
    sampling_rate = 1 / tsamp
    cutoff_frequency = 0.004  # Cutoff frequency in Hz
    signal_data = high_pass_filter(signal_data, cutoff_frequency, sampling_rate)

    # Fourier transform of the signal data
    signal_fft = np.fft.rfft(signal_data)
    filtered_responses = np.zeros((len(filter_widths), nsamp))

    # Apply matched filtering for each filter width
    for i, width in enumerate(filter_widths):
        kernel = generate_boxcar_kernel(t, width, slope=1000)
        kernel_fft = np.fft.rfft(kernel)
        response_fft = signal_fft * np.conj(kernel_fft)
        filtered_response = np.fft.irfft(response_fft)
        filtered_responses[i] = filtered_response / np.std(filtered_response)

    # Identify significant responses
    significant_indices = np.where(filtered_responses >= detection_threshold)
    detection_times = t[significant_indices[1]]
    detection_strengths = filtered_responses[significant_indices]
    detection_dms = np.full(len(detection_strengths), dm)

    return detection_times, detection_dms, detection_strengths

def run_all_matched_filtering(dm_trials_dir, output_dir, detection_threshold=5):
    # Locate .dat and .inf files
    dm_files = [f for f in os.listdir(dm_trials_dir) if f.endswith(".dat")]
    inf_file = next((f for f in os.listdir(dm_trials_dir) if f.endswith(".inf")), None)

    if not inf_file:
        print(f"No .inf file found in {dm_trials_dir}.")
        return

    observation_info = parse_inf_file(os.path.join(dm_trials_dir, inf_file))
    tsamp = observation_info.get("tsamp", None)

    if not tsamp:
        print("Sampling time not found in .inf file.")
        return

    all_times = []
    all_dms = []
    all_strengths = []

    for dm_file in dm_files:
        dm_filepath = os.path.join(dm_trials_dir, dm_file)
        # plot_time_series(dm_filepath, tsamp, output_dir)  # Plot the original time series

        match = re.search(r"_DM([0-9.]+)\.dat", dm_file)
        
        if match:
            dm = float(match.group(1))
            detection_times, detection_dms, detection_strengths = run_matched_filtering(
                dm_filepath, tsamp, dm, detection_threshold
            )

            all_times.extend(detection_times)
            all_dms.extend(detection_dms)
            all_strengths.extend(detection_strengths)
        else:
            print(f"Could not extract DM from filename: {dm_file}")

    # Plot results
    plot_results(all_times, all_dms, all_strengths, observation_info, output_dir)

def plot_results(times, dms, strengths, observation_info, output_dir):
    times = np.array(times)
    dms = np.array(dms)
    strengths = np.array(strengths)

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 3, height_ratios=[1, 3], width_ratios=[1, 1, 1], wspace=0.3)

    # Top-left: Histogram of Signal-to-Noise
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(strengths, bins=20, color='black', edgecolor='black')
    ax1.set_yscale('log')
    ax1.set_xlabel("Signal-to-Noise")
    ax1.set_ylabel("Number of Pulses")

    # Top-center: Histogram of DM
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(dms, bins=20, color='black', edgecolor='black')
    ax2.set_xlabel("DM (pc cm$^{-3}$)")
    ax2.set_ylabel("Number of Pulses")

    # Top-right: Signal-to-Noise vs. DM
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(dms, strengths, color='black', s=1)
    ax3.set_xlabel("DM (pc cm$^{-3}$)")
    ax3.set_ylabel("Signal-to-Noise")

    # Bottom panel: Time vs. DM scatter plot
    ax4 = fig.add_subplot(gs[1, :])
    scatter = ax4.scatter(times, dms, c=strengths, cmap='viridis', s=5)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("DM (pc cm$^{-3}$)")
    fig.colorbar(scatter, ax=ax4, label="Detection Strength")

    # Metadata
    fig.suptitle(f"Single Pulse Results for '{output_dir}'", fontsize=14, fontweight='bold')
    overview_path = os.path.join(output_dir, "matched_filter_overview.png")
    plt.savefig(overview_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Overview plot saved as {overview_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run matched filtering on dedispersed .dat files.")
    parser.add_argument("input_dir", help="Directory containing .dat and .inf files")
    parser.add_argument("output_dir", help="Directory to save output plots")
    parser.add_argument("--threshold", type=float, default=5, help="Detection threshold for S/N")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_all_matched_filtering(args.input_dir, args.output_dir, args.threshold)
