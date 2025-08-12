import numpy as np
import matplotlib.pyplot as plt
import os
import re

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

def parse_inf_file(inf_file):
    """Parses .inf file to retrieve metadata including sampling time."""
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

def plot_dm_time_plane(data_file, tsamp, output_dir):
    """Plots the DM-Time plane (original data) to visualize baseline variations."""
    signal_data = np.fromfile(data_file, dtype="float32")
    nsamp = len(signal_data)
    t = np.arange(nsamp) * tsamp  # Time array

    plt.figure(figsize=(12, 4))
    plt.plot(t, signal_data, color='black', lw=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("DM-Time Plane (Original Data)")
    dm_time_path = os.path.join(output_dir, "dm_time_plane.png")
    plt.savefig(dm_time_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"DM-Time plane plot saved as {dm_time_path}")

def run_matched_filtering(data_file, tsamp, dm, output_dir, detection_threshold=5):
    # Filter widths in seconds
    filter_widths = np.array([2, 3, 4, 6, 9, 14, 20, 30, 45, 70, 100, 150, 220, 300]) * tsamp

    # Load dedispersed data and set up time axis
    signal_data = np.fromfile(data_file, dtype="float32")[::-1]
    nsamp = len(signal_data)
    t = np.arange(nsamp) * tsamp  # Time array

    # Fourier transform of the signal data
    signal_fft = np.fft.rfft(signal_data)
    filtered_responses = np.zeros((len(filter_widths), nsamp))

    # Set up subplots: split into 3 subplots for shorter, medium, and longer widths
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Filtered Responses for DM: {dm} (Different Boxcar Widths)", fontsize=14)

    # Separate widths into short, medium, and long categories
    width_categories = [filter_widths[:5], filter_widths[5:10], filter_widths[10:]]
    axes_titles = ["Short Widths", "Medium Widths", "Long Widths"]

    for ax, widths, title in zip(axes, width_categories, axes_titles):
        for width in widths:
            kernel = generate_boxcar_kernel(t, width, slope=1000)
            kernel_fft = np.fft.rfft(kernel)
            response_fft = signal_fft * np.conj(kernel_fft)
            filtered_response = np.fft.irfft(response_fft)
            normalized_response = filtered_response / np.std(filtered_response)

            # Plot with transparency
            ax.plot(t, normalized_response, lw=0.5, alpha=0.7, label=f'Width: {width:.2f} s')

        ax.set_ylabel("Filtered Response")
        ax.legend(fontsize="small", loc="upper right")
        ax.set_title(title)

    axes[-1].set_xlabel("Time (s)")
    filtered_response_path = os.path.join(output_dir, "filtered_responses_grouped.png")
    plt.savefig(filtered_response_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grouped filtered responses plot saved as {filtered_response_path}")

    # Identify significant responses
    significant_indices = np.where(filtered_responses >= detection_threshold)
    detection_times = t[significant_indices[1]]
    detection_strengths = filtered_responses[significant_indices]
    detection_dms = np.full(len(detection_strengths), dm)

    return detection_times, detection_dms, detection_strengths

def run_single_matched_filtering(data_file, output_dir, detection_threshold=5):
    # Locate the corresponding .inf file to extract metadata
    inf_file = data_file.replace(".dat", ".inf")
    metadata = parse_inf_file(inf_file)

    # Check if tsamp was found
    tsamp = metadata.get("tsamp")
    if tsamp is None:
        print("Sampling time (tsamp) not found in the .inf file.")
        return

    # Plot the DM-Time plane
    plot_dm_time_plane(data_file, tsamp=tsamp, output_dir=output_dir)

    # Extract DM value from the filename
    match = re.search(r"_DM([0-9.]+)\.dat", data_file)
    if match:
        dm = float(match.group(1))
        detection_times, detection_dms, detection_strengths = run_matched_filtering(
            data_file, tsamp, dm, output_dir, detection_threshold=detection_threshold
        )
    else:
        print(f"Could not extract DM from filename: {data_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run matched filtering and visualize DM-Time and filtered responses.")
    parser.add_argument("data_file", help="Path to the .dat file to process")
    parser.add_argument("output_dir", help="Directory to save output plots")
    parser.add_argument("--threshold", type=float, default=5, help="Detection threshold for S/N")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_single_matched_filtering(args.data_file, args.output_dir, args.threshold)
