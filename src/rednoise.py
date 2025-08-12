import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt

def high_pass_filter(data, cutoff_freq, sampling_rate, order=3):
    """
    Apply a high-pass filter to the data.
    """
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normalized_cutoff, btype="high", analog=False)
    print(f"Cutoff: {cutoff_freq}, Coefficients: b={b}, a={a}")

    return filtfilt(b, a, data)

def compute_power_spectrum(signal, sampling_time):
    """
    Compute the power spectrum of a signal.
    """
    fft_result = np.fft.rfft(signal)
    power_spectrum = np.abs(fft_result) ** 2
    frequency_array = np.fft.rfftfreq(len(signal), sampling_time)
    return frequency_array, power_spectrum

def inject_gaussian_pulse(signal, amplitude, center_time, pulse_width, sampling_time):
    """
    Inject a Gaussian pulse into a signal.
    """
    num_samples = len(signal)
    time_array = np.arange(num_samples) * sampling_time
    sigma = pulse_width / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
    gaussian_pulse = amplitude * np.exp(-0.5 * ((time_array - center_time) / sigma) ** 2)
    return signal + gaussian_pulse

def analyze_multiple_cutoffs(input_file, output_dir, cutoff_frequencies):
    # Define sampling time and observation duration
    sampling_time = 16 * 16 * 6 * 5.12e-6  # Sampling time
    total_observation_time = 3600.0  # Total observation duration in seconds
    sampling_rate = 1 / sampling_time  # Sampling rate (Hz)

    # Load data from file
    raw_data = np.fromfile(input_file, dtype="float32")
    num_samples = len(raw_data)
    time_array = np.linspace(0, total_observation_time, num=num_samples)

    # Handle NaNs: Replace with zeros to ensure FFT works as before
    raw_data = np.nan_to_num(raw_data, nan=0.0)

    # Inject Gaussian pulse
    pulse_amplitude = 15
    pulse_center_time = 1800.0  # Inject pulse at the middle of the observation (in seconds)
    pulse_width = 100  # Pulse width (FWHM) in seconds
    injected_signal = inject_gaussian_pulse(raw_data.copy(), pulse_amplitude, pulse_center_time, pulse_width, sampling_time)

    power_reduction_results = []
    snr_results = []
    filtered_time_series = {}

    for cutoff_frequency in cutoff_frequencies:
        # Apply high-pass filter
        filtered_data = high_pass_filter(raw_data, cutoff_frequency, sampling_rate)
        filtered_injected_signal = high_pass_filter(injected_signal, cutoff_frequency, sampling_rate)

        # Compute power spectrum and reduction
        freq_orig, power_orig = compute_power_spectrum(raw_data, sampling_time)
        freq_filt, power_filt = compute_power_spectrum(filtered_data, sampling_time)
        low_freq_mask = freq_orig < cutoff_frequency
        power_reduction = (np.sum(power_orig[low_freq_mask]) - np.sum(power_filt[low_freq_mask])) / np.sum(power_orig[low_freq_mask]) * 100

        # SNR calculation
        pulse_start = int((pulse_center_time - 3 * pulse_width) / sampling_time)
        pulse_end = int((pulse_center_time + 3 * pulse_width) / sampling_time)

        noise_std = np.std(np.concatenate((filtered_injected_signal[:pulse_start], filtered_injected_signal[pulse_end:])))
        pulse_peak_amplitude = np.max(filtered_injected_signal[pulse_start:pulse_end]) - np.median(filtered_injected_signal)

        snr_filtered = pulse_peak_amplitude / noise_std

        power_reduction_results.append((cutoff_frequency, power_reduction))
        snr_results.append((cutoff_frequency, snr_filtered))
        filtered_time_series[cutoff_frequency] = (time_array, filtered_injected_signal, pulse_start, pulse_end)

        print(f"Cutoff Frequency: {cutoff_frequency:.4f} Hz")
        print(f"Power Reduction: {power_reduction:.2f}%")
        print(f"SNR (Filtered): {snr_filtered:.2f}")

    # Create overview plots
    plot_overview(power_reduction_results, snr_results, filtered_time_series, output_dir)

def plot_overview(power_reduction_results, snr_results, filtered_time_series, output_dir):
    cutoff_frequencies = [r[0] for r in power_reduction_results]
    power_reductions = [r[1] for r in power_reduction_results]
    snr_filtered = [r[1] for r in snr_results]

    # Overview plot with metrics and time series
    num_filters = len(cutoff_frequencies)
    fig = plt.figure(figsize=(12, 6 + num_filters * 3))

    # Metrics subplot
    ax1 = plt.subplot2grid((num_filters + 2, 1), (0, 0), rowspan=2)
    ax1.plot(cutoff_frequencies, power_reductions, label="Power Reduction (%)", marker="o", color="blue")
    ax1.set_xlabel("Cutoff Frequency (Hz)")
    ax1.set_ylabel("Power Reduction (%)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Secondary axis for SNR
    ax2 = ax1.twinx()
    ax2.plot(cutoff_frequencies, snr_filtered, label="SNR (Filtered)", marker="o", color="orange", linestyle="--")
    ax2.set_ylabel("SNR", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))

    # Add time series plots for each filter
    for i, cutoff_frequency in enumerate(cutoff_frequencies):
        ax = plt.subplot2grid((num_filters + 2, 1), (i + 2, 0))
        time_array, filtered_injected_signal, pulse_start, pulse_end = filtered_time_series[cutoff_frequency]
        ax.plot(time_array, filtered_injected_signal, label=f"Filtered Signal w/ Pulse (Cutoff = {cutoff_frequency:.4f} Hz)", alpha=0.7, color="red")
        ax.axvspan(time_array[pulse_start], time_array[pulse_end], color="yellow", alpha=0.3, label="Pulse Region")
        ax.set_xlim([time_array[0], time_array[-1]])
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Time (s)")
        ax.legend()

    plt.tight_layout()
    overview_path = os.path.join(output_dir, "cutoff_frequency_overview_with_time_series_and_pulse.png")
    plt.savefig(overview_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Overview plot with time series and pulse saved at: {overview_path}")

if __name__ == "__main__":
    input_file = "/home/euflash-dkuiper/lotaas_reprocessing/output/L600877_SAP001_B027_ff/DM_trials/L600877_SAP001_B027_ff_DM45.0.dat"
    output_directory = "./output_plots/"
    cutoff_frequencies = [0.002, 0.003, 0.004, 0.005, 0.0075, 0.01]
    os.makedirs(output_directory, exist_ok=True)
    analyze_multiple_cutoffs(input_file, output_directory, cutoff_frequencies)
