"""Boxcar matched filtering by rolling sums, against a robust noise scale.

The statistic for width w is the sum over the window, centred and divided by
a median-absolute-deviation estimate of that rolling sum's own scale. Summing
w independent samples scales the noise by sqrt(w), so the MAD carries the
sqrt(w) normalisation and the result is directly a signal-to-noise ratio.

Three properties matter here, and none of them held for the circular-FFT
implementation this replaces (measured in review/2026-09-22/diagnostics.json):

  * A bright pulse no longer suppresses itself. Dividing by the standard
    deviation of a response that contains the signal put a 100-sigma pulse of
    300 s at a statistic of 4.09, below the detection threshold of 5. The MAD
    is unmoved until a pulse fills more than half the observation.
  * Windows are rectangular, so a pulse matching the window recovers the full
    sqrt(w) gain. The tanh-smoothed kernel returned 0.707 of the ideal
    statistic at width one, where most single pulses are found, and was not
    monotonic in width.
  * Rolling sums do not wrap. Circular convolution reported an event in the
    last ten samples of a series as 81 detections in the first 0.1 seconds.

Dedispersion upstream is still circular, which is a separate wrap and is
handled separately: whatever occupied the first samples of the low-frequency
channels reappears at the end of each trial, so that tail is excluded from
the search. See wrap_contaminated_samples.

Rolling sums are also roughly an order of magnitude cheaper than one inverse
FFT per width per DM trial, which was the pipeline's throughput bottleneck.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from functools import lru_cache

# Scale factor taking a median absolute deviation to a Gaussian sigma.
MAD_TO_SIGMA = 1.4826
# Windows drawn to estimate one noise scale. The precision of a scale from n
# samples goes as 1/sqrt(2n), so 65536 windows fix it to about 0.3%, which
# moves a threshold of 5 by 0.014. Two medians over every window of every
# width would otherwise cost more than the rolling sums themselves.
SCALE_SAMPLE = 65536


def _no_detections():
    empty = np.array([], dtype=float)
    return empty, empty, empty, np.array([], dtype=int)


def robust_scale(values, rng=None):
    """Centre and noise scale of a series, which its own pulses cannot inflate.

    A median absolute deviation resists contamination up to half the samples.
    Beyond that no in-band estimate of the noise is meaningful, and such a
    signal is a bandpass or gain fault rather than an astrophysical pulse.

    Long series are estimated from a random subsample. Random draws are used
    rather than a stride because a stride can beat against periodic RFI and
    bias the very quantity being estimated.
    """
    if rng is not None and values.size > SCALE_SAMPLE:
        values = values[rng.integers(0, values.size, SCALE_SAMPLE)]
    centre = float(np.median(values))
    scale = float(np.median(np.abs(values - centre))) * MAD_TO_SIGMA
    if scale > 0:
        return centre, scale
    # More than half the windows share one value: constant, or so heavily
    # quantised that the median deviation vanishes. Fall back to the mean of
    # the deviations that are non-zero, and report nothing if there are none.
    deviation = np.abs(values - centre)
    non_zero = deviation[deviation > 0]
    if non_zero.size == 0:
        return centre, 0.0
    return centre, float(np.mean(non_zero)) * MAD_TO_SIGMA


def boxcar_statistic(cumulative, width, rng=None):
    """Signal-to-noise for every width-w window lying wholly inside the data."""
    total = cumulative[width:] - cumulative[:-width]
    centre, scale = robust_scale(total, rng)
    if scale <= 0:
        return None
    return (total - centre) / scale


def heaviside_step(t, step_time, slope):
    """Smoothed Heaviside step function. Retained for regression comparison."""
    return 0.5 + 0.5 * np.tanh(slope * (t - step_time))

def generate_boxcar_kernel(t, width, slope):
    """The superseded smoothed boxcar kernel. Retained for regression comparison."""
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
        if not np.isfinite(max_duration) or max_duration <= 0:
            raise ValueError('Maximum pulse duration must be finite and positive')
        min_width = 1  # Smallest width to test in samples
        max_width = int(max_duration / (tsamp * downsample))
        if max_width < 1:
            return np.array([], dtype=int)
        
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


def filter_widths_for(nsamp, tsamp, downsample, max_duration=600):
    """Widths to search, bounded so every window has enough siblings to
    estimate a noise scale from. Half the series leaves at least nsamp/2
    windows, which is ample for a median."""
    return compute_filter_widths(tsamp, downsample,
                                 max_duration=min(max_duration, nsamp * tsamp * downsample / 2))


DISPERSION_CONSTANT = 2.41e-4


def wrap_contaminated_samples(dm, nu_min, nu_max, tsamp, downsample=1):
    """Trailing samples of a trial that circular dedispersion has polluted.

    Dedispersion advances each channel by its own delay with a circular
    shift, so whatever occupied the first delay samples of the low-frequency
    channels reappears at the end of the series. Broadband interference in
    the first three samples of a beam produces a response of 56 sigma in the
    last 1089 samples at DM 100, against 13 in the uncontaminated middle.
    Real pulses are unaffected: they are recovered at the right time, only
    with less bandwidth as the sweep runs past the end of the observation.
    """
    if not (nu_min and nu_max) or nu_min <= 0 or nu_max <= 0:
        return 0
    delay = abs(dm) * (min(nu_min, nu_max) ** -2 - max(nu_min, nu_max) ** -2) / DISPERSION_CONSTANT
    return int(np.ceil(delay / (tsamp * downsample)))


def run_matched_filtering(data_file, tsamp, dm, downsample=1, detection_threshold=5,
                          seed=0, valid_samples=None, max_duration=600):
    signal_data = np.fromfile(data_file, dtype="float32")
    if valid_samples is not None:
        # Drop the polluted tail before anything else, so it cannot raise a
        # detection nor inflate the noise scale the rest is measured against.
        if valid_samples < 2:
            raise ValueError(
                f"Circular dedispersion pollutes the whole trial at DM {dm}: the "
                f"sweep is longer than the observation. Shorten the DM plan.")
        signal_data = signal_data[:valid_samples]
    nsamp = signal_data.size
    if nsamp < 2 or not np.isfinite(signal_data).all():
        raise ValueError(f"Invalid DM trial: {data_file}")
    if np.ptp(signal_data) == 0:
        return _no_detections()
    # Seeded per call, so re-searching one trial reproduces its candidates.
    rng = np.random.default_rng(seed)
    widths = filter_widths_for(nsamp, tsamp, downsample, max_duration)
    # One pass in float64: partial sums of float32 drift over millions of samples.
    cumulative = np.empty(nsamp + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(signal_data, dtype=np.float64, out=cumulative[1:])
    starts, strengths, detected_widths = [], [], []
    for width in widths:
        width = int(width)
        if width > nsamp:
            continue
        statistic = boxcar_statistic(cumulative, width, rng)
        if statistic is None:
            continue
        selected = np.flatnonzero(statistic >= detection_threshold)
        if not selected.size:
            continue
        # Report the centre of the window, the arrival time a centred kernel gave.
        starts.append(selected + (width - 1) / 2.0)
        strengths.append(statistic[selected])
        detected_widths.append(np.full(selected.size, width, dtype=int))
    if not starts:
        return _no_detections()
    samples = np.concatenate(starts)
    strengths = np.concatenate(strengths)
    return (samples * tsamp * downsample, np.full(samples.size, dm), strengths,
            np.concatenate(detected_widths))


def _dm_axis(axis, dms, which="x"):
    """Scale a DM axis so its labels are readable for the range actually searched.

    A log scale was applied unconditionally. Over a narrow range it produces
    minor ticks in scientific notation that overlap into an unreadable smear,
    and it cannot show DM 0 at all. Log only earns its place when the range
    spans more than a decade, which the production plan does and a targeted
    search does not.
    """
    values = np.asarray(dms, dtype=float)
    positive = values[values > 0]
    decades = (positive.max() / positive.min()) if positive.size and positive.min() > 0 else 1.
    scale = "log" if decades >= 10 else "linear"
    (axis.set_xscale if which == "x" else axis.set_yscale)(scale)
    if scale == "linear":
        return
    # Plain numbers at every decade, including the sub-unit DMs that a
    # ScalarFormatter rounds to "0".
    formatter = matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:g}")
    target = axis.xaxis if which == "x" else axis.yaxis
    target.set_major_formatter(formatter)
    target.set_minor_formatter(matplotlib.ticker.NullFormatter())


def run_all_matched_filtering(dm_trials_dir, tsamp, output_dir, observation_info,
                              dedispersion_plan, detection_threshold=5,
                              nu_min=None, nu_max=None, trim_wrap=True, max_duration=600):
    """Runs CPU-based matched filtering across all DM trials.

    With the band limits available, the tail that circular dedispersion has
    polluted is excluded from each trial. That costs real sensitivity at the
    end of an observation, growing with DM: 0.4% of a one-hour beam at DM
    150, 3% at DM 1000, 30% at DM 10000. Set trim_wrap False to search the
    whole series and accept the wrapped interference instead.
    """
    
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
            valid = None
            if trim_wrap and nu_min and nu_max:
                polluted = wrap_contaminated_samples(dm, nu_min, nu_max, tsamp, downsample)
                total = os.path.getsize(dm_filepath) // 4
                valid = total - polluted
                if valid < 2:
                    raise ValueError(
                        f"The dispersion sweep at DM {dm} exceeds the observation; "
                        f"{polluted} of {total} samples would be discarded. "
                        f"Shorten the DM plan for this observation length.")
            detection_times, detection_dms, detection_strengths, detection_widths_samples = run_matched_filtering(
            dm_filepath, tsamp, dm, downsample, detection_threshold, valid_samples=valid,
            max_duration=max_duration
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
    _dm_axis(ax2, dms)
    ax2.set_ylabel("Number of Pulses")

    # Top-right: Signal-to-Noise vs. DM
    ax3 = fig.add_subplot(gs[0, 2])
    display_stride=max(1,int(np.ceil(len(times)/100000)))
    ax3.scatter(dms[::display_stride], strengths[::display_stride], color='black', s=1)
    ax3.set_xlabel("DM (pc cm$^{-3}$)")
    _dm_axis(ax3, dms)
    ax3.set_ylabel("Signal-to-Noise")

    # Middle panel: Time vs. DM scatter plot
    ax4 = fig.add_subplot(gs[1, :])
    scatter = ax4.scatter(times[::display_stride], dms[::display_stride], c=strengths[::display_stride], cmap='viridis', s=5)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("DM (pc cm$^{-3}$)")
    _dm_axis(ax4, dms, which='y')
    fig.colorbar(scatter, ax=ax4, label="Detection Strength")

    # Bottom panel: Summed S/N by DM
    ax5 = fig.add_subplot(gs[2, :])
    # Per-bar widths: the plan's DM spacing changes by a factor 200 across
    # ranges, so one global width either leaves gaps or overlaps.
    if len(sorted_dms) > 1:
        steps = np.diff(sorted_dms)
        bar_width = np.concatenate([steps, steps[-1:]])
    else:
        bar_width = 0.1
    ax5.bar(sorted_dms, summed_sn, width=bar_width, color='blue', alpha=0.7, edgecolor='black')
    ax5.set_xlabel("DM (pc cm$^{-3}$)")
    _dm_axis(ax5, dms)
    ax5.set_ylabel("Summed Signal-to-Noise")
    ax5.set_title("Summed Signal-to-Noise as a Function of DM")

    # Title and metadata using observation_info
    fig.text(0.5, 0.99, f"Source: {observation_info['Object']}", ha='center', va='top', fontsize=10)
    fig.text(0.5, 0.97, f"Telescope: {observation_info['Telescope']}   Instrument: {observation_info['Instrument']}", ha='center', va='top', fontsize=10)
    fig.text(0.5, 0.95, f"Observation Date: {observation_info['Observation Date']}", ha='center', va='top', fontsize=10)
    fig.text(0.5, 0.93, f"N positives: {len(times)}   Sampling time: {np.round(tsamp * 1e3, 2)} ms   Frequency Range: {observation_info['Frequency Range (MHz)']}", ha='center', va='top', fontsize=10)
    # The beam, not the absolute path of wherever this happened to run.
    fig.suptitle(f"Single Pulse Results: {os.path.basename(os.path.normpath(output_dir))}",
                 fontsize=14, fontweight='bold', y=1.02)

    # Save the figure
    overview_path = os.path.join(output_dir, "all_matched_filter_overview.png")
    plt.savefig(overview_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Overview plot saved as {overview_path}")
