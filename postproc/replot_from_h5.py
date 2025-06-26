import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

def replot_from_h5(h5_path, save_path=None):
    with h5py.File(h5_path, "r") as f:
        # Load dedispersed data and dmt using correct dataset names
        dedispersed = f["data_freq_time"][:]
        dmt = f.get("data_dm_time", None)
        if dmt is not None:
            dmt = dmt[:]

        # Load attributes
        snr = f.attrs["snr"]
        dm = f.attrs["dm"]
        width = f.attrs["width"]
        tcand = f.attrs["tcand"]
        sample_number = f.attrs["cand_id"].split("_")[-1]
        tsamp = f.attrs["tsamp"]
        nchans = f.attrs["nchans"]
        foff = f.attrs["foff"]
        fch1 = f.attrs["fch1"]
        filename = f.attrs["filelist"] if isinstance(f.attrs["filelist"], str) else f.attrs["filelist"][0]

    # Frequency axis
    frequency_axis = np.flip(fch1 + np.arange(nchans) * foff)

    # Manually zap bad frequency channels
    zap_channels = [117,118,132 ,133, 153]  # Example: zap first 3 and last channel

    # Option 1: Set to zero
    dedispersed[:, zap_channels] = np.nan

    # Option 2: Set to NaN for cleaner color scaling
    # dedispersed[:, zap_channels] = np.nan

    # Time axis
    time_series = np.nansum(dedispersed, axis=1)
    time_axis = np.arange(len(time_series)) * tsamp + tcand
    sn_time_series = time_series * (snr / np.max(time_series))

    # DM-Time axes
    time_size, dm_size = dedispersed.shape[0], dmt.shape[0] if dmt is not None else 256
    dm_time_axis = np.arange(dedispersed.shape[0]) * tsamp + tcand
    dm_values = np.linspace(dm - 5, dm + 5, dm_size)

    # Plot
    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(4, 6, figure=fig,
                  width_ratios=[1.7, 1.7, 1.7, 1.7, 3, 0.2],
                  height_ratios=[0.35, 0.35, 1, 5.0],
                  wspace=0.3, hspace=0.0)

    # Row 0: Observation Info
    ax_obs = fig.add_subplot(gs[0, 0:5])
    ax_obs.axis("off")
    obs_info = (
        f"File: {filename}   |   DM: {dm:.2f} pc cm⁻³   |   Width: {width} samples   |   "
        f"S/N: {snr:.2f}   |   Time: {tcand:.3f} s   |   Sample #: {sample_number}"
    )
    ax_obs.text(0.5, 0.5, obs_info, ha="center", va="center", fontsize=10, family="monospace")

    # Row 1: Placeholder FETCH (you can add probs here)
    ax_fetch = fig.add_subplot(gs[1, 0:5])
    ax_fetch.axis("off")
    ax_fetch.text(0.5, 0.5, "FETCH Probabilities: A: -- | B: -- | C: --", ha="center", va="center", fontsize=9, family="monospace")

    # Row 2: Time Series
    ax_ts = fig.add_subplot(gs[2, 0:3])
    ax_ts.plot(time_axis, sn_time_series, color="black", lw=1)
    ax_ts.set_ylabel("S/N")
    ax_ts.set_xticks([])

    # Compute vmin and vmax from the dedispersed data (after zapping)
    vmin, vmax = np.nanpercentile(dedispersed, [20, 99.9])  # Adjust these if needed
    # Row 3: Frequency-Time
    ax_ft = fig.add_subplot(gs[3, 0:3])
    im0 = ax_ft.imshow(
        dedispersed.T,
        aspect="auto",
        cmap="viridis",
        extent=[
            tcand,
            tcand + dedispersed.shape[0] * tsamp,
            frequency_axis.min(),
            frequency_axis.max(),
        ],
        vmin=vmin,
        vmax=vmax,
    )
    ax_ft.set_xlabel("Time (s)")
    ax_ft.set_ylabel("Frequency (MHz)")

    # Row 3: DMT
    if dmt is not None:
        ax_dmt = fig.add_subplot(gs[3, 3:5])
        im1 = ax_dmt.imshow(dmt, aspect="auto", cmap="viridis",
                            extent=[dm_time_axis.min(), dm_time_axis.max(), dm_values.min(), dm_values.max()])
        ax_dmt.set_xlabel("Time (s)")
        ax_dmt.set_ylabel("DM (pc cm⁻³)")

        # Row 3: Colorbar
        cbar_ax = fig.add_subplot(gs[3, 5])
        cbar = fig.colorbar(im1, cax=cbar_ax)
        cbar.set_label("Flux")

    # Save path
    if save_path is None:
        base = os.path.splitext(h5_path)[0]
        save_path = base + "_replot.png"

    fig.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.07)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved replot to: {save_path}")

replot_from_h5("/home/euflash-dkuiper/lotaas_reprocessing/testing/candidate_plots/DM60.7_Width1_SNR7.56_Time2877.916_Sample365946/cand_tstart_57980.044444444444_tcand_2877.9164470_dm_60.70000_snr_7.56000.h5")