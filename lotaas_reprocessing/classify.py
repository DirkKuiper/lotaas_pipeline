import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import ssl
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from your.candidate import Candidate
from your.utils.math import normalise
from your.candidate import crop
from fetch.utils import get_model
from lotaas_reprocessing.filterbank import FilterbankFile
from lotaas_reprocessing.numpy_utils import compute_rfi_mask
from matplotlib.gridspec import GridSpec
import pygedm
from astropy.coordinates import SkyCoord
from astropy import units as u
from psrqpy import QueryATNF
from lotaas_reprocessing.sap_beam_map import plot_sap_beam_layout

# Slack Bot Token & Channel ID
SLACK_BOT_TOKEN = "xoxb-513966140291-8603128801253-OIZMLciSFmNefi4An84YDNKE"  # Replace with your real token
CHANNEL_ID = "C08HHTN8CTG"  # Replace with your Slack channel ID

# Initialize Slack Client
client = WebClient(token=SLACK_BOT_TOKEN, ssl=ssl._create_unverified_context())
logger = logging.getLogger(__name__)

def send_slack_notification(image_path, title, text):
    """Uploads an image to Slack and sends a message."""
    try:
        result = client.files_upload_v2(
            channels=CHANNEL_ID,
            file=image_path,
            title=title,
            initial_comment=text
        )
        print(f"Successfully sent {title} to Slack!")

    except SlackApiError as e:
        print(f"Error sending {title} to Slack: {e}")

def classify_candidates(filterbank_file, candidate_file, output_dir, observation_info=None):
    """
    Classifies candidates using FETCH and only saves plots if at least one model predicts > 0.5.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load candidate file
    if not os.path.exists(candidate_file):
        print(f"Candidate file {candidate_file} not found. Skipping classification.")
        return

    candidates_df = pd.read_csv(candidate_file, delim_whitespace=True)
    candidates_df.columns = candidates_df.columns.str.strip().str.lower()  # Normalize column names

    # Limit to max 5 candidates per DM (per beam file)
    candidates_df = (
        candidates_df
        .sort_values("s/n", ascending=False)  # prioritize higher S/N
        .groupby("dm", group_keys=False)
        .head(5)
    )
    # Count how many were dropped per DM (if any)
    dm_counts = candidates_df["dm"].value_counts()
    original_df = pd.read_csv(candidate_file, delim_whitespace=True)
    original_df.columns = original_df.columns.str.strip().str.lower()

    original_dm_counts = (
        original_df
        .assign(dm=lambda df: df["dm"].round(6))
        .groupby("dm")["dm"]
        .count()
    )

    # Find DMs where trimming happened
    dropped_info = []
    for dm_val, original_count in original_dm_counts.items():
        new_count = dm_counts.get(dm_val, 0)
        if original_count > 5 and new_count < original_count:
            dropped_info.append(f"DM={dm_val:.2f}: kept 5 of {original_count}")

    if dropped_info:
        drop_summary = "Limited candidates per DM:\n" + "\n".join(dropped_info)
        print(drop_summary)

        # Optional: Ping this to Slack
        try:
            client.chat_postMessage(
                channel=CHANNEL_ID,
                text=f"[{os.path.basename(candidate_file)}]\n{drop_summary}"
            )
        except SlackApiError as e:
            print(f"Failed to send DM drop info to Slack: {e}")

    for _, row in candidates_df.iterrows():
        dm = row["dm"]
        tcand = row["time"]
        width = int(row["filter_width"])
        snr = row["s/n"]
        sample_number = int(row["sample"])

        if dm < 3:
            print(f"Skipping Candidate with DM={dm} (below DM threshold).")
            continue

        if snr <= 7:
            print(f"Skipping Candidate with DM={dm} (below S/N threshold).")
            continue

        # Dynamically adjust sizes
        time_size, freq_size, dm_size = 256, 256, 256

        # Candidate object
        cand = Candidate(
            fp=filterbank_file,
            dm=dm,
            tcand=tcand,
            width=width,
            label=-1,
            snr=snr,
            min_samp=256,
            device=0,
        )

        print(f"Processing Candidate: DM={dm}, Time={tcand}s, Width={width}, SNR={snr}")

        cand.get_chunk()
        cand.dmtime(dmsteps=256)
        cand.dedisperse()

        # Load filterbank
        fil = FilterbankFile(filterbank_file, "read")
        f_start, delta_f, nchan = fil.fch1, fil.foff, fil.nchans
        frequency_axis = np.flip(f_start + np.arange(nchan) * delta_f)

        # Decimate, crop, and normalize FT
        cand.decimate(key="ft", axis=0, pad=True, decimate_factor=max(1, width // 2), mode="median")
        cand.dedispersed = crop(cand.dedispersed, cand.dedispersed.shape[0] // 2 - time_size // 2, time_size, 0)
        cand.decimate(key="ft", axis=1, pad=True, decimate_factor=cand.dedispersed.shape[1] // freq_size, mode="median")
        cand.resize(key="ft", size=freq_size, axis=1, anti_aliasing=True, mode="constant")
        cand.dedispersed = normalise(cand.dedispersed)

        # **Restore Proper `dmt` Handling**
        # Reshape DM-Time array
        time_decimate_factor = max(1, width // 2)  # Ensure it's at least 1
        cand.decimate(key="dmt", axis=1, pad=True, decimate_factor=time_decimate_factor, mode="median")

        # Crop along the time axis
        crop_start_sample_dmt = cand.dmt.shape[1] // 2 - time_size // 2
        cand.dmt = crop(cand.dmt, crop_start_sample_dmt, time_size, axis=1)

        # Crop along the DM axis
        crop_start_dm = cand.dmt.shape[0] // 2 - dm_size // 2
        cand.dmt = crop(cand.dmt, crop_start_dm, dm_size, axis=0)

        # Resize
        cand.resize(key="dmt", size=dm_size, axis=1, anti_aliasing=True, mode="constant")

        # Normalize `dmt`
        cand.dmt = normalise(cand.dmt)

        # Prepare data for FETCH classification
        X = np.reshape(cand.dedispersed, (1, 256, 256, 1))
        Y = np.reshape(cand.dmt, (1, 256, 256, 1))  # Ensure `dmt` is included

        # Load FETCH models
        model_names = ["a", "b", "c", "d", "e", "f"]
        fetch_models = {name: get_model(name) for name in model_names}

        # Classify candidates
        fetch_probabilities = {}
        for model_name, fetch_model in fetch_models.items():
            prob = fetch_model.predict([X, Y], batch_size=1, verbose=0)[0, 1]
            fetch_probabilities[model_name] = prob

        # Get highest probability
        highest_prob = max(fetch_probabilities.values())
        is_candidate_real = highest_prob > 0.5

        print("\nFETCH Classification Probabilities:")
        for model_name, probability in fetch_probabilities.items():
            print(f"Model {model_name.upper()}: {probability:.4f}")

        print(f"Classified as Astrophysical? {'YES' if is_candidate_real else 'NO'}")

        # If no model predicts > 0.5, SKIP saving and plotting
        if not is_candidate_real:
            print(f"Skipping Candidate DM={dm}, Width={width} (No FETCH model > 0.5)")
            continue

        # Create output directory for this candidate
        cand_dir = os.path.join(
            output_dir, 
            f"DM{dm}_Width{width}_SNR{snr}_Time{tcand:.3f}_Sample{sample_number}"
        )
        os.makedirs(cand_dir, exist_ok=True)

        # Save classification results
        classification_result = os.path.join(cand_dir, "classification_result.txt")
        with open(classification_result, "w") as f:
            f.write("FETCH Classification Probabilities:\n")
            for model_name, probability in fetch_probabilities.items():
                f.write(f"{model_name.upper()}: {probability:.4f}\n")
            f.write(f"Highest Probability: {highest_prob:.4f}\n")
            f.write(f"Classified as Astrophysical? {'YES' if is_candidate_real else 'NO'}\n")
        print(f"Saved classification results to {classification_result}")

       # Prepare Output Directory
        cand_dir = os.path.join(output_dir, f"DM{dm}_Width{width}_SNR{snr}_Time{tcand:.3f}_Sample{sample_number}")
        os.makedirs(cand_dir, exist_ok=True)

        # Prepare axis values for DMT and time series plots
        dm_time_axis = np.arange(cand.dmt.shape[1]) * cand.tsamp + tcand
        dm_values = np.linspace(dm - 5, dm + 5, dm_size)

        time_series = np.sum(cand.dedispersed, axis=1)
        time_axis = np.arange(len(time_series)) * cand.tsamp + tcand

        # Convert to Galactic coordinates
        skycoord = SkyCoord(observation_info["RA (J2000)"], observation_info["DEC (J2000)"], unit=(u.hourangle, u.deg))
        l = skycoord.galactic.l.deg
        b = skycoord.galactic.b.deg

        # Estimate max Galactic DM using a very large distance (e.g. 1e5 pc)
        dm_ne2001, _ = pygedm.dist_to_dm(l, b, 5e4, method='ne2001')
        dm_ymw16, _ = pygedm.dist_to_dm(l, b, 5e4, method='ymw16')

        galactic_info = (
            f"RA (J2000): {observation_info['RA (J2000)']}   |   DEC (J2000): {observation_info['DEC (J2000)']}\n"
            f"Galactic l: {l:.2f}°, b: {b:.2f}°\n"
            f"Max DM (NE2001): {dm_ne2001:.1f}   |   Max DM (YMW16): {dm_ymw16:.1f}"
        )

         # Prepare coordinate strings in hh:mm:ss and dd:mm:ss for PSRCAT query
        ra_str = skycoord.ra.to_string(unit=u.hour, sep=':', pad=True, precision=2)
        dec_str = skycoord.dec.to_string(unit=u.deg, sep=':', alwayssign=True, pad=True, precision=2)

        # Query ATNF catalog for known pulsars within 1° of the pointing position
        query = QueryATNF(
            params=['PSRJ', 'RAJ', 'DECJ', 'DM'],
            coord1=ra_str,
            coord2=dec_str,
            radius=5.0  # degrees
        )

        # Extract table of matching pulsars
        known_psrs_df = query.table.to_pandas()

        # Build a string summary
        if not known_psrs_df.empty:
            pulsar_summary = "\n".join(
                [f"{row['PSRJ']}  RAJ: {row['RAJ']}  DECJ: {row['DECJ']}  DM: {row['DM']}" for _, row in known_psrs_df.iterrows()]
            )
            psr_info_text = f"Known Pulsars within 1°:\n{pulsar_summary}"
        else:
            psr_info_text = "No known pulsars within 1°"

        # Limit number of pulsars shown
        max_psrs = 5
        if not known_psrs_df.empty:
            shown = known_psrs_df.head(max_psrs)
            pulsar_summary = "\n".join(
                [f"{row['PSRJ']}  RAJ: {row['RAJ']}  DECJ: {row['DECJ']}  DM: {row['DM']}" for _, row in shown.iterrows()]
            )
            extra = f"\n(+{len(known_psrs_df)-max_psrs} more...)" if len(known_psrs_df) > max_psrs else ""
            psr_info_text = f"Known Pulsars within 5°:\n{pulsar_summary}{extra}"
        else:
            psr_info_text = "No known pulsars within 5°"

        combined_path = os.path.join(cand_dir, f"diagnostic_DM{dm}_Width{width}.png")
        filename = os.path.basename(filterbank_file)
        filterbank_id = os.path.splitext(filename)[0]

        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(5, 6, figure=fig,
                    width_ratios=[1.7, 1.7, 1.7, 1.7, 3, 0.2],
                    height_ratios=[0.35, 0.35, 1, 5.0, 1],  # Obs | FETCH | Plots
                    wspace=0.3, hspace=0.0)

        # --- Row 0: Observation Info (cols 0–4) ---
        ax_obs = fig.add_subplot(gs[0, 0:5])
        ax_obs.axis("off")
        obs_info = (
            f"File: {filterbank_id}   |   DM: {dm:.2f} pc cm⁻³   |   Width: {width} samples   |   "
            f"S/N: {snr:.2f}   |   Time: {tcand:.3f} s   |   Sample #: {sample_number}"
        )
        ax_obs.text(0.5, 0.5, obs_info, ha="center", va="center", fontsize=10, family="monospace")

        # --- Row 1: FETCH Probs (cols 0–4, inline) ---
        ax_fetch = fig.add_subplot(gs[1, 0:5])
        ax_fetch.axis("off")
        fetch_text = "FETCH Probabilities:   " + "   |   ".join([f"{k.upper()}: {v:.2f}" for k, v in fetch_probabilities.items()])
        ax_fetch.text(0.5, 0.5, fetch_text, ha="center", va="center", fontsize=9, family="monospace")

        ax_gal = fig.add_subplot(gs[2, 3:5])
        ax_gal.axis("off")
        ax_gal.text(0.5, 0.5, galactic_info, ha="center", va="center", fontsize=9, family="monospace")

        # --- Row 2, col 0–3: Time Series ---
        sn_time_series = time_series * (snr / np.max(time_series))
        ax_ts = fig.add_subplot(gs[2, 0:3])
        ax_ts.plot(time_axis, sn_time_series, color="black", lw=1)
        ax_ts.set_ylabel("S/N")
        ax_ts.set_xticks([])

        # --- Row 2, col 0–3: Frequency-Time (overlayed below TS) ---
        ax_ft = fig.add_subplot(gs[3, 0:3])
        im0 = ax_ft.imshow(cand.dedispersed.T, aspect="auto", cmap="viridis",
                        extent=[tcand, tcand + time_size * cand.tsamp, frequency_axis.min(), frequency_axis.max()])
        ax_ft.set_xlabel("Time (s)")
        ax_ft.set_ylabel("Frequency (MHz)")

        # --- Row 2, col 4: DMT ---
        ax_dmt = fig.add_subplot(gs[3, 3:5])
        im1 = ax_dmt.imshow(cand.dmt, aspect="auto", cmap="viridis",
                            extent=[dm_time_axis.min(), dm_time_axis.max(), dm_values.min(), dm_values.max()])
        ax_dmt.set_xlabel("Time (s)")
        ax_dmt.set_ylabel("DM (pc cm⁻³)")

        # Row 4, col 0–5: Known Pulsars
        ax_psr_info = fig.add_subplot(gs[4, 0:5])
        ax_psr_info.axis("off")
        ax_psr_info.text(0.5, 0.5, psr_info_text, ha="center", va="center", fontsize=9, family="monospace", wrap=True)

        # --- Save Layout (no tight_layout!) ---
        fig.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.07)
        plt.savefig(combined_path, dpi=300)
        plt.close()
        print(f"Final diagnostic plot saved: {combined_path}")

        # **Send Plots to Slack**
        send_slack_notification(combined_path, f"Diagnostic - DM {dm}, Width {width}", f"Candidate S/N={snr} in {filterbank_id}")
        # Save candidate to HDF5
        cand.save_h5(cand_dir)

    print("Finished processing all candidates.")