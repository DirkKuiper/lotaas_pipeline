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
from matplotlib.gridspec import GridSpec
import pygedm
from astropy.coordinates import SkyCoord
from astropy import units as u

# Import DB utils
from db.db_utils import insert_beam_run, update_beam_run, insert_detection

# Import LOTAAS matching
from lotaas_reprocessing.lotaas_matcher import load_reference, lotaas_within, match_redetection

SLACK_BOT_TOKEN = "xoxb-513966140291-8603128801253-OIZMLciSFmNefi4An84YDNKE"
CHANNEL_ID = "C08HHTN8CTG"

client = WebClient(token=SLACK_BOT_TOKEN, ssl=ssl._create_unverified_context())
logger = logging.getLogger(__name__)

# --- test mode toggles (env) ---
TEST_FORCE_PLOTS = True
TEST_FORCE_SLACK = False
print(f"[DEBUG] TEST_FORCE_PLOTS={TEST_FORCE_PLOTS}, TEST_FORCE_SLACK={TEST_FORCE_SLACK}")
def send_slack_message(text):
    try:
        client.chat_postMessage(channel=CHANNEL_ID, text=text)
        print("Slack message sent.")
    except SlackApiError as e:
        print(f"Slack error: {e}")

def classify_candidates(filterbank_file, candidate_file, output_dir, observation_info=None):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(candidate_file):
        print(f"Candidate file {candidate_file} not found. Skipping.")
        return

    candidates_df = pd.read_csv(candidate_file, delim_whitespace=True)
    candidates_df.columns = candidates_df.columns.str.strip().str.lower()

    beam_id = os.path.basename(filterbank_file)
    observation_date = observation_info.get("Observation Date", "Unknown")
    output_dir = os.path.abspath(output_dir)
    log_file = os.path.join(output_dir, "processing.log")

    beam_run_id = insert_beam_run(
        beam_id=beam_id,
        observation_date=observation_date,
        output_dir=output_dir,
        log_file=log_file,
        code_version="v1.0"
    )

    try:
        # Pointing coordinate
        skycoord = SkyCoord(
            observation_info["RA (J2000)"],
            observation_info["DEC (J2000)"],
            unit=(u.hourangle, u.deg)
        )

        # Load sealed LOTAAS ref and filter to local neighborhood
        _ = load_reference()  # ensure cached
        local_lotaas = lotaas_within(skycoord, radius_arcmin=3.0)
        highest_snr = 0

        # Reduce candidate list: top 5 per DM, sorted by S/N
        candidates_df = (
            candidates_df
            .sort_values("s/n", ascending=False)
            .groupby("dm", group_keys=False)
            .head(5)
        )

        model_names = ["a", "b", "c", "d", "e", "f"]
        fetch_models = {name: get_model(name) for name in model_names}

        # Dict to store best S/N redetection per pulsar
        redetections_best = {}
        num_redetections = 0
        slack_messages = []

        for _, row in candidates_df.iterrows():
            dm = row["dm"]
            tcand = row["time"]
            width = int(row["filter_width"])
            snr = row["s/n"]
            sample_number = int(row["sample"])

            if dm < 10 or snr <= 7:
                continue

            if snr > highest_snr:
                highest_snr = snr

            # Check redetection against LOTAAS reference
            m = match_redetection(dm_cand=dm, local_ref=local_lotaas,
                      dm_abs_tol=5.0, dm_rel_tol=0.10)

            # test-mode flag for downstream plotting/labels
            is_redetection_test = False
            redetect_meta = None

            if m["is_match"]:
                psr_name = m["psr"]
                if psr_name not in redetections_best or snr > redetections_best[psr_name]["snr"]:
                    redetections_best[psr_name] = {
                        "dm": dm,
                        "snr": snr,
                        "width": width,
                        "sep_arcsec": m.get("sep_arcsec"),
                        "dm_ref": m.get("dm_ref"),
                    }
                if not TEST_FORCE_PLOTS:
                    # normal behavior: skip redetections
                    continue
                else:
                    # test mode: continue through to plotting branch, annotate as redetection
                    is_redetection_test = True
                    redetect_meta = {
                        "psr": psr_name,
                        "dm_ref": m.get("dm_ref"),
                        "sep_arcsec": m.get("sep_arcsec"),
                    }

            # Classify non-redetections with FETCH
            time_size, freq_size, dm_size = 256, 256, 256

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
            cand.get_chunk()
            cand.dmtime(dmsteps=256)
            cand.dedisperse()

            fil = FilterbankFile(filterbank_file, "read")
            f_start, delta_f, nchan = fil.fch1, fil.foff, fil.nchans
            frequency_axis = np.flip(f_start + np.arange(nchan) * delta_f)

            # Decimate, crop, normalize
            cand.decimate(key="ft", axis=0, pad=True,
                          decimate_factor=max(1, width // 2), mode="median")
            cand.dedispersed = crop(
                cand.dedispersed,
                cand.dedispersed.shape[0] // 2 - time_size // 2,
                time_size, 0
            )
            cand.decimate(key="ft", axis=1, pad=True,
                          decimate_factor=cand.dedispersed.shape[1] // freq_size,
                          mode="median")
            cand.resize(key="ft", size=freq_size, axis=1,
                        anti_aliasing=True, mode="constant")
            cand.dedispersed = normalise(cand.dedispersed)

            time_decimate_factor = max(1, width // 2)
            cand.decimate(key="dmt", axis=1, pad=True,
                          decimate_factor=time_decimate_factor, mode="median")
            crop_start_sample_dmt = cand.dmt.shape[1] // 2 - time_size // 2
            cand.dmt = crop(cand.dmt, crop_start_sample_dmt, time_size, axis=1)
            crop_start_dm = cand.dmt.shape[0] // 2 - dm_size // 2
            cand.dmt = crop(cand.dmt, crop_start_dm, dm_size, axis=0)
            cand.resize(key="dmt", size=dm_size, axis=1,
                        anti_aliasing=True, mode="constant")
            cand.dmt = normalise(cand.dmt)

            # FETCH classification
            X = np.reshape(cand.dedispersed, (1, 256, 256, 1))
            Y = np.reshape(cand.dmt, (1, 256, 256, 1))
            fetch_probs = {name: model.predict([X, Y], batch_size=1, verbose=0)[0, 1]
                           for name, model in fetch_models.items()}
            highest_prob = max(fetch_probs.values())
            if highest_prob <= 0.5:
                continue

            if not is_redetection_test:
                insert_detection(
                    beam_id=beam_id,
                    candidate_dm=dm,
                    snr=snr,
                    width_samples=width,
                    detection_type="candidate",
                    classification_probability=highest_prob
                )
            else:
                # optional: record as a test redetection preview
                insert_detection(
                    beam_id=beam_id,
                    candidate_dm=dm,
                    snr=snr,
                    width_samples=width,
                    detection_type="known_pulsar_test",
                    pulsar_name=redetect_meta["psr"] if redetect_meta else None,
                    classification_probability=highest_prob
                )

            # Galactic context
            l = skycoord.galactic.l.deg
            b = skycoord.galactic.b.deg
            dm_ne2001, _ = pygedm.dist_to_dm(l, b, 5e4, method='ne2001')
            dm_ymw16, _ = pygedm.dist_to_dm(l, b, 5e4, method='ymw16')
            galactic_info = (
                f"RA: {observation_info['RA (J2000)']}  DEC: {observation_info['DEC (J2000)']} | "
                f"l: {l:.2f} b: {b:.2f}\n"
                f"Max DM NE2001: {dm_ne2001:.1f} | Max DM YMW16: {dm_ymw16:.1f}"
            )

            dm_time_axis = np.arange(cand.dmt.shape[1]) * cand.tsamp + tcand
            dm_values = np.linspace(dm - 5, dm + 5, dm_size)
            time_series = np.sum(cand.dedispersed, axis=1)
            time_series = time_series * (snr / np.max(time_series))
            time_axis = np.arange(len(time_series)) * cand.tsamp + tcand

            # Figure
            fig = plt.figure(figsize=(14, 10))
            gs = GridSpec(5, 6, figure=fig,
                          width_ratios=[1.7] * 4 + [3, 0.2],
                          height_ratios=[0.35, 0.35, 1, 5, 1],
                          wspace=0.3, hspace=0)

            ax_obs = fig.add_subplot(gs[0, 0:5]); ax_obs.axis("off")
            ax_obs.text(0.5, 0.5,
                        f"{beam_id} DM={dm:.2f} Width={width} S/N={snr:.2f}",
                        ha="center", va="center", fontsize=10, family="monospace")
            if is_redetection_test and redetect_meta:
                ax_obs.text(0.5, 0.15,
                    f"[TEST MODE] Redetection of {redetect_meta['psr']}  "
                    + (f"ref DM={redetect_meta['dm_ref']:.2f}  " if redetect_meta.get("dm_ref") is not None else "")
                    + (f"sep={redetect_meta['sep_arcsec']:.0f}\"" if redetect_meta.get("sep_arcsec") is not None else ""),
                    ha="center", va="center", fontsize=9, family="monospace", color="crimson"
                )

            ax_fetch = fig.add_subplot(gs[1, 0:5]); ax_fetch.axis("off")
            ax_fetch.text(0.5, 0.5,
                          "FETCH: " + " | ".join([f"{k}:{v:.2f}" for k, v in fetch_probs.items()]),
                          ha="center", va="center", fontsize=9, family="monospace")

            ax_gal = fig.add_subplot(gs[2, 3:5]); ax_gal.axis("off")
            ax_gal.text(0.5, 0.5, galactic_info, ha="center", va="center",
                        fontsize=9, family="monospace")

            ax_ts = fig.add_subplot(gs[2, 0:3])
            ax_ts.plot(time_axis, time_series, color="black")
            ax_ts.set_ylabel("S/N"); ax_ts.set_xticks([])

            ax_ft = fig.add_subplot(gs[3, 0:3])
            ax_ft.imshow(cand.dedispersed.T, aspect="auto", cmap="viridis",
                         extent=[tcand, tcand+time_size*cand.tsamp,
                                 frequency_axis.min(), frequency_axis.max()])
            ax_ft.set_xlabel("Time(s)"); ax_ft.set_ylabel("Freq(MHz)")

            ax_dmt = fig.add_subplot(gs[3, 3:5])
            ax_dmt.imshow(cand.dmt, aspect="auto", cmap="viridis",
                          extent=[dm_time_axis.min(), dm_time_axis.max(),
                                  dm_values.min(), dm_values.max()])
            ax_dmt.set_xlabel("Time(s)"); ax_dmt.set_ylabel("DM")

            ax_psr = fig.add_subplot(gs[4, 0:5]); ax_psr.axis("off")
            if not local_lotaas.empty:
                show = local_lotaas.head(6).copy()
                show["DMshow"] = show["dm_ref"].round(3)
                psr_list = "\n".join(
                    [f"{r['psr']}  DM:{r['DMshow']}  sep:{r['sep_arcsec']:.1f}\""
                     for _, r in show.iterrows()]
                )
            else:
                psr_list = "No LOTAAS sources near pointing"
            ax_psr.text(0.5, 0.5, psr_list, ha="center", va="center",
                        fontsize=9, family="monospace")

            fig.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.07)
            out_path = os.path.join(output_dir, f"DM{dm}_Width{width}_SNR{snr}.png")
            plt.savefig(out_path, dpi=300); plt.close()

            # Slack notify for new candidate
            try:
                if (not is_redetection_test) or (is_redetection_test and TEST_FORCE_SLACK):
                    client.files_upload_v2(
                        channels=CHANNEL_ID,
                        initial_comment=(
                            ("*New candidate detected!*\n" if not is_redetection_test else "*[TEST] Redetection preview*\n")
                            + f"{beam_id}\n"
                            + f"DM = {dm:.2f} pc/cm³\n"
                            + f"S/N = {snr:.2f}\n"
                            + f"Width = {width} samples\n"
                            + f"FETCH max probability = {highest_prob:.2f}"
                            + (f"\nMatched {redetect_meta['psr']} (ref {redetect_meta['dm_ref']:.2f}, "
                            f"sep {redetect_meta['sep_arcsec']:.0f}\")" if is_redetection_test and redetect_meta else "")
                        ),
                        file=out_path,
                        title=os.path.basename(out_path)
                    )
                    print(f"Slack image sent for {'candidate' if not is_redetection_test else 'TEST redetection'} DM={dm:.2f}, SNR={snr:.2f}")
            except SlackApiError as e:
                print(f"Slack error when sending image: {e}")

        # Insert and announce best redetections per pulsar
        for psr_name, info in redetections_best.items():
            insert_detection(
                beam_id=beam_id,
                candidate_dm=info["dm"],
                snr=info["snr"],
                width_samples=info["width"],
                detection_type="known_pulsar",
                pulsar_name=psr_name
            )
            slack_messages.append(
                f"*Redetected:* {psr_name}  DM={info['dm']:.2f}"
                + (f" (ref {info['dm_ref']:.2f})" if info.get("dm_ref") else "")
                + f"  S/N={info['snr']:.2f}  W={info['width']}  sep={info.get('sep_arcsec',0):.0f}\""
            )
            num_redetections += 1

        update_beam_run(
            row_id=beam_run_id,
            outcome="classified",
            num_candidates=len(candidates_df),
            num_redetections=num_redetections,
            highest_snr=highest_snr
        )

        if slack_messages:
            send_slack_message("\n".join(slack_messages))

        print("Finished processing.")

    except Exception as e:
        update_beam_run(
            row_id=beam_run_id,
            outcome="error",
            error_message=str(e)
        )
        raise