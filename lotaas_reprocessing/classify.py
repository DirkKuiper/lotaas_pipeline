import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from your.candidate import Candidate
from your.utils.math import normalise
from your.candidate import crop
from fetch.utils import get_model
from lotaas_reprocessing.filterbank import FilterbankFile
from matplotlib.gridspec import GridSpec
import pygedm
from astropy.coordinates import SkyCoord
from astropy import units as u
from psrqpy import QueryATNF

# Import DB utils
from db.db_utils import insert_beam_run, update_beam_run, insert_detection

logger = logging.getLogger(__name__)


def send_slack_message(text):
    # Notifications are deliberately local; cluster runs never send messages.
    logger.info(text)


def classify_candidates(filterbank_file, candidate_file, output_dir, observation_info=None):
    os.makedirs(output_dir, exist_ok=True)
    observation_info = observation_info or {}
    beam_id = os.path.basename(filterbank_file)
    observation_date = observation_info.get("Observation Date", "Unknown")
    output_dir = os.path.abspath(output_dir)
    log_file = os.path.join(output_dir, "processing.log")

    beam_run_id = insert_beam_run(
        beam_id=beam_id,
        observation_date=observation_date,
        output_dir=output_dir,
        log_file=log_file,
        code_version=os.environ.get("LOTAAS_RUN_FINGERPRINT", "unversioned")
    )

    try:
        candidates_df = pd.read_csv(candidate_file, sep=r"\s+")
        candidates_df.columns = candidates_df.columns.str.strip().str.lower()
        candidates_df = candidates_df[(candidates_df["dm"] >= 10) & (candidates_df["s/n"] > 7)]
        if candidates_df.empty:
            update_beam_run(beam_run_id, outcome="no_candidates", num_candidates=0,
                           num_redetections=0, highest_snr=0)
            return
        skycoord = SkyCoord(
            observation_info["RA (J2000)"],
            observation_info["DEC (J2000)"],
            unit=(u.hourangle, u.deg)
        )
        ra_str = skycoord.ra.to_string(unit=u.hour, sep=':', pad=True, precision=2)
        dec_str = skycoord.dec.to_string(unit=u.deg, sep=':', alwayssign=True, pad=True, precision=2)

        query = QueryATNF(
            params=['PSRJ', 'RAJ', 'DECJ', 'DM'],
            coord1=ra_str,
            coord2=dec_str,
            radius=5.0
        )
        known_psrs_df = query.table.to_pandas()
        dm_tolerance = 0.5
        highest_snr = 0

        candidates_df = (
            candidates_df
            .sort_values("s/n", ascending=False)
            .groupby("dm", group_keys=False)
            .head(5)
        )

        model_names = ["a", "b", "c", "d", "e", "f"]
        fetch_models = None

        # Dict to store highest S/N redetections per pulsar
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

            matched_psr = None
            if not known_psrs_df.empty:
                for _, psr_row in known_psrs_df.iterrows():
                    if abs(dm - psr_row["DM"]) <= dm_tolerance:
                        matched_psr = psr_row
                        break

            if matched_psr is not None:
                psr_name = matched_psr["PSRJ"]
                if psr_name not in redetections_best or snr > redetections_best[psr_name]["snr"]:
                    redetections_best[psr_name] = {
                        "dm": dm,
                        "snr": snr,
                        "width": width,
                        "time": tcand,
                        "sample": sample_number,
                    }
                continue  # Skip further processing for redetections

            if fetch_models is None:
                fetch_models = {name: get_model(name) for name in model_names}

            # Proceed with classification of non-pulsar candidates
            time_size, freq_size, dm_size = 256, 256, 256

            cand = Candidate(
                fp=filterbank_file,
                dm=dm,
                tcand=tcand,
                width=width,
                label=-1,
                snr=snr,
                min_samp=256,
                device=-1,
            )
            cand.get_chunk()
            cand.dmtime(dmsteps=256)
            cand.dedisperse()

            fil = FilterbankFile(filterbank_file, "read")
            f_start, delta_f, nchan = fil.fch1, fil.foff, fil.nchans
            fil.close()
            frequency_axis = np.flip(f_start + np.arange(nchan) * delta_f)

            # Decimate, crop, and normalize FT
            cand.decimate(key="ft", axis=0, pad=True, decimate_factor=max(1, width // 2), mode="median")
            cand.dedispersed = crop(cand.dedispersed, cand.dedispersed.shape[0] // 2 - time_size // 2, time_size, 0)
            cand.decimate(key="ft", axis=1, pad=True, decimate_factor=max(1, cand.dedispersed.shape[1] // freq_size), mode="median")
            cand.resize(key="ft", size=freq_size, axis=1, anti_aliasing=True, mode="constant")
            cand.dedispersed = normalise(cand.dedispersed)

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
            if not np.isfinite(X).all() or not np.isfinite(Y).all():
                raise ValueError(f"Nonfinite FETCH input at DM={dm}, time={tcand}")

            fetch_probs = {name: model.predict([X,Y], batch_size=1, verbose=0)[0,1]
                           for name, model in fetch_models.items()}
            if not all(np.isfinite(p) and 0 <= p <= 1 for p in fetch_probs.values()):
                raise ValueError(f"Invalid FETCH probabilities at DM={dm}, time={tcand}")
            highest_prob = max(fetch_probs.values())

            if highest_prob <= 0.5:
                continue

            insert_detection(
                beam_id=beam_id,
                beam_run_id=beam_run_id,
                time_seconds=tcand,
                sample_number=sample_number,
                candidate_dm=dm,
                snr=snr,
                width_samples=width,
                detection_type="candidate",
                classification_probability=highest_prob
            )

            # Galactic info
            l = skycoord.galactic.l.deg
            b = skycoord.galactic.b.deg
            dm_ne2001, _ = pygedm.dist_to_dm(l, b, 5e4, method='ne2001')
            dm_ymw16, _ = pygedm.dist_to_dm(l, b, 5e4, method='ymw16')
            galactic_info = (
                f"RA: {observation_info['RA (J2000)']}  DEC: {observation_info['DEC (J2000)']} | "
                f"l: {l:.2f} b: {b:.2f}\n"
                f"Max DM NE2001: {dm_ne2001:.1f} | Max DM YMW16: {dm_ymw16:.1f}"
            )

            # Both arrays have been time-decimated and cropped around the event.
            # Display relative time, including the decimation factor, rather than
            # labelling the beginning of the cropped window as the event time.
            plot_tsamp = cand.tsamp * time_decimate_factor
            dm_time_axis = (np.arange(cand.dmt.shape[1]) - (cand.dmt.shape[1]-1)/2) * plot_tsamp
            dm_values = np.linspace(dm - 5, dm + 5, dm_size)
            time_series = np.sum(cand.dedispersed, axis=1)
            time_series = time_series * (snr / np.max(time_series))
            time_axis = (np.arange(len(time_series)) - (len(time_series)-1)/2) * plot_tsamp

            # Figure
            fig = plt.figure(figsize=(14,10))
            gs = GridSpec(5,6,figure=fig,
                          width_ratios=[1.7]*4+[3,0.2],
                          height_ratios=[0.35,0.35,1,5,1],
                          wspace=0.3,hspace=0)

            ax_obs = fig.add_subplot(gs[0,0:5])
            ax_obs.axis("off")
            ax_obs.text(0.5,0.5,
                f"{beam_id} DM={dm:.2f} Width={width} S/N={snr:.2f} Time={tcand:.3f}s",
                ha="center",va="center",fontsize=10,family="monospace")

            ax_fetch = fig.add_subplot(gs[1,0:5])
            ax_fetch.axis("off")
            ax_fetch.text(0.5,0.5,
                "FETCH: "+" | ".join([f"{k}:{v:.2f}" for k,v in fetch_probs.items()]),
                ha="center",va="center",fontsize=9,family="monospace")

            ax_gal = fig.add_subplot(gs[2,3:5])
            ax_gal.axis("off")
            ax_gal.text(0.5,0.5,galactic_info,ha="center",va="center",fontsize=9,family="monospace")

            ax_ts = fig.add_subplot(gs[2,0:3])
            ax_ts.plot(time_axis,time_series,color="black")
            ax_ts.set_ylabel("S/N")
            ax_ts.set_xticks([])

            ax_ft = fig.add_subplot(gs[3,0:3])
            ax_ft.imshow(cand.dedispersed.T,aspect="auto",cmap="viridis",
                extent=[time_axis.min(),time_axis.max(),frequency_axis.min(),frequency_axis.max()])
            ax_ft.set_xlabel("Time relative to candidate (s)")
            ax_ft.set_ylabel("Freq(MHz)")

            ax_dmt = fig.add_subplot(gs[3,3:5])
            ax_dmt.imshow(cand.dmt,aspect="auto",cmap="viridis",origin="lower",
                extent=[dm_time_axis.min(),dm_time_axis.max(),dm_values.min(),dm_values.max()])
            ax_dmt.set_xlabel("Time relative to candidate (s)")
            ax_dmt.set_ylabel("DM")

            ax_psr = fig.add_subplot(gs[4,0:5])
            ax_psr.axis("off")
            if not known_psrs_df.empty:
                psr_list = "\n".join(
                    [f"{r['PSRJ']} RA:{r['RAJ']} DEC:{r['DECJ']} DM:{r['DM']}" for _,r in known_psrs_df.iterrows()]
                )
            else:
                psr_list = "No known pulsars"
            ax_psr.text(0.5,0.5,psr_list,ha="center",va="center",fontsize=9,family="monospace")

            fig.subplots_adjust(left=0.05,right=0.98,top=0.96,bottom=0.07)
            out_path = os.path.join(output_dir,f"DM{dm}_Width{width}_SNR{snr}.png")
            plt.savefig(out_path,dpi=300)
            plt.close()

        # Insert and announce only the best redetections per pulsar
        for psr_name, info in redetections_best.items():
            insert_detection(
                beam_id=beam_id,
                beam_run_id=beam_run_id,
                time_seconds=info["time"],
                sample_number=info["sample"],
                candidate_dm=info["dm"],
                snr=info["snr"],
                width_samples=info["width"],
                detection_type="known_pulsar",
                pulsar_name=psr_name
            )
            slack_messages.append(
                f"*Redetected:* {psr_name}  DM={info['dm']:.2f}  highest S/N={info['snr']:.2f}  Width={info['width']}"
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
