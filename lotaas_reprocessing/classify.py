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

# Cone fetched from the ATNF catalogue. Generous: the query is cheap and a
# complete local catalogue costs nothing to filter.
CATALOGUE_RADIUS_DEG = 5.0
# Separation within which a known pulsar can plausibly account for a detection
# in this beam. The veto used the full 5-degree query cone and matched on DM
# alone, so a new source sharing a DM with any pulsar in a 78 square-degree
# field was recorded as a redetection of it. Override with
# LOTAAS_ATNF_VETO_RADIUS_DEG once the tied-array beam response is measured.
VETO_RADIUS_DEG = float(os.environ.get("LOTAAS_ATNF_VETO_RADIUS_DEG", "1.0"))


def dm_time_plane(cand, decimate, time_size=256, dmsteps=256, range_dm=5.0):
    """The decimated, time-cropped DM-time plane FETCH receives.

    `your` dedisperses the whole chunk (twice the dispersion sweep, 221,579
    samples at DM 8,000) at 256 trial DMs and the result is then decimated and
    cropped to 256 samples around the pulse. That took ~40 s per high-DM
    candidate and left a few RFI-rich beams classifying for over an hour while
    the GPUs idled. Only the columns that survive the crop are computed here,
    accumulating channels in the same order and precision, so the result is
    bit-identical. Where the crop would reach the median padding added for
    decimation, or a delay reaches the chunk length, the original computation
    is used.
    """
    nt, nf = cand.data.shape
    padded = nt + (-nt) % decimate
    decimated = padded // decimate
    start = decimated // 2 - time_size // 2
    dm_list = cand.dm + np.linspace(-float(range_dm), float(range_dm), dmsteps)
    freqs = np.asarray(cand.chan_freqs)
    delays = np.round(4148808.0 * dm_list[:, None] * (1 / (freqs[0]) ** 2 - 1 / (freqs[None, :]) ** 2)
                      / 1000 / cand.native_tsamp).astype("int64")
    # The window only pays when it is a small part of the chunk (wide pulses
    # force chunks barely longer than the crop).
    if not (decimated > start + time_size and start >= 0 and (start + time_size) * decimate <= nt
            and 2 * time_size * decimate <= nt and np.abs(delays).max() < nt):
        cand.dmtime(dmsteps=dmsteps, range_dm=range_dm)
        cand.decimate(key="dmt", axis=1, pad=True, decimate_factor=decimate, mode="median")
        return crop(cand.dmt, cand.dmt.shape[1] // 2 - time_size // 2, time_size, axis=1)
    first, width = start * decimate, time_size * decimate
    rows = np.ascontiguousarray(cand.data.T)
    plane = np.zeros((dmsteps, width), dtype=np.float32)
    # your rolls each channel right by its delay: out[t] = data[(t - d) mod nt].
    # Channels are added in order, as there, so every sum is bit-identical.
    for channel in range(nf):
        row = rows[channel]
        for step in range(dmsteps):
            begin = (first - delays[step, channel]) % nt
            end = begin + width
            if end <= nt:
                plane[step] += row[begin:end]
            else:
                split = nt - begin
                plane[step, :split] += row[begin:]
                plane[step, split:] += row[:end - nt]
    return plane.reshape(dmsteps, time_size, decimate).mean(2)


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
            radius=CATALOGUE_RADIUS_DEG
        )
        known_psrs_df = query.table.to_pandas()
        if not known_psrs_df.empty:
            catalogue = SkyCoord(known_psrs_df["RAJ"].values, known_psrs_df["DECJ"].values,
                                 unit=(u.hourangle, u.deg))
            known_psrs_df = known_psrs_df.assign(
                separation_deg=skycoord.separation(catalogue).deg)
            # Vetoing on DM alone across the whole query cone labelled genuinely
            # new sources as redetections: a pulsar degrees away cannot produce
            # a detection in a tied-array beam, and at DM < 150 the grid is
            # dense enough that some catalogue entry usually falls within the
            # DM tolerance.
            # The veto uses the nearby set; the plot footer keeps the whole
            # cone. A pulsar 1.2 degrees away cannot account for a detection
            # in this beam, but a human judging the candidate still wants to
            # know it is there.
            catalogue_psrs_df = known_psrs_df
            known_psrs_df = known_psrs_df[
                known_psrs_df["separation_deg"] <= VETO_RADIUS_DEG].reset_index(drop=True)
        else:
            catalogue_psrs_df = known_psrs_df
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
                # Nearest matching pulsar, not whichever the catalogue listed
                # first, so the attribution names the plausible source.
                close = known_psrs_df[(known_psrs_df["DM"] - dm).abs() <= dm_tolerance]
                if not close.empty:
                    matched_psr = close.loc[close["separation_deg"].idxmin()]

            if matched_psr is not None:
                psr_name = matched_psr["PSRJ"]
                if psr_name not in redetections_best or snr > redetections_best[psr_name]["snr"]:
                    redetections_best[psr_name] = {
                        "dm": dm,
                        "snr": snr,
                        "width": width,
                        "time": tcand,
                        "sample": sample_number,
                        "separation_deg": float(matched_psr["separation_deg"]),
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
            time_decimate_factor = max(1, width // 2)  # Ensure it's at least 1
            cand.dmt = dm_time_plane(cand, time_decimate_factor, time_size, dm_size)
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

            # The DM-time plane is already decimated and cropped in time.
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
                # Record the rejection. A bare `continue` left no plot, no row
                # and no log line, so a candidate the classifier discarded was
                # indistinguishable in the outputs from one never found. An
                # injected pulse recovered at S/N 14.7 scored 0.374 here while
                # three fainter ones from the same beam scored above 0.97, and
                # nothing recorded that it had been considered at all.
                logger.info(
                    "FETCH rejected DM=%.2f t=%.3f S/N=%.2f width=%d (max p=%.3f)",
                    dm, tcand, snr, width, highest_prob)
                insert_detection(
                    beam_id=beam_id,
                    beam_run_id=beam_run_id,
                    time_seconds=tcand,
                    sample_number=sample_number,
                    candidate_dm=dm,
                    snr=snr,
                    width_samples=width,
                    detection_type="rejected",
                    classification_probability=highest_prob,
                )
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
            # Per-channel robust normalisation for display. A handful of
            # channels carry a persistent offset or several times the typical
            # noise, and on a shared colour scale they stripe the waterfall
            # and take the dynamic range with them. That matters most for a
            # marginal candidate, which is exactly what the plot is for.
            waterfall = np.asarray(cand.dedispersed, dtype=float).T   # (freq, time)
            channel_median = np.median(waterfall, axis=1, keepdims=True)
            channel_scale = np.median(np.abs(waterfall - channel_median),
                                      axis=1, keepdims=True) * 1.4826
            # A flat channel normalises to zero rather than to a division error.
            channel_scale[~np.isfinite(channel_scale) | (channel_scale <= 0)] = np.inf
            waterfall = np.nan_to_num((waterfall - channel_median) / channel_scale)
            # Robust limits, so one surviving spike cannot flatten the rest.
            vmin, vmax = np.percentile(waterfall, [1, 99])

            # Profile measured against its own off-pulse noise, rather than
            # rescaled so its maximum equals the reported S/N. A reader needs
            # to see how far the pulse stands out here, not be told again; and
            # the old scaling silently keyed on whatever the largest sample
            # was, noise spike included.
            time_series = waterfall.sum(axis=0)
            baseline = np.median(time_series)
            spread = np.median(np.abs(time_series - baseline)) * 1.4826
            time_series = (time_series - baseline) / (spread if spread > 0 else 1.)
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
            ax_ts.axhline(0,color="0.7",lw=.5)
            ax_ts.set_ylabel("S/N in window")
            ax_ts.set_xticks([])

            ax_ft = fig.add_subplot(gs[3,0:3])
            # Orientation is unchanged from the original: the transposed array
            # with the default origin puts each channel at its own frequency.
            ax_ft.imshow(waterfall,aspect="auto",cmap="viridis",
                vmin=vmin,vmax=vmax,
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
            if not catalogue_psrs_df.empty:
                psr_list = "\n".join(
                    [f"{r['PSRJ']} RA:{r['RAJ']} DEC:{r['DECJ']} DM:{r['DM']}"
                     f" sep:{r['separation_deg']:.2f}deg"
                     + ("" if r['separation_deg'] <= VETO_RADIUS_DEG else " (outside veto radius)")
                     for _,r in catalogue_psrs_df.iterrows()]
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
                f"*Redetected:* {psr_name}  DM={info['dm']:.2f}  highest S/N={info['snr']:.2f}"
                f"  Width={info['width']}  separation={info['separation_deg']:.3f} deg"
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
