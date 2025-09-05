# lotaas_reprocessing/classify_post_detection.py
import os
import re
import logging
import ssl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from numpy.polynomial import Polynomial
from matplotlib.gridspec import GridSpec
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from astropy.coordinates import SkyCoord
from astropy import units as u
import pygedm
from psrqpy import QueryATNF

from your.candidate import Candidate
from your.utils.math import normalise
from your.candidate import crop
from fetch.utils import get_model

from lotaas_reprocessing.filterbank import FilterbankFile
from lotaas_reprocessing.numpy_utils import compute_rfi_mask  # CPU fallback OK
from lotaas_reprocessing.cupy_utils import fourier_domain_dedispersion  # uses GPU if available
from lotaas_reprocessing.sap_beam_map import plot_sap_beam_layout, matched_filter_sn_at_idx

logger = logging.getLogger(__name__)

# Slack setup
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "xoxb-513966140291-8603128801253-OIZMLciSFmNefi4An84YDNKE")
CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID", "C08HHTN8CTG")
_client = WebClient(token=SLACK_BOT_TOKEN, ssl=ssl._create_unverified_context()) if SLACK_BOT_TOKEN else None


def _send_slack_notification(image_path, title, text):
    if not _client:
        return
    try:
        _client.files_upload_v2(
            channels=CHANNEL_ID,
            file=image_path,
            title=title,
            initial_comment=text
        )
        print(f"[slack] Sent: {title}")
    except SlackApiError as e:
        print(f"[slack] Error: {e}")


def _packed_float_to_sexagesimal_string(val, is_ra=True):
    val = float(val)
    hh_or_dd = int(val // 10000)
    mm = int((val % 10000) // 100)
    ss = val % 100
    if is_ra:
        return f"{hh_or_dd:02d}:{mm:02d}:{ss:05.2f}"
    else:
        return f"{hh_or_dd:02d}:{mm:02d}:{ss:04.1f}"


def _load_settings(path="settings.yaml"):
    defaults = dict(
        rfi_block_size=256,
        rfi_averaging_factor=128,
        detection_threshold=5,
    )
    if not os.path.exists(path):
        return defaults
    with open(path, "r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader) or {}
    return {**defaults, **cfg}


def classify_candidates(
    filterbank_file: str,
    candidate_file: str,
    output_dir: str,
    observation_info=None,
    candidate_dm: float = None,
    tcand: float = None,
    snr: float = None,
    width: int = None,
    send_to_slack: bool = True,
    sample_idx: int = None,
):
    """
    Post-detection classification and diagnostics.

    Parameters
    ----------
    filterbank_file : str
        Path to the .fil file of the *triggering beam*.
    candidate_file : str
        Either a whitespace .cands file OR a path to something else (ignored if single-candidate args given).
    output_dir : str
        Directory to write outputs into (will be created).
    observation_info : dict or None
        Optional dict with RA/DEC strings used in the header panel.
    candidate_dm, tcand, snr, width : optional
        If provided, process *this one candidate* directly. If omitted, we will try to read `candidate_file` as a .cands file.
    send_to_slack : bool
        If True and SLACK_BOT_TOKEN is configured, upload the final diagnostic plot.
    sample_idx : int or None
        If provided, re-measure S/N at this exact sample index to match the detector.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Settings to mirror pipeline preprocessing
    settings = _load_settings("settings.yaml")
    block_size = int(settings["rfi_block_size"])
    BAD_CHAN = 138

    # Load candidates
    rows = []
    if all(v is not None for v in (candidate_dm, tcand, snr, width)):
        rows.append(
            dict(dm=float(candidate_dm), time=float(tcand), width=int(width), **{"s/n": float(snr), "sample": -1})
        )
        print("[classify] Using single-candidate arguments passed in.")
    else:
        if not os.path.exists(candidate_file):
            print(f"[classify] Candidate file not found: {candidate_file}")
            return
        try:
            df = pd.read_csv(candidate_file, delim_whitespace=True, comment="#")
            df.columns = df.columns.str.strip().str.lower()
            if "filter_width" in df.columns and "width" not in df.columns:
                df["width"] = df["filter_width"]
            if "s/n" not in df.columns and "snr" in df.columns:
                df["s/n"] = df["snr"]
            req = {"dm", "time", "width", "s/n"}
            if not req.issubset(df.columns):
                raise ValueError(f"Missing columns in {candidate_file}. Need at least {req}")
            for _, r in df.iterrows():
                rows.append(dict(
                    dm=float(r["dm"]),
                    time=float(r["time"]),
                    width=int(r["width"]),
                    **{"s/n": float(r["s/n"]), "sample": int(r.get("sample", -1))}
                ))
            print(f"[classify] Loaded {len(rows)} candidates from {candidate_file}")
        except Exception as e:
            print(f"[classify] Failed to parse {candidate_file}: {e}")
            return

    # Frequency axis from filterbank header
    fil = FilterbankFile(filterbank_file, "read")
    f_start, delta_f, nchan = fil.fch1, fil.foff, fil.nchans
    frequency_axis = np.flip(f_start + np.arange(nchan) * delta_f)

    # SAP directory (…/SAP###)
    sap_dir = os.path.dirname(os.path.dirname(filterbank_file))
    combined_cands_file = os.path.join(sap_dir, "combined_all_candidates.cands")
    if os.path.exists(combined_cands_file):
        try:
            combined_df = pd.read_csv(
                combined_cands_file,
                delim_whitespace=True,
                comment="#",
                header=None,
                names=["beam", "dm", "snr", "time", "sample", "width"],
            )
        except Exception:
            combined_df = pd.DataFrame(columns=["beam", "dm", "snr", "time", "sample", "width"])
    else:
        combined_df = pd.DataFrame(columns=["beam", "dm", "snr", "time", "sample", "width"])

    # Load FETCH models once
    model_names = ["a", "b", "c", "d", "e", "f"]
    fetch_models = {name: get_model(name) for name in model_names}

    # Process each candidate
    for row in rows:
        dm = row["dm"]
        t0 = row["time"]
        width_samp = int(row["width"])
        snr0 = row["s/n"]
        sample_number = int(row.get("sample", -1))

        if dm < 3:
            print(f"[classify] Skip DM={dm:.2f} (<3)")
            continue
        if snr0 <= 6.5:
            print(f"[classify] Skip DM={dm:.2f} (S/N {snr0:.2f} <= 6.5)")
            continue

        print(f"[classify] Candidate: DM={dm:.2f}, t={t0:.3f}s, width={width_samp}, S/N={snr0:.2f}")

        # Build Candidate for FETCH features
        cand = Candidate(
            fp=filterbank_file,
            dm=dm,
            tcand=t0,
            width=width_samp,
            label=-1,
            snr=snr0,
            min_samp=256,
            device=0,
        )
        cand.get_chunk()
        cand.dmtime(dmsteps=256)
        cand.dedisperse()

        # Prepare FT (dedispersed waterfall)
        time_size, freq_size, dm_size = 256, 256, 256
        dec_t = max(1, width_samp // 2)

        cand.decimate(key="ft", axis=0, pad=True, decimate_factor=dec_t, mode="median")
        cand.dedispersed = crop(cand.dedispersed, cand.dedispersed.shape[0] // 2 - time_size // 2, time_size, 0)
        cand.decimate(key="ft", axis=1, pad=True, decimate_factor=cand.dedispersed.shape[1] // freq_size, mode="median")
        cand.resize(key="ft", size=freq_size, axis=1, anti_aliasing=True, mode="constant")
        cand.dedispersed = normalise(cand.dedispersed)

        # Prepare DMT image
        cand.decimate(key="dmt", axis=1, pad=True, decimate_factor=dec_t, mode="median")
        crop_start_sample_dmt = cand.dmt.shape[1] // 2 - time_size // 2
        cand.dmt = crop(cand.dmt, crop_start_sample_dmt, time_size, axis=1)
        crop_start_dm = cand.dmt.shape[0] // 2 - dm_size // 2
        cand.dmt = crop(cand.dmt, crop_start_dm, dm_size, axis=0)
        cand.resize(key="dmt", size=dm_size, axis=1, anti_aliasing=True, mode="constant")
        cand.dmt = normalise(cand.dmt)

        # FETCH inference
        X = np.reshape(cand.dedispersed, (1, 256, 256, 1))
        Y = np.reshape(cand.dmt, (1, 256, 256, 1))
        fetch_probabilities = {}
        for mname, mdl in fetch_models.items():
            fetch_probabilities[mname] = float(mdl.predict([X, Y], batch_size=1, verbose=0)[0, 1])
        highest_prob = max(fetch_probabilities.values())
        is_real = highest_prob > 0.5

        print("[classify] FETCH probabilities:", " | ".join(f"{k}:{v:.2f}" for k, v in fetch_probabilities.items()))
        print(f"[classify] Astrophysical? {'YES' if is_real else 'NO'}")

        if not is_real:
            print("[classify] Skipping save/plots (no model > 0.5).")
            continue

        # Cross-beam dedispersion + matched-filter S/N at candidate time
        beam_snrs = {}
        beam_positions = {}
        beam_limit = 75
        count = 0
        for beam_dir in sorted(os.listdir(sap_dir)):
            if not beam_dir.startswith("B"):
                continue
            if count >= beam_limit:
                break
            beam_path = os.path.join(sap_dir, beam_dir)
            flist = [f for f in os.listdir(beam_path) if f.endswith(".fil")]
            if not flist:
                continue
            beam_fil = os.path.join(beam_path, flist[0])
            try:
                bf = FilterbankFile(beam_fil, "read")
                ts = bf.tsamp
                nu = np.flipud(bf.frequencies)
                raw = np.flipud(bf.get_spectra(0, bf.nspec).T)

                # RFI mask + replace with noise
                m = compute_rfi_mask(raw, block_size)
                raw = raw / np.median(raw) - 1
                if 0 <= BAD_CHAN < raw.shape[0]:
                    m[BAD_CHAN, :] = True

                masked = raw.copy()
                masked[m] = np.nan
                mu = np.nanmean(masked)
                sig = np.nanstd(masked)
                noise = np.random.normal(mu, sig, masked.shape)
                masked[m] = noise[m]

                # per-channel poly2 detrend
                detr = np.empty_like(masked)
                xx = np.arange(masked.shape[1])
                for i in range(masked.shape[0]):
                    p2 = Polynomial.fit(xx, masked[i, :], deg=2)
                    detr[i, :] = masked[i, :] - p2(xx)

                I_f = fourier_domain_dedispersion(detr.astype(np.float32), ts, nu, [dm])
                dedisp = np.fft.irfft(I_f, n=detr.shape[1], axis=1).real[0]

                # exact index if provided, else recompute from time
                if sample_idx is not None:
                    idx = int(sample_idx)
                else:
                    t_axis = np.arange(len(dedisp)) * ts
                    idx = int(np.argmin(np.abs(t_axis - t0)))

                sn, _ = matched_filter_sn_at_idx(dedisp, ts, width_samp, idx)
                beam_snrs[beam_dir] = float(sn)

                # Optional positions from header
                try:
                    ra_val = bf.header.get("src_raj", None)
                    dec_val = bf.header.get("src_dej", None)
                    if ra_val is not None and dec_val is not None:
                        ra_str = _packed_float_to_sexagesimal_string(ra_val, is_ra=True)
                        dec_str = _packed_float_to_sexagesimal_string(dec_val, is_ra=False)
                        sky = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))
                        beam_positions[int(beam_dir[1:])] = (sky.ra.deg, sky.dec.deg)
                except Exception as e:
                    print(f"[classify] Beam {beam_dir} RA/DEC parse failed: {e}")

                count += 1
            except Exception as e:
                print(f"[classify] Beam {beam_dir} failed: {e}")

        # Output directory for this candidate
        cand_dir = os.path.join(
            output_dir,
            f"DM{dm}_Width{width_samp}_SNR{snr0}_Time{t0:.3f}_Sample{sample_number}"
        )
        os.makedirs(cand_dir, exist_ok=True)

        # Galactic / PSRCAT info (if observation_info provided with RA/DEC strings)
        if observation_info and "RA (J2000)" in observation_info and "DEC (J2000)" in observation_info:
            skycoord = SkyCoord(observation_info["RA (J2000)"], observation_info["DEC (J2000)"], unit=(u.hourangle, u.deg))
            l = skycoord.galactic.l.deg
            b = skycoord.galactic.b.deg
            dm_ne2001, _ = pygedm.dist_to_dm(l, b, 5e4, method="ne2001")
            dm_ymw16, _ = pygedm.dist_to_dm(l, b, 5e4, method="ymw16")
            galactic_info = (
                f"RA (J2000): {observation_info['RA (J2000)']}   |   DEC (J2000): {observation_info['DEC (J2000)']}\n"
                f"Galactic l: {l:.2f}°, b: {b:.2f}°\n"
                f"Max DM (NE2001): {dm_ne2001:.1f}   |   Max DM (YMW16): {dm_ymw16:.1f}"
            )
        else:
            skycoord = None
            galactic_info = "No RA/DEC provided."

        # Known pulsars (within 5° if we have coords)
        if skycoord is not None:
            ra_str = skycoord.ra.to_string(unit=u.hour, sep=":", pad=True, precision=2)
            dec_str = skycoord.dec.to_string(unit=u.deg, sep=":", alwayssign=True, pad=True, precision=2)
            try:
                query = QueryATNF(params=["PSRJ", "RAJ", "DECJ", "DM"], coord1=ra_str, coord2=dec_str, radius=5.0)
                known_psrs_df = query.table.to_pandas()
            except Exception:
                known_psrs_df = pd.DataFrame()
        else:
            known_psrs_df = pd.DataFrame()

        if not known_psrs_df.empty:
            shown = known_psrs_df.head(5)
            pulsar_summary = "\n".join(
                [f"{row['PSRJ']}  RAJ: {row['RAJ']}  DECJ: {row['DECJ']}  DM: {row['DM']}"
                 for _, row in shown.iterrows()]
            )
            extra = f"\n(+{len(known_psrs_df)-5} more...)" if len(known_psrs_df) > 5 else ""
            psr_info_text = f"Known Pulsars within 5°:\n{pulsar_summary}{extra}"
        else:
            psr_info_text = "No known pulsars within 5°"

        # ======= Plot =======
        dm_time_axis = np.arange(cand.dmt.shape[1]) * cand.tsamp + t0
        dm_values = np.linspace(dm - 5, dm + 5, 256)
        time_series = np.sum(cand.dedispersed, axis=1)
        time_axis = np.arange(len(time_series)) * cand.tsamp + t0

        filename = os.path.basename(filterbank_file)
        filterbank_id = os.path.splitext(filename)[0]
        combined_path = os.path.join(cand_dir, f"diagnostic_DM{dm}_Width{width_samp}.png")

        fig = plt.figure(figsize=(14, 15))
        gs = GridSpec(5, 6, figure=fig,
                      width_ratios=[1.7, 1.7, 1.7, 1.7, 3, 0.2],
                      height_ratios=[0.35, 0.35, 1, 5.0, 7],
                      wspace=0.3, hspace=0.0)

        # Row 0: Observation info
        ax_obs = fig.add_subplot(gs[0, 0:5]); ax_obs.axis("off")
        obs_info = (f"File: {filterbank_id}   |   DM: {dm:.2f} pc cm⁻³   |   "
                    f"Width: {width_samp} samples   |   S/N: {snr0:.2f}   |   "
                    f"Time: {t0:.3f} s   |   Sample #: {sample_number}")
        ax_obs.text(0.5, 0.5, obs_info, ha="center", va="center", fontsize=10, family="monospace")

        # Row 1: FETCH probs
        ax_fetch = fig.add_subplot(gs[1, 0:5]); ax_fetch.axis("off")
        fetch_text = "FETCH Probabilities:   " + "   |   ".join([f"{k.upper()}: {v:.2f}" for k, v in fetch_probabilities.items()])
        ax_fetch.text(0.5, 0.5, fetch_text, ha="center", va="center", fontsize=9, family="monospace")

        # Row 2 (left): time-series S/N-scaled
        ax_ts = fig.add_subplot(gs[2, 0:3])
        sn_time_series = time_series * (snr0 / max(np.max(time_series), 1e-6))
        ax_ts.plot(time_axis, sn_time_series, lw=1)
        ax_ts.set_ylabel("S/N")
        ax_ts.set_xticks([])

        # Row 3 (left): FT waterfall
        ax_ft = fig.add_subplot(gs[3, 0:3])
        ax_ft.imshow(
            cand.dedispersed.T, aspect="auto", cmap="viridis",
            extent=[t0, t0 + 256 * cand.tsamp, frequency_axis.min(), frequency_axis.max()]
        )
        ax_ft.set_xlabel("Time (s)"); ax_ft.set_ylabel("Frequency (MHz)")

        # Row 3 (mid): DMT
        ax_dmt = fig.add_subplot(gs[3, 3:5])
        im1 = ax_dmt.imshow(
            cand.dmt, aspect="auto", cmap="viridis",
            extent=[dm_time_axis.min(), dm_time_axis.max(), dm_values.min(), dm_values.max()]
        )
        ax_dmt.set_xlabel("Time (s)"); ax_dmt.set_ylabel("DM (pc cm⁻³)")
        cbar_ax = fig.add_subplot(gs[3, 5])
        cbar = fig.colorbar(im1, cax=cbar_ax); cbar.set_label("Flux")

        # Row 2 right / info
        ax_gal = fig.add_subplot(gs[2, 3:5]); ax_gal.axis("off")
        ax_gal.text(0.5, 0.5, galactic_info, ha="center", va="center", fontsize=9, family="monospace")

        # Row 4 (left): cross-beam layout + colorbar
        ax_cross = fig.add_subplot(gs[4, 0:3])
        sm = plot_sap_beam_layout(ax_cross, beam_snrs, positions=beam_positions, cmap="plasma")
        ax_cross.set_title("Cross-Beam Detections", fontsize=10, y=0.85)
        cbar2_ax = fig.add_axes([
            ax_cross.get_position().x0 - 0.04,
            ax_cross.get_position().y0 - 0.05,
            ax_cross.get_position().width,
            0.015
        ])
        fig.colorbar(sm, cax=cbar2_ax, orientation="horizontal", label="S/N")

        # Row 4 (mid-right): known pulsars
        ax_psr = fig.add_subplot(gs[4, 3:5]); ax_psr.axis("off")
        ax_psr.text(0.5, 0.5, psr_info_text, ha="center", va="center", fontsize=9, family="monospace", wrap=True)

        fig.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.07)
        plt.savefig(combined_path, dpi=300); plt.close()
        print(f"[classify] Saved diagnostic: {combined_path}")

        # Save classification summary
        classification_result = os.path.join(cand_dir, "classification_result.txt")
        with open(classification_result, "w") as f:
            f.write("FETCH Classification Probabilities:\n")
            for k, v in fetch_probabilities.items():
                f.write(f"{k.upper()}: {v:.4f}\n")
            f.write(f"Highest Probability: {highest_prob:.4f}\n")
            f.write(f"Classified as Astrophysical? {'YES' if is_real else 'NO'}\n")
        print(f"[classify] Wrote: {classification_result}")

        if send_to_slack:
            _send_slack_notification(
                combined_path,
                f"Diagnostic - DM {dm}, Width {width_samp}",
                f"Candidate S/N={snr0} in {filterbank_id}"
            )

        # Save candidate HDF5
        cand.save_h5(cand_dir)

    print("[classify] Finished processing candidates.")