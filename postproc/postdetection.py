#!/usr/bin/env python3
# postproc/postdetection.py

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
import re
from numpy.polynomial import Polynomial
from astropy.coordinates import SkyCoord
from astropy import units as u

# --- ensure repo root on path (so sibling package imports work) ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# -----------------------------------------------------------------

from lotaas_reprocessing import filterbank
from lotaas_reprocessing.numpy_utils import fourier_domain_dedispersion, compute_rfi_mask
from lotaas_reprocessing.matched_filter import run_matched_filtering
from lotaas_reprocessing.classify_post_detection import classify_candidates
from lotaas_reprocessing.plotting import get_observation_info
from lotaas_reprocessing.sap_beam_map import plot_sap_beam_layout, matched_filter_sn_at_idx


def load_settings(path="settings.yaml"):
    # Safe defaults if keys are missing
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


def packed_float_to_sexagesimal_string(val, is_ra=True):
    """Convert PSRFITS-style packed RA/DEC floats to sexagesimal strings."""
    val = float(val)
    hh_or_dd = int(val // 10000)
    mm = int((val % 10000) // 100)
    ss = val % 100
    if is_ra:
        return f"{hh_or_dd:02d}:{mm:02d}:{ss:05.2f}"
    else:
        return f"{hh_or_dd:02d}:{mm:02d}:{ss:04.1f}"


def main():
    # ========== Input ==========
    if len(sys.argv) < 3:
        print("Usage: python postdetection.py <input_fil> <dm>")
        sys.exit(1)

    fil_file = sys.argv[1]
    candidate_dm = float(sys.argv[2])

    base_fname = os.path.basename(fil_file).replace(".fil", "")
    beam_dir = os.path.dirname(fil_file)
    sap_dir = os.path.abspath(os.path.join(beam_dir, ".."))
    output_dir = os.path.join(beam_dir, "Output", f"DM{candidate_dm:.1f}")
    os.makedirs(output_dir, exist_ok=True)

    # ========== Settings (match main pipeline) ==========
    settings = load_settings("settings.yaml")
    block_size = int(settings["rfi_block_size"])
    rfi_avg_factor = int(settings["rfi_averaging_factor"])
    detect_threshold = float(settings["detection_threshold"])
    BAD_CHAN = 138  # same as pipeline

    print(f"[postdetect] Using settings: block_size={block_size}, avg_factor={rfi_avg_factor}, "
          f"threshold={detect_threshold}, bad_chan={BAD_CHAN}")

    # ========== Step 1: Load Filterbank + Preprocess (IDENTICAL to pipeline) ==========
    print(f"[postdetect] Loading and preprocessing: {fil_file}")
    fil = filterbank.FilterbankFile(fil_file, "read")
    tsamp = fil.tsamp
    nu = np.flipud(fil.frequencies)  # MHz, high->low
    data = np.flipud(fil.get_spectra(0, fil.nspec).T)  # (nchan, nsamp)
    t = np.arange(data.shape[1]) * tsamp

    # RFI mask (same block size)
    mask = compute_rfi_mask(data, block_size)
    # Normalize
    data = data / np.median(data) - 1
    # Mask known bad channel (same)
    if BAD_CHAN >= 0 and BAD_CHAN < data.shape[0]:
        mask[BAD_CHAN, :] = True

    # Replace masked values with Gaussian noise computed from masked_data stats (same)
    masked_data = data.copy()
    masked_data[mask] = np.nan
    mu = np.nanmean(masked_data)
    sig = np.nanstd(masked_data)
    noise = np.random.normal(mu, sig, masked_data.shape)
    masked_data[mask] = noise[mask]

    # Per-channel polynomial detrend (deg=2), same as pipeline
    detrended = np.empty_like(masked_data)
    x = np.arange(masked_data.shape[1])
    for i in range(masked_data.shape[0]):
        p = Polynomial.fit(x, masked_data[i, :], deg=2)
        detrended[i, :] = masked_data[i, :] - p(x)
    masked_data = detrended

    # Observation info (for plots/headers)
    observation_info = get_observation_info(fil, block_size, np.sum(mask) / mask.size, factor=rfi_avg_factor)

    # ========== Step 2: Dedisperse at Input DM (explicit irfft length) ==========
    print(f"[postdetect] Dedispersing at DM={candidate_dm}")
    I_f = fourier_domain_dedispersion(masked_data.astype(np.float32), tsamp, nu, [candidate_dm])
    dedispersed = np.fft.irfft(I_f, n=masked_data.shape[1], axis=1).real[0].astype("float32")  # (time,)

    dat_path = os.path.join(output_dir, f"{base_fname}_DM{candidate_dm:.1f}.dat")
    dedispersed.tofile(dat_path)
    print(f"[postdetect] Wrote dedispersed timeseries -> {dat_path}")

    # ========== Step 3: Run Matched Filtering (same threshold) ==========
    print("[postdetect] Running matched filtering...")
    detection_times, detection_dms, strengths, widths = run_matched_filtering(
        dat_path, tsamp, candidate_dm, downsample=1, detection_threshold=detect_threshold
    )

    if len(detection_times) == 0:
        print("[postdetect] No candidates detected above threshold.")
        sys.exit(0)

    idx_top = int(np.argmax(strengths))
    tcand = float(detection_times[idx_top])
    snr = float(strengths[idx_top])
    width = int(widths[idx_top])
    print(f"[postdetect] Top candidate: time={tcand:.3f}s, DM={candidate_dm:.2f}, "
          f"S/N={snr:.2f}, width={width} samples")

    # exact sample index (avoid off-by-one later)
    sample_idx = int(np.argmin(np.abs(t - tcand)))

    # ========== Step 4: Classification (pass exact index) ==========
    classified_output_dir = os.path.join(output_dir, "classified")
    os.makedirs(classified_output_dir, exist_ok=True)
    print("[postdetect] Classifying candidate...")
    classify_candidates(
        fil_file, dat_path, classified_output_dir, observation_info,
        candidate_dm, tcand, snr, width,
        send_to_slack=True, sample_idx=sample_idx,   # <- rely on your updated function
    )

    # ========== Step 5: Cross-Beam Dedispersion + Matched-Filter S/N (limit to a few beams) ==========
    print("[postdetect] Evaluating S/N in nearby beams...")

    # How many beams to test
    BEAM_LIMIT = 73

    # Try to extract the triggering beam index (e.g., B009 -> 9)
    m = re.search(r"/B(\d{3})/", fil_file)
    trigger_beam_idx = int(m.group(1)) if m else None

    # Build the candidate list of beams to process:
    beam_dirs = sorted([d for d in os.listdir(sap_dir) if d.startswith("B")])

    def beam_num(bname: str) -> int:
        try:
            return int(bname[1:])
        except Exception:
            return 10_000

    if trigger_beam_idx is not None:
        # Prefer symmetric neighbors around the triggering beam (e.g., B008, B009, B010)
        sorted_by_distance = sorted(beam_dirs, key=lambda b: abs(beam_num(b) - trigger_beam_idx))
        beam_dirs_to_process = sorted_by_distance[:BEAM_LIMIT]
    else:
        # Fallback: just first N beams
        beam_dirs_to_process = beam_dirs[:BEAM_LIMIT]

    beam_snrs = {}
    beam_positions = {}   # MUST be floats in degrees: {beam_num: (ra_deg, dec_deg)}

    for beam_dir in beam_dirs_to_process:
        beam_path = os.path.join(sap_dir, beam_dir)
        fil_files = [f for f in os.listdir(beam_path) if f.endswith(".fil")]
        if not fil_files:
            continue

        beam_fil = os.path.join(beam_path, fil_files[0])
        try:
            bf = filterbank.FilterbankFile(beam_fil, "read")
            ts = bf.tsamp
            nu_other = np.flipud(bf.frequencies)
            raw = np.flipud(bf.get_spectra(0, bf.nspec).T)

            # --- SAME preprocessing as earlier ---
            msk = compute_rfi_mask(raw, block_size)
            raw = raw / np.median(raw) - 1
            if 0 <= BAD_CHAN < raw.shape[0]:
                msk[BAD_CHAN, :] = True
            masked = raw.copy()
            masked[msk] = np.nan
            mu2, sig2 = np.nanmean(masked), np.nanstd(masked)
            noise2 = np.random.normal(mu2, sig2, masked.shape)
            masked[msk] = noise2[msk]

            # poly-2 detrend per channel
            detr = np.empty_like(masked)
            xx = np.arange(masked.shape[1])
            for i in range(masked.shape[0]):
                p2 = Polynomial.fit(xx, masked[i, :], deg=2)
                detr[i, :] = masked[i, :] - p2(xx)

            I_f2 = fourier_domain_dedispersion(detr.astype(np.float32), ts, nu_other, [candidate_dm])
            dd = np.fft.irfft(I_f2, n=detr.shape[1], axis=1).real[0]

            # use exact index and matched-filter S/N
            sn, _ = matched_filter_sn_at_idx(dd, ts, width, sample_idx)
            beam_snrs[beam_dir] = float(sn)

            # ---- POSITIONS AS FLOATS (DEGREES) ----
            try:
                ra_val = bf.header.get("src_raj", None)
                dec_val = bf.header.get("src_dej", None)
                if ra_val is not None and dec_val is not None:
                    # convert packed floats (e.g., 014022.08, +132229.08) to sexagesimal strings
                    ra_str = packed_float_to_sexagesimal_string(ra_val, is_ra=True)
                    dec_str = packed_float_to_sexagesimal_string(dec_val, is_ra=False)
                    sky = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))
                    beam_positions[int(beam_dir[1:])] = (float(sky.ra.deg), float(sky.dec.deg))
            except Exception as e:
                print(f"[postdetect] Beam {beam_dir} RA/DEC parse failed: {e}")

        except Exception as e:
            print(f"[postdetect] Failed on {beam_dir}: {e}")

    # ========== Step 6: Save Beam Layout Plot ==========
    if beam_snrs:
        fig, ax = plt.subplots(figsize=(7, 6))
        sm = plot_sap_beam_layout(ax, beam_snrs, positions=beam_positions, cmap="plasma")
        ax.set_title(f"Beam S/N Layout\nDM={candidate_dm:.1f}, t={tcand:.3f}s")
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("S/N")
        beam_plot_path = os.path.join(output_dir, f"beam_layout_DM{candidate_dm:.1f}.png")
        plt.savefig(beam_plot_path, dpi=300)
        plt.close(fig)
        print(f"[postdetect] Beam layout saved: {beam_plot_path}")
    else:
        print("[postdetect] No beam S/Ns computed.")

    print("[postdetect] Done.")


if __name__ == "__main__":
    main()