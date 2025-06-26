#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from lotaas_reprocessing import filterbank
from lotaas_reprocessing.numpy_utils import fourier_domain_dedispersion, compute_rfi_mask
from lotaas_reprocessing.matched_filter import run_matched_filtering
from lotaas_reprocessing.classify import classify_candidates
from lotaas_reprocessing.plotting import get_observation_info
from lotaas_reprocessing.sap_beam_map import plot_sap_beam_layout

# ========== Input ==========
if len(sys.argv) < 3:
    print("Usage: python postdetection_pipeline.py <input_fil> <dm>")
    sys.exit(1)

fil_file = sys.argv[1]  # e.g., B027.fil
candidate_dm = float(sys.argv[2])
base_fname = os.path.basename(fil_file).replace(".fil", "")
sap_dir = os.path.abspath(os.path.join(os.path.dirname(fil_file), ".."))
output_dir = os.path.join(os.path.dirname(fil_file), "Output", f"DM{candidate_dm:.1f}")
os.makedirs(output_dir, exist_ok=True)

# ========== Step 1: Load Filterbank + Preprocess ==========
print(f"Loading and preprocessing: {fil_file}")
fil = filterbank.FilterbankFile(fil_file, "read")
tsamp = fil.tsamp
nu = np.flipud(fil.frequencies)
data = np.flipud(fil.get_spectra(0, fil.nspec).T)

mask = compute_rfi_mask(data, block_size=256)
data = data / np.median(data) - 1
masked_data = data.copy()
masked_data[mask] = np.random.normal(np.nanmean(data), np.nanstd(data), data.shape)[mask]

observation_info = get_observation_info(fil, 256, np.sum(mask)/mask.size, factor=128)

# ========== Step 2: Dedisperse at Input DM ==========
print(f"Dedispersing at DM={candidate_dm}")
I_f = fourier_domain_dedispersion(masked_data, tsamp, nu, [candidate_dm])
dedispersed = np.real(np.fft.irfft(I_f, axis=1))[0]  # shape: (time,)

dat_path = os.path.join(output_dir, f"{base_fname}_DM{candidate_dm:.1f}.dat")
dedispersed.astype("float32").tofile(dat_path)

# ========== Step 3: Run Matched Filtering ==========
print("Running matched filtering...")
detection_times, detection_dms, strengths, widths = run_matched_filtering(
    dat_path, tsamp, candidate_dm, downsample=1, detection_threshold=5
)

if len(detection_times) == 0:
    print("No candidates detected above threshold.")
    sys.exit(0)

# Pick top detection
idx_top = np.argmax(strengths)
tcand = detection_times[idx_top]
snr = strengths[idx_top]
width = widths[idx_top]
print(f"Top candidate: time={tcand:.3f}s, DM={candidate_dm}, S/N={snr:.2f}, width={width} samples")

# ========== Step 4: Run Classification ==========
classified_output_dir = os.path.join(output_dir, "classified")
os.makedirs(classified_output_dir, exist_ok=True)

print("Classifying candidate...")
classify_candidates(fil_file, dat_path, classified_output_dir, observation_info, candidate_dm, tcand, snr, width)

# ========== Step 5: Cross-Beam Dedispersion and S/N Extraction ==========
print("Evaluating S/N in all beams...")
beam_snrs = {}
beam_positions = {}

for beam_dir in sorted(os.listdir(sap_dir)):
    if not beam_dir.startswith("B"):
        continue
    beam_path = os.path.join(sap_dir, beam_dir)
    fil_files = [f for f in os.listdir(beam_path) if f.endswith(".fil")]
    if not fil_files:
        continue
    full_path = os.path.join(beam_path, fil_files[0])
    try:
        other_fil = filterbank.FilterbankFile(full_path, "read")
        ts = other_fil.tsamp
        nu_other = np.flipud(other_fil.frequencies)
        data_other = np.flipud(other_fil.get_spectra(0, other_fil.nspec).T)
        data_other = data_other / np.median(data_other) - 1
        masked_other = data_other

        I_f = fourier_domain_dedispersion(masked_other, ts, nu_other, [candidate_dm])
        dedispersed = np.real(np.fft.irfft(I_f, axis=1))[0]
        t = np.arange(len(dedispersed)) * ts
        closest_idx = np.argmin(np.abs(t - tcand))
        sn = dedispersed[closest_idx] / np.std(dedispersed)
        beam_snrs[beam_dir] = sn
        beam_positions[beam_dir] = (
            other_fil.header.get("src_raj", 0),
            other_fil.header.get("src_dej", 0),
        )
    except Exception as e:
        print(f"Failed on {beam_dir}: {e}")

# ========== Step 6: Save Beam Layout Plot ==========
fig, ax = plt.subplots(figsize=(7, 6))
sm = plot_sap_beam_layout(ax, beam_snrs, positions=beam_positions, cmap="plasma")
ax.set_title(f"Beam S/N Layout\nDM={candidate_dm}, t={tcand:.3f}s")
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("S/N")

beam_plot_path = os.path.join(output_dir, f"beam_layout_DM{candidate_dm:.1f}.png")
plt.savefig(beam_plot_path, dpi=300)
print(f"Beam layout saved: {beam_plot_path}")
