import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
from your.candidate import Candidate
from your.utils.math import normalise
from your.candidate import crop
from astropy.time import Time

# Import your filterbank reader
sys.path.append("/home/euflash-dkuiper/lotaas_reprocessing/src")
import filterbank

# --- File + candidate info ---
fil_path = "/home/euflash-dkuiper/lotaas_reprocessing/explore/higher_res.fil"
dm = 60.7
tcand = 2877.916
width = 1

# Optional: set output PNG path
outfile = os.path.splitext(fil_path)[0] + f"_DM{dm}_tcand{tcand:.3f}.png"

# Load metadata
fil = filterbank.FilterbankFile(fil_path, "read")
tsamp = fil.tsamp
nchans = fil.nchans
foff = fil.foff
fch1 = fil.fch1
fil.close()

# Candidate object
cand = Candidate(fp=fil_path, dm=dm, tcand=tcand, width=width, label=-1, snr=-1, min_samp=256, device=0)
cand.get_chunk()
cand.dmtime(dmsteps=256)
cand.dedisperse()

# Format + normalize
time_size, freq_size, dm_size = 256, 256, 256
cand.dedispersed = crop(cand.dedispersed, cand.dedispersed.shape[0]//2 - time_size//2, time_size, 0)
cand.dedispersed = normalise(cand.dedispersed)
cand.decimate(key="ft", axis=1, pad=True, decimate_factor=cand.dedispersed.shape[1] // freq_size, mode="median")
cand.resize(key="ft", size=freq_size, axis=1, anti_aliasing=True, mode="constant")
cand.decimate(key="dmt", axis=1, pad=True, decimate_factor=max(1, width // 2), mode="median")
cand.dmt = crop(cand.dmt, cand.dmt.shape[1]//2 - time_size//2, time_size, axis=1)
cand.dmt = crop(cand.dmt, cand.dmt.shape[0]//2 - dm_size//2, dm_size, axis=0)
cand.resize(key="dmt", size=dm_size, axis=1, anti_aliasing=True, mode="constant")
cand.dmt = normalise(cand.dmt)

# Axes
frequency_axis = np.flip(fch1 + np.arange(nchans) * foff)
time_series = np.sum(cand.dedispersed, axis=1)
time_axis = np.arange(len(time_series)) * tsamp + tcand
sn_time_series = time_series #* (1 / np.max(time_series))
dm_time_axis = np.arange(cand.dedispersed.shape[0]) * tsamp + tcand
dm_values = np.linspace(dm - 5, dm + 5, dm_size)

# Plot
fig = plt.figure(figsize=(14, 7))
gs = GridSpec(4, 6, figure=fig,
              width_ratios=[1.7, 1.7, 1.7, 1.7, 3, 0.2],
              height_ratios=[0.35, 0.35, 1, 5.0],
              wspace=0.3, hspace=0.0)

ax_obs = fig.add_subplot(gs[0, 0:5])
ax_obs.axis("off")
ax_obs.text(0.5, 0.5,
    f"File: {os.path.basename(fil_path)}   |   DM: {dm:.2f} pc cm⁻³   |   Width: {width} samples   |   Time: {tcand:.3f} s",
    ha="center", va="center", fontsize=10, family="monospace")

ax_fetch = fig.add_subplot(gs[1, 0:5])
ax_fetch.axis("off")
ax_fetch.text(0.5, 0.5, "FETCH Probabilities: A: -- | B: -- | C: --",
              ha="center", va="center", fontsize=9, family="monospace")

ax_ts = fig.add_subplot(gs[2, 0:3])
ax_ts.plot(time_axis, sn_time_series, color="black", lw=1)
ax_ts.set_ylabel("S/N")
ax_ts.set_xticks([])

vmin, vmax = np.nanpercentile(cand.dedispersed, [20, 99.9])
ax_ft = fig.add_subplot(gs[3, 0:3])
im0 = ax_ft.imshow(cand.dedispersed.T, aspect="auto", cmap="viridis",
                   extent=[tcand, tcand + time_size * tsamp, frequency_axis.min(), frequency_axis.max()],
                   vmin=vmin, vmax=vmax)
ax_ft.set_xlabel("Time (s)")
ax_ft.set_ylabel("Frequency (MHz)")

ax_dmt = fig.add_subplot(gs[3, 3:5])
im1 = ax_dmt.imshow(cand.dmt, aspect="auto", cmap="viridis",
                    extent=[dm_time_axis.min(), dm_time_axis.max(), dm_values.min(), dm_values.max()])
ax_dmt.set_xlabel("Time (s)")
ax_dmt.set_ylabel("DM (pc cm⁻³)")

cbar_ax = fig.add_subplot(gs[3, 5])
fig.colorbar(im1, cax=cbar_ax).set_label("Flux")

fig.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.07)
plt.savefig(outfile, dpi=300)
print(f"Saved plot to: {outfile}")