import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from your.candidate import Candidate
from your.utils.math import normalise
from your.candidate import crop
from fetch.utils import get_model
from src.filterbank import FilterbankFile

def classify_candidates(filterbank_file, candidate_file, output_dir):
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

    for _, row in candidates_df.iterrows():
        dm = row["dm"]
        tcand = row["time"]
        width = int(row["filter_width"])
        snr = row["s/n"]
        sample_number = int(row["sample"])

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
        cand_dir = os.path.join(output_dir, f"DM_{dm}_Width_{width}")
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

        # Generate Frequency-Time (FT) plot
        plt.figure(figsize=(8, 6))
        plt.imshow(cand.dedispersed.T, aspect="auto", cmap="viridis",
                   extent=[tcand, tcand + time_size * cand.tsamp, frequency_axis.min(), frequency_axis.max()])
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (MHz)")
        plt.colorbar(label="Flux (Arb. Units)")
        plt.title(f"FT Plot - DM {dm}, Width {width}")
        plt.savefig(os.path.join(cand_dir, f"ft_plot_DM{dm}_Width{width}.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Generate DM-Time (DMT) plot
        dm_time_axis = np.arange(cand.dmt.shape[1]) * cand.tsamp + tcand
        dm_values = np.linspace(dm - 5, dm + 5, dm_size)

        plt.figure(figsize=(8, 6))
        plt.imshow(cand.dmt, aspect="auto", interpolation="none", cmap="viridis",
                   extent=[dm_time_axis.min(), dm_time_axis.max(), dm_values.min(), dm_values.max()])
        plt.xlabel("Time (s)")
        plt.ylabel("Dispersion Measure (pc cm⁻³)")
        plt.colorbar(label="Flux (Arb. Units)")
        plt.title(f"DM-Time Plot - DM {dm}, Width {width}")
        plt.savefig(os.path.join(cand_dir, f"dmt_plot_DM{dm}_Width{width}.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Generate Time Series plot
        time_series = np.sum(cand.dedispersed, axis=1)
        time_axis = np.arange(len(time_series)) * cand.tsamp + tcand

        plt.figure(figsize=(8, 4))
        plt.plot(time_axis, time_series, color="black", lw=1)
        plt.xlabel("Time (s)")
        plt.ylabel("Flux (Arb. Units)")
        plt.title(f"Pulse Profile - DM {dm}, Width {width}")
        ts_plot_path = os.path.join(cand_dir, f"time_series_DM{dm}_Width{width}.png")
        plt.savefig(ts_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved Time Series plot as {ts_plot_path}")

        # Save candidate to HDF5
        cand.save_h5(cand_dir)

    print("Finished processing all candidates.")