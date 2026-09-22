import os
import sys
import numpy as np
import yaml
from lotaas_reprocessing import matched_filter, cluster
import shutil
from pathlib import Path
from lotaas_reprocessing.dm_plan import dm_values, dm_label


def validate_trials(metadata, directory):
    base = Path(metadata["filename"]).stem
    samples = metadata["samples_processed"]
    expected = {}
    for plan in metadata["dedispersion_plan"]:
        for dm in dm_values(plan):
            name = f"{base}_DM{dm_label(dm)}.dat"
            if name in expected:
                raise ValueError("DM plan produces colliding filenames")
            expected[name] = (samples // plan["downsample"]) * 4
    actual = {p.name: p.stat().st_size for p in Path(directory).glob("*.dat")}
    if actual != expected:
        raise ValueError("DM trials are incomplete, mixed between beams, or have incorrect sample counts")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 pipeline_cpu.py <output_directory>")
        sys.exit(1)

    output_dir = sys.argv[1]

    # Load metadata from GPU stage
    metadata_path = os.path.join(output_dir, "metadata.yaml")
    if not os.path.exists(metadata_path):
        print(f"Metadata file {metadata_path} not found. Exiting.")
        sys.exit(1)

    with open(metadata_path, "r") as fp:
        metadata = yaml.load(fp, Loader=yaml.FullLoader)

    tsamp = metadata["tsamp"]
    dedispersion_plan = metadata["dedispersion_plan"]
    observation_info = metadata["observation_info"]
    fname = metadata["filename"]

    dm_trials_dir = os.path.join(output_dir, "DM_trials")

    try:
        validate_trials(metadata, dm_trials_dir)
        # Run matched filtering
        matched_filter.run_all_matched_filtering(
            dm_trials_dir, tsamp, output_dir, observation_info, dedispersion_plan,
            nu_min=metadata.get("nu_min"), nu_max=metadata.get("nu_max")
        )

        # Define file paths
        all_candidates_file = os.path.join(output_dir, "all_detected_candidates.cands")
        clustered_output_file = os.path.join(output_dir, "clustered_candidates.txt")

        # Run clustering
        print("Running clustering on detected candidates...")
        cluster.cluster_candidates(all_candidates_file, clustered_output_file,
                                   plan=dedispersion_plan)
        print("Clustering completed.")

        # Run classification
        classified_output_dir = os.path.join(output_dir, "candidate_plots")
        os.makedirs(classified_output_dir, exist_ok=True)

        print("Running classification on clustered candidates...")
        from lotaas_reprocessing import classify
        classify.classify_candidates(fname, clustered_output_file, classified_output_dir, observation_info)
        print("Classification completed.")

        # Only clean up if everything above succeeded
        print(f"Removing temporary DM trials directory: {dm_trials_dir}")
        shutil.rmtree(dm_trials_dir, ignore_errors=True)
        print("DM trials directory removed.")

        print(f"Keeping provenance metadata: {metadata_path}")

    except Exception as e:
        # Handle pipeline errors
        print(f"Pipeline encountered an error: {e}")
        raise

    finally:
        print("CPU pipeline exited; see status above.")
