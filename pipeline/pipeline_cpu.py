import os
import sys
import numpy as np
import yaml
from lotaas_reprocessing import matched_filter, cluster, classify
import shutil

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
        # Run matched filtering
        matched_filter.run_all_matched_filtering(
            dm_trials_dir, tsamp, output_dir, observation_info, dedispersion_plan
        )

        # Define file paths
        all_candidates_file = os.path.join(output_dir, "all_detected_candidates.cands")
        clustered_output_file = os.path.join(output_dir, "clustered_candidates.txt")

        # Run clustering
        print("Running clustering on detected candidates...")
        cluster.cluster_candidates(all_candidates_file, clustered_output_file)
        print("Clustering completed.")

        # Run classification
        classified_output_dir = os.path.join(output_dir, "candidate_plots")
        os.makedirs(classified_output_dir, exist_ok=True)

        print("Running classification on clustered candidates...")
        classify.classify_candidates(fname, clustered_output_file, classified_output_dir, observation_info)
        print("Classification completed.")

        # Only clean up if everything above succeeded
        print(f"Removing temporary DM trials directory: {dm_trials_dir}")
        shutil.rmtree(dm_trials_dir, ignore_errors=True)
        print("DM trials directory removed.")

        print(f"Removing temporary metadata file: {metadata_path}")
        try:
            os.remove(metadata_path)
            print("Metadata file removed.")
        except Exception as e:
            print(f"Skipping metadata file removal ({e}).")

    except Exception as e:
        # Handle pipeline errors
        print(f"Pipeline encountered an error: {e}")

    finally:
        print("CPU pipeline completed (with or without errors).")