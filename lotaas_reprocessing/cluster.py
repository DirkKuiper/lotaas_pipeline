import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os

# Define a hard limit for number of candidates
MAX_CANDIDATES = 1_000_000  

# Define dedispersion plan
DEDISPERSION_PLAN = [
    (0.000, 150.600, 0.10),
    (150.600, 289.800, 0.30),
    (289.800, 511.800, 0.50),
    (511.800, 1010.800, 1.00),
    (1010.800, 2014.800, 2.00),
    (2014.800, 4469.800, 5.00),
    (4469.800, 8939.800, 10.00),
    (8939.800, 10019.800, 20.00),
]

def get_ddm(dm_value):
    """Finds the corresponding ddm value for a given DM based on dedispersion plan."""
    for low_dm, high_dm, ddm in DEDISPERSION_PLAN:
        if low_dm <= dm_value < high_dm:
            return ddm
    return 1.0  # Default to 1.0 if not found (shouldn't happen)

def cluster_candidates(candidate_file, output_file):
    """
    Perform DBSCAN clustering first, then filter clusters by S/N threshold.
    Normalize DM values by ddm to avoid bias due to uneven spacing in the dedispersion plan.
    """
    # Ensure the candidate file exists
    if not os.path.exists(candidate_file):
        print(f"Candidate file {candidate_file} not found. Skipping clustering.")
        return

    # Load the candidate file
    df = pd.read_csv(candidate_file, delim_whitespace=True, comment="#",
                     names=["DM", "S/N", "Time", "Sample", "Filter_Width"])

    if df.empty:
        print("No candidates found. Exiting clustering...")
        return
    
    # Check if data is too large to process
    num_candidates = len(df)
    if num_candidates > MAX_CANDIDATES:
        print(f"Too many candidates ({num_candidates} > {MAX_CANDIDATES}). Data may be bad. Exiting clustering...")
        return

    # Normalize DM values by ddm
    df["DM_scaled"] = df["DM"] / df["DM"].apply(get_ddm)

    # Stack normalized features for clustering (DM_scaled and Time)
    X = df[["DM_scaled", "Time"]].values  

    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=5, min_samples=2).fit(X)
    df["Cluster"] = clustering.labels_

    # Now filter by S/N ≥ 6 (but preserve cluster structure)
    valid_clusters = set(df[df["S/N"] >= 6]["Cluster"])  # Find clusters with at least one strong S/N
    df = df[df["Cluster"].isin(valid_clusters)].reset_index(drop=True)  # Keep only those clusters

    if df.empty:
        print("No valid clusters with S/N ≥ 6 found. Exiting clustering...")
        return

    # Find the highest S/N pulse in each remaining cluster
    cluster_centers = []
    for cluster in valid_clusters:
        cluster_points = df[df["Cluster"] == cluster]
        highest_sn_point = cluster_points.loc[cluster_points["S/N"].idxmax()]
        cluster_centers.append(highest_sn_point)

    # Convert to DataFrame
    cluster_centers_df = pd.DataFrame(cluster_centers)

    # Save clustered candidates
    cluster_centers_df.to_csv(output_file, sep="\t", index=False)
    print(f"Clustered candidates saved as {output_file}")

    # Plot DM vs Time
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Time"], df["DM"], c=df["Cluster"], cmap="tab10", alpha=0.6, label="Detected Pulses")
    plt.scatter(cluster_centers_df["Time"], cluster_centers_df["DM"], 
                c='red', edgecolors='black', marker='*', s=200, label="Cluster Centers (Highest S/N)")
    plt.xlabel("Time (s)")
    plt.ylabel("DM (pc/cm³)")
    plt.title("DM vs Time with Cluster Centers Highlighted")
    plt.legend()

    # Save the plot
    output_filename = os.path.join(os.path.dirname(output_file), "dm_vs_time_clusters.png")
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure saved as {output_filename}")