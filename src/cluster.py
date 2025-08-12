import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os

def cluster_candidates(candidate_file, output_file):
    """
    Perform DBSCAN clustering first, then filter clusters by S/N threshold.
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

    # Stack features for clustering (DM and Time)
    X = df[["DM", "Time"]].values  

    # Perform DBSCAN clustering first (even on low S/N)
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