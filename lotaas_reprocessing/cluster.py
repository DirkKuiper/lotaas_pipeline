import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os

# Define a hard limit for number of candidates
MAX_CANDIDATES = 25_000_000


def cluster_labels(points, eps=5.):
    """Exact epsilon-connected components for DBSCAN with min_samples=2.

    Every point with a neighbor is a core point when min_samples=2 (self
    counts). Thus there are no non-core border points: isolated points are
    noise, and the other clusters are graph components. Points in a square
    of side eps/sqrt(2) form a clique. Union these cells using exact nearest
    distances, without materializing the quadratic neighborhood graph.
    """
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all() or eps<=0:
        raise ValueError('Expected finite two-dimensional coordinates and eps > 0')
    if len(points)==0:return np.empty(0,dtype=int)
    keys,inverse=np.unique(np.floor(points/(eps/np.sqrt(2))).astype(np.int64),axis=0,return_inverse=True)
    count=np.bincount(inverse)
    order=np.argsort(inverse,kind='stable')
    groups=np.split(order,np.cumsum(count)[:-1])
    coordinates=[points[g] for g in groups]
    bounds=[(p.min(axis=0),p.max(axis=0)) for p in coordinates]
    parent=np.arange(len(keys));sizes=count.copy()
    lookup={tuple(key):i for i,key in enumerate(keys)}
    trees={}
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def join(i,j):
        i,j=root(i),root(j)
        if i==j:return
        if sizes[i]<sizes[j]:i,j=j,i
        parent[j]=i;sizes[i]+=sizes[j]
    offsets=[(dx,dy) for dx in range(3) for dy in range(-2,3) if dx>0 or dy>0]
    for i,key in enumerate(keys):
        for dx,dy in offsets:
            j=lookup.get((key[0]+dx,key[1]+dy))
            if j is None or root(i)==root(j):continue
            lo1,hi1=bounds[i];lo2,hi2=bounds[j]
            gap=np.maximum(0,np.maximum(lo1-hi2,lo2-hi1))
            if gap@gap>eps*eps:continue
            small,large=(i,j) if count[i]<=count[j] else (j,i)
            if large not in trees:trees[large]=cKDTree(coordinates[large])
            distance,_=trees[large].query(coordinates[small],k=1,distance_upper_bound=np.nextafter(eps,np.inf))
            if np.any(distance<=eps):join(i,j)
    roots=np.array([root(i) for i in range(len(keys))])[inverse]
    unique,first=np.unique(roots,return_index=True)
    mapping={r:k for k,r in enumerate(r for r in unique[np.argsort(first)] if sizes[r]>=2)}
    cell_labels=np.array([mapping.get(root(i),-1) for i in range(len(keys))])
    return cell_labels[inverse]

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
    df = pd.read_csv(candidate_file, sep=r"\s+", comment="#",
                     names=["DM", "S/N", "Time", "Sample", "Filter_Width"])

    if df.empty:
        df.to_csv(output_file, sep="\t", index=False)
        print("No candidates found. Exiting clustering...")
        return
    
    # Check if data is too large to process
    num_candidates = len(df)
    if num_candidates > MAX_CANDIDATES:
        raise RuntimeError(f"Too many candidates ({num_candidates} > {MAX_CANDIDATES}). Data may be bad. Exiting clustering...")
        return

    # Normalize DM values by ddm
    df["DM_scaled"] = df["DM"] / df["DM"].apply(get_ddm)

    # Stack normalized features for clustering (DM_scaled and Time)
    X = df[["DM_scaled", "Time"]].values  

    # Perform DBSCAN clustering
    df["Cluster"] = cluster_labels(X, eps=5)

    # Now filter by S/N ≥ 6 (but preserve cluster structure)
    valid_clusters = set(df[df["S/N"] >= 6]["Cluster"])  # Find clusters with at least one strong S/N
    df = df[df["Cluster"].isin(valid_clusters)].reset_index(drop=True)  # Keep only those clusters

    if df.empty:
        df.to_csv(output_file, sep="\t", index=False)
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
    display=df.iloc[::max(1, int(np.ceil(len(df)/100000)))]
    plt.scatter(display["Time"], display["DM"], c=display["Cluster"], cmap="tab10", alpha=0.6, label="Detected Pulses (display sample)")
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
