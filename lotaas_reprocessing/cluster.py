import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os

from lotaas_reprocessing.dm_plan import dm_values

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

# Default plan, matching settings.yaml. Callers that know which plan produced
# the trials should pass it, so the grid used to cluster is the grid searched.
DEFAULT_PLAN = [
    {"low_dm": 0.000, "high_dm": 150.600, "ddm": 0.10},
    {"low_dm": 150.600, "high_dm": 289.800, "ddm": 0.30},
    {"low_dm": 289.800, "high_dm": 511.800, "ddm": 0.50},
    {"low_dm": 511.800, "high_dm": 1010.800, "ddm": 1.00},
    {"low_dm": 1010.800, "high_dm": 2014.800, "ddm": 2.00},
    {"low_dm": 2014.800, "high_dm": 4469.800, "ddm": 5.00},
    {"low_dm": 4469.800, "high_dm": 8939.800, "ddm": 10.00},
    {"low_dm": 8939.800, "high_dm": 10019.800, "ddm": 20.00},
]

# Neighbourhood in which two detections are taken to be the same event.
# Five trials matches the DM tolerance the original DM/ddm scaling gave inside
# a single plan range. Half a second covers the residual dispersion sweep
# across five trials (0.054 s at the finest spacing) and a boxcar of up to
# that width, while keeping pulses seconds apart distinct: the original
# tolerance of five seconds merged a source repeating every four seconds
# into one candidate.
DM_EPS_TRIALS = 5.0
TIME_EPS_SECONDS = 0.5

# A beam yielding more distinct events than this is RFI, not astronomy.
MAX_CLUSTERS = 2_000_000


def dm_trial_position(dms, plan=None):
    """Position of each DM on the concatenated trial grid, counting from zero.

    The earlier coordinate was DM/ddm, which is not monotonic across plan
    boundaries. DM 50.2 and DM 150.6 both mapped to 502, so unrelated events
    at opposite ends of the low-DM search collided whenever they fell close
    in time. Adjacent trials DM 150.5 and DM 150.6 mapped to 1505 and 502, so
    one pulse straddling a boundary was split in two. Counting trials is
    monotonic and uniformly spaced, which is what a fixed epsilon needs.
    """
    plan = plan or DEFAULT_PLAN
    dms = np.asarray(dms, dtype=float)
    edges = np.array([entry["low_dm"] for entry in plan] + [plan[-1]["high_dm"]])
    steps = np.array([entry["ddm"] for entry in plan])
    counts = np.array([len(dm_values(entry)) for entry in plan])
    starts = np.concatenate([[0], np.cumsum(counts)])[:len(plan)]
    which = np.clip(np.searchsorted(edges, dms, side="right") - 1, 0, len(plan) - 1)
    return starts[which] + (dms - edges[which]) / steps[which]


def get_ddm(dm_value):
    """DM step at a given DM. Retained for the review diagnostics."""
    for entry in DEFAULT_PLAN:
        if entry["low_dm"] <= dm_value < entry["high_dm"]:
            return entry["ddm"]
    return 1.0


def cluster_candidates(candidate_file, output_file, plan=None,
                       dm_eps=DM_EPS_TRIALS, time_eps=TIME_EPS_SECONDS):
    """Group threshold crossings into events, then keep clusters reaching S/N 6.

    Detections are clustered on the trial grid and in arrival time, each axis
    divided by its own tolerance so a single epsilon is dimensionless. Mixing
    trial index with seconds under one epsilon, as before, gave the time axis
    whatever scale the DM axis happened to have.
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

    # Position on the trial grid, monotonic across plan boundaries.
    df["DM_scaled"] = dm_trial_position(df["DM"].values, plan)

    # Each axis in units of its own tolerance, so one epsilon is dimensionless.
    X = np.column_stack([df["DM_scaled"].values / dm_eps,
                         df["Time"].values / time_eps])

    labels = cluster_labels(X, eps=1.)

    # A point with no neighbour is its own event, not a member of one shared
    # bucket. Treating the DBSCAN noise label as a cluster collapsed every
    # isolated detection in a beam into a single reported candidate.
    isolated = labels == -1
    if isolated.any():
        labels = labels.copy()
        labels[isolated] = labels.max() + 1 + np.arange(isolated.sum())
    df["Cluster"] = labels

    # Now filter by S/N ≥ 6 (but preserve cluster structure)
    valid_clusters = set(df[df["S/N"] >= 6]["Cluster"])  # Find clusters with at least one strong S/N
    df = df[df["Cluster"].isin(valid_clusters)].reset_index(drop=True)  # Keep only those clusters

    if df.empty:
        df.to_csv(output_file, sep="\t", index=False)
        print("No valid clusters with S/N ≥ 6 found. Exiting clustering...")
        return

    if len(valid_clusters) > MAX_CLUSTERS:
        raise RuntimeError(
            f"Too many distinct events ({len(valid_clusters)} > {MAX_CLUSTERS}); "
            "inspect RFI before retrying")

    # Highest S/N detection in each cluster, in one pass rather than a scan
    # per cluster: isolated events now make clusters numerous.
    cluster_centers_df = df.loc[df.groupby("Cluster")["S/N"].idxmax()].reset_index(drop=True)

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
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Figure saved as {output_filename}")
