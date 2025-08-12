#!/bin/bash
#SBATCH --job-name=lotaas_download            # Job name
#SBATCH --output=logs/lotaas_download/%j.log  # Output log file
#SBATCH --error=logs/lotaas_download/%j.log   # Error log file
#SBATCH --cores=1                             # Run on a single task (1 core)
#SBATCH --nodes=1                             # Run on a single node
#SBATCH --time=02:00:00                       # Max runtime (adjust as needed)
#SBATCH --mem=16G                             # Memory allocation

# Check if a WebDAV link and macaroon are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <WebDAV link> <Macaroon>"
    exit 1
fi

WEBDAV_LINK="$1"
MACAROON="$2"

# Extract OBSID, SAP, and BEAM number from the URL

echo "WEBDAV_LINK: '$WEBDAV_LINK'"
echo "OBS_SAP_BEAM: '$OBS_SAP_BEAM'"
echo "Saving to: /project/euflash/Data/$OBS_SAP_BEAM"

# Print job details
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Allocated on $(hostname)"
echo "Number of nodes: ${SLURM_JOB_NUM_NODES:-1}"
echo "Number of tasks: ${SLURM_NTASKS:-1}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-1}"
echo "Memory per node: ${SLURM_MEM_PER_NODE:-N/A}"
echo "Starting LOTAAS data download at $(date)"

# Create temporary working directory in scratch space
SCRATCH_DIR="$TMPDIR/lotaas_download"
mkdir -p "$SCRATCH_DIR"

# Move into scratch directory
cd "$SCRATCH_DIR"

# Start timer
START_TIME=$(date +%s)

# Execute wget command with tar extraction directly in scratch space
wget -qO- --header="authorization: bearer $MACAROON" $WEBDAV_LINK | tar xvf - -C "$SCRATCH_DIR"

# End timer
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "Download and extraction completed in ${DURATION} seconds."

# Copy and flatten output directory
OUTPUT_DIR="/project/euflash/Data/$OBS_SAP_BEAM"
mkdir -p "$OUTPUT_DIR"
find "$SCRATCH_DIR" -type f -exec mv -t "$OUTPUT_DIR" {} +

OBSID=$(echo "$FILENAME" | awk -F'_' '{print $1}')
SAP=$(echo "$FILENAME" | awk -F'_' '{print $2}')
SAP_DIR="/project/euflash/Data/$OBSID/$SAP"

# Define a tracking file
SAP_TRACK_FILE="/project/euflash/Data/current_sap.txt"

# Overwrite the file with the latest SAP directory (since only one SAP at a time)
echo "$SAP_DIR" > "$SAP_TRACK_FILE"

# Print finish time
echo "Files copied to $OUTPUT_DIR (flattened structure)"
echo "Finished at $(date)"

# Job automatically cleans up $TMPDIR when finished
