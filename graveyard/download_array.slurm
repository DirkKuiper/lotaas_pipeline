#!/bin/bash
#SBATCH --job-name=lotaas_download
#SBATCH --output=logs/lotaas_download_%A_%a.log
#SBATCH --error=logs/lotaas_download_%A_%a.log
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --array=0-0  # Placeholder — dynamically updated below

# --- BEGIN STANDALONE SETUP ---

# Input arguments
INPUT_FILE="$1"
MACAROON="$2"

if [ -z "$INPUT_FILE" ] || [ -z "$MACAROON" ]; then
    echo "Usage: sbatch submit_downloads_array.slurm <input_file> <macaroon_token>"
    exit 1
fi

# Ensure logs directory exists
mkdir -p logs

# Clean input file — remove empty lines and carriage returns
CLEANED_INPUT_FILE="${INPUT_FILE}_cleaned"
if [ ! -f "$CLEANED_INPUT_FILE" ]; then
    echo "Cleaning input file..."
    sed '/^$/d' "$INPUT_FILE" | tr -d '\r' > "$CLEANED_INPUT_FILE"
fi

# Dynamically count lines (only for SLURM_ARRAY_TASK_ID=0)
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    NUM_URLS=$(wc -l < "$CLEANED_INPUT_FILE")
    echo "Detected $NUM_URLS URLs to download."

    if [ "$NUM_URLS" -eq 0 ]; then
        echo "No URLs found in input file. Exiting."
        exit 1
    fi

    # Submit the real array job and exit this "dummy" one
    echo "Submitting array job for URLs..."
    sbatch --array=0-$(($NUM_URLS - 1)) "$0" "$INPUT_FILE" "$MACAROON"
    echo "Submitted array job for download tasks."
    exit 0
fi

# --- BEGIN ACTUAL DOWNLOAD TASKS ---

# Extract URL based on task ID
TASK_ID=$SLURM_ARRAY_TASK_ID
WEBDAV_URL=$(sed -n "$((TASK_ID + 1))p" "$CLEANED_INPUT_FILE" | xargs)

if [ -z "$WEBDAV_URL" ]; then
    echo "No URL found at task ID $TASK_ID"
    exit 1
fi

echo "[$(date)] Starting download for task $TASK_ID"
echo "URL: $WEBDAV_URL"

# Create unique scratch directory
SCRATCH_DIR="/scratch/$USER/lotaas_download_${TASK_ID}_$RANDOM"
mkdir -p "$SCRATCH_DIR"
cd "$SCRATCH_DIR" || exit 1

# Download and extract
wget -qO- --header="authorization: bearer $MACAROON" "$WEBDAV_URL" | tar xvf - -C "$SCRATCH_DIR"

# Extract OBS_SAP_BEAM
FILENAME=$(find "$SCRATCH_DIR" -type f | head -n 1 | xargs basename)
if [[ "$FILENAME" =~ ^(L[0-9]+_SAP[0-9]+_B[0-9]+)_ ]]; then
    OBS_SAP_BEAM="${BASH_REMATCH[1]}"
else
    echo "Failed to parse OBS_SAP_BEAM from $FILENAME"
    rm -r "$SCRATCH_DIR"
    exit 1
fi

# Output directory setup
OUTPUT_DIR="/project/euflash/Data/$OBS_SAP_BEAM"
mkdir -p "$OUTPUT_DIR"
find "$SCRATCH_DIR" -type f -exec mv -t "$OUTPUT_DIR" {} +

# SAP tracking
OBSID=$(echo "$OBS_SAP_BEAM" | awk -F'_' '{print $1}')
SAP=$(echo "$OBS_SAP_BEAM" | awk -F'_' '{print $2}')
SAP_DIR="/project/euflash/Data/$OBSID/$SAP"
echo "$SAP_DIR" > "/project/euflash/Data/current_sap.txt"

echo "Files moved to $OUTPUT_DIR"
echo "Task $TASK_ID completed at $(date)"

# Clean up
rm -r "$SCRATCH_DIR"