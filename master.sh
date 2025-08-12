#!/bin/bash
#SBATCH --job-name=master_pipeline
#SBATCH --output=logs/master_pipeline_%j.log
#SBATCH --error=logs/master_pipeline_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

echo "Starting master pipeline at $(date)"
mkdir -p logs

# MODE A: Automatic staging
if [ "$#" -eq 1 ]; then
    SRM_LIST="$1"
    STAGE_DIR="staging/$(basename "$SRM_LIST" .txt)_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$STAGE_DIR"

    echo "Running staging step..."
    python3 staging/stage_and_extract.py "$SRM_LIST" "$STAGE_DIR"
    if [ $? -ne 0 ]; then
        echo "Staging failed. Exiting."
        exit 1
    fi

    INPUT_FILE="$STAGE_DIR/webdav_links.txt"
    MACAROON_FILE="$STAGE_DIR/macaroon.txt"
    MACAROON=$(cat "$MACAROON_FILE")

    CLEANUP_STAGE_DIR="$STAGE_DIR"

# MODE B: Manual input of already staged files
elif [ "$#" -eq 2 ]; then
    INPUT_FILE="$1"
    MACAROON="$2"
    CLEANUP_STAGE_DIR=""  # No staging dir to clean up

else
    echo "Usage:"
    echo "  sbatch master.sh <srm_list.txt>"
    echo "  sbatch master.sh <webdav_links.txt> <macaroon.txt>"
    exit 1
fi

# Step 1: Clean and sort the input file
CLEANED_INPUT_FILE="${INPUT_FILE}_cleaned"
echo "Cleaning input file '$INPUT_FILE'..."
sed -e 's/\r$//' -e '/^$/d' "$INPUT_FILE" | sort -t '_' -k3,3n > "$CLEANED_INPUT_FILE"

NUM_URLS=$(wc -l < "$CLEANED_INPUT_FILE")
if [ "$NUM_URLS" -eq 0 ]; then
    echo "Error: No URLs found in cleaned input file!"
    exit 1
fi

echo "Cleaned input file created: $CLEANED_INPUT_FILE with $NUM_URLS URLs"

FIRST_URL=$(head -n 1 "$CLEANED_INPUT_FILE")
FILENAME=$(basename "$FIRST_URL")
OBSID=$(echo "$FILENAME" | awk -F'_' '{print $1}')
SAP=$(echo "$FILENAME" | awk -F'_' '{print $2}')
SAP_DIR="/project/euflash/Data/$OBSID/$SAP"

SAP_LOG_DIR="logs/${OBSID}_${SAP}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SAP_LOG_DIR"
echo "Using SAP log directory: $SAP_LOG_DIR"

# Step 2 & 3: Downloading and Downsampling
DL_DS_JOB_ID=$(sbatch --parsable \
  --array=0-$(($NUM_URLS - 1)) \
  --output="$SAP_LOG_DIR/dl_downsample/%A_%a.log" \
  --error="$SAP_LOG_DIR/dl_downsample/%A_%a.log" \
  bin/download_and_downsample_array.slurm "$CLEANED_INPUT_FILE" "$MACAROON")

# Step 4: Flatfielding
FLATFIELD_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:$DL_DS_JOB_ID \
  --output="$SAP_LOG_DIR/flatfield/%j.log" \
  --error="$SAP_LOG_DIR/flatfield/%j.log" \
  bin/run_flatfielding.slurm "$SAP_DIR")

if [ -z "$FLATFIELD_JOB_ID" ]; then
    echo "Error: Failed to submit flatfielding job."
    exit 1
fi

echo "Flatfielding submitted with job ID: $FLATFIELD_JOB_ID"

# Step 5a: GPU pipeline
echo "Submitting GPU pipeline after flatfielding completes..."
GPU_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:$FLATFIELD_JOB_ID \
  --output="$SAP_LOG_DIR/gpu_pipeline/%j.log" \
  --error="$SAP_LOG_DIR/gpu_pipeline/%j.log" \
  bin/run_gpu_pipeline.slurm "$SAP_DIR")

if [ -z "$GPU_JOB_ID" ]; then
    echo "Error: Failed to submit GPU pipeline job."
    exit 1
fi

# Step 5b: CPU pipeline (array job, e.g. 0-72 for 73 beams)
echo "Submitting CPU pipeline array job after GPU pipeline completes..."
CPU_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:$GPU_JOB_ID \
  --array=0-72 \
  --output="$SAP_LOG_DIR/cpu_pipeline/%A_%a.log" \
  --error="$SAP_LOG_DIR/cpu_pipeline/%A_%a.log" \
  bin/run_cpu_pipeline.slurm "$SAP_DIR")

if [ -z "$CPU_JOB_ID" ]; then
    echo "Error: Failed to submit CPU pipeline job."
    exit 1
fi

# Step 6: Cleanup job
echo "Submitting cleanup job after full pipeline completes..."
CLEANUP_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:$CPU_JOB_ID \
  --output="$SAP_LOG_DIR/cleanup/%j.log" \
  --error="$SAP_LOG_DIR/cleanup/%j.log" \
  bin/cleanup.slurm "$SAP_DIR" "$SAP_LOG_DIR" "$CLEANUP_STAGE_DIR")

if [ -z "$CLEANUP_JOB_ID" ]; then
    echo "Error: Failed to submit cleanup job."
    exit 1
fi

echo "Cleanup job submitted with job ID: $CLEANUP_JOB_ID"
echo "Master pipeline script completed."