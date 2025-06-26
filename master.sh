#!/bin/bash
#SBATCH --job-name=master_pipeline
#SBATCH --output=logs/master_pipeline_%j.log
#SBATCH --error=logs/master_pipeline_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00

echo "Starting master pipeline at $(date)"
mkdir -p logs

INPUT_FILE="$1"
MACAROON="$2"

if [ -z "$INPUT_FILE" ] || [ -z "$MACAROON" ]; then
    echo "Usage: sbatch master_pipeline.slurm <input_file> <macaroon_token>"
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

# Step 5: Unified GPU+CPU pipeline
echo "Submitting unified full pipeline after flatfielding completes..."
FULL_PIPELINE_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:$FLATFIELD_JOB_ID \
  --output="$SAP_LOG_DIR/full_pipeline/%j.log" \
  --error="$SAP_LOG_DIR/full_pipeline/%j.log" \
  bin/run_full_pipeline.slurm "$SAP_DIR")

if [ -z "$FULL_PIPELINE_JOB_ID" ]; then
    echo "Error: Failed to submit full pipeline job."
    exit 1
fi

echo "Unified pipeline submitted with job ID: $FULL_PIPELINE_JOB_ID"

# Step 6: Log-moving job
echo "Submitting log-moving job after full pipeline completes..."
MOVE_LOGS_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:$FULL_PIPELINE_JOB_ID \
  --output="$SAP_LOG_DIR/move_logs/%j.log" \
  --error="$SAP_LOG_DIR/move_logs/%j.log" \
  bin/move_logs.slurm "$SAP_DIR" "$SAP_LOG_DIR")

if [ -z "$MOVE_LOGS_JOB_ID" ]; then
    echo "Error: Failed to submit log-moving job."
    exit 1
fi

echo "Log-moving job submitted with job ID: $MOVE_LOGS_JOB_ID"

echo "Master pipeline script completed."