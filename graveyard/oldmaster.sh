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

# **Step 1: Submit Download Jobs**
INPUT_FILE="$1"
MACAROON="$2"

if [ -z "$INPUT_FILE" ] || [ -z "$MACAROON" ]; then
    echo "Usage: sbatch master_pipeline.slurm <input_file> <macaroon_token>"
    exit 1
fi

# **Step 1: Clean and Sort Input File**
CLEANED_INPUT_FILE="${INPUT_FILE}_cleaned"
echo "Cleaning input file '$INPUT_FILE'..."

sed -e 's/\r$//' -e '/^$/d' "$INPUT_FILE" | sort -t '_' -k3,3n > "$CLEANED_INPUT_FILE"

# Count number of non-empty lines
NUM_URLS=$(wc -l < "$CLEANED_INPUT_FILE")
if [ "$NUM_URLS" -eq 0 ]; then
    echo "Error: No URLs found in cleaned input file!"
    exit 1
fi

echo "Cleaned input file created: $CLEANED_INPUT_FILE with $NUM_URLS URLs"

# Parse first URL → extract SAP_DIR
FIRST_URL=$(head -n 1 "$CLEANED_INPUT_FILE")
FILENAME=$(basename "$FIRST_URL")
OBSID=$(echo "$FILENAME" | awk -F'_' '{print $1}')
SAP=$(echo "$FILENAME" | awk -F'_' '{print $2}')
SAP_DIR="/project/euflash/Data/$OBSID/$SAP"

echo "Parsed SAP_DIR: $SAP_DIR"

# Create SAP-specific log directory
SAP_LOG_DIR="logs/${OBSID}_${SAP}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SAP_LOG_DIR"
echo "Using SAP log directory: $SAP_LOG_DIR"

# **Step 2: Submit Download Array Job**
echo "Submitting download array job for $NUM_URLS URLs..."

DOWNLOAD_ARRAY_JOB_ID=$(sbatch --parsable \
  --array=0-$(($NUM_URLS - 1)) \
  --output="$SAP_LOG_DIR/download_array/%A_%a.log" \
  --error="$SAP_LOG_DIR/download_array/%A_%a.log" \
  bin/submit_downloads.slurm "$CLEANED_INPUT_FILE" "$MACAROON")

if [ -z "$DOWNLOAD_ARRAY_JOB_ID" ]; then
    echo "Error: Failed to submit download jobs."
    exit 1
fi

echo "Download array job ID: $DOWNLOAD_ARRAY_JOB_ID"

# **Step 5: Submit Downsampling**
DOWNSAMPLE_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:$DOWNLOAD_ARRAY_JOB_ID \
  --output="$SAP_LOG_DIR/downsample/%j.log" \
  --error="$SAP_LOG_DIR/downsample/%j.log" \
  bin/downsample_array.slurm "$SAP_DIR")

if [ -z "$DOWNSAMPLE_JOB_ID" ]; then
    echo "Error: Failed to submit downsampling job."
    exit 1
fi

echo "Downsampling submitted with job ID: $DOWNSAMPLE_JOB_ID"

# **Step 7: Submit Flatfielding**
echo "Submitting flatfielding for: $SAP_DIR"
FLATFIELD_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:$DOWNSAMPLE_JOB_ID \
  --output="$SAP_LOG_DIR/flatfield/%j.log" \
  --error="$SAP_LOG_DIR/flatfield/%j.log" \
  bin/run_flatfielding.slurm "$SAP_DIR")

if [ -z "$FLATFIELD_JOB_ID" ]; then
    echo "Error: Failed to submit flatfielding job."
    exit 1
fi

echo "Flatfielding submitted with job ID: $FLATFIELD_JOB_ID"


# **Step 8: Submit GPU Pipeline After Flatfielding**
echo "Submitting GPU pipeline after flatfielding completes..."

GPU_PIPELINE_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:$FLATFIELD_JOB_ID \
  --output="$SAP_LOG_DIR/gpu_pipeline/%j.log" \
  --error="$SAP_LOG_DIR/gpu_pipeline/%j.log" \
  bin/run_gpu_pipeline.slurm "$SAP_DIR")

if [ -z "$GPU_PIPELINE_JOB_ID" ]; then
    echo "Error: Failed to submit GPU pipeline job."
    exit 1
fi

echo "GPU pipeline submitted with job ID: $GPU_PIPELINE_JOB_ID"

# **Step 9: Submit CPU Pipeline After GPU pipeline**
echo "Submitting CPU pipeline after flatfielding GPU pipeline..."

CPU_PIPELINE_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:$GPU_PIPELINE_JOB_ID \
  --output="$SAP_LOG_DIR/cpu_pipeline/%A_%a.log" \
  --error="$SAP_LOG_DIR/cpu_pipeline/%A_%a.log" \
  bin/run_cpu_pipeline.slurm "$SAP_DIR" "$SAP_LOG_DIR")

if [ -z "$CPU_PIPELINE_JOB_ID" ]; then
    echo "Error: Failed to submit GPU pipeline job."
    exit 1
fi

echo "CPU pipeline submitted with job ID: $CPU_PIPELINE_JOB_ID"

# **Step 10: Submit log-moving job after CPU pipeline completes**
echo "Submitting log-moving job after CPU pipeline completes..."

MOVE_LOGS_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:$CPU_PIPELINE_JOB_ID \
  --output="$SAP_LOG_DIR/move_logs/%j.log" \
  --error="$SAP_LOG_DIR/move_logs/%j.log" \
  bin/move_logs.slurm "$SAP_DIR" "$SAP_LOG_DIR")

if [ -z "$MOVE_LOGS_JOB_ID" ]; then
    echo "Error: Failed to submit log-moving job."
    exit 1
fi

echo "Log-moving job submitted with job ID: $MOVE_LOGS_JOB_ID"

echo "Master pipeline script completed."