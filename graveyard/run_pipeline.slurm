#!/bin/bash
#SBATCH --job-name=pipeline_processing
#SBATCH --output=logs/pipeline_%A_%a.log  # Log output file for each array task
#SBATCH --error=logs/pipeline_%A_%a.log   # Error log file for each array task
#SBATCH --nodes=1                         # Run on a single node
#SBATCH --ntasks=1                        # One task per array job
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task
#SBATCH -p gpu_a100_7c                   # Use the correct GPU partition
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH --time=02:00:00                   # Max runtime
#SBATCH --mem=32G                         # Memory allocation per task
#SBATCH --array=0-72                      # Adjust dynamically based on number of files

# Print job details
echo "SLURM Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on $(hostname) with ${SLURM_CPUS_PER_TASK} CPUs and 1 GPU"
echo "Starting pipeline processing at $(date)"

# Ensure logs directory exists
mkdir -p logs

# Path to the Singularity container
SIF_PATH="$HOME/vm.sif"

# SAP Directory (provided as an argument)
SAP_DIR="$1"

# Validate input
if [ -z "$SAP_DIR" ]; then
    echo "Usage: sbatch run_pipeline.slurm <SAP_directory>"
    exit 1
fi

# Find all .fil files in the SAP directory, but EXCLUDE BEAM12
FIL_FILES=($(find "$SAP_DIR" -type f -name "downsampled_L*_SAP*_BEAM*_32bit_ff.fil" ! -name "*BEAM12*" | sort))

# Ensure the task ID does not exceed the number of available files
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#FIL_FILES[@]}" ]; then
    echo "Task ID exceeds available .fil files. Exiting."
    exit 1
fi

# Select the file based on SLURM array task ID
INPUT_FILE="${FIL_FILES[$SLURM_ARRAY_TASK_ID]}"
echo "Processing file: $INPUT_FILE"

# Extract the beam directory from the file path
BEAM_DIR=$(dirname "$INPUT_FILE")
OUTPUT_DIR="$BEAM_DIR/output"

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Run the pipeline inside the Singularity container, passing the output directory
singularity exec --nv \
    --bind "$SAP_DIR:$SAP_DIR" \
    --bind /project:/project \
    --bind /home/euflash-dkuiper/your:/opt/your \
    "$SIF_PATH" \
    python3 pipeline.py "$INPUT_FILE" "$OUTPUT_DIR"

echo "Processing completed for $INPUT_FILE at $(date)"