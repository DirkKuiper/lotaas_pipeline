#!/bin/bash
#SBATCH --job-name=downsample_array
#SBATCH --output=logs/downsample/%A_%a.log  # Log output file
#SBATCH --error=logs/downsample/%A_%a.log   # Error log file
#SBATCH --nodes=1                           # Run on a single node
#SBATCH --ntasks=1                          # One task per array job
#SBATCH --cpus-per-task=1                   # Number of CPU cores per task
#SBATCH --time=02:00:00                     # Max runtime (adjust as needed)
#SBATCH --mem=8G                            # Memory allocation per task
#SBATCH --array=0-73                        # 74 beams (adjust dynamically)

# Print job details
echo "SLURM Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on $(hostname) with ${SLURM_CPUS_PER_TASK} CPUs"
echo "Starting downsampling at $(date)"

# Ensure logs directory exists
mkdir -p logs

# Path to the Singularity container
SIF_PATH="containers/vm.sif"

# Parent directory (provided as an argument)
PARENT_DIR="$1"

# Validate input
if [ -z "$PARENT_DIR" ]; then
    echo "Usage: sbatch downsample_array.slurm <parent_directory>"
    exit 1
fi

# Find all subdirectories inside the parent directory (assumes they are named beams)
BEAM_DIRS=($(find "$PARENT_DIR" -mindepth 1 -maxdepth 1 -type d | sort))

# Ensure the task ID does not exceed the number of available beam directories
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#BEAM_DIRS[@]}" ]; then
    echo "Task ID exceeds available beam directories. Exiting."
    exit 1
fi

# Select the beam directory based on SLURM array task ID
INPUT_DIR="${BEAM_DIRS[$SLURM_ARRAY_TASK_ID]}"
echo "Processing beam directory: $INPUT_DIR"

# Find the first PSRFITS file in the directory
PSRFITS_FILE=$(find "$INPUT_DIR" -type f -name "*.fits" | sort | head -n 1)

# Check if a file was found
if [ -z "$PSRFITS_FILE" ]; then
    echo "No PSRFITS file found in $INPUT_DIR"
    exit 1
fi

echo "Found PSRFITS file: $PSRFITS_FILE"

# Get filename without path and remove '_2bit' or '_8bit' if present
FILENAME=$(basename "$PSRFITS_FILE")
CLEAN_FILENAME=$(echo "$FILENAME" | sed -E 's/_(2bit|8bit)//' | sed 's/__/_/g')

# Construct new output filename
OUTPUT_FILE="$INPUT_DIR/downsampled_${CLEAN_FILENAME%.fits}_32bit.fil"

# Run the Python downsampling script inside the Singularity container
REPO_DIR="$PWD"

singularity exec \
  --bind "$REPO_DIR":"$REPO_DIR" \
  --bind "$INPUT_DIR":"$INPUT_DIR" \
  --env PYTHONPATH="$REPO_DIR" \
  "$SIF_PATH" \
  python3 "$REPO_DIR/preproc/downsample_psrfits2fil_32bit.py" \
    --fscrunch 4 --tscrunch 16 -o "$OUTPUT_FILE" "$PSRFITS_FILE"

# Check if downsampling was successful and output file exists
if [ -f "$OUTPUT_FILE" ]; then
    echo "Downsampling completed. Output saved to $OUTPUT_FILE"
    echo "Deleting original file: $PSRFITS_FILE"
    rm "$PSRFITS_FILE"
    echo "Original file deleted. Finished at $(date)"
else
    echo "Error: Downsampling failed or output file not found."
    exit 1
fi