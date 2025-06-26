#!/bin/bash
#SBATCH --job-name=downsample_psrfits
#SBATCH --output=logs/downsample_%j.log  # Log output file
#SBATCH --error=logs/downsample_%j.log   # Error log file
#SBATCH --cores=1                        # Run on a single task (1 core)
#SBATCH --nodes=1                        # Run on a single node
#SBATCH --time=02:00:00                  # Max runtime (adjust as needed)
#SBATCH --mem=16G                        # Memory allocation                    

# Print job details
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Allocated on $(hostname)"
echo "Number of nodes: ${SLURM_JOB_NUM_NODES:-1}"
echo "Number of tasks: ${SLURM_NTASKS:-1}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-1}"
echo "Memory per node: ${SLURM_MEM_PER_NODE:-N/A}"
echo "Starting downsampling at $(date)"

# Ensure logs directory exists
mkdir -p logs

# Path to the Singularity container
SIF_PATH="$HOME/vm.sif"

# Input directory (required argument)
INPUT_DIR="$1"

# Validate input
if [ -z "$INPUT_DIR" ]; then
    echo "Usage: sbatch run_downsampling.slurm <input_directory>"
    exit 1
fi

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
OUTPUT_FILE="$INPUT_DIR/downsampled_${CLEAN_FILENAME%.fits}_32bit.fits"

# Run the Python downsampling script inside the Singularity container
singularity exec --bind "$INPUT_DIR":"$INPUT_DIR" "$SIF_PATH" python3 downsample_psrfits.py --fscrunch 4 --tscrunch 16 -o "$OUTPUT_FILE" "$PSRFITS_FILE"

# Check if downsampling was successful and output file exists
if [ -f "$OUTPUT_FILE" ]; then
    echo "Downsampling completed. Output saved to $OUTPUT_FILE"
    echo "Deleting original file: $PSRFITS_FILE"
    rm "$PSRFITS_FILE"
    echo "Original file deleted."
else
    echo "Error: Downsampling failed or output file not found."
    exit 1
fi