#!/bin/bash
#SBATCH --job-name=flatfielding
#SBATCH --output=logs/flatfield_%j.log  # Log output file
#SBATCH --error=logs/flatfield_%j.log   # Error log file
#SBATCH --nodes=1                       # Run on a single node
#SBATCH --ntasks=1                      # One task
#SBATCH --cpus-per-task=8               # Number of CPU cores to use
#SBATCH --time=12:00:00                 # Adjust max runtime as needed
#SBATCH --mem=12G                       # Memory allocation per task

# Print job details
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Allocated on $(hostname)"
echo "Number of nodes: ${SLURM_JOB_NUM_NODES:-1}"
echo "Number of tasks: ${SLURM_NTASKS:-1}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-1}"
echo "Memory per node: ${SLURM_MEM_PER_NODE:-N/A}"

# Ensure logs directory exists
mkdir -p logs

# Path to Singularity container (modify if needed)
SIF_PATH="containers/vm.sif"

# SAP Directory (provided as an argument)
SAP_DIR="$1"

# Validate input
if [ -z "$SAP_DIR" ]; then
    echo "Usage: sbatch run_flatfielding.slurm <SAP_directory>"
    exit 1
fi

# Check if the directory exists
if [ ! -d "$SAP_DIR" ]; then
    echo "Error: SAP directory '$SAP_DIR' does not exist!"
    exit 1
fi

REPO_DIR="$PWD"
SIF_PATH="$REPO_DIR/containers/vm.sif"

singularity exec \
    --bind "$SAP_DIR:$SAP_DIR" \
    --bind "$REPO_DIR:$REPO_DIR" \
    --bind /project:/project \
    --env PYTHONPATH="$REPO_DIR" \
    "$SIF_PATH" \
    python3 "$REPO_DIR/preproc/flatfield_fil.py" "$SAP_DIR"

echo "Flatfielding completed at $(date)"