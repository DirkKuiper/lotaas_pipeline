#!/bin/bash
#SBATCH --job-name=CPU_pipeline
#SBATCH --output=logs/CPU_pipeline/%A_%a.log  # Log output file for each array task
#SBATCH --error=logs/CPU_pipeline/%A_%a.log   # Error log file for each array task
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00                        # Max runtime
#SBATCH --mem=32G                             # Memory allocation per task
#SBATCH --array=0-72                          # Adjust dynamically based on the number of beams

# Print job details
echo "SLURM Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on $(hostname) with ${SLURM_CPUS_PER_TASK} CPUs"
echo "Starting CPU pipeline processing at $(date)"

# Ensure logs directory exists
mkdir -p logs

# Path to the Singularity container
SIF_PATH="containers/vm.sif"

# SAP Directory (provided as an argument)
SAP_DIR="$1"

# Validate input
if [ -z "$SAP_DIR" ]; then
    echo "Usage: sbatch run_cpu_pipeline.slurm <SAP_directory>"
    exit 1
fi

# Debugging: show what we are working with
echo "Running find command in SAP_DIR: $SAP_DIR"
echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "Hostname: $(hostname)"
echo "Time: $(date)"

# Try finding output directories up to 5 times in case of NFS delay
for attempt in {1..5}; do
    echo "Attempt $attempt: Looking for output directories..."
    OUTPUT_DIRS=($(find "$SAP_DIR" -type d -name "output" 2>/dev/null | sort))
    if [ "${#OUTPUT_DIRS[@]}" -gt 0 ]; then
        echo "Found ${#OUTPUT_DIRS[@]} output directories."
        break
    fi
    echo "No output directories found. Sleeping for 10 seconds..."
    sleep 10
done

if [ "${#OUTPUT_DIRS[@]}" -eq 0 ]; then
    echo "Error: Could not find any output directories in $SAP_DIR after retries."
    ls -ld "$SAP_DIR" || echo "Directory inaccessible."

    # Extra: list immediate subdirectories for debugging permissions
    echo "Debug: Listing subdirectories of $SAP_DIR to check permissions..."
    **find "$SAP_DIR" -maxdepth 2 -type d -exec ls -ld {} \; 2>&1 | tee /dev/stderr**

    exit 1
fi

# Show what was found
printf 'Output directories:\n%s\n' "${OUTPUT_DIRS[@]}"

# Validate task ID
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#OUTPUT_DIRS[@]}" ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID exceeds available output directories (${#OUTPUT_DIRS[@]})."
    exit 1
fi

# Select the output directory based on SLURM array task ID
OUTPUT_DIR="${OUTPUT_DIRS[$SLURM_ARRAY_TASK_ID]}"
echo "Processing output directory: $OUTPUT_DIR"

# Check if metadata file exists before running CPU pipeline
METADATA_FILE="$OUTPUT_DIR/metadata.yaml"
if [ ! -f "$METADATA_FILE" ]; then
    echo "Error: Metadata file not found in $OUTPUT_DIR! Skipping."
    exit 1
fi

# Run the **CPU pipeline** inside the Singularity container
REPO_DIR="$PWD"
SIF_PATH="$REPO_DIR/containers/vm.sif"

singularity exec --nv \
    --bind "$SAP_DIR:$SAP_DIR" \
    --bind "$REPO_DIR:$REPO_DIR" \
    --bind /project:/project \
    --bind /home/euflash-dkuiper/your:/opt/your \
    --env PYTHONPATH="$REPO_DIR" \
    "$SIF_PATH" \
    python3 "$REPO_DIR/pipeline/pipeline_cpu.py" "$OUTPUT_DIR"

echo "CPU Processing completed for $OUTPUT_DIR at $(date)"