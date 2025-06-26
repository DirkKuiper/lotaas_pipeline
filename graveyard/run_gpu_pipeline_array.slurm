#!/bin/bash
#SBATCH --job-name=GPU_pipeline
#SBATCH --output=logs/GPU_pipeline/%A_%a.log
#SBATCH --error=logs/GPU_pipeline/%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -p gpu_a100_22c,gpu_a100_7c
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --array=0-72

echo "SLURM Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on $(hostname) with ${SLURM_CPUS_PER_TASK} CPUs and 2 GPUs"
echo "Starting GPU pipeline processing at $(date)"

# Ensure logs directory exists
mkdir -p logs

# Path to the Singularity container
SIF_PATH="containers/vm.sif"

# SAP Directory (provided as an argument)
SAP_DIR="$1"

if [ -z "$SAP_DIR" ]; then
    echo "Usage: sbatch run_gpu_pipeline.slurm <SAP_directory>"
    exit 1
fi

# Find all .fil files (exclude BEAM12)
FIL_FILES=($(find "$SAP_DIR" -type f -name "downsampled_L*_SAP*_BEAM*_32bit_ff.fil" ! -name "*BEAM12*" | sort))

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#FIL_FILES[@]}" ]; then
    echo "Task ID exceeds available .fil files. Exiting."
    exit 1
fi

INPUT_FILE="${FIL_FILES[$SLURM_ARRAY_TASK_ID]}"
echo "Processing file: $INPUT_FILE"

BEAM_DIR=$(dirname "$INPUT_FILE")
OUTPUT_DIR="$BEAM_DIR/output"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# ------------------- GPU MEMORY PROFILING -------------------
# Create a unique GPU monitoring log
GPU_LOG="$OUTPUT_DIR/gpu_memory_usage_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

echo "Starting GPU memory monitoring..."
nvidia-smi --query-compute-apps=timestamp,pid,process_name,gpu_uuid,used_memory \
           --format=csv \
           -l 2 > "$GPU_LOG" &
GPU_MON_PID=$!
echo "GPU monitor PID: $GPU_MON_PID"
# ------------------------------------------------------------

REPO_DIR="$PWD"
SIF_PATH="$REPO_DIR/containers/vm.sif"

singularity exec --nv \
    --bind "$SAP_DIR:$SAP_DIR" \
    --bind "$REPO_DIR:$REPO_DIR" \
    --bind /project:/project \
    --env PYTHONPATH="$REPO_DIR" \
    "$SIF_PATH" \
    python3 "$REPO_DIR/pipeline/pipeline_gpu.py" "$INPUT_FILE" "$OUTPUT_DIR"

# ------------------- END MONITORING -------------------
echo "Stopping GPU memory monitoring (PID: $GPU_MON_PID)"
kill "$GPU_MON_PID"
wait "$GPU_MON_PID" 2>/dev/null
echo "GPU memory log saved to $GPU_LOG"
# ------------------------------------------------------

echo "GPU Processing completed for $INPUT_FILE at $(date)"