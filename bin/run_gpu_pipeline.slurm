#!/bin/bash
#SBATCH --job-name=GPU_pipeline_all
#SBATCH --output=logs/GPU_pipeline/%A.log
#SBATCH --error=logs/GPU_pipeline/%A.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -p gpu_a100_22c,gpu_a100_7c
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mem=32G

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on $(hostname) with ${SLURM_CPUS_PER_TASK} CPUs and 2 GPUs"
echo "Starting full GPU pipeline processing at $(date)"

# Ensure logs directory exists
mkdir -p logs

# Path to the Singularity container
REPO_DIR="$PWD"
SIF_PATH="$REPO_DIR/containers/vm.sif"

# SAP Directory (provided as an argument)
SAP_DIR="$1"

if [ -z "$SAP_DIR" ]; then
    echo "Usage: sbatch run_gpu_pipeline_all.slurm <SAP_directory>"
    exit 1
fi

# Find all .fil files (exclude BEAM12)
FIL_FILES=($(find "$SAP_DIR" -type f -name "downsampled_L*_SAP*_BEAM*_32bit_ff.fil" ! -name "*BEAM12*" | sort))

if [ ${#FIL_FILES[@]} -eq 0 ]; then
    echo "No .fil files found. Exiting."
    exit 1
fi

# Loop through all .fil files
for ((i=0; i<${#FIL_FILES[@]}; i++)); do
    INPUT_FILE="${FIL_FILES[$i]}"
    echo "Processing file $((i+1))/${#FIL_FILES[@]}: $INPUT_FILE"

    BEAM_DIR=$(dirname "$INPUT_FILE")
    OUTPUT_DIR="$BEAM_DIR/output"
    mkdir -p "$OUTPUT_DIR"
    echo "Output directory: $OUTPUT_DIR"

    # GPU memory monitoring
    GPU_LOG="$OUTPUT_DIR/gpu_memory_usage_${SLURM_JOB_ID}_beam$(basename "$BEAM_DIR").log"
    echo "Starting GPU memory monitoring..."
    nvidia-smi --query-compute-apps=timestamp,pid,process_name,gpu_uuid,used_memory \
               --format=csv \
               -l 2 > "$GPU_LOG" &
    GPU_MON_PID=$!
    echo "GPU monitor PID: $GPU_MON_PID"

    # Run container
    singularity exec --nv \
        --bind "$SAP_DIR:$SAP_DIR" \
        --bind "$REPO_DIR:$REPO_DIR" \
        --bind /project:/project \
        --env PYTHONPATH="$REPO_DIR" \
        "$SIF_PATH" \
        python3 "$REPO_DIR/pipeline/pipeline_gpu.py" "$INPUT_FILE" "$OUTPUT_DIR"

    # Stop monitoring
    echo "Stopping GPU memory monitoring (PID: $GPU_MON_PID)"
    kill "$GPU_MON_PID"
    wait "$GPU_MON_PID" 2>/dev/null
    echo "GPU memory log saved to $GPU_LOG"
    echo "Done with $INPUT_FILE at $(date)"
    echo "--------------------------------------------------"
done

echo "All GPU processing completed at $(date)"