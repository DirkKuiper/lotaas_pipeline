#!/bin/bash
#SBATCH --job-name=full_pipeline
#SBATCH --output=logs/full_pipeline/%A.log
#SBATCH --error=logs/full_pipeline/%A.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -p gpu_a100_22c,gpu_a100_7c
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mem=32G

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on $(hostname) with ${SLURM_CPUS_PER_TASK} CPUs and 2 GPUs"
echo "Starting full pipeline processing at $(date)"

# Set project repo and container
REPO_DIR="$PWD"
SIF_PATH="$REPO_DIR/containers/vm.sif"

# Argument: SAP directory
SAP_DIR="$1"
if [ -z "$SAP_DIR" ]; then
    echo "Usage: sbatch run_gpu_pipeline_all.slurm <SAP_directory>"
    exit 1
fi

# Find all .fil files (excluding BEAM12)
FIL_FILES=($(find "$SAP_DIR" -type f -name "downsampled_L*_SAP*_BEAM*_32bit_ff.fil" ! -name "*BEAM12*" | sort))

if [ ${#FIL_FILES[@]} -eq 0 ]; then
    echo "No .fil files found in $SAP_DIR. Exiting."
    exit 1
fi

# Loop through all .fil files and process
for INPUT_FILE in "${FIL_FILES[@]}"; do
    echo "--------------------------------------------------"
    echo "Processing file: $INPUT_FILE"

    BEAM_DIR=$(dirname "$INPUT_FILE")
    OUTPUT_DIR="$BEAM_DIR/output"
    mkdir -p "$OUTPUT_DIR"
    echo "Output directory: $OUTPUT_DIR"

    # GPU memory monitoring
    GPU_LOG="$OUTPUT_DIR/gpu_memory_${SLURM_JOB_ID}_$(basename "$BEAM_DIR").log"
    echo "Starting GPU memory monitoring..."
    nvidia-smi --query-compute-apps=timestamp,pid,process_name,gpu_uuid,used_memory \
               --format=csv \
               -l 2 > "$GPU_LOG" &
    GPU_MON_PID=$!
    echo "GPU monitor PID: $GPU_MON_PID"

    # Run full pipeline (RFI → dedispersion → matched filter → classify)
    singularity exec --nv \
        --bind "$SAP_DIR:$SAP_DIR" \
        --bind "$REPO_DIR:$REPO_DIR" \
        --bind /project:/project \
        --bind /home/euflash-dkuiper/your:/opt/your \
        --env PYTHONPATH="$REPO_DIR" \
        "$SIF_PATH" \
        python3 "$REPO_DIR/pipeline/pipeline.py" "$INPUT_FILE" "$OUTPUT_DIR"

    # Stop GPU monitoring
    echo "Stopping GPU monitoring (PID: $GPU_MON_PID)"
    kill "$GPU_MON_PID"
    wait "$GPU_MON_PID" 2>/dev/null
    echo "GPU memory log saved to $GPU_LOG"
    echo "Finished processing $INPUT_FILE at $(date)"
done

echo "All beam processing completed at $(date)"