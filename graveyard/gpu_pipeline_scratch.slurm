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

mkdir -p logs

# Path to Singularity container
SIF_PATH="$HOME/vm.sif"

# SAP Directory (provided as argument)
SAP_DIR="$1"

if [ -z "$SAP_DIR" ]; then
    echo "Usage: sbatch run_gpu_pipeline.slurm <SAP_directory>"
    exit 1
fi

# Get .fil files, excluding BEAM12
FIL_FILES=($(find "$SAP_DIR" -type f -name "downsampled_L*_SAP*_BEAM*_32bit_ff.fil" ! -name "*BEAM12*" | sort))

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#FIL_FILES[@]}" ]; then
    echo "Task ID exceeds available .fil files. Exiting."
    exit 1
fi

INPUT_FILE="${FIL_FILES[$SLURM_ARRAY_TASK_ID]}"
echo "Selected file: $INPUT_FILE"

# Extract file and beam directory
INPUT_FILENAME=$(basename "$INPUT_FILE")
BEAM_DIR=$(dirname "$INPUT_FILE")
OUTPUT_DIR="$BEAM_DIR/output"
mkdir -p "$OUTPUT_DIR"

### === SCRATCH SETUP === ###
SCRATCH_DIR="$TMPDIR/gpu_pipeline_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SCRATCH_DIR"
echo "Scratch directory: $SCRATCH_DIR"

# Copy input file to scratch
echo "Copying input file to scratch space..."
cp "$INPUT_FILE" "$SCRATCH_DIR/"
SCRATCH_INPUT="$SCRATCH_DIR/$INPUT_FILENAME"

# Prepare output dir in scratch
SCRATCH_OUTPUT="$SCRATCH_DIR/output"
mkdir -p "$SCRATCH_OUTPUT"

### === RUN GPU PIPELINE === ###
echo "Running GPU pipeline in Singularity container..."
singularity exec --nv \
    --bind "$SCRATCH_DIR:$SCRATCH_DIR" \
    --bind /project:/project \
    --bind /home/euflash-dkuiper/your:/opt/your \
    "$SIF_PATH" \
    python3 pipeline_gpu.py "$SCRATCH_INPUT" "$SCRATCH_OUTPUT"

# Copy results back to final output directory
echo "Copying results back to $OUTPUT_DIR..."
cp -r "$SCRATCH_OUTPUT/"* "$OUTPUT_DIR/"

echo "GPU Processing completed for $INPUT_FILE at $(date)"