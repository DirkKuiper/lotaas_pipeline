#!/bin/bash
#SBATCH --job-name=dl_downsample_array
#SBATCH --output=logs/dl_downsample/%A_%a.log
#SBATCH --error=logs/dl_downsample/%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:30:00
#SBATCH --mem=12G
#SBATCH --array=0-73  # Adjust dynamically at submission

mkdir -p logs

INPUT_FILE="$1"
MACAROON="$2"

if [ -z "$INPUT_FILE" ] || [ -z "$MACAROON" ]; then
    echo "Usage: sbatch download_and_downsample_array.slurm <input_file> <macaroon_token>"
    exit 1
fi

# Ensure task ID is valid
TOTAL_LINES=$(wc -l < "$INPUT_FILE")
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_LINES" ]; then
    echo "Array task ID exceeds number of lines in input file"
    exit 1
fi

# Get URL and define output paths
URL=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$INPUT_FILE")
FILENAME=$(basename "$URL")
OBS_SAP_BEAM=$(echo "$FILENAME" | awk -F'_' '{print $1"/"$2"/"$3}')
OUTPUT_DIR="/project/euflash/Data/$OBS_SAP_BEAM"
mkdir -p "$OUTPUT_DIR"

# Define base repo path and container path BEFORE switching to scratch dir
REPO_DIR="$PWD"
SIF_PATH="$REPO_DIR/containers/vm.sif"

# Download and extract
SCRATCH_DIR="$TMPDIR/download_extract"
mkdir -p "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

echo "Downloading $URL..."
wget -qO- --header="authorization: bearer $MACAROON" "$URL" | tar xvf - -C "$SCRATCH_DIR"

# Move extracted files to OUTPUT_DIR
find "$SCRATCH_DIR" -type f -exec mv -t "$OUTPUT_DIR" {} +

# Find PSRFITS file
PSRFITS_FILE=$(find "$OUTPUT_DIR" -type f -name "*.fits" | sort | head -n 1)
if [ -z "$PSRFITS_FILE" ]; then
    echo "No PSRFITS file found after download!"
    exit 1
fi

echo "Found PSRFITS file: $PSRFITS_FILE"

# Downsample
FILENAME=$(basename "$PSRFITS_FILE")
CLEAN_FILENAME=$(echo "$FILENAME" | sed -E 's/_(2bit|8bit)//' | sed 's/__/_/g')
OUTPUT_FILE="$OUTPUT_DIR/downsampled_${CLEAN_FILENAME%.fits}_32bit.fil"

singularity exec \
  --bind "$REPO_DIR":"$REPO_DIR" \
  --bind "$OUTPUT_DIR":"$OUTPUT_DIR" \
  --env PYTHONPATH="$REPO_DIR" \
  "$SIF_PATH" \
  python3 "$REPO_DIR/preproc/downsample_psrfits2fil_32bit.py" \
    --fscrunch 4 --tscrunch 16 -o "$OUTPUT_FILE" "$PSRFITS_FILE"

# Validate output
if [ -f "$OUTPUT_FILE" ]; then
    echo "Downsampling complete: $OUTPUT_FILE"
    echo "Removing original PSRFITS file: $PSRFITS_FILE"
    rm "$PSRFITS_FILE"
    echo "Done at $(date)"
else
    echo "Downsampling failed: $OUTPUT_FILE not found!"
    exit 1
fi