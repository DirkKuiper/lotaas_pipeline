#!/bin/bash
#SBATCH --job-name=lotaas_download_array  # Job name
#SBATCH --output=logs/lotaas_download/%A_%a.log  # Array log output
#SBATCH --error=logs/lotaas_download/%A_%a.log   # Array error log
#SBATCH --nodes=1   # Run on a single node
#SBATCH --ntasks=1  # One task per array job
#SBATCH --cpus-per-task=1  # Single core per task
#SBATCH --time=00:30:00  # Max runtime
#SBATCH --mem=8G  # Memory allocation per task
#SBATCH --array=0-73  # Adjust dynamically based on number of URLs

# Ensure logs directory exists
mkdir -p logs

# Input file with WebDAV URLs
INPUT_FILE="$1"
MACAROON="$2"

# Validate input
if [ -z "$INPUT_FILE" ] || [ -z "$MACAROON" ]; then
    echo "Usage: sbatch lotaas_download_array.slurm <input_file> <macaroon_token>"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found!"
    exit 1
fi

CLEANED_INPUT_FILE="$INPUT_FILE"  # Already cleaned before submission
echo "Reading from pre-cleaned input: $CLEANED_INPUT_FILE"

# **STEP 2: Read URLs properly**
TOTAL_LINES=$(wc -l < "$CLEANED_INPUT_FILE")
if [ "$TOTAL_LINES" -eq 0 ]; then
    echo "Error: Cleaned input file is empty! Check formatting."
    cat "$CLEANED_INPUT_FILE"  # Debugging
    exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_LINES" ]; then
    echo "Error: No URL found for array task ID $SLURM_ARRAY_TASK_ID (Only $TOTAL_LINES lines available)"
    exit 1
fi

WEBDAV_LINK=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CLEANED_INPUT_FILE")

# **STEP 3: Validate URL**
if [[ -z "$WEBDAV_LINK" || ! "$WEBDAV_LINK" =~ ^https?:// ]]; then
    echo "Error: No valid URL found for array task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Starting download for: $WEBDAV_LINK"

# Extract OBSID, SAP, and BEAM number from the URL
FILENAME=$(basename "$WEBDAV_LINK")
OBS_SAP_BEAM=$(echo "$FILENAME" | awk -F'_' '{print $1"/"$2"/"$3}')

echo "Saving to: /project/euflash/Data/$OBS_SAP_BEAM"

# **STEP 4: Download & Extract Data**
SCRATCH_DIR="$TMPDIR/lotaas_download"
mkdir -p "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

START_TIME=$(date +%s)
wget -qO- --header="authorization: bearer $MACAROON" "$WEBDAV_LINK" | tar xvf - -C "$SCRATCH_DIR"
END_TIME=$(date +%s)

DURATION=$((END_TIME - START_TIME))
echo "Download completed in ${DURATION} seconds."

# **STEP 5: Move Extracted Files to Final Location**
OUTPUT_DIR="/project/euflash/Data/$OBS_SAP_BEAM"
mkdir -p "$OUTPUT_DIR"
find "$SCRATCH_DIR" -type f -exec mv -t "$OUTPUT_DIR" {} +

# **STEP 6: Track SAP Directory**
OBSID=$(echo "$FILENAME" | awk -F'_' '{print $1}')
SAP=$(echo "$FILENAME" | awk -F'_' '{print $2}')
SAP_DIR="/project/euflash/Data/$OBSID/$SAP"

SAP_TRACK_FILE="/project/euflash/Data/current_sap.txt"
echo "$SAP_DIR" > "$SAP_TRACK_FILE"

echo "Download and file transfer completed for: $WEBDAV_LINK"
echo "Files moved to $OUTPUT_DIR"