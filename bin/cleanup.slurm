#!/bin/bash
#SBATCH --job-name=cleanup
#SBATCH --output=logs/cleanup_%j.log
#SBATCH --error=logs/cleanup_%j.log
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --mem=1G

SAP_DIR="$1"
SAP_LOG_DIR="$2"
STAGE_DIR="$3"
MASTER_LOG="$4"

if [ -z "$SAP_DIR" ] || [ -z "$SAP_LOG_DIR" ]; then
    echo "Usage: sbatch cleanup.slurm <SAP_DIR> <SAP_LOG_DIR> [STAGE_DIR] [MASTER_LOG]"
    exit 1
fi

echo "Starting cleanup for SAP directory: $SAP_DIR"
echo "--------------------------------------------"

# Clean beam directories (keep only 'output/')
for BEAM_DIR in "$SAP_DIR"/B???; do
    if [ -d "$BEAM_DIR" ]; then
        echo "Cleaning beam directory: $BEAM_DIR"
        for ITEM in "$BEAM_DIR"/*; do
            if [[ "$ITEM" == "$BEAM_DIR/output" ]]; then
                continue
            fi
            echo "Removing: $ITEM"
            rm -rf "$ITEM"
        done
    fi
done

# Move SAP-level logs into SAP_DIR
echo "Moving logs from $SAP_LOG_DIR to $SAP_DIR/logs/"
mkdir -p "$SAP_DIR/logs"
rsync -a "$SAP_LOG_DIR/" "$SAP_DIR/logs/"

# Move master pipeline log if specified and exists
if [ -n "$MASTER_LOG" ] && [ -f "$MASTER_LOG" ]; then
    echo "Moving master pipeline log: $MASTER_LOG"
    mv "$MASTER_LOG" "$SAP_DIR/logs/"
else
    echo "No master pipeline log found or not passed. Skipping move."
fi

# Remove intermediate log directory
echo "Removing intermediate log directory: $SAP_LOG_DIR"
rm -rf "$SAP_LOG_DIR"

# Remove staging directory
if [ -n "$STAGE_DIR" ] && [ -d "$STAGE_DIR" ]; then
    echo "Removing staging directory: $STAGE_DIR"
    rm -rf "$STAGE_DIR"
fi

echo "Cleanup job completed at $(date)"