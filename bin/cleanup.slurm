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

if [ -z "$SAP_DIR" ] || [ -z "$SAP_LOG_DIR" ]; then
    echo "Usage: sbatch cleanup.slurm <SAP_DIR> <SAP_LOG_DIR> [STAGE_DIR]"
    exit 1
fi

echo "Moving logs from $SAP_LOG_DIR to $SAP_DIR/logs/"

mkdir -p "$SAP_DIR/logs"
rsync -a "$SAP_LOG_DIR/" "$SAP_DIR/logs/"

echo "Removing intermediate log directory $SAP_LOG_DIR"
rm -rf "$SAP_LOG_DIR"

if [ -n "$STAGE_DIR" ] && [ -d "$STAGE_DIR" ]; then
    echo "Removing staging directory $STAGE_DIR"
    rm -rf "$STAGE_DIR"
fi

echo "Cleanup job completed."