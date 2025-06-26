#!/bin/bash
#SBATCH --job-name=move_logs
#SBATCH --output=logs/move_logs_%j.log
#SBATCH --error=logs/move_logs_%j.log
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --mem=1G

SAP_DIR="$1"
SAP_LOG_DIR="$2"

if [ -z "$SAP_DIR" ] || [ -z "$SAP_LOG_DIR" ]; then
    echo "Usage: sbatch move_logs.slurm <SAP_DIR> <SAP_LOG_DIR>"
    exit 1
fi

echo "Moving logs from $SAP_LOG_DIR to $SAP_DIR/logs/"

mkdir -p "$SAP_DIR/logs"
rsync -a "$SAP_LOG_DIR/" "$SAP_DIR/logs/"

echo "Removing intermediate log directory $SAP_LOG_DIR"
rm -rf "$SAP_LOG_DIR"

