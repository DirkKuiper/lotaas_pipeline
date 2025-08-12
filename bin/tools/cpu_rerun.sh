#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   cpu_rerun.sh <cleanup_or_cpu_jobid> [--beams 7,13,18 | --all] [--sap-dir /project/euflash/Data/OBSID/SAP] [--preserve-logs]
# If --beams is omitted, it will auto-rerun ONLY the failed CPU indices found via sacct.

JID=""
BEAMS=""             # e.g., "7,13,18" or "0-72"
SAP_DIR_OVERRIDE=""
PRESERVE_LOGS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --beams)        BEAMS="$2"; shift 2;;
    --all)          BEAMS="0-72"; shift;;
    --sap-dir)      SAP_DIR_OVERRIDE="$2"; shift 2;;
    --preserve-logs) PRESERVE_LOGS=1; shift;;
    -h|--help)      echo "Usage: $0 <cleanup_or_cpu_jobid> [--beams 7,13,18 | --all] [--sap-dir PATH] [--preserve-logs]"; exit 0;;
    *)              if [[ -z "$JID" ]]; then JID="$1"; shift; else echo "Unexpected arg: $1"; exit 1; fi;;
  esac
done
[[ -n "$JID" ]] || { echo "Need a job id."; exit 1; }

divider(){ printf '%*s\n' 80 | tr ' ' '-'; }

echo ">>> Inspecting job $JID"
SC=$(scontrol show job "$JID" 2>/dev/null || true)
if [[ -z "$SC" ]]; then
  echo "Warning: scontrol can't see $JID (finished?). Falling back to sacct where possible."
fi

# Pull dependency line (to find CPU array parent if this is cleanup)
DEP_LINE=$(echo "$SC" | awk -F'Dependency=' '/Dependency=/{print $2}' | tr -d ' ')
CPU_PARENT="$JID"
if echo "${DEP_LINE:-}" | grep -q '_\*'; then
  CPU_PARENT=$(echo "$DEP_LINE" | grep -oE '[0-9]+_\*' | head -n1 | cut -d'_' -f1)
  echo "Found CPU array parent from dependency: $CPU_PARENT"
fi

# Figure out SAP_LOG_DIR from cleanup StdOut (or from a CPU child if possible)
CLEAN_STDOUT=$(echo "$SC" | awk -F= '/^ *StdOut=/{print $2}')
SAP_LOG_DIR=""
if [[ -n "${CLEAN_STDOUT:-}" ]]; then
  # logs/<SAP>/cleanup/<jid>.log -> parent’s parent is logs/<SAP>
  SAP_LOG_DIR=$(dirname "$(dirname "$CLEAN_STDOUT")")
fi

# If beams not provided, collect FAILED indices for the CPU array parent
if [[ -z "$BEAMS" ]]; then
  echo ">>> Collecting failed CPU indices from $CPU_PARENT"
  FAILED=$(sacct -j "$CPU_PARENT" -X --noheader --format=JobID,State 2>/dev/null \
    | awk '/_[0-9]+/ {jid=$1; state=$2; if (state !~ /COMPLETED/) { split(jid,a,"_"); print a[2]; }}' \
    | sort -n | paste -sd, -)
  if [[ -z "$FAILED" ]]; then
    echo "No failed indices found. Use --all or --beams to force a rerun."
    exit 1
  fi
  BEAMS="$FAILED"
fi
echo "Beams to rerun: $BEAMS"

# Derive SAP_DIR if not supplied
if [[ -n "$SAP_DIR_OVERRIDE" ]]; then
  SAP_DIR="$SAP_DIR_OVERRIDE"
else
  # Infer from SAP_LOG_DIR basename like L1285512_SAP002_20250811_001637 -> OBSID=L1285512, SAP=SAP002
  if [[ -z "$SAP_LOG_DIR" ]]; then
    echo "Cannot infer SAP paths (no cleanup StdOut). Provide --sap-dir."
    exit 1
  fi
  SAP_TAG=$(basename "$SAP_LOG_DIR")     # e.g., L1285512_SAP002_20250811_001637
  OBSID=$(echo "$SAP_TAG" | awk -F'_' '{print $1}')
  SAP=$(echo "$SAP_TAG"   | awk -F'_' '{print $2}')
  SAP_DIR="/project/euflash/Data/${OBSID}/${SAP}"
fi
echo "SAP_DIR: $SAP_DIR"
echo "SAP_LOG_DIR: ${SAP_LOG_DIR:-<unknown>}"
divider

# Ensure log dirs exist; optionally preserve old logs in a timestamped subfolder
CPU_LOG_DIR="$SAP_LOG_DIR/cpu_pipeline"
CLEAN_LOG_DIR="$SAP_LOG_DIR/cleanup"
mkdir -p "$CPU_LOG_DIR" "$CLEAN_LOG_DIR"

if [[ $PRESERVE_LOGS -eq 1 ]]; then
  TS=$(date +%Y%m%d_%H%M%S)
  PRES_DIR="$SAP_LOG_DIR/attempts/run-$TS"
  mkdir -p "$PRES_DIR/cpu_pipeline" "$PRES_DIR/cleanup"
  echo "Preserving previous logs under: $PRES_DIR"
  # No moving now; we just keep new logs separate if you want. (Optionally copy/mv here.)
fi

# Submit CPU array rerun
echo "Submitting CPU rerun..."
jid_cpu=$(sbatch --parsable \
  --job-name=cpu_pipeline_rerun \
  --output="$CPU_LOG_DIR/%A_%a.log" \
  --error="$CPU_LOG_DIR/%A_%a.log" \
  --array="$BEAMS" \
  bin/run_cpu_pipeline.slurm "$SAP_DIR")
echo "CPU rerun array job: $jid_cpu"

# Figure out the original master log for this SAP (so cleanup can move it)
SAP_TAG=$(basename "$SAP_LOG_DIR")   # e.g. L1285512_SAP002_20250811_001637

# Find the newest master log that mentions this SAP_TAG
MASTER_LOG_PATH=$(ls -t logs/master_pipeline_*.log 2>/dev/null \
  | xargs -r grep -l -m1 "$SAP_TAG" || true)

if [[ -z "${MASTER_LOG_PATH:-}" ]]; then
  echo "Warning: could not find original master log mentioning $SAP_TAG; cleanup will run without moving it."
fi

# Chain cleanup after CPU success
echo "Submitting cleanup afterok:$jid_cpu ..."
jid_clean=$(sbatch --parsable \
  --job-name=cleanup_rerun \
  --output="$CLEAN_LOG_DIR/%j.log" \
  --error="$CLEAN_LOG_DIR/%j.log" \
  --dependency=afterok:$jid_cpu \
  bin/cleanup.slurm "$SAP_DIR" "$SAP_LOG_DIR" "" "${MASTER_LOG_PATH:-}")
echo "Cleanup job: $jid_clean"
divider

# Cancel the old stuck cleanup, if that's what JID is
OLD_STATE=$(sacct -j "$JID" -X --noheader --format=State 2>/dev/null | head -n1 || true)
OLD_NAME=$(sacct -j "$JID" -X --noheader --format=JobName 2>/dev/null | head -n1 || true)
OLD_REASON=$(squeue -j "$JID" -o "%.40R" 2>/dev/null | awk 'NR==2{print}' || true)

if [[ "$OLD_NAME" =~ cleanup ]]; then
  if [[ "$OLD_STATE" =~ PENDING ]] && [[ "$OLD_REASON" =~ DependencyNeverSatisfied ]]; then
    echo "Cancelling old stuck cleanup $JID ($OLD_STATE: $OLD_REASON)"
    scancel "$JID" || true
  else
    echo "Not cancelling $JID (name=$OLD_NAME state=$OLD_STATE reason=${OLD_REASON:-N/A})"
  fi
fi

echo "Done. Track with: sacct -j $jid_cpu,$jid_clean -X --format=JobID,JobName,State,ExitCode,Reason"