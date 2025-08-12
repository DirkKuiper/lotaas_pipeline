#!/usr/bin/env bash
set -euo pipefail

usage() { echo "Usage: $0 <JOBID>"; exit 1; }
[[ $# -eq 1 ]] || usage
JID="$1"

divider() { printf '%*s\n' 80 | tr ' ' '-'; }

echo ">>> Job summary (sacct)"
sacct -j "$JID" -X --format=JobID,JobName,State,ExitCode,Reason,Start,End,Elapsed || true
divider

echo ">>> SLURM job details (scontrol)"
SC=$(scontrol show job "$JID" | sed 's/ *$//' || true)
echo "$SC"
divider

# Extract cleanup's SAP_LOG_DIR for fallback log path construction
CLEAN_STDOUT=$(echo "$SC" | awk -F= '/^ *StdOut=/{print $2}')
SAP_LOG_DIR=""
if [[ -n "${CLEAN_STDOUT:-}" ]]; then
  # logs/<SAP>/cleanup/<job>.log -> take parent of 'cleanup'
  CLEAN_DIR=$(dirname "$CLEAN_STDOUT")              # .../logs/<SAP>/cleanup
  SAP_LOG_DIR=$(dirname "$CLEAN_DIR")               # .../logs/<SAP>
fi

# Extract dependency info (if any)
DEP_LINE=$(echo "$SC" | awk -F'Dependency=' '/Dependency=/{print $2}' | tr -d ' ')
if [[ -n "${DEP_LINE:-}" ]]; then
  echo ">>> Dependencies"
  echo "$DEP_LINE"
  DEP_JOBS=$(echo "$DEP_LINE" \
    | sed 's/afterok://g; s/afterany://g; s/afternotok://g; s/singleton//g; s/;/\n/g' \
    | tr ',' '\n' | tr ':' '\n' | tr -d ' ' \
    | grep -E '^[0-9]+$' || true)
  if [[ -n "${DEP_JOBS:-}" ]]; then
    echo
    echo ">>> Upstream job states"
    for dj in $DEP_JOBS; do
      sacct -j "$dj" -X --format=JobID,JobName,State,ExitCode,Reason,Elapsed || true
    done
  fi
  divider
fi

echo ">>> Array inspection (if applicable)"
ARR_LINES=$(sacct -j "$JID" -X --noheader --format=JobID,State 2>/dev/null || true)
BASE_FOR_CHILD="$JID"

if [[ -z "$ARR_LINES" || "$(echo "$ARR_LINES" | wc -l)" -le 1 ]]; then
  if echo "${DEP_LINE:-}" | grep -q '_\*'; then
    ARR_PARENT=$(echo "$DEP_LINE" | grep -oE '[0-9]+_\*' | head -n1 | cut -d'_' -f1)
    if [[ -n "${ARR_PARENT:-}" ]]; then
      echo "This job waits on array: $ARR_PARENT"
      ARR_LINES=$(sacct -j "$ARR_PARENT" -X --noheader --format=JobID,State 2>/dev/null || true)
      BASE_FOR_CHILD="$ARR_PARENT"
      echo "(showing array parent instead)"
    fi
  fi
fi

if [[ -n "$ARR_LINES" ]]; then
  FAILED=$(echo "$ARR_LINES" \
    | awk '/_[0-9]+/ {jid=$1; state=$2; if (state !~ /COMPLETED/) { split(jid,a,"_"); print a[2]; }}' \
    | sort -n | paste -sd, -)
  if [[ -n "${FAILED:-}" ]]; then
    echo "Failed indices: $FAILED"
    ONE=$(echo "$FAILED" | tr ',' '\n' | head -n1)
    CHILD_ID="${BASE_FOR_CHILD}_${ONE}"
    echo
    echo ">>> Example failed task: ${CHILD_ID}"

    # Try scontrol first (works only if still active)
    if scontrol show jobid -dd "$CHILD_ID" >/tmp/.jd.$$ 2>/dev/null; then
      META=$(cat /tmp/.jd.$$ | egrep 'StdOut=|StdErr=|WorkDir=|Command=' || true)
      rm -f /tmp/.jd.$$
      echo "$META"
      STDOUT_FILE=$(echo "$META" | awk -F= '/^StdOut=/ {print $2}')
      STDERR_FILE=$(echo "$META" | awk -F= '/^StdErr=/ {print $2}')
    else
      echo "(task no longer active; using inferred log path)"
      STDOUT_FILE=""
      STDERR_FILE=""
    fi

    # If we couldn't get paths from scontrol, derive from cleanup’s SAP_LOG_DIR
    if [[ -z "${STDOUT_FILE:-}" && -n "${SAP_LOG_DIR:-}" ]]; then
      CAND="${SAP_LOG_DIR}/cpu_pipeline/${BASE_FOR_CHILD}_${ONE}.log"
      if [[ -f "$CAND" ]]; then
        STDOUT_FILE="$CAND"
      fi
    fi
    if [[ -z "${STDERR_FILE:-}" && -n "${STDOUT_FILE:-}" ]]; then
      STDERR_FILE="$STDOUT_FILE"
    fi

    showed=0
    if [[ -n "${STDOUT_FILE:-}" && -f "$STDOUT_FILE" ]]; then
      echo "Log file: $STDOUT_FILE"
      echo ">>> Last 20 lines:"
      divider
      tail -n 20 "$STDOUT_FILE" || true
      divider
      showed=1
    fi
    if [[ $showed -eq 0 && -n "${STDERR_FILE:-}" && -f "$STDERR_FILE" ]]; then
      echo "Log file: $STDERR_FILE"
      echo ">>> Last 20 lines:"
      divider
      tail -n 20 "$STDERR_FILE" || true
      divider
      showed=1
    fi
    if [[ $showed -eq 0 ]]; then
      echo "No readable stdout/stderr file found. Checked:"
      echo "  - scontrol paths (if available)"
      echo "  - ${SAP_LOG_DIR}/cpu_pipeline/${BASE_FOR_CHILD}_${ONE}.log"
    fi
  else
    echo "No failed indices found (or not an array)."
  fi
else
  echo "No array information available."
fi
divider

echo "Done."