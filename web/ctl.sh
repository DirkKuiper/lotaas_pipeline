#!/usr/bin/env bash
# Start, stop and inspect the web layer on the head node.
#   web/ctl.sh start | stop | restart | status | url | logs
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${LOTAAS_WEB_PYTHON:-$HOME/.venvs/lotaas-web/bin/python}
cd "$HERE"
DATA=$("$PYTHON" -c 'from web import config; print(config.load().data)')
PIDFILE=$DATA/web.pid
LOG=$DATA/logs/web.log
mkdir -p "$DATA/logs"

running() { [[ -f $PIDFILE ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case ${1:-status} in
  start)
    if running; then echo "Already running (PID $(cat "$PIDFILE"))"; exit 0; fi
    setsid nohup "$PYTHON" -m web serve >> "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    sleep 2
    running && echo "Started (PID $(cat "$PIDFILE")); log $LOG" || { echo "Failed to start; see $LOG"; exit 1; } ;;
  stop)
    if running; then kill "$(cat "$PIDFILE")"; echo "Stopped"; else echo "Not running"; fi
    rm -f "$PIDFILE" ;;
  restart)
    "$0" stop; sleep 1; "$0" start ;;
  status)
    if running; then echo "Running (PID $(cat "$PIDFILE"))"; else echo "Not running"; fi ;;
  url)
    "$PYTHON" -m web url ;;
  logs)
    tail -n 50 -f "$LOG" ;;
  *)
    echo "usage: $0 start|stop|restart|status|url|logs"; exit 2 ;;
esac
