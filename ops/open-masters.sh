#!/bin/bash
# Open the persistent SSH control masters the campaign dispatches through, for the nodes given (default: all),
# and a second master on each ingest node for euroflash.ingest (ingest-<node>; INGEST_NODES, default
# efc-cpu-04..06), so ingest never takes the search tier's sessions (sshd allows 10 per connection).
# Needs a usable identity once: your forwarded agent (ssh-add on your own machine) or the unattended key
# registered on the nodes. The masters persist after that. Masters that are already up are left alone.
#     ops/open-masters.sh [efc-bfc-01 efc-cpu-03 ...]
nodes=${*:-efc-gpu-00 efc-gpu-01 efc-bfc-00 efc-bfc-01 efc-bfc-02 efc-bfc-03 efc-cpu-00 efc-cpu-01 efc-cpu-02 efc-cpu-03 efc-cpu-04 efc-cpu-05 efc-cpu-06}
ingest_nodes=${INGEST_NODES:-efc-cpu-04 efc-cpu-05 efc-cpu-06}
control=$HOME/.ssh/control
mkdir -p "$control"
failures=0
open_master() {     # node socket
  if ssh -S "$2" -O check "$1" 2>/dev/null; then
    echo "up      $1 ($(basename "$2"))"
  elif ssh -o ControlMaster=yes -o ControlPersist=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
          -o StrictHostKeyChecking=accept-new -S "$2" -fN "$1"; then
    echo "opened  $1 ($(basename "$2"))"
  else
    echo "FAILED  $1 ($(basename "$2"))"; failures=1
  fi
}
for node in $nodes; do
  short=${node#efc-}; short=${short//-/}
  open_master "$node" "$control/lotaas-$short"         # as euroflash.cluster.ssh_args names it
done
for node in $ingest_nodes; do
  case " $nodes " in *" $node "*) ;; *) [ -n "$*" ] && continue ;; esac
  short=${node#efc-}; short=${short//-/}
  open_master "$node" "$control/ingest-$short"         # as euroflash.ingest.Ingest.ssh_args names it
done
exit $failures
