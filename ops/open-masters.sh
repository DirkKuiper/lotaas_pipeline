#!/bin/bash
# Open the persistent SSH control masters the campaign dispatches through, for the nodes given (default: all).
# Needs a usable identity once: your forwarded agent (ssh-add on your own machine) or the unattended key
# registered on the nodes. The masters persist after that. Masters that are already up are left alone.
#     ops/open-masters.sh [efc-bfc-01 efc-cpu-03 ...]
nodes=${*:-efc-gpu-00 efc-gpu-01 efc-bfc-00 efc-bfc-01 efc-bfc-02 efc-bfc-03 efc-cpu-00 efc-cpu-01 efc-cpu-02 efc-cpu-03 efc-cpu-04 efc-cpu-05 efc-cpu-06}
control=$HOME/.ssh/control
mkdir -p "$control"
failures=0
for node in $nodes; do
  short=${node#efc-}; socket=$control/lotaas-${short//-/}      # as euroflash.cluster.ssh_args names it
  if ssh -S "$socket" -O check "$node" 2>/dev/null; then
    echo "up      $node"
  elif ssh -o ControlMaster=yes -o ControlPersist=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
          -o StrictHostKeyChecking=accept-new -S "$socket" -fN "$node"; then
    echo "opened  $node"
  else
    echo "FAILED  $node"; failures=1
  fi
done
exit $failures
