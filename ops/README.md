# Continuous campaign operations

The live configuration is `~/.config/lotaas/campaign.json`. Cron starts
`ops/supervise.py` each minute and at boot. Its exclusive lock allows only one
supervisor. It restarts a failed driver after 30 seconds; the campaign's own
lock prevents a second driver. Existing cluster processes are adopted after
a driver restart. `campaign/supervisor.json` records the live PID and heartbeat.

Create `campaign/STOP` before stopping the driver to prevent automatic restart.
The driver finishes its current work and waits for its cluster processes.
Remove STOP to resume through the next cron invocation. Never launch a second
driver against the same state database.

`production-settings.yaml` preserves the earlier production search range and
classification thresholds; it imposes no width or candidate-count cut.
Timeouts record failed beams and retain their input for investigation.
Completed CPU beams finalize immediately when cross-beam vetoing is disabled.
The cluster importer updates the central ledger every minute from consistent
snapshots, while final result files are collected before input reclamation.

Staging uses both a hard SAP limit and an outstanding-file target. Late files
can keep their SAP open while replacement SAPs are admitted, within disk,
prepared-queue and SAP limits. The target may be exceeded by at most the size
of one newly admitted SAP.

SSH reconnection remains an administrator prerequisite on this cluster:
sshd uses `/etc/ssh/authorized_keys/%u` and SSSD, and ignores user
`~/.ssh/authorized_keys`. Register the public key from
`~/.ssh/id_ed25519_lotaas_cluster.pub` for this account on the allowed compute
nodes, restricted to the head node's source address. Do not copy a private key
to any compute node. Test with `IdentityAgent=none` and `ControlPath=none`.

Ingest runs on CPU nodes (`--ingest-nodes`, `euroflash/ingest.py`): each LTA
archive is streamed from SURF through tar into the node's RAM-backed /tmp,
converted there, and only its 32-bit filterbank is copied into
`prepared/data` on /shared; the head then flatfields and dispatches as
before. The nodes run the code the driver started with, sent once to
`~/lotaas-ingest/<digest>` on each node, so a change to conversion code
reaches them at the next driver restart. Ingest has its own SSH master per
node (`~/.ssh/control/ingest-<node>`), so it never takes the search tier's
sessions; `ops/open-masters.sh` opens both kinds (after a cluster reboot, run
it with your agent forwarded). A node ssh cannot reach rests for 10 minutes;
with none left the head ingests as before, so a missing master only costs
head load. Failed jobs keep their output in `campaign/logs/ingest/<item>.log`.

The Judge's interference lines are kept in `campaign/judge-lines.json`, so a
driver restart reads them instead of working them out for the whole campaign
(45 minutes). Deleting the file is safe; the next start rebuilds it.
