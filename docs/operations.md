# Operations

Running a campaign, from the archive to an announced candidate. Commands are
given from the repository root.

## Nodes

Non-BFC compute nodes are `efc-cpu-00`–`efc-cpu-07` and `efc-gpu-00`–`efc-gpu-01`.
Each GPU node has two RTX PRO 6000 Blackwell Server Edition GPUs (about 96 GB
each), 96 physical cores and about 750 GiB RAM. Storage on the compute nodes is
local: do not assume `/home` or `/shared/results` is shared. BFC nodes are
excluded from the dispatcher.

**`efc-gpu-00` cannot currently run CUDA.** `modprobe -n -v nvidia_uvm` reports
a `modulejail` block and host `cuInit(0)` returns 999. An administrator must
enable the UVM module; installing another container will not repair a host
condition. Until it is fixed, only `efc-gpu-01` is usable, which puts both GPUs
in every capacity estimate on one machine.

`euroflash.cluster` probes each node before splitting a batch, using the
presence of `/dev/nvidia-uvm` alongside `nvidia-smi`. An unusable node gets a
recorded dispatch failure explaining why, the batch is split across the rest,
and a run with no usable node fails before transferring anything.
`--skip-health-check` restores the old unconditional behaviour.

The allowlist exists to keep work off the BFC nodes and stays an opt-in. Set
`LOTAAS_ALLOWED_NODES` to a comma-separated list to widen it.

Forward your SSH agent to the head node. A persistent connection avoids
depending on the forwarded agent for the length of a long job:

```bash
mkdir -p ~/.ssh/control
ssh -M -S ~/.ssh/control/lotaas-gpu01 -o ControlPersist=86400 -fN efc-gpu-01
```

An SSH master expiring during overnight tape staging caused the first full-SAP
dispatch to fail, which is why the control persistence above is 24 hours.

## Inventory and staging

Start with LT5_004. The recovered inventory holds 64,507 distinct archive URLs
from the account's complete StageIT request history. This is a **known-file
inventory, not a verified complete LTA catalogue**. Files are `unknown` until
availability is checked; successful retrieval updates that state. Historical
placeholder names ending in `_sample.tar` are excluded.

A public DBView audit on 22 September found 222,072 valid named PULP metadata
records, including legacy products and other data types. Normalising the
archive hash suffix gives 221,857 logical filenames, of which 157,574 have no
matching archive URL in the recovered history. These records do not prove tape
availability, nor that every format is supported by this PSRFITS pipeline. The
reconciliation is in `catalogue/valid-pulp/audit.json` under the results root.

```bash
python3 -m euroflash.catalogue --project LT5_004 \
  --cache RESULTS_ROOT/catalogue/valid-pulp \
  --inventory RESULTS_ROOT/lt5_004-inventory.txt

python3 -m euroflash.inventory --project lt5_004 \
  --output RESULTS_ROOT/lt5_004-inventory.txt --ledger CAMPAIGN.sqlite
```

Then stage and retrieve:

```bash
python3 staging/stage_and_extract.py srm_list.txt stage-directory --submit-only
python3 -m euroflash.stage_campaign stage-directory \
  --destination RESULTS_ROOT/data/batch --ledger CAMPAIGN.sqlite --workers 2
```

Each staging directory persists its request ID; rerunning against the same
directory resumes that request rather than submitting another. The retriever
downloads files as they come online, uses the macaroon whose path caveat covers
each URL, checks HTTP lengths, computes SHA256 receipts, extracts only safe
regular FITS members and records retrieval failures. It refuses to call a
partial staging request a complete success. The original tar and FITS are kept.

Choose complete SAP batches including central beams 13–73, and keep the staging
window bounded — do not stage a whole campaign at once.

Known request IDs: `991613` is the pilot; `991614` is the 61-central-beam
validation (observation archive ID L1163405, PSRFITS observation ID L559289 —
both identifiers matter), all 61 retrieved and flatfielded; `991618` covers the
remaining 161 products of that three-SAP observation, which must be grouped
with the first SAP's central beams before its outer and incoherent beams are
processed.

### If a download returns 403

`euroflash.access_check` probes one URL per directory in a request and reports
which macaroon was selected, its path caveats and the HTTP status, printing no
token:

```bash
python3 -m euroflash.access_check 991619 --output access-check.json
```

Where a probe is refused it also reports whether another macaroon would have
been accepted, which separates a token-selection fault from a real denial. See
[capacity.md](capacity.md) for the request 991619 case.

## Preparing and running a batch

```bash
python3 -m euroflash.run --input EXTRACTED_FITS --work PREPARED \
  --ledger CAMPAIGN.sqlite --prepare-only --preprocess-workers 4

python3 -m euroflash.cluster --input PREPARED/data --work RESULTS \
  --ledger CAMPAIGN.sqlite --nodes efc-gpu-01 --control-dir ~/.ssh/control \
  --run-name lt5-batch-001 --gpus 0,1 --cpu-workers 24
```

The local runner also processes already prepared files with `--prepared`, and
`--backend cpu` for validation. `--pilot` permits an incomplete central-beam
set and `--max-samples N` reads only a prefix. `euroflash/pilot.yaml`
deliberately uses a small DM range: **pilot output is not production survey
coverage.** Production uses the eight-range `settings.yaml` plan, 4,407 trials.
`--convert-only` converts files as they arrive without flatfielding an
incomplete SAP; per-file conversion fingerprints let `--prepare-only` resume
those outputs. Every provided beam is searched, but the central beams define
the flatfield.

The dispatcher copies a source snapshot and the image to each node, distributes
beams, starts one search worker per GPU, bounds the CPU queue, and copies node
results and database snapshots back. It records every attempt with its exit
status, timing, device, log path and output sizes, including dispatch and
transfer failures that happen before a remote pipeline starts. The head's
ledger imports attempts idempotently under the node/run identifier.

Do not edit the source snapshot of a running job. After an interrupted SSH
session, inspect remote PIDs and ledger state before restarting anything.

## Run identity and resume

A run fingerprint is the content of the code, the settings, the image and the
run options. It deliberately excludes absolute paths, modification times and
the composition of the batch, so the same work has the same identity on the
head and on a node, and adding a beam to a batch does not invalidate the
others. Which bytes a beam name stood for is recorded separately, by the SHA256
archive receipts.

Output directories are unique per beam and fingerprint. A stage is skipped only
if its recorded outputs still exist at their recorded sizes. Failures keep
their diagnostics; a retry never erases the record of what failed.

## Monitoring

```bash
python3 -m euroflash.monitor --run-name lt5-batch-001 --work RESULTS \
  --ledger CAMPAIGN.sqlite --control-dir ~/.ssh/control
```

The monitor reads a consistent SQLite backup over SSH every minute, merges it
idempotently, and stops when the remote runner releases its execution lock.

SQLite writes stay local to each node and a backup snapshot is merged on the
head. Do not point one writer at a database over cross-node filesystem locking
that has not been verified.

## Reporting

```bash
python3 -m euroflash.status CAMPAIGN.sqlite
```

The ledger holds `inputs`, `archive_receipts`, `archive_beams`, stage
`attempts`, classifier `beam_runs`, `detections`, `slack_notifications` and
imported-node mappings. Archive receipts preserve request IDs, URLs, sizes and
SHA256 hashes. Beam mappings connect archive pipeline IDs to the
observation/SAP/beam names inside the FITS. The status report reconciles those
identities against successful production runs and excludes pilots. This is
identity history, not a claim that two archive replicas are byte-identical.

Backfill older downloads with:

```bash
python3 -m euroflash.provenance --input DOWNLOADED_DIR --ledger CAMPAIGN.sqlite
```

## Reclaiming disk

A beam writes about 3.6 GB across 8,814 trial files, removed only when its
search succeeds. Keeping them on failure is deliberate — it lets the search be
repeated without redoing dedispersion — but it is not free. The GPU stage
refuses a beam whose trials will not fit rather than filling the disk part way
through, and stranded trials can be reclaimed:

```bash
python3 -m euroflash.reclaim --work COLLECTED_OR_NODE_WORK \
  --ledger CAMPAIGN.sqlite          # add --apply to delete
```

It removes only trials whose beam has since been classified successfully under
the same fingerprint, or whose last attempt failed longer ago than the
retention window (seven days by default). It reads the ledger read-only and
reports before it deletes.

## Candidate notifications

Candidate plots are posted to `#lotaas-cands` from the head node, by a step
deliberately outside the search:

```bash
python3 -m postproc.notify_candidates RESULTS --ledger CAMPAIGN.sqlite
```

`--watch` keeps posting while a cluster job runs, alongside `euroflash.monitor`;
`--dry-run` prints the messages without sending them.

The notifier uses only the standard library, for two reasons worth keeping.
Adding `slack_sdk` would mean rebuilding the image, and the image is part of
the run fingerprint, so a rebuild would invalidate resume state and the
provenance of finished work. Posting from the classifier would also put a Slack
token on the compute nodes, which the credential rules do not allow. Because
the plots and ledger rows are the durable record, notification can be replayed
for work already finished.

Each candidate is announced once. The send key is the beam, DM, width and S/N
rather than the file path, so re-searching a beam under a new fingerprint does
not repeat an announcement. Only candidates with a matching `candidate` row are
posted, which keeps injection tests and other synthetic plots out of the
channel unless `--unrecorded` asks for them, and those are labelled. `--limit`
(25 by default) bounds one pass so a first run over a backlog does not empty it
into the channel at once. A Slack outage is logged and retried and never fails
a search.

Credentials follow the StageIT convention; see [SECURITY.md](../SECURITY.md).
The bot needs `chat:write`, `files:write` and `channels:read`, and must be a
member of the channel, which a sending run checks before it posts.

**A Slack post is an alert, not a validated detection.** See
[science.md](science.md) for what the candidate rules do and do not establish.
