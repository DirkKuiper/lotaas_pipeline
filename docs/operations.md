# Operations

Running a campaign, from the archive to an announced candidate. Commands are
given from the repository root.

## Nodes

Non-BFC compute nodes are `efc-cpu-00`–`efc-cpu-07` and `efc-gpu-00`–`efc-gpu-01`.
Each GPU node has two RTX PRO 6000 Blackwell Server Edition GPUs (about 96 GB
each), 96 physical cores and about 750 GiB RAM. Storage on the compute nodes is
local: do not assume `/home` or `/shared/results` is shared. BFC nodes are
excluded from the dispatcher.

**Both GPU nodes run CUDA since 23 September 2026.** `efc-gpu-00` could not
until then: `modulejail` blocked `nvidia_uvm`. It has `/dev/nvidia-uvm` now and
both of its GPUs pass CUDA memory checks, so four GPUs on two nodes are usable.

**CPU nodes:** `efc-cpu-00`–`efc-cpu-06` were reachable on 23 September, each
with 96 physical cores, ~750 GiB RAM and apptainer; `efc-cpu-07` (and
`efc-cpu-08`, which is in DNS) answered "No route to host". `/shared/results`
is NFS-mounted from the head on every node. `/home` on a node is its own local
disk (~830 GB root filesystem), and `/scratch` (2.9 TB) belongs to root, so
runs are staged under `/home/$USER/lotaas-runs` on each node.

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
for node in efc-gpu-00 efc-gpu-01 efc-cpu-0{0..6}; do
  ssh -M -S ~/.ssh/control/lotaas-$(echo ${node#efc-} | tr -d -) -o ControlPersist=yes -fN $node
done
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

## Running the campaign continuously

`euroflash.campaign` keeps the whole campaign moving without hand-offs. It
holds a rolling window of SAP-sized StageIT requests in flight. It takes each
file as soon as **dCache** reports it on disk: download, SHA256 receipt,
extraction, conversion to a 32-bit filterbank. It then deletes the tar and
PSRFITS. A SAP is flatfielded once every beam is converted, in the background
so retrieval continues. Its unflattened filterbanks are then removed, and with
`--dispatch-nodes` it is searched on the GPU nodes with `euroflash.cluster`.

Once searched, a beam's flatfielded filterbank is removed unless something
was found in it: a candidate FETCH accepted, or a periodic fold that is
neither RFI-flagged nor a catalogued pulsar. Those beams are marked `kept`
for review, and `status.json` reports how many and how many terabytes. In the
first 136 coherent beams searched this was about 5%. Beams whose search
failed keep theirs for a retry. `--keep-prepared` keeps every searched beam.
Search products stay under `results/`: candidate plots, periodic folds, and
the node ledger.

```bash
python3 -m euroflash.campaign run --root RESULTS_ROOT/campaign \
  --inventory RESULTS_ROOT/lt5_004-inventory.txt --ledger CAMPAIGN.sqlite \
  --max-staging-saps 8 --staging-schedule RESULTS_ROOT/campaign/staging-schedule.json \
  --max-prepared-saps 100 --min-free-tb 10 --download-workers 4 \
  --dispatch-nodes efc-gpu-00 efc-gpu-01 --gpus 0,1 --workers-per-gpu 3 --cpu-workers 24 \
  --dispatch-saps 2 --batches-per-node 2 \
  --cpu-nodes efc-cpu-00 efc-cpu-01 efc-cpu-02 efc-cpu-03 efc-cpu-04 efc-cpu-05 efc-cpu-06 \
  --cpu-tier-workers 64 --control-dir ~/.ssh/control \
  --image CONTAINERS/euroflash-runtime.sif --settings RELEASE/settings.yaml

python3 -m euroflash.campaign status --root RESULTS_ROOT/campaign
```

Run the driver from a **pinned release worktree**, never from a checkout being
edited. Each dispatch copies the driver's own checkout to the nodes, and every
Python file under the code folders is part of the search fingerprint. On 23
September, edits between dispatches gave the campaign five fingerprints in one
day:

```bash
git worktree add ~/lotaas-release RELEASE_COMMIT
cd ~/lotaas-release && setsid nohup python3 -m euroflash.campaign run … &
```

### Two tiers: GPU nodes and CPU nodes

With `--cpu-nodes`, each batch is split between two machines:

1. **GPU node:** dedispersion and the single-pulse search (matched filter and
   clustering). As each beam finishes, the node marks it ready in
   `work/handoff/`.
2. **Head:** relays every ready beam to a CPU node through SSH pipes and then
   deletes it from the GPU node. What is relayed is the filterbank, clusters,
   metadata and periodic trials, about 5.2 GB. Nothing is written to shared
   storage on the way.
3. **CPU node:** FETCH classification and the periodic search, beam by beam as
   they arrive. A periodic search prunes every trial no sifted-best peak points
   to, freeing ~4 GB at once. When the batch is complete, the cross-beam veto
   runs, then folding and finalisation.

The GPU node is free for the next batch as soon as its own stages end. The
head records this in `results/<run>/<node>.gpu-done`, and the driver then
starts another batch there. Two batches, one on each GPU node, run at once. A
slow beam holds only its CPU node.

- CPU nodes are taken through flock slots in `campaign/.cpu-slots/`, one batch
  per node, shared by concurrent runs.
- A node without apptainer or 300 GB free is skipped for 10 minutes.
- The head keeps at most `--relay-backlog` (40) relayed-but-unsearched beams on
  a CPU node.
- Each node keeps its own ledger. Both snapshots are merged into the campaign
  ledger and results are collected from the CPU node.

Without `--cpu-nodes`, a GPU node runs every stage itself, as before.

**Stage limits.** Every stage runs in its own process group, and a stage past
its limit is killed, children included:

| Stage | Limit |
| --- | --- |
| dedisperse | 30 min |
| single_pulse | 60 min |
| sp_classify | 45 min |
| periodicity | 60 min |
| periodicity_fold | 30 min |
| classify | 15 min |

The beam fails and its SAP goes to `attention`, keeping the filterbank for a
retry. On 23 September two FETCH runs over 600 s-wide clusters held a batch for
five hours. `euroflash.run --stage-timeout STAGE=SECONDS` overrides a limit.

### The staging window experiment

`--staging-schedule` changes `--max-staging-saps` at set times. Download
workers stay at four: transfers of online files are fast, and SURF has
answered HTTP 429. The experiment runs windows of 8, 16 and 32 SAPs for 30–36
hours each; the schedule file is `campaign/staging-schedule.json`:

```json
{"phases": [
  {"start": "2026-09-23T06:00:00Z", "max_staging_saps": 8,  "label": "window-8"},
  {"start": "2026-09-24T12:00:00Z", "max_staging_saps": 16, "label": "window-16"},
  {"start": "2026-09-26T00:00:00Z", "max_staging_saps": 32, "label": "window-32"}]}
```

Each change is logged as a `staging_window` event and shown in `status.json`.
The report gives delivered bytes (SHA256 receipts of successful downloads),
files, complete SAPs, 429/503 throttling and restages per phase. It gives each
phase whole and again without its first hours, while earlier requests still
arrive:

```bash
python3 -m euroflash.staging_report --root RESULTS_ROOT/campaign --ledger CAMPAIGN.sqlite \
  --schedule RESULTS_ROOT/campaign/staging-schedule.json --ramp-hours 6
```

Do not read `files_converted_last_24h` in `status.json` as throughput. It
counts files still in the `converted` state, not those already searched. On
23 September it showed 887 while the receipts showed about 2,500–2,950 files
(13–15 TB) a day.

Run it detached (`setsid nohup … &`) so it survives a closed session. State is
kept in `campaign-state.sqlite` under the root, and a restart resumes where it
stopped. Create `STOP` in the root, or send SIGTERM, to stop after the current
step. `status.json` shows files and SAPs by state, active requests, the last
24 hours' conversions, free space, SAPs needing attention and recent events.

Everything is bounded:

- `--max-staging-saps`: requests in flight.
- `--max-prepared-saps`: flatfielded SAPs waiting for a search.
- `--min-free-tb`: below this, no new SAP is admitted.

**Beam 12 is the incoherent beam and is excluded everywhere.** Each SAP holds
74 beams: 73 coherent tied-array beams and one incoherent sum of the stations,
which the archive files under `incoherentstokes/`. In every SAP retrieved so
far that is beam 12. Its sensitivity, beam pattern and RFI response differ, and
it is not part of the central flatfield. Flatfielded against the coherent mean,
it overran the 25-million-crossing single-pulse guard.

- The campaign never stages it.
- `euroflash.run` never converts or searches it; neither does
  `euroflash.cluster`.
- Any archive member under `incoherentstokes/` is excluded whatever its number.
- `--exclude-beams` changes the list; the default is `12`.

SAPs without all central beams 13–73 in the inventory are marked
`incomplete` and never admitted. A central beam that cannot be retrieved
after `--max-submissions` requests or `--max-failures` download attempts puts
its SAP in `attention`. HTTP 429 or 503 from SURF pauses downloads for
`--throttle-seconds` without counting against the file.

**StageIT's "online" is only a hint.** Request 991619 was reported fully online
ten minutes after submission, while dCache still held every file on tape only.
The driver asks the dCache namespace API for each file's locality, using the
request's own macaroon, and downloads only files that are `ONLINE` or
`ONLINE_AND_NEARLINE`. Files still on tape `--restage-after-hours` after
StageIT reported the request finished are requested again. A download refused
with 401/403 goes back to waiting for dCache rather than failing.

A driver restarted while its cluster run is still going adopts that run and
follows it to completion instead of dispatching the same SAPs again.

Dispatch needs SSH to the GPU node without the forwarded agent that a detached
process loses. Keep the control master alive, and re-create it if it has
expired, as described under Nodes. A dispatch that cannot reach the node puts
its SAPs back and backs off. Staging continues until `--max-prepared-saps` SAPs
are waiting.

### Staging and preparing by hand

The single-request tools remain for targeted work:

```bash
python3 staging/stage_and_extract.py srm_list.txt stage-directory --submit-only
python3 -m euroflash.stage_campaign stage-directory \
  --destination RESULTS_ROOT/data/batch --ledger CAMPAIGN.sqlite --workers 2 --delete-archive
```

Each staging directory persists its request ID; rerunning against the same
directory resumes that request rather than submitting another. The retriever
downloads files as they come online and checks HTTP lengths. It computes
SHA256 receipts, extracts only safe regular FITS members and records retrieval
failures. It refuses to call a partial staging request a complete success.
`--delete-archive` removes each tar once extracted; the receipt keeps its
size and hash.

Choose complete SAP batches including central beams 13–73, and keep the staging
window bounded — do not stage a whole campaign at once.

Known request IDs:

- `991613` is the pilot.
- `991614` is the 61-central-beam validation: all 61 retrieved and flatfielded.
  Its observation archive ID is L1163405 and its PSRFITS observation ID
  L559289; both identifiers matter.
- `991618` covers the remaining 161 products of that three-SAP observation.
  They must be grouped with the first SAP's central beams before its outer
  and incoherent beams are processed.
- `991619` was reported online but never left tape.

### If a download returns 403

The usual cause is a file that is not on disk: SURF's WebDAV door refuses to
read from tape with `403 Permission denied`, whatever the token. Check the
file's locality first:

```bash
curl -s -H "Authorization: Bearer $MACAROON" \
  "https://dcacheview.grid.surfsara.nl:22880/api/v1/namespace/pnfs/grid.sara.nl/data/lofar/ops/projects/lt5_004/OBS/FILE.tar?locality=true"
```

`NEARLINE` means tape only: request it again, and report a request StageIT
calls online if dCache disagrees. For request 991619, every sampled file was
`NEARLINE`. Its single macaroon covered the whole projects tree and read other
observations' files, and another request's working token was refused on these
files. So it was not a token-selection fault, as first suspected.
`euroflash.access_check` still reports each directory's selected macaroon, its
path caveats and HTTP status, without printing a token:

```bash
python3 -m euroflash.access_check 991619 --output access-check.json
```

## Preparing and running a batch

```bash
python3 -m euroflash.run --input EXTRACTED_FITS --work PREPARED \
  --ledger CAMPAIGN.sqlite --prepare-only --preprocess-workers 4 --delete-raw

python3 -m euroflash.cluster --input PREPARED/data --work RESULTS \
  --ledger CAMPAIGN.sqlite --nodes efc-gpu-01 --control-dir ~/.ssh/control \
  --run-name lt5-batch-001 --gpus 0,1 --workers-per-gpu 3 --cpu-workers 24
```

Conversion sums the raw 2-bit values over each 16-sample group as integers
before applying scale, offset and weights. That is 25× less floating-point
work than scaling every sample, and a full beam takes ~14 s instead of
~240–300 s. The output agrees with the per-sample definition to float32
rounding; rows with non-finite scales or `--dc` still take that path.
`--delete-raw` deletes each PSRFITS and its tar only after its conversion has
succeeded, and records the deletion in the extraction marker, so
reconciliation and retrieval resume treat it as deliberate. Flatfielding uses
every converted beam in the SAP directory, including those from earlier
`--convert-only` runs whose raw data is already gone.

The local runner also processes already prepared files with `--prepared`, and
`--backend cpu` for validation. `--pilot` permits an incomplete central-beam
set and `--max-samples N` reads only a prefix. `euroflash/pilot.yaml`
deliberately uses a small DM range: **pilot output is not production survey
coverage.** Production uses the eight-range `settings.yaml` plan, 4,407 trials.
`--convert-only` converts files as they arrive without flatfielding an
incomplete SAP; per-file conversion fingerprints let `--prepare-only` resume
those outputs. Every provided beam is searched, but the central beams define
the flatfield.

The dispatcher copies a source snapshot and the image to each node and
distributes beams. It runs `--workers-per-gpu` dedispersion workers on each GPU
(default 3), bounds the CPU queue, and copies node results and database
snapshots back. One worker left a card idle for ~70% of its slot while the
same process masked, detrended, plotted and wrote on the CPU. Three sharing a
GPU measured 26.8 s per beam against 47.4 s, at ~13 GB of GPU memory each.
`--skip-trials` leaves failed beams' DM trials (~4.7 GB each) on the node
instead of copying them back; a retry regenerates them. `--cleanup-remote`
removes the run's input, work and source copy from the node once its results
and ledger are collected. The dispatcher records every attempt with its exit
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

With periodicity enabled, a representative full beam writes about 4.88 GB of
unique trial payload (shared trials are hard-linked), removed only when its
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

### Raw data left by earlier retrievals

Tars and PSRFITS from `stage_campaign` runs made before raw data was deleted
automatically can be removed once they are no longer needed:

```bash
python3 -m euroflash.reclaim --raw RESULTS_ROOT/data --ledger CAMPAIGN.sqlite   # add --apply
```

An archive qualifies when the ledger shows its beam converted, with the
filterbank still on disk, or searched in production, or when it is the
excluded incoherent beam. Anything without such evidence is kept. Receipts
and extraction markers stay. On 23 September this reclaimed 1.79 TB: all 178
archives of the earlier manual staging.

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


## Combined single-pulse and periodicity searches

`settings.yaml` enables both branches. The ledger records `dedisperse`,
`single_pulse`, `periodicity`, and the aggregate `classify` completion. Disabling
`periodicity.enabled` retains the single-pulse-only path. Each CPU branch has its
own process, log, timing and checkpoint; a failure does not suppress the other
search. `euroflash.status` reports periodicity completion by pilot/production
scope. All enabled branches must succeed before trial cleanup.

The runtime writes `metadata.json` for the standard-library host orchestrator,
plus `metadata.yaml` for science tools. SHA256 trial manifests cover both DM
directories. Missing, truncated or same-size corrupted trials trigger GPU
regeneration; existing hard links are replaced on regeneration. A successful
search branch can resume independently while its peer retries.

Periodicity outputs include raw/sifted JSONL, a per-DM coverage report, folded
candidate JSONL, and PNG/NPZ diagnostics. The shortlist is bounded by `max_folds`.
Every template of every trial is searched. A trial with more than
`max_candidates_per_trial` peaks keeps its strongest distinct frequencies, one
per Fourier bin. A beam with more than `max_candidates_per_beam` sifts its
strongest, halving that set if sifting would exceed `max_sift_comparisons`.
The periodicity summary records every count: peaks found, dropped in crowded
trials, and left unsifted, with the reason. Unsifted rows stay in the sifted
JSONL. Aborting instead meant a bright pulsar (J0323+3944) ended the search at
DM 25.7 with nothing above it covered. Raw evidence already written remains
available on any other failure.

Post completed periodic plots explicitly from the head node:

```bash
python3 -m postproc.notify_periodicity RESULTS --ledger CAMPAIGN.sqlite \
  --limit 1 --dry-run
```

Remove `--dry-run` to send. `--known-only` selects catalogue associations;
`--include-pilot` permits clearly labelled validation plots. Sending requires a
successful periodicity ledger attempt and intact completion products. Tokens
stay on the head node and are never copied to the compute nodes. Notifications
are recorded once per beam/run/plot.
