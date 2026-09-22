# EuroFlash port

This checkout is based on `DirkKuiper/lotaas_pipeline` main at `117cdd6` (the
StageIT/LTA API version). Use the commands below from the repository root.
The old Slurm entry points remain for reference; EuroFlash commands do not
need Slurm. The classifier itself sends no Slack notifications; candidate
plots are posted from the head node by `postproc.notify_candidates`
(see **Candidate notifications**).

## Runtime and nodes

Build the complete runtime:

```bash
APPTAINER_TMPDIR="$HOME/.cache/lotaas-build" apptainer build --fakeroot \
  --mksquashfs-args '-processors 4' containers/euroflash-runtime.sif containers/euroflash.def
```

`euroflash.def` pins the main science packages and the YOUR/FETCH commits.
The image uses Python 3.11, CuPy/CUDA 12.8 libraries and CPU TensorFlow for
classification. Run CuPy numerical tests with `apptainer exec --nv`; a head-node
pass with GPU tests skipped is not GPU validation. `containers/euroflash.lock`
records the installed environment and should be regenerated for a new build.
`euroflash-runtime.def` is only the incremental recipe used to repair the first
CUDA build; the main definition contains the same runtime path correction.

Available non-BFC compute nodes are `efc-cpu-00`–`efc-cpu-07` and
`efc-gpu-00`–`efc-gpu-01`. Each GPU node has two RTX PRO 6000 Blackwell Server
Edition GPUs (about 96 GB each), 96 physical CPU cores and about 750 GiB RAM.
Storage on the compute nodes is local. Do not assume `/home` or `/shared/results`
is shared. BFC nodes are excluded from the cluster runner.

The September 21 check found CUDA working on `efc-gpu-01`. On `efc-gpu-00`,
`modprobe -n -v nvidia_uvm` reports a `modulejail` block and host `cuInit(0)`
returns 999. An administrator must enable the UVM module there. Installing
another container will not repair that host condition. Until it is repaired,
use only `efc-gpu-01` for CUDA work.

Forward the laptop's SSH agent to the head node. Optional persistent connections
avoid repeatedly depending on the forwarded agent:

```bash
mkdir -p ~/.ssh/control
ssh -M -S ~/.ssh/control/lotaas-gpu01 -o ControlPersist=86400 -fN efc-gpu-01
```

Private keys and StageIT credentials are never copied into the compute image or
source archive. A private StageIT config is at `~/.config/lotaas/stagingrc`, or
set `LOTAAS_STAGING_CONFIG`. Its contents are `api_token = ...`; use mode 0600.
Credentials embedded in the original public checkout were removed from the
working files and the path is now ignored, but the token remains in the
history of three pushed branches and **is live until rotated at the LTA**.
See [SECURITY.md](../SECURITY.md).

## LTA staging and inventory

Start with LT5_004. The recovered inventory currently contains 64,507 distinct
archive URLs from the account's complete StageIT request history. This is a
**known-file inventory, not a verified complete LTA catalogue**. Files are
marked `unknown` until availability is checked; successful retrieval updates
that state. Historical placeholder names ending in `_sample.tar` are excluded.

The September 22 public DBView audit found 222,072 valid named PULP metadata
records, including legacy products and different data types. Normalizing the
archive hash suffix gives 221,857 logical filenames; 157,574 have no matching
archive URL in the recovered history. These metadata records do not prove tape
availability, nor that every format is supported by this PSRFITS pipeline.
Cached pages and the detailed reconciliation are in
`/shared/results/dkuiper/lotaas/catalogue/valid-pulp/audit.json`.

```bash
python3 -m euroflash.catalogue --project LT5_004 \
  --cache /shared/results/dkuiper/lotaas/catalogue/valid-pulp \
  --inventory /shared/results/dkuiper/lotaas/lt5_004-inventory.txt
```

```bash
python3 -m euroflash.inventory --project lt5_004 \
  --output /shared/results/dkuiper/lotaas/lt5_004-inventory.txt \
  --ledger /shared/results/dkuiper/lotaas/campaign.sqlite
python3 staging/stage_and_extract.py srm_list.txt stage-directory --submit-only
python3 -m euroflash.stage_campaign stage-directory \
  --destination /shared/results/dkuiper/lotaas/data/batch \
  --ledger /shared/results/dkuiper/lotaas/campaign.sqlite --workers 2
```

Each staging directory persists its request ID. Rerun against the same directory
to resume that request instead of submitting another. The streaming campaign
retriever downloads files as they become online, uses site-specific macaroons,
checks HTTP lengths, computes SHA256 receipts, extracts only safe regular FITS
members and records retrieval failures. It refuses to call a partial staging
request a complete success. The original tar and FITS are retained.

Choose complete SAP batches including central beams 13–73. Keep a bounded
staging window; do not stage the whole campaign at once. The pilot request is
`991613`; the 61-central-beam validation request is `991614`, observation archive
ID L1163405, whose PSRFITS observation ID is L559289. Both identifiers matter.
All 61 were retrieved and flatfielded successfully. Request `991618` covers the
remaining 161 products of this three-SAP observation. Data from that request go
to `data/lt5_004-observation-remaining`; they must be grouped with the first SAP's
central beams before processing its remaining outer/incoherent beams.

## Local preparation and multi-node execution

```bash
python3 -m euroflash.run --input /path/to/extracted-fits \
  --work /path/to/prepared-batch --ledger /path/to/campaign.sqlite \
  --prepare-only --preprocess-workers 4
python3 -m euroflash.cluster --input /path/to/prepared-batch/data \
  --work /path/to/collected-results --ledger /path/to/campaign.sqlite \
  --nodes efc-gpu-01 --control-dir ~/.ssh/control \
  --run-name lt5-batch-001 --gpus 0,1 --cpu-workers 24
```

The local runner also processes already prepared files with `--prepared`, and
can use `--backend cpu` for validation. `--pilot` permits an incomplete central
beam set; add `--max-samples N` for a prefix test. `euroflash/pilot.yaml` deliberately
uses only a small DM range. **Pilot outputs are not production survey coverage.**
Production uses the unchanged eight-range `settings.yaml` plan (4,407 trials).
`--convert-only` converts files as they arrive without flatfielding an incomplete
SAP. Per-file conversion fingerprints let `--prepare-only` resume those outputs.
Every provided beam is processed; the central beams define the flatfield.

The cluster runner copies a source snapshot and image, distributes beams across
the selected nodes, starts one search worker per GPU, bounds the CPU queue,
and copies node results and database snapshots back. It records each attempt,
exit status, timing, device, log path and output file sizes. The head's campaign
DB imports attempts idempotently using the node/run identifier. Output paths
are unique per beam and code/settings fingerprint. Successful stage outputs are
checked before resuming; failures keep their diagnostics. Do not edit source
snapshots of running jobs. On an interrupted SSH session, inspect remote PIDs
and ledger state before restarting anything.
Dispatch/transfer failures are also recorded, including failures before a remote
pipeline starts. An SSH master expiring during overnight tape staging caused the
first full-SAP dispatch to fail; the September 22 retry uses a 24-hour master.

## Tracking and validation

Reclaim DM trials stranded by failed beams; about 3.6 GB each, and the CPU
stage only removes them on success:

```bash
python3 -m euroflash.reclaim --work COLLECTED_OR_NODE_WORK \
  --ledger /shared/results/dkuiper/lotaas/campaign.sqlite     # --apply to delete
```

```bash
python3 -m euroflash.status /shared/results/dkuiper/lotaas/campaign.sqlite
apptainer exec --cleanenv --env PYTHONPATH="$PWD" containers/euroflash-runtime.sif \
  python -m pytest -q tests
# Run this on a functioning GPU node:
apptainer exec --nv --cleanenv --env PYTHONPATH="$PWD" containers/euroflash-runtime.sif \
  python -m pytest -q -rs tests
```

The ledger contains `inputs`, `archive_receipts`, `archive_beams`, stage
`attempts`, classifier `beam_runs`, `detections` and imported-node mappings.
Archive receipts preserve request IDs, URLs, sizes and SHA256 hashes. Beam
mappings connect archive pipeline IDs to the original observation/SAP/beam
names inside the FITS files. The status report reconciles these identities
against successful production runs and excludes pilots. This is identity
history, not an assertion that different archive replicas are byte-identical.
Backfill older downloads with `python3 -m euroflash.provenance --input
DOWNLOADED_DIRECTORY --ledger CAMPAIGN.sqlite`; preparation also reconciles
available extraction receipts. Every failure has its own attempt
record; retries do not erase it. Per-beam metadata and run.json preserve the
processing choices. SQLite writes remain local to each node; a backup snapshot
is merged on the head. Do not run one SQLite writer database over unverified
cross-node filesystem locking.

The search uses rectangular rolling sums against a median-absolute-deviation
noise scale. Dividing each response by its own standard deviation let a pulse
inflate the denominator measuring it, so a 100-sigma pulse of 300 s scored
4.09 against a threshold of 5 and the long half of the width ladder was
computed but undetectable; the tanh-smoothed kernel returned 0.707 of the
ideal statistic at width one, where most single pulses are found; and the
circular convolution reported an event in the last ten samples as 81
detections in the first 0.1 seconds. Recovery is now 1.000 at every width
tested, none of those 81 remain, the statistic is still calibrated to unit
variance on Gaussian noise so the thresholds of 5 and 7 keep their meaning,
and a full-length trial searches 3.7x faster. **The CPU-stage timings below
predate this change and need re-measuring.**

Validated corrections include 2-bit values decoded as 0–3 (the old unpacking
left-shifted them), frequency headers derived from DAT_FREQ, time-domain
averaging before high-DM dedispersion, odd-length inverse FFTs, CPU failure exit
codes, bounded beam memory and distinct concurrent output directories. Tests
check dispersed pulse recovery, CPU/GPU agreement, the matched-filter statistic,
trial completeness, token/site association and failed/retried attempts.
DM values are generated with decimal arithmetic so adjacent plan ranges cannot
overwrite a boundary trial through floating-point rounding. Short pilot filter
windows are bounded by the observation duration. Constant signals yield no events.

Candidates are clustered on their position in the concatenated trial grid
rather than on DM/ddm, which was not monotonic across plan boundaries: DM 50.2
and DM 150.6 both mapped to 502, while adjacent trials DM 150.5 and DM 150.6
mapped to 1505 and 502. The DM and time axes each carry their own tolerance,
five trials and half a second, instead of sharing one epsilon of five that
merged a source repeating every four seconds into a single candidate. Points
with no neighbour are their own events; treating the DBSCAN noise label as a
cluster had reduced every isolated detection in a beam to one reported row.

The clustering implementation uses exact epsilon-connected components, which
are equivalent to the original DBSCAN with `min_samples=2`. It groups points in
small cells and tests neighboring cells with nearest-neighbor queries instead of
building a quadratic neighborhood graph. Tests compare its labels with sklearn
DBSCAN, including duplicate and boundary points. The candidate ceiling is 25
million per beam and is checked before plotting; compact numeric arrays replace
Python tuples to control memory usage. Every candidate remains in the
science table, while diagnostic scatter plots display at most 100,000 points.

Dedispersion is still circular Fourier dedispersion. A pulse in the last
samples of a beam is still recovered at the right time, with less bandwidth
as its sweep runs past the end, but whatever occupied the first samples of
the low-frequency channels reappears at the end of each trial: broadband
interference in the first three samples gives 56 sigma in the last 1089
samples at DM 100, against 13 in the uncontaminated middle. The search now
drops that tail, which costs about 0.4% of a one-hour beam at DM 150, 3% at
DM 1000 and 30% at DM 10000; pass `trim_wrap=False` to search the whole
series instead. The clustering and classification policy still requires
scientific review before final survey candidate claims. A partial flatfield
benchmark cannot establish survey sensitivity.

### User-supplied recovery target

The reference plot identifies **L603682, SAP1, beam 5**, at DM 26.20,
2465.653 seconds (sample 313524), S/N 12.48 and width one sample. Public
observation metadata maps L603682 to LT5_004 pointing LOTAAS-P1712C,
observed on 2017-08-17; its reprocessed archive pipeline is **L1263256**.
Request **991620** stages SAP1 beam 5 plus central beams 13–73. The downloaded
PSRFITS filenames must confirm the observation/SAP mapping before preparation.

The running continuation script is
`/shared/results/dkuiper/lotaas/benchmarks/l603682-recovery/run_when_ready.py`.
It waits for all 62 retrieval receipts, converts and flatfields the complete
central set, and searches beam 5 with the full production DM plan on GPU 1 of
efc-gpu-01. Its current state is in `progress.json` beside the script. Download
and workflow logs are `logs/stage-l603682-recovery.log` and
`logs/l603682-recovery-workflow.log` under the results root. Waiting for tape
does not establish pulse recovery.

`python3 -m euroflash.recovery --target TARGET.json --work COLLECTED_NODE_WORK
--output REPORT.json` checks threshold crossings, cluster representatives and
accepted detections separately. The target tolerances are 0.5 pc/cm³ in DM and
0.25 seconds in arrival time. A matching accepted detection and successful
search/classification stages are required for a recovery pass; the reference
S/N and FETCH scores are not imposed on new results. The report identifies the
run fingerprint and refuses ambiguous result directories.

## Candidate notifications

Candidate plots are posted to the Slack channel `#lotaas-cands` from the head
node, by a step that is deliberately outside the search:

```bash
python3 -m postproc.notify_candidates \
  /shared/results/dkuiper/lotaas/lt5-full-sap-results \
  --ledger /shared/results/dkuiper/lotaas/campaign.sqlite
```

Add `--watch` to keep posting while a cluster job runs, alongside
`euroflash.monitor`; `--dry-run` prints the messages without sending them.

The notifier is in `postproc/` and uses only the standard library, for two
reasons that are worth keeping. Adding `slack_sdk` would mean rebuilding
`euroflash-runtime.sif`, and the image identity is part of the run fingerprint
that records how each beam was searched, so a rebuild would invalidate resume
state and the provenance of finished work. Posting from the classifier would
also put a Slack token on the compute nodes, which the credential rules here do
not allow. Because the plots and the ledger rows are the durable record,
notification can be replayed for work that has already finished; the classifier
in `lotaas_reprocessing/classify.py` is unchanged and still only writes plots
and database rows.

Credentials follow the StageIT convention: `~/.config/lotaas/slackrc` at mode
0600, or `LOTAAS_SLACK_CONFIG`, containing `bot_token` and `channel_id`;
`SLACK_BOT_TOKEN` and `SLACK_CHANNEL_ID` override the file. The bot needs
`chat:write`, `files:write` and `channels:read`, and must be a member of the
channel, which a sending run checks before it posts. No token is stored in the
repository, the source snapshot or the image.

Each candidate is announced once. The send key is the beam, DM, width and S/N,
not the file path, so re-searching a beam under a new fingerprint does not
repeat an announcement; sends are recorded in `slack_notifications` in the
campaign ledger. Only candidates with a matching `candidate` row in that ledger
are posted, which keeps injection tests and other synthetic plots out of the
channel unless `--unrecorded` asks for them, and those are labelled. `--limit`
(25 by default) bounds one pass, so a first run over a large backlog does not
empty it into the channel at once. A Slack outage is logged and retried, and
never fails a search: the pipeline does not depend on notification succeeding.

**A Slack post is an alert, not a validated detection.** The candidate rules are
unchanged from the classifier except for the redetection veto (DM >= 10,
S/N > 7, no ATNF pulsar within both 0.5 pc/cm3 and
`LOTAAS_ATNF_VETO_RADIUS_DEG`, one degree by default, FETCH probability
> 0.5). The veto previously matched on DM alone across the whole 5-degree
query cone, recording genuinely new sources as redetections, and the clustering and classification policy
still require the scientific review noted above.

## One-year capacity

```bash
python3 -m euroflash.capacity --beams 64507 --gpus 2 \
  --utilization 0.7 --gpu-seconds MEASURED_GPU_SECONDS \
  --cpu-seconds MEASURED_CPU_SECONDS --cpu-workers 24 \
  --bytes-per-beam 4885500000 --download-mbps 500
```

At this inventory size the target is 177 beams/day. With two GPUs and an assumed
70% usable duty cycle, the GPU budget is 684 seconds per beam (1,369 seconds
with four GPUs). Two complete one-hour pilot beams each took about 40 seconds
for GPU preprocessing/dedispersion. End-to-end timing, tape availability, full
batch throughput and complete catalogue coverage must also be established.
Capacity output labels partial measurements as a lower bound and never treats
the transfer time alone as an end-to-end forecast.

Measured pilot GPU stages took 39.2–39.6 seconds; filtering/classification took
199–210 seconds after filter-kernel caching (previously about 429 seconds).
Thirty concurrent-batch conversions took 225–264 seconds per beam with 12
workers, and the complete 61-beam flatfield took 224 seconds total. Downloading
online files averaged about 505 MB/s per transfer, but the 61-file tape request
took roughly 11 hours. Tape delivery is a separate constraint from HTTP speed.

`benchmarks/measured-capacity.json` records partial extrapolations at both 64,507
and 222,072 products. The measured compute stages suggest roughly 24 and 81 days
respectively at 70% utilization, using two GPUs, 24 CPU search workers and 12
conversion workers. These are lower bounds, not validated campaign forecasts:
the larger count includes mixed formats and sustained tape throughput is unproven.
The complete 61-beam run is at
`/shared/results/dkuiper/lotaas/lt5-full-sap-results`.

The first complete SAP search finished in 1,445 seconds on two GPUs: 54 beams
completed with `no_candidates`; seven hit the previous one-million-candidate
ceiling. All seven subsequently completed under `lt5-candidate-retry-20260922`,
using two CPU workers for these larger jobs. The combined result is 61
successfully searched central beams, across two code fingerprints; historical
failures remain in the ledger. `benchmarks/full-sap-validation.json` records
each successful run and checks that its collected output files have the
recorded sizes. This does not establish full three-SAP coverage or sensitivity.
The updated stage-capacity calculation is in `benchmarks/full-sap-capacity.json`:
median GPU time is 39.8 seconds and the slowest successful CPU stage took 260.8
seconds. At 70% duty cycle, the measured stages imply lower bounds of about
22 days for 64,507 files and 74 days for 222,072 products. Sustained tape rate
remains unmeasured, so the end-to-end forecast is deliberately unset.
An independent retry of the largest tested beam, L559289/SAP000/beam53,
completed with 9,345,098 threshold crossings: 39.5 seconds for dedispersion and
254.9 seconds for filtering/classification. Its evidence is in
`benchmarks/real-compact-validation/validation-report.json` under the results root.

A full-hour injection test recovered pulses at DM 83.2/time 800 s and DM
2204.8/time 2300 s after 1,184,234 threshold crossings. FETCH accepted the first
(probability 0.99968) and rejected the second. These are artificial test signals,
kept outside the campaign database. The recovery report is
`benchmarks/injection-scalable-validation/recovery-report.json` under the results
root. This is a functional check, not a complete sensitivity study.

Request `991619` is a bounded throughput test of 666 files from three complete
observations (about 3.25 TB). StageIT reported them online while SURF returned HTTP
403 "Permission denied for GET on path /lt5_004/1261459/...", though a
byte-range GET on an earlier request still succeeded (206). That is what a
path-scoped macaroon does when presented for a directory it does not cover:
this request spans three observations, so StageIT issues one macaroon per
path, and the downloader chose whichever expired last. Tokens are now ordered
by whether their path caveat covers the URL, and a 401 or 403 falls through
to the remaining tokens. Confirm against the live service before rerunning
the throughput test:

```bash
python3 -m euroflash.access_check 991619 --output access-check.json
```

It probes one URL per directory and reports the chosen macaroon's path
caveats and the HTTP status, printing no token. The diagnosis rests on the
recorded evidence in `benchmarks/download-access-check.json` and has not yet
been confirmed against SURF.

For live coverage while a node job is running:

```bash
python3 -m euroflash.monitor --run-name lt5-candidate-retry-20260922 \
  --work /shared/results/dkuiper/lotaas/lt5-failed-beams-retry \
  --ledger /shared/results/dkuiper/lotaas/campaign.sqlite \
  --control-dir ~/.ssh/control
```

The monitor reads a consistent SQLite backup over SSH every minute, merges it
idempotently and stops when the remote runner releases its execution lock.
