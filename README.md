# LOTAAS reprocessing pipeline

A single-pulse and periodicity search over LOTAAS beams from the LOFAR LTA,
run on the EuroFlash cluster. It stages and retrieves PSRFITS from the archive,
converts and flatfields it, dedisperses on GPU, searches with a boxcar matched
filter plus a zero-acceleration FFT/harmonic-summing search, clusters the
crossings and classifies single-pulse survivors with FETCH, and folds periodic
candidates into retained diagnostic plots. Each search has its own checkpoint
in the campaign ledger.

There is no Slurm. Work is dispatched over SSH to named GPU nodes, and each
stage runs inside an Apptainer image that is part of the run's identity.

## Layout

```
euroflash/              campaign orchestration: staging, dispatch, ledger, reporting
pipeline/               per-beam dedispersion, independent searches, finalization
preproc/                PSRFITS to filterbank conversion, and flatfielding
lotaas_reprocessing/    the science: dedispersion, matched filter, clustering, classifier
postproc/               Slack notification, and post-detection analysis
staging/                StageIT client and the staging entry point
db/                     per-beam classifier records
containers/             Apptainer definition and build lock
tests/                  the test suite
docs/                   operations, science, capacity
```

## Build the runtime

```bash
APPTAINER_TMPDIR="$HOME/.cache/lotaas-build" apptainer build --fakeroot \
  --mksquashfs-args '-processors 4' \
  containers/euroflash-runtime.sif containers/euroflash.def
```

`containers/euroflash.lock` records the installed environment and should be
regenerated for a new build. The image is hashed by content into every run
fingerprint, so rebuilding it starts a new generation of work: finished
results keep their provenance, but in-flight resume state does not carry over.

## Quick start

Run the campaign continuously: stage a rolling window of SAPs, retrieve each file
as soon as dCache has it on disk, convert it and delete the raw data, flatfield
each complete SAP and search it on the GPU nodes.

```bash
setsid nohup python3 -m euroflash.campaign run --root RESULTS_ROOT/campaign \
  --inventory RESULTS_ROOT/lt5_004-inventory.txt --ledger CAMPAIGN.sqlite \
  --dispatch-nodes efc-gpu-01 --control-dir ~/.ssh/control > campaign.log 2>&1 &
python3 -m euroflash.campaign status --root RESULTS_ROOT/campaign
```

Or retrieve a batch, prepare it locally, then dispatch the search by hand:

```bash
python3 staging/stage_and_extract.py srm_list.txt stage-directory --submit-only
python3 -m euroflash.stage_campaign stage-directory \
  --destination DATA_DIR --ledger CAMPAIGN.sqlite --workers 2

python3 -m euroflash.run --input DATA_DIR --work PREPARED \
  --ledger CAMPAIGN.sqlite --prepare-only --preprocess-workers 4 --delete-raw

python3 -m euroflash.cluster --input PREPARED/data --work RESULTS \
  --ledger CAMPAIGN.sqlite --nodes efc-gpu-01 --control-dir ~/.ssh/control \
  --run-name my-batch --gpus 0,1 --workers-per-gpu 3 --cpu-workers 24
```

Then report, and announce candidates from the head node:

```bash
python3 -m euroflash.status CAMPAIGN.sqlite
python3 -m postproc.notify_candidates RESULTS --ledger CAMPAIGN.sqlite
```

## Tests

```bash
apptainer exec --cleanenv --env PYTHONPATH="$PWD" \
  containers/euroflash-runtime.sif python -m pytest -q tests
```

GPU tests skip on the head node. A pass there is not GPU validation; run the
suite with `--nv` on a working GPU node to exercise CuPy:

```bash
apptainer exec --nv --cleanenv --env PYTHONPATH="$PWD" \
  containers/euroflash-runtime.sif python -m pytest -q -rs tests
```

## Documentation

- [docs/operations.md](docs/operations.md) — nodes, staging, running a campaign,
  monitoring, notifications, and reclaiming disk
- [docs/science.md](docs/science.md) — what the search does, what has been
  validated, and the limitations that still stand
- [docs/capacity.md](docs/capacity.md) — measured stage timings and what they
  do and do not imply for a one-year campaign
- [docs/periodicity-validation.md](docs/periodicity-validation.md) — combined GPU runs,
  real pulsar recovery, failure recovery and measured periodicity cost
- [SECURITY.md](SECURITY.md) — credential handling, and a token still awaiting
  rotation

## A note on scope

Results from this pipeline are search output, not vetted detections. A Slack
post is an alert. The clustering and classification policy still needs
scientific review before any survey candidate claim, and no measurement here
establishes survey sensitivity. `docs/science.md` is specific about which
claims are supported and which are not.

The Slurm entry points this repository began with (`master.sh`, `bin/*.slurm`,
`pipeline/pipeline.py`) targeted a cluster no longer in use and were removed
once the EuroFlash path became the only one exercised. They remain in git
history before the cleanup commit.
