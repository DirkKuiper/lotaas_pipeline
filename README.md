# LOTAAS reprocessing pipeline

A single-pulse search over LOTAAS beams from the LOFAR LTA, run on the
EuroFlash cluster. It stages and retrieves PSRFITS from the archive, converts
and flatfields it, dedisperses on GPU, searches with a boxcar matched filter,
clusters the crossings and classifies survivors with FETCH, recording every
stage in a campaign ledger.

There is no Slurm. Work is dispatched over SSH to named GPU nodes, and each
stage runs inside an Apptainer image that is part of the run's identity.

## Layout

```
euroflash/              campaign orchestration: staging, dispatch, ledger, reporting
pipeline/               the two per-beam stages, dedispersion then search
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

Retrieve a batch, prepare it locally, then dispatch the search:

```bash
python3 staging/stage_and_extract.py srm_list.txt stage-directory --submit-only
python3 -m euroflash.stage_campaign stage-directory \
  --destination DATA_DIR --ledger CAMPAIGN.sqlite --workers 2

python3 -m euroflash.run --input DATA_DIR --work PREPARED \
  --ledger CAMPAIGN.sqlite --prepare-only --preprocess-workers 4

python3 -m euroflash.cluster --input PREPARED/data --work RESULTS \
  --ledger CAMPAIGN.sqlite --nodes efc-gpu-01 --control-dir ~/.ssh/control \
  --run-name my-batch --gpus 0,1 --cpu-workers 24
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
