# Capacity

What has been measured, and what it does and does not imply for a one-year
campaign.

## Current measurements, 22–23 September

These replace the older stage timings further down, which are kept for their
provenance. Measured on efc-gpu-01 and the head node with the current code and
the production image; each number is for one hour-long beam.

| stage | as it was | now |
| --- | --- | --- |
| conversion, PSRFITS → filterbank | 213 s alone, 240–300 s at 12-way | **~14 s** (integer-first reduction) |
| GPU stage, per GPU | 47–52 s per beam, GPU busy ~15 s | **26.8 s per beam** with 3 workers sharing each GPU |
| periodicity search | 227 s (FFT 88 s of it) | **128–131 s** (fast FFT lengths, no false edge peaks) |
| single-pulse matched filter | ~65 s | unchanged |
| FETCH, per candidate | 1.4 s at DM 10 → 40 s at DM 8,000; ~23 s per beam to load | unchanged; ~3.4 s per beam on the measured SAP |
| download | ~480–530 MB/s per stream | unchanged; 4 streams, HTTP 429 now backs off |
| raw data kept per beam | ~11.3 GB (tar + FITS + two filterbanks) | receipts only; flatfielded filterbank until searched |

The periodicity search is memory-bound. On one 96-core node, periodicity
throughput peaks at ~24 concurrent searches and falls beyond. The smooth-length
FFTs double that peak. Add capacity with more nodes, not more workers per node.

**Tape recall decides the campaign length.** On 22 September, 595 of 600
randomly sampled LT5_004 files were `NEARLINE` (tape only). One ~60-file request
at a time completed in 8–11 hours: about 0.65–1.1 TB a day, or roughly a year
for the ~315 TB inventory. After these changes compute needs about 6 days on two
GPUs. `euroflash.campaign` now keeps several SAP requests in flight; the rate
it sustains is the number to measure next.

## The calculation

```bash
python3 -m euroflash.capacity --beams 64507 --gpus 2 \
  --utilization 0.7 --gpu-seconds MEASURED_GPU_SECONDS \
  --cpu-seconds MEASURED_CPU_SECONDS --cpu-workers 24 \
  --bytes-per-beam 4885500000 --download-mbps 500
```

At 64,507 files the target is 177 beams/day. With two GPUs at an assumed 70%
duty cycle the GPU budget is 684 seconds per beam, or 1,369 with four GPUs.
The tool labels partial measurements as lower bounds and never treats transfer
time alone as an end-to-end forecast.

## Measured stages

| stage | measurement |
| --- | --- |
| GPU preprocessing and dedispersion | 39.2–39.6 s per beam, median 39.8 over a full SAP |
| filtering and classification (CPU) | 199–210 s, slowest successful 260.8 s |
| conversion | 225–264 s per beam, 12 workers, 30 concurrent |
| flatfield, 61 beams | 224 s total |
| download of online files | about 505 MB/s per transfer |
| tape delivery, 61-file request | roughly 11 hours |

Tape delivery is a separate constraint from HTTP speed, and the binding one.

The first complete SAP search finished in 1,445 s on two GPUs: 54 beams
returned `no_candidates` and seven hit the then one-million-candidate ceiling.
All seven completed under `lt5-candidate-retry-20260922` with two CPU workers.
The combined result is 61 successfully searched central beams across two code
fingerprints; the historical failures remain in the ledger.
`benchmarks/full-sap-validation.json` records each successful run and checks
its collected outputs against the recorded sizes. This does not establish
three-SAP coverage or sensitivity.

An independent retry of the largest tested beam, L559289/SAP000/beam53,
completed with 9,345,098 threshold crossings: 39.5 s dedispersion and 254.9 s
filtering and classification. Evidence in
`benchmarks/real-compact-validation/validation-report.json`.

## Forecasts, and why they are lower bounds

`benchmarks/measured-capacity.json` and `benchmarks/full-sap-capacity.json`
record extrapolations at 64,507 and 222,072 products: roughly **22 days** and
**74 days** at 70% utilisation with two GPUs, 24 CPU search workers and 12
conversion workers.

These are lower bounds on compute alone, not campaign forecasts:

- The 222,072 count includes mixed formats this PSRFITS pipeline may not read.
- Sustained tape throughput is unmeasured, so the end-to-end forecast is
  deliberately unset.
- Both GPUs in the estimate live on `efc-gpu-01`, since `efc-gpu-00` cannot
  initialise CUDA. There is no redundancy behind these numbers.
- The CPU stage that dominates them has since changed by 3.7×.

## Injection tests

A full-hour injection recovered pulses at DM 83.2/800 s and DM 2204.8/2300 s
after 1,184,234 threshold crossings. FETCH accepted the first (0.99968) and
rejected the second. Report in
`benchmarks/injection-scalable-validation/recovery-report.json`. These are
artificial signals kept outside the campaign database, and a functional check
rather than a sensitivity study.

## The throughput test that measured nothing

Request `991619` was a bounded throughput test of 666 files from three complete
observations, about 3.25 TB. StageIT reported all 666 online ten minutes after
submission. SURF returned HTTP 403 `Permission denied for GET on path
/lt5_004/1261459/...`.

The first diagnosis was a path-scoped macaroon presented for a directory it did
not cover. The live service disproved it on 22 September:

- The request carries one macaroon, scoped to the whole projects tree.
- That macaroon reads other observations' files.
- Another request's working macaroon is refused on these files.
- dCache's namespace API reports every sampled file of the request as
  `NEARLINE`: on tape, never staged.

The door refuses reads from tape with 403. The token ordering added for the
first diagnosis is harmless, but the fix is to trust dCache locality rather
than StageIT's online list. `euroflash.campaign` does that. These observations
have since been requested again through the driver and are coming online.

## Periodicity capacity measurement

Measure the complete periodic CPU path inside the runtime:

```bash
apptainer exec --cleanenv --env PYTHONPATH="$PWD" \
  containers/euroflash-runtime.sif python -m euroflash.periodicity_benchmark \
  PERIODIC_DM_TRIALS --metadata METADATA_YAML --max-trials 100 \
  --output periodicity-benchmark.json
```

Omit `--max-trials` for a full-beam measurement. A bounded sample spans the
numeric DM grid and is labelled as sampled coverage. This includes reading,
search, sifting, folding, plotting and product writing; it excludes GPU
production and the single-pulse branch. Benchmark work never updates a campaign
ledger. The per-beam summary also separates search, sift and fold timings.

At 457,728 input samples the production settings generate 7,856 periodic trials,
sharing 1,844 exact `(DM, downsample)` pairs with the 4,407 single-pulse trials.
Unique trial payload grows from 3.60 GB to 4.88 GB per beam, excluding small
metadata files. The combined disk guard includes this extra payload. Worker
counts must be chosen from full-beam measurements, including peak process RSS,
RFI candidate volume and the retained-trial backlog. The historical timings
above remain unsuitable as measurements of this combined implementation.


On 22 September, a two-GPU/two-CPU-worker combined run on L559289/SAP000 beams
13 and 15 measured 51.3–51.9 s for GPU production/validation, 73.7–74.0 s for
single-pulse searching, and **231.4–235.7 s for periodicity**, including folds and
plots. Each searched all 7,856 periodic trials. The periodic process peak RSS
was 1.24–1.25 GB. See [the validation record](periodicity-validation.md) for
scope and exact evidence. These measurements do not establish sustained
24-worker throughput or an archive-to-results campaign duration.


A full-grid run on the bright real pulsar J0323+3944 additionally measured
49.5 s for GPU production/validation, 71.8 s for the single-pulse branch and
248.3 s for periodicity. The latter included 29,903 raw peaks, 12.1 s sifting,
and 16 folds; peak process RSS was 1.265 GB. This used unflatfielded pilot input,
so it checks bright-source workload and recovery, not production flatfield
sensitivity. The bounded sifting allowance is 100 million comparisons, with
50,000 raw candidates per beam; overflow still fails explicitly.
