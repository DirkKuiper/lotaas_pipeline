# Capacity

What has been measured, and what it does and does not imply for a one-year
campaign.

> **The stage timings below predate the matched-filter rewrite.** That change
> made a full-length trial 3.7× faster and altered what the search detects, so
> every CPU-stage number here is stale and every derived forecast with it. They
> are kept because they are the last measurements actually taken; re-measure
> before planning on them.

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

## Outstanding: the throughput test

Request `991619` is a bounded throughput test of 666 files from three complete
observations, about 3.25 TB. StageIT reported them online while SURF returned
HTTP 403 `Permission denied for GET on path /lt5_004/1261459/...`, though a
byte-range GET on an earlier request still succeeded (206).

That is the signature of a path-scoped macaroon presented for a directory it
does not cover. The request spans three observations, so StageIT issues one
macaroon per path, and the downloader chose whichever expired last. Tokens are
now ordered by whether their path caveat covers the URL, and a 401 or 403 falls
through to the remaining tokens.

**This diagnosis rests on the recorded evidence in
`benchmarks/download-access-check.json` and has not been confirmed against the
live service.** Confirm before rerunning the throughput test:

```bash
python3 -m euroflash.access_check 991619 --output access-check.json
```

Until sustained tape and download throughput are measured, no end-to-end
campaign duration is supportable.
