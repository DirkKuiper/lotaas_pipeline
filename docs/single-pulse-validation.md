# Single-pulse search and viewer validation, 23 September 2026

The production single-pulse plan has 3,617 trials, ending at DM 3019.8
(exclusive configured boundary 3020). Search templates are capped at one
second before clustering. FETCH remains required for non-catalogued candidates
that survive the local-noise check; no broad-event bypass was deployed.
The periodicity configuration was preserved.

Changes are installed in both `lotaas-ops` (the current driver checkout) and
`lotaas-production` (the supervisor's configured restart checkout), including
their production settings. Already dispatched jobs retain frozen sources and
settings. They are not interrupted or relabelled.

Validation:

- Runtime test suite: 184 passed, one GPU-specific test skipped on the head
  node. The changes were exercised using the CPU dedispersion implementation;
  this run is not a new GPU performance or agreement measurement.
- Web tests: 34 passed. Zoom, display averaging and frequency grouping leave
  local candidate S/N unchanged. Missing-band tails are excluded from the
  whole usable view. Broad and band-limited pulse diagnostics remain readable
  for historical data.
- An independent synthetic filterbank ran through the actual dedispersion
  entry point and CPU search stage: a 0.7-second pulse injected at DM 2500
  was recovered at DM 2501, width 0.8 seconds, search S/N 32.356 and local
  S/N 27.936. The injected persistently noisy channel 9 and configured channel
  7 were correctly recorded in filterbank order. Its DM grid was deliberately
  limited to three trials for this integration check.
- Synthetic regression cases cover high-DM scattering, band-limited bursts,
  a local noise increase, insufficient reference windows, and the one-second
  template limit at all downsampling factors. No full-survey completeness or
  false-positive rate is inferred from these cases.
- The live viewer was restarted and checked by authenticated HTTP requests.
  Candidate `8f37b18d1663` (L559289 SAP000 B002, DM 9679.8) reports local S/N
  1.02265 at its original 3.0199-second width, with 69 reference windows, for
  both a two-second view and the whole usable view with extra averaging.
  Its original search/FETCH scores and stored snippet remain unchanged.

The local check measures the reported event, rather than taking the highest
peak within several pulse widths. It uses a fixed neighbourhood and measured
same-width noise; the older viewer's 4.08 versus 1.43 scores changed with zoom
and are not directly comparable to the new at-event statistic. The historical
snippet does not reproduce a significant event under this check. The cutoff
of local S/N 5 is an explicit starting policy, not a calibrated false-alarm
probability or an established optimum.
