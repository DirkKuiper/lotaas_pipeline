# What the search does, and what it establishes

This is the honest account of the algorithm and its limits. It distinguishes
what has been measured from what has only been made to run.

## The per-beam path

1. **Convert.** PSRFITS to 32-bit filterbank, scrunching 4 in frequency and 16
   in time. 2-bit values decode as 0–3; the original unpacking left-shifted
   them. Frequency headers derive from `DAT_FREQ`.
2. **Flatfield.** Central beams 13–73 define the flatfield for the whole SAP.
   Beams are accumulated one at a time to bound memory; a beam extending past
   the flatfield time grid is an error rather than silently padded.
3. **Mask and detrend.** An RFI mask from block-normalised variance, skew and
   kurtosis, plus the always-masked channels in `settings.yaml`. Masked samples
   are replaced with noise, then each channel is detrended with a degree-2
   polynomial.
4. **Dedisperse.** Fourier-domain dedispersion over the eight-range plan,
   4,407 trials, on GPU. Time samples are averaged before the FFT; frequency
   bins are never subsampled. DM grids are generated with decimal arithmetic so
   adjacent plan ranges cannot overwrite a boundary trial through floating-point
   rounding.
5. **Search.** A boxcar matched filter over 16 widths, as rolling sums against
   a median-absolute-deviation noise scale.
6. **Cluster.** Threshold crossings grouped into events on the trial grid.
7. **Classify.** Survivors above DM 10 and S/N 7, vetted against ATNF, scored
   by six FETCH models, plotted and recorded.

## The matched filter

The statistic for width *w* is the sum over the window, centred and divided by
a MAD estimate of that rolling sum's own scale. Summing *w* samples scales the
noise by √*w*, so the MAD carries the normalisation and the result is directly
a signal-to-noise ratio.

Three properties matter, and none held for the circular-FFT implementation this
replaced. All three were measured; see `review/2026-09-22/diagnostics.json`.

| | before | now |
| --- | --- | --- |
| 100σ pulse lasting 300 s | statistic 4.09, below the threshold of 5 | detected |
| recovery at width 1 | 0.707 of the ideal boxcar, non-monotonic in width | 1.000 at every width tested |
| event in the last ten samples | 81 detections in the first 0.1 s | none |
| full-length trial | — | 3.7× faster |

The old normalisation divided each response by the standard deviation of a
response that contained the signal, so a pulse inflated the denominator
measuring it. The width ladder runs to 600 s, so the entire long half of the
search space was computed and could not be detected.

The statistic remains calibrated to unit variance on Gaussian noise, so the
thresholds of 5 in the search and 7 in the classifier keep their meaning. The
noise scale is drawn from 65,536 random windows, fixing it to about 0.3%; draws
are seeded per call so re-searching a trial reproduces its candidates.

## Clustering

Candidates carry their position on the concatenated trial grid. The earlier
coordinate, DM/ddm, was not monotonic across plan boundaries: DM 50.2 and DM
150.6 both mapped to 502, so unrelated events collided, while adjacent trials
DM 150.5 and DM 150.6 mapped to 1505 and 502, so one pulse straddling a
boundary was split.

The DM and time axes each carry their own tolerance — five trials, and half a
second — instead of sharing a single epsilon of five that merged a source
repeating every four seconds into one candidate. Points with no neighbour are
their own events; treating the DBSCAN noise label as a cluster had reduced
every isolated detection in a beam to a single reported row.

The implementation computes exact epsilon-connected components, equivalent to
DBSCAN with `min_samples=2`, by grouping points into cells and testing
neighbouring cells with nearest-neighbour queries rather than building a
quadratic graph. Tests compare its labels against sklearn DBSCAN including
duplicate and boundary points. Every candidate stays in the science table;
diagnostic scatter plots display at most 100,000 points.

## Classification

Candidates above DM 10 and S/N 7 are checked against ATNF. The veto requires
both a DM match within 0.5 pc/cm³ **and** an angular separation within
`LOTAAS_ATNF_VETO_RADIUS_DEG`, one degree by default. It previously matched on
DM alone across the whole 5-degree query cone, so a new source sharing a DM
with any pulsar in 78 square degrees was recorded as a redetection of it. The
plot footer still lists the full cone, with separations, so a human keeps the
context the veto does not use.

Survivors are scored by six FETCH models and accepted above 0.5. **A rejection
is now recorded** as a `rejected` detection row with its probability, and
logged. It previously left no plot, no row and no log line, so a discarded
candidate was indistinguishable from one never found.

## Known limitations

**Dedispersion still wraps.** It is circular Fourier dedispersion. A pulse in
the last samples of a beam is still recovered at the right time, with less
bandwidth as its sweep runs past the end — that was checked. The defect is
that whatever occupied the first samples of the low-frequency channels
reappears at the end of each trial: broadband interference in the first three
samples gives 56σ in the last 1089 samples at DM 100, against 13σ in the
uncontaminated middle. The search drops that tail, which costs about 0.4% of a
one-hour beam at DM 150, 3% at DM 1000 and 30% at DM 10000. `trim_wrap=False`
searches the whole series instead.

**FETCH's verdict depends on the reported boxcar width.** The preprocessing
decimation is keyed to the search's best-fit width, which is a noisy estimate
of the pulse width. An injected pulse recovered at S/N 14.7 scored 0.374 and
was rejected at width 20, and 0.926 at width 9 — the same event, the same
beam. Three fainter injections in that beam scored 0.975 to 0.999. Rejections
are now recorded so this is auditable, but the sensitivity itself is not fixed.

**Channel weighting is uniform.** Six to seven channels run 1.8–9.4× the
typical noise. The RFI mask keys on block-normalised statistics, which removes
the per-channel signature that marks a *persistently* bad channel: the worst
channel in the band is caught only 14% of the time, and is masked in practice
only because it is listed in `bad_channels`. Uniform channel summing costs
about 7% S/N against the inverse-variance optimum. Extending `bad_channels`, or
an automatic per-channel noise cut, would recover most of it.

**The detection floor has been probed, not characterised.** A ladder of
injections at nominal S/N 6, 7, 8, 10 and 14 into a real beam recovered four of
five, at measured S/N 7.45, 7.67, 9.19 and 14.70; the nominal-7 injection fell
below the clustering threshold. That is one beam, one pulse width and one DM.

**No measurement here establishes survey sensitivity.** A partial flatfield
benchmark does not, and neither does a functional injection test. The
clustering and classification policy still requires scientific review before
any survey candidate claim.

## What the tests cover

Dispersed pulse recovery at the right time; CPU/GPU agreement; the matched
filter's calibration, width recovery, edge behaviour and determinism; the four
clustering cases above; trial completeness and rejection of mixed or truncated
trials; the DM grid's exclusive endpoints; the wrap-tail exclusion; the ATNF
veto in both directions; resume that rechecks outputs and records errors;
fingerprint stability across image copies and batch growth; token/site
association and macaroon fallback; tar traversal; node health; disk reclamation;
and the notifier's send-once, limit and credential behaviour.

Constant signals yield no events. Short pilot filter windows are bounded by the
observation duration.

## Reference recovery target

The reference plot identifies **L603682, SAP1, beam 5** at DM 26.20,
2465.653 s (sample 313524), S/N 12.48, width one sample. Public metadata maps
L603682 to LT5_004 pointing LOTAAS-P1712C, observed 2017-08-17; its reprocessed
archive pipeline is **L1263256**. Request **991620** stages SAP1 beam 5 plus
central beams 13–73. The downloaded PSRFITS filenames must confirm the
observation/SAP mapping before preparation.

```bash
python3 -m euroflash.recovery --target TARGET.json \
  --work COLLECTED_NODE_WORK --output REPORT.json
```

It checks threshold crossings, cluster representatives and accepted detections
separately, with tolerances of 0.5 pc/cm³ and 0.25 s. A pass requires a
matching accepted detection and successful search and classification stages;
the reference S/N and FETCH scores are not imposed on new results. The report
identifies the run fingerprint and refuses an ambiguous result directory.

Waiting for tape does not establish pulse recovery.
