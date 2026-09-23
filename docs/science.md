# What the search does, and what it establishes

This is the honest account of the algorithm and its limits. It distinguishes
what has been measured from what has only been made to run.

## The per-beam path

Each SAP has 74 beams: 73 coherent tied-array beams and one incoherent beam,
beam 12, which the archive files under `incoherentstokes/`. The incoherent beam
is excluded from conversion and search. The central coherent beams 13–73
define each SAP's flatfield.

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
4. **Dedisperse.** Fourier-domain dedispersion over the six-range single-pulse plan,
   3,617 trials through DM 3019.8, on GPU. Time samples are averaged before the FFT; frequency
   bins are never subsampled. DM grids are generated with decimal arithmetic so
   adjacent plan ranges cannot overwrite a boundary trial through floating-point
   rounding.
5. **Search.** A boxcar matched filter over up to 16 widths, capped at one second
   before clustering, as rolling sums against a median-absolute-deviation noise scale.
6. **Cluster.** Threshold crossings grouped into events on the trial grid.
7. **Classify.** Survivors above DM 10 and S/N 7, vetted against ATNF, scored
   by six FETCH models, plotted and recorded.
8. **Periodicity.** A separate DM grid shares preprocessing and exact
   `(DM, downsample)` trials with the single-pulse search. Its CPU stage runs
   fractional-frequency FFT harmonic sums, sifts candidates, refines periods
   and folds a bounded shortlist. Single-pulse and periodicity attempts have
   independent ledger checkpoints; `classify` succeeds only after both enabled
   searches and their retained products are complete.

## Periodicity search and products

The search excludes the DM-dependent circular-dedispersion tail before estimating
noise or searching. Nonfinite, constant, truncated and wholly contaminated trials
fail explicitly. `periodicity_coverage.json` records valid duration, trimming and
accessible period limits for every DM. At the usual 7.864 ms input sampling, the
Nyquist period is 15.729 ms; at downsampling 128 it is 2.013 s. The 16 ms configured
lower limit does not imply that all DMs can reach it. Preprocessing has already
averaged 16 native samples; this is not a millisecond-pulsar search.

A twice-padded real FFT evaluates half-bin frequencies. For an H-harmonic sum,
fundamental templates are spaced at 1/(2 H T), with each harmonic evaluated at
its nearest half-bin frequency. Every harmonic stays within one quarter of an
independent Fourier bin. This avoids the former large loss when integer-only
harmonic samples missed fractional-bin signals. Search boundaries are included.
Each mean-subtracted trial is zero-padded to the next 7-smooth length before
the FFT (`fft_fast_lengths`). Lengths with a large prime factor otherwise fall
back to a several-times slower algorithm; padding keeps every sample and only
makes the Fourier grid at most ~2% finer.

Noise means are estimated from medians of approximately independent native FFT
powers, corrected for the exact finite-sample exponential median. Blocks grow
from 31 bins at low frequency to `red_noise_window_bins` (257 by default). No
block is shorter than 31 bins, and the real-only Nyquist coefficient, which is
not exponentially distributed, is excluded. Before 23 September neither held.
Trials whose length left a final block of one bin, the Nyquist term,
estimated the noise at the top of the band from that single value, sometimes
~1000× too low. The whitened power there rose ~100-fold. The result was false
periodicities at each trial's own Nyquist period (2 × downsample × tsamp) and
its sub-harmonics, at unrelated DMs. These filled the fold shortlist of every
normal beam checked; on beam 13, 42 raw peaks fell to none. The
ranking statistic is **minus log10 of the nominal Gamma(H, 1) survival
probability**, computed stably for integer H. The default threshold is 12; it
is not Gaussian sigma. See the [SciPy gamma survival definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammaincc.html).
Noise estimation, Fourier correlations, coloured noise and RFI mean these are
nominal per-template values, not calibrated survey false-alarm probabilities.
White-noise and coloured-noise injections are regression checks; neither proves
survey completeness or universal sensitivity.

The raw JSONL is streamed before sifting. The per-trial, per-beam and
comparison limits bound what is **retained**, not what is searched.

- **Per trial.** Every template of every trial is searched. A trial with more
  than `max_candidates_per_trial` (100) peaks keeps its strongest distinct
  frequencies, one per Fourier bin.
- **Per beam.** The strongest `max_candidates_per_beam` (20,000) are sifted.
  If their neighbourhoods would exceed `max_sift_comparisons` (200 million),
  that set is halved until it fits.
- **Recording.** Candidates left unsifted keep their rows, labelled with the
  reason. The coverage report and summary record peaks found, dropped and
  unsifted per trial and per beam.

These limits replace hard failures. Under them, the brightest sources in the
survey ended the search: J0323+3944 put up to 1,190 peaks in a trial and
aborted its beam at DM 25.7, leaving nothing above covered. The same beam now
completes all 7,856 trials. The pulsar's harmonics sift into one group, folded
at P = 3.031791 s, DM 26.2, with a catalogue match.

Sifting checks adjacent period buckets, including harmonic relationships, and
indexes each DM step separately, so coarse high-DM trials do not expand every
low-DM neighbourhood. Any configured RFI
line in a summed harmonic flags the candidate; flagged evidence is retained.

Products retained after successful cleanup are:

- `periodicity_raw_candidates.jsonl`: frequency, period, DM, sampling, valid
  duration, harmonic count, nominal statistic and RFI harmonics for every peak.
- `periodicity_candidates.jsonl`: raw rows with group, relationship and ranking
  annotations. `best_candidates` counts group representatives, not discoveries.
- `periodicity_coverage.json`: actual valid data and searched period bounds.
- `periodicity_folded_candidates.jsonl` and `periodicity_plots/`: up to 16 group
  representatives, ordered with unflagged candidates first. Each has a local
  period refinement, profile, time-phase diagnostic, PNG and numerical NPZ.
  The top unflagged fold also receives a frequency-phase panel from the original
  filterbank. Catalogue context matches position, DM and period; it never creates
  or vetoes a detection. Catalogue unavailability is recorded without losing
  search products. Periods are topocentric; catalogue matching allows Doppler
  differences and propagates spin frequency where a catalogue derivative exists.
- `periodicity_summary.json`: completion manifest, configuration, counts, stage
  timings and process peak RSS. The latter includes the whole periodicity process,
  including folds and catalogue loading, and is not a GPU memory measurement.

The shortlist is a bounded follow-up policy, not a claim that all candidates were
folded. All raw and sifted rows remain available. Fold chi-square and profile bin
noise are diagnostic quantities and are not promoted to calibrated significance.
Acceleration searches and an FFA are not implemented; this is explicitly a
zero-acceleration FFT search.

Trial SHA256 manifests cover both DM directories and detect same-size corruption
on resume. A dedispersion retry atomically replaces reused hard links. Successful
single-pulse and periodicity checkpoints survive a failure in the other branch;
either branch is attempted even if the other fails. Cleanup requires both
completion manifests and all named fold products. Periodicity notifications are
an explicit, idempotent head-node operation, with pilots labelled as validation.

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

Survivors are scored by six FETCH models and accepted above 0.5. Their DM–time
input is computed only over the 256 decimated samples FETCH receives, not the
whole chunk of twice the dispersion sweep that `your` dedisperses at 256 DMs.
Channels are summed in the same order and precision, so the input is
bit-identical, and the original path remains wherever the crop would reach
decimation padding. At DM 3,000 and above this took ~40 s per candidate. A few
RFI-rich beams, with 20–100 candidates near DM 10,000, classified for over an
hour while the GPUs idled. **A rejection
is now recorded** as a `rejected` detection row with its probability, and
logged. It previously left no plot, no row and no log line, so a discarded
candidate was indistinguishable from one never found.

## Known limitations

**Single-pulse policy, 23 September.** The search uses a one-second maximum
boxcar, configured by `single_pulse.max_width_seconds`, and DM <3020.
An intrinsically wider event can still trigger a shorter template, with lost
sensitivity; this is a template limit, not a width-based proof of interference.
The historical search reached DM 10019.8 and used widths up to 600 seconds.

Before dedispersion, a channel whose temporal MAD exceeds four times the local
33-channel median is masked. Constant/nonfinite channels are also masked.
The statistic uses at most 8192 reproducibly sampled time points. Configured,
automatic and persisted channel indices use filterbank order; the pipeline's
internal reversed frequency axis is translated explicitly. FETCH now honours
the persisted channel mask instead of reading those bad channels back in.

Each clustered event gets an additional S/N estimate at its reported time and
width, using non-overlapping same-width reference windows nearby, outside a
two-width guard. Fewer than 32 usable windows gives unknown. The production
threshold is five: weaker checks are retained as `unconfirmed`, with the
original search score and evidence in `single_pulse_evidence.json`, and are
not sent to FETCH. Unknown checks still proceed. In-range survivors continue
through FETCH; there is no separate classifier-free queue for broad events.
These local scores are not calibrated false-alarm probabilities, and the
threshold is a starting policy rather than a measured survey optimum.

Synthetic tests cover scattered high-DM and band-limited injections, persistent
channel interference, a local noise increase, width-cap propagation through
the CPU stage, and incomplete reference data. They do not establish survey
completeness or the full false-positive reduction on the archive.

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

**Channel weighting is uniform among usable channels.** Previously, six to seven channels ran 1.8–9.4× the
typical noise. The RFI mask keys on block-normalised statistics, which removes
the per-channel signature that marks a *persistently* bad channel: the worst
channel in the band is caught only 14% of the time, and is masked in practice
only because it is listed in `bad_channels`. Uniform channel summing costs
about 7% S/N against the inverse-variance optimum. The persistent-noise cut
above addresses the worst channels; remaining channel weights are still uniform.

**The detection floor has been probed, not characterised.** A ladder of
injections at nominal S/N 6, 7, 8, 10 and 14 into a real beam recovered four of
five, at measured S/N 7.45, 7.67, 9.19 and 14.70; the nominal-7 injection fell
below the clustering threshold. That is one beam, one pulse width and one DM.

**No measurement here establishes survey sensitivity.** A partial flatfield
benchmark does not, and neither does a functional injection test. The
clustering and classification policy still requires scientific review before
any survey candidate claim.

**Periodicity limitations.** Effective sampling and pulse-width sensitivity
vary by DM. The implementation is zero acceleration, with topocentric periods,
no FFA and no automated periodic-candidate classifier. Red-noise whitening and
thresholds need continuing validation across real observing conditions. A
catalogue match plus a folded diagnostic supports a recovery check, not a new
pulsar claim.

## What the tests cover

Dispersed pulse recovery at the right time; CPU/GPU agreement; the matched
filter's calibration, width recovery, edge behaviour and determinism; the four
clustering cases above; trial completeness and rejection of mixed or truncated
trials; the DM grid's exclusive endpoints; the wrap-tail exclusion; the ATNF
veto in both directions; resume that rechecks outputs and records errors;
fingerprint stability across image copies and batch growth; token/site
association and macaroon fallback; tar traversal; node health; disk reclamation;
and the notifier's send-once, limit and credential behaviour.

Constant single-pulse inputs yield no events; a constant periodic trial fails
explicitly because it has no measurable noise. Short pilot filter windows are
bounded by the observation duration.

Periodicity adds fractional-bin, boundary, noise, RFI-harmonic, dispersed-train,
long-period, folding, manifest-corruption and independent-branch recovery tests.
[The periodicity validation record](periodicity-validation.md) includes full-grid
combined runs and recovery of J0323+3944 from real data.

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

The periodic source J0323+3944 has now been recovered with both targeted and
full DM grids from this beam; see [the validation record](periodicity-validation.md).
The supplied individual pulse at 2465.653 s has not been recovered as an accepted
single-pulse detection by these runs. The complete flatfield reference remains
unavailable; the successful periodic recovery used unflatfielded pilot input.
