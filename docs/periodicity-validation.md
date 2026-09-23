# Periodicity validation — 22 September 2026

The zero-acceleration periodicity branch runs alongside single-pulse searching.
It is enabled in `settings.yaml`. The final GPU test suite passed **101 tests,
with no skips**, inside the existing runtime on `efc-gpu-01`.

The complete machine-readable evidence is at
`/shared/results/dkuiper/lotaas/benchmarks/periodicity-validation/validation-report.json`.
This contains exact run fingerprints, retained-product verification, timings,
noise/injection results, failure-recovery checks and the Slack upload receipt.
All validation runs use separate pilot ledgers; they do not add production
coverage to the campaign database.

## Full-grid combined operation

Two existing, fully prepared one-hour beams from L559289/SAP000 were processed
concurrently with two GPUs and two CPU workers. Both branches and aggregate
completion succeeded; every named retained product was verified after collection,
and both trial directories were removed only after success.

| Beam | GPU stage, including validation | Single-pulse stage | Periodicity stage | Periodic trials | Raw periodic peaks | Folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 13 | 51.9 s | 74.0 s | 235.7 s | 7,856 | 97 | 14 |
| 15 | 51.3 s | 73.7 s | 231.4 s | 7,856 | 55 | 5 |

The periodic process peak RSS was 1.24–1.25 GB. These are measured costs for
these beams at this concurrency, including folding and plot writing. They are
not a sustained 24-worker cluster benchmark or a campaign-duration forecast.

## Real pulsar recovery and notification

**PSR J0323+3944 was recovered in real observation L603682/SAP1/beam5**, with no
injected signal. The recovered topocentric period was **3.031792594 s**, at
**DM 26.3 pc cm⁻³**. The catalogue gives DM 26.1898 and a propagated period near
3.032072434 s; the catalogue position is 0.00110 degrees from the beam centre.
Catalogue periods are barycentric, while this search reports topocentric periods.

The diagnostic shows a narrow integrated pulse, persistence across the hour,
and consistent pulse phase across the 119–151 MHz band. Period refinement,
folded profiles, subintegrations, frequency-phase data and the PNG are retained.
The plot was posted to **#lotaas-cands**, Slack file **F0C3FD9K2AF**, explicitly
labelled as validation.

The plot first came from a targeted periodic DM range of 25.5–26.9 and a
single-pulse range of 26.0–26.4. A subsequent run with the final code searched
the **complete production grids**: 4,407 single-pulse DM trials and 7,856
periodic trials. Both branches succeeded and recovered the same period and DM.
The full periodic branch retained 29,903 peaks, sifted them into 16 groups and
folded all 16, taking 248.3 s and 1.265 GB process peak RSS. All products were
verified after collection and both trial directories were cleaned.

Both pulsar runs above used unflatfielded data: the complete central-beam set was
still unavailable. An earlier one-reference-beam partial flatfield lost the
periodic detection after pipeline processing and caused the single-pulse
candidate guard to fire. The follow-up comparison below separates that failed
preparation experiment from an ensemble flatfield.

Both search stages succeeded in the targeted and full-grid pulsar runs. The single-pulse
branch returned no accepted detections; this test does not claim recovery of
the separately supplied single pulse at 2465.653 s.

## Flatfield comparison and newly identified production limits

An identical-settings comparison used the original beam, the earlier beam52-only
flatfield, and a flatfield averaged from all **30 available central beams**.
All inputs cover the full observation and share the same time/frequency grid.
The periodic search covered DM 20.0–32.9 in steps of 0.1, with the production
period range, harmonics and threshold. Diagnostic folds held the period at
3.031792594 s and DM at 26.3, with identical phase bins and plot scales.

| Preparation | Pulsar recovered | Maximum nearby H=16 nominal −log₁₀ p | Fixed profile peak / bin noise |
| --- | --- | ---: | ---: |
| Unflatfielded | Yes | 560.31 | 39.72 |
| One reference beam | No | 2.82 | 1.19 |
| Mean of 30 reference beams | Yes | 2716.77 | 100.66 |

The ensemble flatfield improves the processed profile peak by **2.53 times**.
The pulse remains visible in independent, channel-normalized input folds for
all three preparations. In the single-reference case, residual interference
survives masking and inflates the global replacement-noise standard deviation
to about 324 times the median robust channel-noise estimate; that ratio is 1.45
for the 30-beam flatfield. The one-reference failure therefore does not show
that flatfielding intrinsically removes this pulsar.

**The current production workload limits fail on the 30-beam case.** Its first
run exceeded 50,000 retained peaks at DM 28.6. A diagnostic rerun retained
59,698 peaks across all 130 trials, but exceeded the 100-million sifting-work
limit. For the completed comparison, all three cases used isolated diagnostic
limits of 100,000 peaks and 500 million sifting comparisons. The ensemble case
then completed searching, sifting and folding in 60.3 s (40.3 s in sifting).
Production settings were not changed. Candidate retention and sifting capacity
must be addressed and validated on the full production DM grid before this
strong flatfielded beam can complete in production.

This remains a **30-of-61-reference pilot**, not validation of the full production
flatfield. The comparison does not claim a single-pulse recovery. Source/input
hashes, exact settings, failure logs, candidate products, fixed folds, PNG/PDF
comparison and reproduction scripts are retained under
`/shared/results/dkuiper/lotaas/benchmarks/periodicity-validation/flatfield-comparison/`.
The main results are `comparison-report.json` and `flatfield-comparison.png`.

## Numerical and recovery checks

- Five independent full-length white-noise trials produced zero candidates at
  the default nominal `-log10(p)` threshold of 12. This is a regression check,
  not a measurement of a 10⁻¹² survey false-alarm rate.
- Narrow periodic injections were recovered at Fourier-bin offsets 0, 0.1,
  0.25, 0.5 and 0.73, with all best fundamentals using 16 harmonics.
- Tests cover dispersed periodic trains, DM mismatch, downsampling, long-period
  signals in coloured noise, search boundaries, summed-harmonic RFI flags,
  invalid data, wrap trimming, bounded candidate/sifting work and retained folds.
- A real-beam failure-recovery run deliberately failed the periodicity process,
  deleted one periodic trial and replaced another with same-size corrupt data.
  Retry regenerated the trials, reused the successful single-pulse checkpoint,
  recovered the pulsar and completed cleanup. A subsequent rerun added no ledger
  attempts. This used the final recovery implementation; subsequent changes
  only revised periodicity sifting and its workload limits.
- Reclamation tests protect active search branches and count shared trial inodes
  once. Notification tests enforce successful-stage evidence, pilot labelling
  and send-once behaviour.

The supported scope remains a topocentric, zero-acceleration FFT search with
bounded diagnostic folding. Acceleration, an FFA, automated periodic-candidate
classification and survey-wide sensitivity characterization are separate work.

## Update, 23 September: limits, speed and a whitening fault

**The production limits no longer end a search.** The production run on
L603682/SAP1/beam5 with all 61 central beams in its flatfield failed after 28.7 s.
J0323+3944's harmonics had exceeded the 50,000-candidate beam limit by DM 25.7,
after 258 of 7,856 trials. The limits now bound what is retained rather than
what is searched; see [science.md](science.md). The same retained production
trials, rerun with the current code and settings:

- **Coverage:** completed all 7,856 trials.
- **Retention:** 95,572 peaks found, 305 crowded trials thinned to 28,831
  retained, and the strongest 20,000 sifted.
- **Recovery:** J0323+3944 folded at P = 3.031791 s, DM 26.2, nominal
  −log₁₀ p 2,637.6, with a catalogue match.
- **Cost:** search 129 s, sift 4.5 s, fold 5.5 s; peak RSS 316 MB.

**The normal-beam shortlist was a whitening fault, not an instrumental signal.**
Every fold of L559289/SAP000 beams 13 and 15 sat at 2^j × tsamp: 0.0315 s, 0.063
s, 0.126 s and up to 16 s, at unrelated DMs. Each lay at the trial's own Nyquist
period (2 × downsample × tsamp) or a sub-harmonic whose top harmonic reached it.

- **No real signal.** Folding the DM 150.3 trial at 512 samples showed no
  periodicity: profile scatter 0.0129 against 0.0139 expected from noise.
- **Cause.** The red-noise estimate's last block was a single bin, the
  real-only Nyquist term, which put the noise at the band edge 0.001–0.4× too
  low.
- **Fix.** Blocks are now at least 31 bins and exclude that term.
- **Result.** On the same beam-13 trials, 42 raw peaks became 0.
- **Regression test.** A series with a zeroed Nyquist component reproduces
  the fault with the old estimator and gives no peaks now.

**Speed.** Zero-padding trials to 7-smooth FFT lengths cut the periodicity
search of beam 13 from 226 s to 128 s. On one node it doubles the throughput
peak.

The test suite passes 132 tests with none skipped on efc-gpu-01 (CUDA), and 131
with the GPU test skipped on the head. None of this is a sensitivity
characterisation.
