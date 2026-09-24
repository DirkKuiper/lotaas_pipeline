# Campaign web layer

A web view of the LT5_004 campaign that sits beside the pipeline and never
inside it: progress and health, staging from the LTA, coverage by SAP and beam,
every candidate, and a page for deciding whether a FETCH positive is real.

```bash
web/ctl.sh start        # serve on 127.0.0.1:8000, index every minute, keep snippets
web/ctl.sh url          # the address with its access token; open it once per browser
```

The head node is shared, so the server binds to localhost and asks for a
token cookie on every request. VS Code's port forwarding (or
`ssh -L 8000:localhost:8000 efc-head`) brings it to a laptop.

## Pages

- **Overview** — driver, dispatch, SSH master and disk health; SAPs and beam
  files by state; hourly throughput; catalogue volumes and remaining-time
  projections for the current inventory, LT5_004 survey and all catalogued
  survey observations; search processes running on the GPU node, flagged past an hour; SAPs
  needing attention; beams kept for review; the latest FETCH positives.
- **Staging** — StageIT requests with their age against the restage and
  timeout limits, files on tape, on disk and failing, time on tape once the
  index has seen files come off it, and the staging log.
- **Coverage** — every SAP with archive and observation IDs, pointing, state,
  beams searched and under which fingerprints; searched SAPs on the sky. A SAP
  page shows its beam layout and each beam's stages; a beam page its clusters
  in DM and time, stage attempts, candidates and diagnostic plots.
- **Single pulse** — FETCH positives, known-pulsar redetections and FETCH
  rejects, filtered by type, S/N, review and beam.
- **Periodic** — folded periodic candidates, filtered by DM, period, statistic,
  review and beam. The default list and dashboard count include only signal
  groups with strong fold evidence. A pulse window selected in one subset
  must repeat in the other: alternating subintegrations and first/second
  halves, each tested in both directions. Every split needs a repeatability
  score of at least 5 and positive support in at least 75% of its intervals.
  This score is a heuristic, not Gaussian significance: the period was fitted
  using the whole observation. There must also be at least three supporting
  DM trials within max(0.5, twice the DM step), at least eight samples per
  period, a best DM of at least 2, and no near-zero-DM response reaching 90%
  of the peak statistic. Where frequency diagnostics exist, at least 60%
  of valid subbands must support the same pulse phase; missing frequency
  diagnostics are explicitly shown and do not substitute for a band check.
  Consistent-DM repeats and integer harmonics share one representative per
  observation/SAP, with pilot and incoherent beams grouped separately.
  **All retained folds** and **Deferred for follow-up** remain available;
  failing the shortlist never labels a candidate false or deletes its data.
  Weak or intermittent sources can be deferred. This policy is checked against
  synthetic noise/pulses and the saved J0323+3944 recovery, and is not a
  survey sensitivity or completeness measurement.
  A period found in 4 or more beams, or in two SAPs, of one
  observation is hidden as RFI unless all its folds share one DM above zero;
  the indexer groups folds whose periods agree within 5×10⁻⁴. Each row shows
  how many beams share its period and the median scatter broadening expected
  at its DM (Bhat et al. 2004) against its period.
- The incoherent beam 12 and pilot runs are left out of both lists and queues
  unless asked for.
- **Review** — each list has its own queue: one candidate at a time, with the
  next one a key away. A
  single-pulse candidate gets its dynamic spectrum at any DM, time and frequency
  scrunch, a channel mask, the undedispersed view with the expected sweep drawn
  in, the on- and off-pulse spectrum, S/N against DM beside the fall-off a real
  pulse of the fitted width would show (Cordes & McLaughlin 2003), the same from
  DM 0 where undispersed RFI peaks, and the DM-time bowtie. It also compares the
  best width with the narrowest a real pulse can be at that DM, the smearing
  within one channel. A periodic candidate gets its folded profile, phase
  against time and frequency, search response against DM, fold refinement and
  the other beams where the same period was folded.
  Verdicts (RFI, noise, known source, astrophysical, unsure) take one key and a
  note; `,` and `.` step through the queue.

The measures on the Verify page describe the snippet; they do not replace a reviewer's judgement.

## What it reads and writes

It reads the campaign state database, the ledger and the results tree, and
never writes to them. Everything it owns is under its data directory
(`/shared/results/dkuiper/lotaas/web` by default):

- `web.sqlite` — an index rebuilt from those sources every minute. It records
  each state change it sees, which is how staging latency is measured. It can
  be deleted at any time.
- `reviews.sqlite` — verdicts and audited candidate removals, which nothing else can rebuild.
  Back it up.
  Its `candidate_removals` table records explicitly removed dashboard candidates
  with the reason, evidence and audit identity. These remain absent after an
  index rebuild. Their original pipeline products and cross-beam evidence stay
  intact; removing the corresponding removal record restores the candidate
  on the next index pass.
- `snippets/` — for each FETCH positive and known-pulsar redetection, the
  stretch the classifier read around it (the dispersion sweep either side, as
  `your` reads it, plus 5 s), as a SIGPROC file at the search's resolution for
  that DM, with a JSON sidecar recording where it came from. About 3–11 MB each.
  They open in FLITS, your_viewer or PRESTO as well.
- `held/` — hard links to the flatfielded filterbanks of prepared and
  dispatched SAPs. A link costs nothing while the campaign's copy exists; when
  the campaign deletes its copy after the search, the link keeps the data until
  the snippets are cut, then goes. A beam the campaign keeps for review is
  released at once. Candidates from before the web layer are cut from any copy
  of their beam still on disk.

The pipeline hashes `pipeline`, `preproc`, `lotaas_reprocessing`, `db`,
`euroflash` and `postproc` into every run fingerprint
(`euroflash.run.CODE_FOLDERS`). This package is outside them, and it runs in
its own environment rather than the runtime image, so changing it never
changes the identity of a search.

## Setting it up

```bash
python3 -m venv --without-pip ~/.venvs/lotaas-web       # Debian's python has no ensurepip
curl -sS https://bootstrap.pypa.io/get-pip.py | ~/.venvs/lotaas-web/bin/python
~/.venvs/lotaas-web/bin/pip install -r web/requirements.txt
```

`~/.config/lotaas/web.toml` overrides any field of `web.config.Config`, for
example:

```toml
port = 8001
snippet_types = ["candidate", "known_pulsar"]
rejected_above = 0.3        # also keep FETCH rejects scoring above 0.3
```

`python -m web index` and `python -m web snippets` run one pass by hand.

## Catalogue and estimates

`observation_catalogue` defaults to
`/shared/results/dkuiper/lotaas/lotaas_observations`, the local clone of
[cbassa/lotaas_observations](https://github.com/cbassa/lotaas_observations).
The indexer checks `observations.csv` against each project's CSV row counts
and exact byte totals, then caches individual beam products in `web.sqlite`.
Changed files are reloaded on the next index pass. The displayed revision
identifies the source snapshot; it is not a live archive-availability check.

The overview and authenticated `/api/forecast` expose separate retrieval and
search projections using the last 24 and six wall-clock hours. Remaining
download bytes use catalogue sizes and recorded receipts. Search progress
uses the first successful production completion per beam; pilots, retries,
and completions under another code version do not add new progress. Counts
remain across production versions and do not establish uniform science
coverage. Receipt history stays counted when a file becomes searched or kept.
Missing rates or file sizes yield an unknown projection, never zero work.

The overview and Coverage page default to all catalogue projects and observation
types (survey, confirmation and unclassified). Their project selector changes
SAP counts, archive counts, progress bars, volume and remaining-time estimates
together. Project totals reconcile to the all-project goal. Operational throughput,
cluster health and review queues remain explicitly campaign-wide.

SAPs are grouped by project, scientific observation ID and SAP number, not the
archive processing version. Completed production searches are matched by archive
URI. A SAP is complete only when all included archives have successful searches;
an archive containing multiple mapped beams requires all of them to finish.
Files outside the active inventory remain visible as `not_queued`. Observations
without parsable SAP identifiers stay in the observation total and are listed
separately in Coverage; their unknown SAP count and search cost are not zero.

Beam totals exclude summary/legacy products without beam identities and configured
excluded beam numbers. Filenames with Stokes identifiers and older gzip products
are included. Duplicate archive products sharing an observation, SAP, beam and part
are retained and flagged for reconciliation. Format compatibility and per-archive
search cost outside the current campaign remain unverified. The full source volume
(including auxiliary products) is shown separately in the estimate details.

The combined pace is the larger of the retrieval and search durations,
assuming overlap; it is a conditional projection, not a finish date or a
hardware-capacity benchmark. Other projects inherit the measured campaign
rate. Catalogue import changes only the web index and never admits new
archive work or changes the running campaign.

## Tests

```bash
~/.venvs/lotaas-web/bin/python -m pytest -q web/tests
```

They build a throwaway campaign, ledger and results tree with the pipeline's
own schema code, and synthetic filterbanks with an injected dispersed pulse.

## Single-pulse diagnostic changes, 23 September

The viewer defaults to a window covering eight candidate widths on each side.
The whole usable snippet excludes the dedispersed tail where the full usable
band is absent. The curved edge in older whole-snippet plots was missing data
from the dispersion delay, not a pulse feature.

Local candidate S/N and best-width measurements use saved sampling, independent
of display averaging or zoom. The local score is evaluated at the candidate
time against at least 32 non-overlapping same-width reference windows. Unknown
means insufficient reference data. Per-channel normalisation uses local
off-pulse samples at the candidate DM. The spectrum uses the measured scatter
of boxcar sums rather than assuming independent samples. Persistent noisy
channels are masked by default; the checkbox allows comparison without that
automatic mask. The search score remains visible as historical provenance.

The fine DM response follows the candidate's expected time shift. The coarse
DM scan can peak on other nearby events and is labelled accordingly. The
Gaussian full-band reference curve and instrumental broadening estimate are
diagnostics, not automatic astrophysical vetoes.

New snippets retain a wider off-pulse margin, use bounded-memory averaging,
and clip at actual observation boundaries instead of inventing constant noise.
The beam's own mask is authoritative when available. `unconfirmed` candidates
remain accessible through the type filter. Production searches stop at one
second and DM 3020; in-range survivors still use FETCH. Historical snippets
and recorded classifier decisions are unchanged.
