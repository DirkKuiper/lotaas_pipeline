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
  files by state; hourly throughput; the pace the slower of staging and search
  sets; search processes running on the GPU node, flagged past an hour; SAPs
  needing attention; beams kept for review; the latest FETCH positives.
- **Staging** — StageIT requests with their age against the restage and
  timeout limits, files on tape, on disk and failing, time on tape once the
  index has seen files come off it, and the staging log.
- **Coverage** — every SAP with archive and observation IDs, pointing, state,
  beams searched and under which fingerprints; searched SAPs on the sky. A SAP
  page shows its beam layout and each beam's stages; a beam page its clusters
  in DM and time, stage attempts, candidates and diagnostic plots.
- **Candidates** — FETCH positives, known-pulsar redetections, FETCH rejects
  and folded periodic candidates, filtered by type, S/N, review and beam.
- **Verify** — one candidate at a time, with the next one a key away. A
  single-pulse candidate gets its dynamic spectrum at any DM, time and frequency
  scrunch, a channel mask, the undedispersed view with the expected sweep drawn
  in, the on- and off-pulse spectrum, S/N against DM beside the fall-off a real
  pulse of the fitted width would show (Cordes & McLaughlin 2003), the same from
  DM 0 where undispersed RFI peaks, and the DM-time bowtie. It also compares the
  best width with the narrowest a real pulse can be at that DM, the smearing
  within one channel. A periodic candidate gets its folded profile, phase
  against time and frequency, search response against DM and fold refinement.
  Verdicts (RFI, noise, known source, astrophysical, unsure) take one key and a
  note; `,` and `.` step through the queue.

A Slack post remains an alert. The measures on the Verify page describe the
snippet; they do not replace a reviewer's judgement.

## What it reads and writes

It reads the campaign state database, the ledger and the results tree, and
never writes to them. Everything it owns is under its data directory
(`/shared/results/dkuiper/lotaas/web` by default):

- `web.sqlite` — an index rebuilt from those sources every minute. It records
  each state change it sees, which is how staging latency is measured. It can
  be deleted at any time.
- `reviews.sqlite` — verdicts, the one record here nothing else can rebuild.
  Back it up.
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
slack_threads = true        # offer "reply in the Slack thread" with a verdict
```

With `slack_threads`, a verdict can be posted as a reply to the candidate's own
Slack message, found from the file ID `postproc.notify_candidates` recorded, with
the same credentials. It is sent only when the reviewer ticks the box.

`python -m web index` and `python -m web snippets` run one pass by hand.

## Tests

```bash
~/.venvs/lotaas-web/bin/python -m pytest -q web/tests
```

They build a throwaway campaign, ledger and results tree with the pipeline's
own schema code, and synthetic filterbanks with an injected dispersed pulse.
