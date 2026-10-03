import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from your.candidate import Candidate
from your.utils.math import normalise
from your.candidate import crop
from fetch.utils import get_model
from lotaas_reprocessing.filterbank import FilterbankFile
from lotaas_reprocessing.own_data import (RFI_BLOCK_SAMPLES, dispersion_ratio, load as load_own, measure as measure_own,
                                         near_edge, rfi_mask as search_rfi_mask, smoothness)
from matplotlib.gridspec import GridSpec
import pygedm
from astropy.coordinates import SkyCoord
from astropy import units as u
from psrqpy import QueryATNF

# Import DB utils
from db.db_utils import insert_beam_run, update_beam_run, insert_detection

logger = logging.getLogger(__name__)

# Cone fetched from the ATNF catalogue. Generous: the query is cheap and a
# complete local catalogue costs nothing to filter.
CATALOGUE_RADIUS_DEG = 5.0
# Separation within which a known pulsar can plausibly account for a detection
# in this beam. The veto used the full 5-degree query cone and matched on DM
# alone, so a new source sharing a DM with any pulsar in a 78 square-degree
# field was recorded as a redetection of it. Override with
# LOTAAS_ATNF_VETO_RADIUS_DEG once the tied-array beam response is measured.
VETO_RADIUS_DEG = float(os.environ.get("LOTAAS_ATNF_VETO_RADIUS_DEG", "1.0"))

# Which clusters reach FETCH; settings.yaml `classification` overrides these.
# Without a width cap or budget every cluster above the floors is classified.
DEFAULT_LIMITS = {"min_dm": 2.0, "min_snr": 7.0, "max_width_seconds": None,
                  "max_fetch_candidates": None, "min_local_snr": None, "min_raw_local_snr": None,
                  "min_own_snr": None, "min_own_fraction": None, "dispersed": None,
                  "fetch_bowtie": None, "fetch_clean": False, "fetch_high_dm": None}

# FETCH's DM-time plane spans at least DM +- this (its only span until 30 September 2026).
FETCH_MIN_RANGE_DM = 5.0
# No pulse reaches this in one channel and sample once channels are scaled to their noise (own_data).
FETCH_MAX_CELL_SIGMA = 50.0
K_DM = 4148.808


def snr_gate(limits, dm):
    """The search S/N a cluster must exceed: min_snr, or at the dispersed route's DMs its own min_snr when lower
    (those clusters only the route judges). dm may be a pandas Series."""
    route = limits.get('dispersed') or {}
    if route.get('min_snr') is None:
        return limits['min_snr']
    low = min(route['min_snr'], limits['min_snr'])
    if hasattr(dm, 'where'):
        return (dm >= route['min_dm']).map({True: low, False: limits['min_snr']})
    return low if dm >= route['min_dm'] else limits['min_snr']


def dispersed_tier(route, width_seconds):
    """The dispersed route's cuts for a cluster this wide: the first of its tiers (by max_width_seconds) the
    cluster fits, or its single set of cuts; None beyond them all."""
    tiers = route.get('tiers') or [{'max_width_seconds': route.get('max_width_seconds', float('inf')),
                                    'min_own_snr': route['min_own_snr'], 'max_ratio': route['max_ratio']}]
    for tier in sorted(tiers, key=lambda t: t['max_width_seconds']):
        if width_seconds is None or width_seconds <= tier['max_width_seconds']:
            return tier
    return None


def high_dm_accepts(high, fetch_probs, measure):
    """Whether FETCH's verdict stands at the DMs of `fetch_high_dm`: at least min_votes of its models above 0.5,
    and on the candidate's own data (measure() gives its dispersion ratio and smoothness, called only once the votes
    are there) a pulse that fades when dedispersed too little and a spectrum smooth across the band.
    Returns (accepted, ratio, smoothness).

    On its bowtie and cleaned input (fetch_bowtie, fetch_clean) FETCH recognised 632 of 634 injected FRB-like bursts
    in the levelled chain test where it had recognised 345, but with one model enough it passed 13.8% of all it
    judged in production (30 September to 2 October 2026, 14,906 beams), 40,099 of its 49,111 positives wider than
    0.15 s and most below DM 100, where a wide broadband burst cannot be told from DM 0. From DM 100, with five of
    six models, smoothness 0.6, the beam's edges left out and ratio 0.8: 608 of the 634 by FETCH and 626 with the
    dispersed route behind it (613 by the route alone), and 0.24 a SAP of those positives still open, 0.48 with the
    route's (13.7 with one model; benchmarks/fetch-flood-2026-10-02). A narrow-band patch at 145 MHz that five
    models passed at DM 1213 (L522694 SAP002 B033) is what the smoothness is for.
    """
    if sum(p > 0.5 for p in fetch_probs.values()) < high.get('min_votes', 1):
        return False, None, None
    try:
        ratio, smooth = measure()
    except Exception as error:
        logger.warning("FETCH's verdict not checked on its own data: %s", error)
        return False, None, None
    accepted = (ratio is not None and ratio <= high.get('max_ratio', 1.0)
                and smooth is not None and smooth >= high.get('min_smoothness', 0.0))
    return accepted, ratio, smooth


def dm_time_plane(cand, decimate, time_size=256, dmsteps=256, range_dm=5.0):
    """The decimated, time-cropped DM-time plane FETCH receives.

    `your` dedisperses the whole chunk (twice the dispersion sweep, 221,579
    samples at DM 8,000) at 256 trial DMs and the result is then decimated and
    cropped to 256 samples around the pulse. That took ~40 s per high-DM
    candidate and left a few RFI-rich beams classifying for over an hour while
    the GPUs idled. Only the columns that survive the crop are computed here,
    accumulating channels in the same order and precision, so the result is
    bit-identical. Where the crop would reach the median padding added for
    decimation, or a delay reaches the chunk length, the original computation
    is used.
    """
    nt, nf = cand.data.shape
    padded = nt + (-nt) % decimate
    decimated = padded // decimate
    start = decimated // 2 - time_size // 2
    dm_list = cand.dm + np.linspace(-float(range_dm), float(range_dm), dmsteps)
    freqs = np.asarray(cand.chan_freqs)
    delays = np.round(4148808.0 * dm_list[:, None] * (1 / (freqs[0]) ** 2 - 1 / (freqs[None, :]) ** 2)
                      / 1000 / cand.native_tsamp).astype("int64")
    # The window only pays when it is a small part of the chunk (wide pulses
    # force chunks barely longer than the crop).
    if not (decimated > start + time_size and start >= 0 and (start + time_size) * decimate <= nt
            and 2 * time_size * decimate <= nt and np.abs(delays).max() < nt):
        cand.dmtime(dmsteps=dmsteps, range_dm=range_dm)
        cand.decimate(key="dmt", axis=1, pad=True, decimate_factor=decimate, mode="median")
        return crop(cand.dmt, cand.dmt.shape[1] // 2 - time_size // 2, time_size, axis=1)
    first, width = start * decimate, time_size * decimate
    rows = np.ascontiguousarray(cand.data.T)
    plane = np.zeros((dmsteps, width), dtype=np.float32)
    # your rolls each channel right by its delay: out[t] = data[(t - d) mod nt].
    # Channels are added in order, as there, so every sum is bit-identical.
    for channel in range(nf):
        row = rows[channel]
        for step in range(dmsteps):
            begin = (first - delays[step, channel]) % nt
            end = begin + width
            if end <= nt:
                plane[step] += row[begin:end]
            else:
                split = nt - begin
                plane[step, :split] += row[begin:]
                plane[step, split:] += row[:end - nt]
    return plane.reshape(dmsteps, time_size, decimate).mean(2)


def fetch_range_dm(freqs, native_tsamp, width, bowtie, time_size=256, floor=FETCH_MIN_RANGE_DM):
    """Half-span of the DM-time plane in which a pulse's arms travel `bowtie` of the way from the plane's centre to
    its corners, and at least `floor`.

    The plane's time pixel is half the boxcar, so its 256 pixels span 128 widths whatever the width. A fixed +-5 DM
    shifts the band's edges by +-0.54 s over 119-151 MHz: a bowtie for a 31 ms pulse (+-35 pixels) but a vertical bar
    for a 252 ms one (+-4 pixels), and at DM >= 1200 dispersion inside a 48.8 kHz channel alone makes 87% of bursts
    at least 0.2 s wide. `your`'s +-DM, which FETCH was trained with, lets the arms leave the plane within a few rows
    and did worst of all. On the 28 September FETCH benchmark (1,290 candidates), 0.5 raised the injected FRBs FETCH
    passes from 23% to 38% (64% with fetch_clean), with pulsar pulses unchanged at 94% (99%); 1.0 passed
    15 of 94 wide junk candidates without fetch_clean (0.5: 4) (benchmarks/fetch-models-2026-09-30).

    Off in production since 2 October 2026: with 0.5, fetch_clean and FETCH on every width, FETCH passed 13.8% of
    what it judged in 14,906 beams instead of 0.6%, four in five of them wide broadband interference
    (ops/production-settings.yaml). That benchmark's junk, 94 candidates of three SAPs, was far too few to show it.
    """
    if not bowtie:
        return floor
    f = np.asarray(freqs, dtype=float)
    sweep_per_dm = K_DM * (f.min() ** -2 - f.max() ** -2)
    pixel = max(1, int(width) // 2) * native_tsamp
    return max(floor, bowtie * (time_size / 2) * pixel / sweep_per_dm)


def clean_chunk(data, good, rfi_mask=False, baseline_pixels=None, pixel=1):
    """The chunk as the search sees its data, roughly: each channel on its own median and noise (1.4826 MAD), cells
    beyond FETCH_MAX_CELL_SIGMA and bad channels at 0, and the mean over good channels subtracted at each sample (the
    search's zero-DM filter). With `rfi_mask`, the channel blocks the search's RFI mask flags (own_data.rfi_mask,
    1000-sample blocks from the chunk's start, whose grid may be offset from the search's) are zeroed before the
    zero-DM. With `baseline_pixels`, each channel's running median over that many `pixel`-sample means is then
    subtracted: single channels high or low for seconds (level bars, row steps) go, a pulse a few pixels long stays.

    FETCH was otherwise given the flatfielded chunk as it is: its noisiest channels set the normalised planes' scale
    and the beam's broadband wander rivalled the burst. Measured on the 28 September benchmark with the bowtie range
    (fetch_range_dm 0.5; benchmarks/fetch-models-2026-09-30, FETCH's a-f, any above 0.5), injected FRBs passed:
    38% as given, 64% cleaned, 77% with the RFI mask, 96% with the mask and a 64-pixel baseline (92% of those
    >= 0.2 s wide); pulsar pulses 94% -> 99%; of 94 wide junk candidates FETCH now sees, 0, 0, 0 and 9 passed.
    """
    step = max(1, len(data) // 8192)
    centre = np.median(data[::step], axis=0)
    scale = 1.4826 * np.median(np.abs(data[::step] - centre), axis=0)
    scale[~good | (scale <= 0)] = 1.0
    z = ((data - centre) / scale).astype(np.float32)
    z[:, ~good] = 0.0
    z[np.abs(z) > FETCH_MAX_CELL_SIGMA] = 0.0
    if rfi_mask:
        z[search_rfi_mask(data, RFI_BLOCK_SAMPLES)] = 0.0
    z[:, good] -= z[:, good].mean(axis=1, keepdims=True)
    if baseline_pixels:
        from scipy.ndimage import median_filter
        pixel = max(1, int(pixel))
        n = len(z) // pixel
        if n >= 4:
            means = z[:n * pixel].reshape(n, pixel, z.shape[1]).mean(axis=1)
            base = np.repeat(median_filter(means, size=(min(int(baseline_pixels), n), 1), mode='nearest'), pixel, axis=0)
            if len(base) < len(z):
                base = np.vstack([base, np.repeat(base[-1:], len(z) - len(base), axis=0)])
            z = (z - base).astype(np.float32)
            z[:, ~good] = 0.0
    return z


def fetch_inputs(filterbank_file, dm, tcand, width, snr, bad_channels=(), time_size=256, freq_size=256, dm_size=256,
                 range_dm=FETCH_MIN_RANGE_DM, bowtie=None, clean=False):
    """The candidate and FETCH's two inputs, the frequency-time plane X and the DM-time plane Y.

    Y spans DM +- range_dm, or wider with `bowtie` (fetch_range_dm; `your`, which FETCH was trained with, spans
    +-DM). With `clean` (True, or clean_chunk's options {rfi_mask, baseline_pixels}) the chunk is first cleaned.
    Returns (cand, X, Y, time_decimate_factor); cand keeps both planes, and cand.fetch_range_dm, for the plot.
    """
    cand = Candidate(
        fp=filterbank_file,
        dm=dm,
        tcand=tcand,
        width=width,
        label=-1,
        snr=snr,
        min_samp=256,
        device=-1,
    )
    cand.get_chunk()
    good = np.ones(cand.data.shape[1], dtype=bool)
    if bad_channels:
        bad = np.asarray(bad_channels, dtype=int)
        if np.any((bad < 0) | (bad >= cand.data.shape[1])):
            raise ValueError('Classifier channel mask is outside the filterbank')
        # Honour the search's file-order channel mask in both FETCH
        # planes; a noisy channel must not reappear at classification.
        good[bad] = False
        if not good.any():
            raise ValueError('No usable channels for classification')
        baseline = np.median(cand.data[::max(1, len(cand.data) // 8192), good])
        cand.data[:, bad] = baseline
    if clean:
        options = clean if isinstance(clean, dict) else {}
        cand.data = clean_chunk(cand.data, good, rfi_mask=bool(options.get('rfi_mask')),
                                baseline_pixels=options.get('baseline_pixels'), pixel=max(1, width // 2))
    range_dm = fetch_range_dm(cand.chan_freqs, cand.native_tsamp, width, bowtie, time_size, floor=range_dm)
    cand.fetch_range_dm = range_dm
    time_decimate_factor = max(1, width // 2)  # Ensure it's at least 1
    cand.dmt = dm_time_plane(cand, time_decimate_factor, time_size, dm_size, range_dm)
    cand.dedisperse()

    # Decimate, crop, and normalize FT
    cand.decimate(key="ft", axis=0, pad=True, decimate_factor=max(1, width // 2), mode="median")
    cand.dedispersed = crop(cand.dedispersed, cand.dedispersed.shape[0] // 2 - time_size // 2, time_size, 0)
    cand.decimate(key="ft", axis=1, pad=True, decimate_factor=max(1, cand.dedispersed.shape[1] // freq_size), mode="median")
    cand.resize(key="ft", size=freq_size, axis=1, anti_aliasing=True, mode="constant")
    cand.dedispersed = normalise(cand.dedispersed)

    # The DM-time plane is already decimated and cropped in time.
    # Crop along the DM axis
    crop_start_dm = cand.dmt.shape[0] // 2 - dm_size // 2
    cand.dmt = crop(cand.dmt, crop_start_dm, dm_size, axis=0)

    # Resize
    cand.resize(key="dmt", size=dm_size, axis=1, anti_aliasing=True, mode="constant")

    # Normalize `dmt`
    cand.dmt = normalise(cand.dmt)

    # Prepare data for FETCH classification
    X = np.reshape(cand.dedispersed, (1, 256, 256, 1))
    Y = np.reshape(cand.dmt, (1, 256, 256, 1))  # Ensure `dmt` is included
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError(f"Nonfinite FETCH input at DM={dm}, time={tcand}")
    return cand, X, Y, time_decimate_factor


def classify_candidates(filterbank_file, candidate_file, output_dir, observation_info=None,
                        limits=None, tsamp=None, evidence=None, bad_channels=(), plan=None,
                        baseline_seconds=None, baseline_widths=None, slow_cap=None):
    """Classify one beam's clusters; returns how many went to FETCH and why others did not.

    A cluster wider than `max_width_seconds` (needs `tsamp`, the native sample
    time), or beyond the `max_fetch_candidates` strongest, is recorded as
    'unclassified'. Locally weak events remain auditable as 'unconfirmed'.
    Known pulsars never use the budget.

    With `min_own_snr` and `min_own_fraction` (and the search's dedispersion
    `plan`), a cluster whose S/N on its own data (own_data.measure, the review
    page's local S/N) is below both is also 'unconfirmed', and FETCH never runs
    on it: 78% of what FETCH rejected on 27 September, 84% of its positives the
    page showed nothing in, and none of 444 pulsar pulses or 51 injected ones.
    Each FETCH model's score is kept with what FETCH judged (model_probabilities).

    With `dispersed` ({min_dm, tiers: [{max_width_seconds, min_own_snr,
    max_ratio}], fetch_max_width_seconds}; or one tier's keys at top level) a
    cluster FETCH rejects at DM >= min_dm that its own data show (own S/N >=
    the tier's min_own_snr for its width) and that fades when dedispersed too
    little (own_data.dispersion_ratio <= max_ratio) is recorded as 'dispersed'
    instead of 'rejected'. Wider than fetch_max_width_seconds at those DMs,
    FETCH is not asked ('unjudged'): only the route judges. Its min_smoothness
    (own_data.smoothness) keeps out a signal in scattered single channels, and
    edge_seconds what comes near the start or end of the beam. With the route's
    own min_snr below min_snr, clusters at its DMs between the two reach it
    too, unjudged by FETCH, under the route's `faint` cuts as well. With min_votes, a cluster FETCH was asked
    about needs at least that many of its models above 0.5 to reach the route. FETCH accepted 23% of FRB-like bursts
    injected at DM 300-2500 that the search found, and almost none wider than
    150 ms, the width most FRBs have at 135 MHz. With min_dm 100, min_own_snr 8,
    max_ratio 0.5 and max_width_seconds 0.5 this route kept 352 of the 485 such
    bursts FETCH rejected in a SAP searched with baseline_widths 8, and 1 of that
    SAP's 3,495 real clusters FETCH rejected at DM >= 100 (28 September 2026).
    """
    limits = dict(DEFAULT_LIMITS, **{k: v for k, v in (limits or {}).items() if k in DEFAULT_LIMITS})
    from lotaas_reprocessing.single_pulse_quality import candidate_key, evidence_route, review_route
    if limits['min_local_snr'] is not None and evidence is None:
        raise ValueError('Local S/N screening requires single_pulse_evidence.json')
    counts = {"fetch": 0, "known_pulsar": 0, "unconfirmed": 0, "unclassified": 0, "own_data": 0, "dispersed": 0,
              "unjudged": 0, "fetch_unsupported": 0}
    own_check = limits['min_own_snr'] is not None and limits['min_own_fraction'] is not None and bool(plan)
    os.makedirs(output_dir, exist_ok=True)
    observation_info = observation_info or {}
    beam_id = os.path.basename(filterbank_file)
    observation_date = observation_info.get("Observation Date", "Unknown")
    output_dir = os.path.abspath(output_dir)
    log_file = os.path.join(output_dir, "processing.log")

    beam_run_id = insert_beam_run(
        beam_id=beam_id,
        observation_date=observation_date,
        output_dir=output_dir,
        log_file=log_file,
        code_version=os.environ.get("LOTAAS_RUN_FINGERPRINT", "unversioned")
    )

    try:
        candidates_df = pd.read_csv(candidate_file, sep=r"\s+")
        candidates_df.columns = candidates_df.columns.str.strip().str.lower()
        candidates_df = candidates_df[(candidates_df["dm"] >= limits["min_dm"])
                                      & (candidates_df["s/n"] > snr_gate(limits, candidates_df["dm"]))]
        if candidates_df.empty:
            update_beam_run(beam_run_id, outcome="no_candidates", num_candidates=0,
                           num_redetections=0, highest_snr=0)
            return counts
        skycoord = SkyCoord(
            observation_info["RA (J2000)"],
            observation_info["DEC (J2000)"],
            unit=(u.hourangle, u.deg)
        )
        ra_str = skycoord.ra.to_string(unit=u.hour, sep=':', pad=True, precision=2)
        dec_str = skycoord.dec.to_string(unit=u.deg, sep=':', alwayssign=True, pad=True, precision=2)
        # The Milky Way's largest DM along this line of sight (NE2001, YMW16): a candidate well beyond
        # it would be extragalactic, and the review queue shows it first.
        l, b = skycoord.galactic.l.deg, skycoord.galactic.b.deg
        dm_ne2001, _ = pygedm.dist_to_dm(l, b, 5e4, method='ne2001')
        dm_ymw16, _ = pygedm.dist_to_dm(l, b, 5e4, method='ymw16')
        dm_galactic = float(max(getattr(dm_ne2001, 'value', dm_ne2001), getattr(dm_ymw16, 'value', dm_ymw16)))

        from lotaas_reprocessing.atnf import query_atnf
        query = query_atnf(
            factory=QueryATNF,
            params=['PSRJ', 'RAJ', 'DECJ', 'DM'],
            coord1=ra_str,
            coord2=dec_str,
            radius=CATALOGUE_RADIUS_DEG
        )
        known_psrs_df = query.table.to_pandas()
        if not known_psrs_df.empty:
            catalogue = SkyCoord(known_psrs_df["RAJ"].values, known_psrs_df["DECJ"].values,
                                 unit=(u.hourangle, u.deg))
            known_psrs_df = known_psrs_df.assign(
                separation_deg=skycoord.separation(catalogue).deg)
            # Vetoing on DM alone across the whole query cone labelled genuinely
            # new sources as redetections: a pulsar degrees away cannot produce
            # a detection in a tied-array beam, and at DM < 150 the grid is
            # dense enough that some catalogue entry usually falls within the
            # DM tolerance.
            # The veto uses the nearby set; the plot footer keeps the whole
            # cone. A pulsar 1.2 degrees away cannot account for a detection
            # in this beam, but a human judging the candidate still wants to
            # know it is there.
            catalogue_psrs_df = known_psrs_df
            known_psrs_df = known_psrs_df[
                known_psrs_df["separation_deg"] <= VETO_RADIUS_DEG].reset_index(drop=True)
        else:
            catalogue_psrs_df = known_psrs_df
        dm_tolerance = 0.5
        highest_snr = 0

        candidates_df = (
            candidates_df
            .sort_values("s/n", ascending=False)
            .groupby("dm", group_keys=False)
            .head(5)
        )

        model_names = ["a", "b", "c", "d", "e", "f"]
        fetch_models = None

        # Dict to store highest S/N redetections per pulsar
        redetections_best = {}

        num_redetections = 0

        for _, row in candidates_df.iterrows():
            dm = row["dm"]
            tcand = row["time"]
            width = int(row["filter_width"])
            snr = row["s/n"]
            sample_number = int(row["sample"])

            if dm < limits["min_dm"] or snr <= snr_gate(limits, dm):
                continue

            if snr > highest_snr:
                highest_snr = snr

            key = candidate_key(dm, tcand, width)
            if limits['min_local_snr'] is not None and key not in evidence:
                raise ValueError(f'Missing local S/N evidence for {key}')
            local = (evidence or {}).get(key, {})

            matched_psr = None
            # A catalogue DM is no evidence of a pulse by itself. The first 48 s
            # of L603674 stepped in level, and nine edges near DM 22.6 with a
            # local S/N of 0.2-3.8 were announced as J0152+0948 redetections.
            # A redetection passes the local check every other event does.
            if not known_psrs_df.empty and not evidence_route(local, limits):
                # Nearest matching pulsar, not whichever the catalogue listed
                # first, so the attribution names the plausible source.
                close = known_psrs_df[(known_psrs_df["DM"] - dm).abs() <= dm_tolerance]
                if not close.empty:
                    matched_psr = close.loc[close["separation_deg"].idxmin()]

            if matched_psr is not None:
                psr_name = matched_psr["PSRJ"]
                if psr_name not in redetections_best or snr > redetections_best[psr_name]["snr"]:
                    redetections_best[psr_name] = {
                        "dm": dm,
                        "snr": snr,
                        "width": width,
                        "time": tcand,
                        "sample": sample_number,
                        "separation_deg": float(matched_psr["separation_deg"]),
                    }
                continue  # Skip further processing for redetections

            reason = review_route(width * tsamp if tsamp else None, local, limits, counts['fetch'])
            if reason:
                counts[reason] += 1
                insert_detection(
                    beam_id=beam_id,
                    beam_run_id=beam_run_id,
                    time_seconds=tcand,
                    sample_number=sample_number,
                    candidate_dm=dm,
                    snr=snr,
                    width_samples=width,
                    detection_type=reason,
                )
                continue
            own = None
            if own_check:
                try:
                    own = measure_own(filterbank_file, dm, tcand, width, plan, bad_channels, baseline_seconds,
                                      baseline_widths, slow_cap)
                except Exception as error:
                    # Unmeasurable (a stretch past the beam's end, say): FETCH decides, as before.
                    logger.warning("Own-data S/N not measured at DM=%.2f t=%.3f: %s", dm, tcand, error)
                    own = None
                if own is not None and own < limits['min_own_snr'] and own < limits['min_own_fraction'] * snr:
                    counts["own_data"] += 1
                    logger.info("Own data show nothing at DM=%.2f t=%.3f S/N=%.2f width=%d: S/N %.1f there",
                                dm, tcand, snr, width, own)
                    insert_detection(
                        beam_id=beam_id,
                        beam_run_id=beam_run_id,
                        time_seconds=tcand,
                        sample_number=sample_number,
                        candidate_dm=dm,
                        snr=snr,
                        width_samples=width,
                        detection_type="unconfirmed",
                    )
                    continue
            route = limits['dispersed']
            width_seconds = width * tsamp if tsamp else None
            # FETCH accepted almost none of the injected bursts wider than 150 ms: at the route's
            # DMs, past its fetch_max_width_seconds, only the route judges a cluster.
            # From fetch_high_dm's DM on FETCH is asked about every width, on its bowtie and cleaned input, and its
            # verdict needs more than one model and the candidate's own data behind it (high_dm_accepts).
            high = limits['fetch_high_dm'] if limits['fetch_high_dm'] and dm >= limits['fetch_high_dm']['min_dm'] else None
            unjudged = bool(route and dm >= route['min_dm'] and (
                (not high and route.get('fetch_max_width_seconds') is not None and width_seconds is not None
                 and width_seconds > route['fetch_max_width_seconds'])
                or snr <= limits['min_snr']))           # below FETCH's gate, let in by the route's own
            time_size, freq_size, dm_size = 256, 256, 256
            fetch_probs = highest_prob = None
            accepted = False
            high_ratio = high_smooth = None
            fetch_bowtie = high.get('bowtie') if high else limits['fetch_bowtie']
            fetch_clean = high.get('clean', False) if high else limits['fetch_clean']
            if unjudged:
                counts["unjudged"] += 1
            else:
                counts["fetch"] += 1
                if fetch_models is None:
                    from lotaas_reprocessing.fetch_models import load_models
                    fetch_models = load_models(model_names, factory=get_model)
                # Proceed with classification of non-pulsar candidates
                cand, X, Y, time_decimate_factor = fetch_inputs(filterbank_file, dm, tcand, width, snr, bad_channels,
                                                                time_size, freq_size, dm_size,
                                                                bowtie=fetch_bowtie, clean=fetch_clean)
                fetch_probs = {name: model.predict([X,Y], batch_size=1, verbose=0)[0,1]
                               for name, model in fetch_models.items()}
                if not all(np.isfinite(p) and 0 <= p <= 1 for p in fetch_probs.values()):
                    raise ValueError(f"Invalid FETCH probabilities at DM={dm}, time={tcand}")
                highest_prob = max(fetch_probs.values())
                accepted = highest_prob > 0.5
                if accepted and high:
                    if high.get('edge_seconds') and near_edge(filterbank_file, dm, tcand, high['edge_seconds']):
                        accepted = False
                    else:
                        def measure():
                            data = load_own(filterbank_file, dm, tcand, width, plan, bad_channels, slow_cap)
                            ratio, _ = dispersion_ratio(data, baseline_seconds=baseline_seconds or 2.0,
                                                        baseline_widths=baseline_widths or 64)
                            return ratio, smoothness(data)
                        accepted, high_ratio, high_smooth = high_dm_accepts(high, fetch_probs, measure)
                    if not accepted:
                        counts["fetch_unsupported"] += 1

            if unjudged or not accepted:
                # Record the rejection. A bare `continue` left no plot, no row
                # and no log line, so a candidate the classifier discarded was
                # indistinguishable in the outputs from one never found. An
                # injected pulse recovered at S/N 14.7 scored 0.374 here while
                # three fainter ones from the same beam scored above 0.97, and
                # nothing recorded that it had been considered at all.
                if not unjudged:
                    logger.info("FETCH rejected DM=%.2f t=%.3f S/N=%.2f width=%d (max p=%.3f, %d models above 0.5)",
                                dm, tcand, snr, width, highest_prob, sum(p > 0.5 for p in fetch_probs.values()))
                tier = dispersed_tier(route, width_seconds) if route and dm >= route['min_dm'] else None
                if (tier and not unjudged and route.get('min_votes')
                        and sum(p > 0.5 for p in fetch_probs.values()) < route['min_votes']):
                    # FETCH was asked and too few of its models saw a pulse: in LT5_004 production every one of
                    # 535 such route candidates was junk, while 52 of the 54 injected bursts the route kept had a vote.
                    tier = None
                if tier and snr <= limits['min_snr'] and route.get('faint'):
                    # Let in by the route's own, lower S/N gate: its stricter cuts too.
                    tier = {'min_own_snr': max(tier['min_own_snr'], route['faint']['min_own_snr']),
                            'max_ratio': min(tier['max_ratio'], route['faint']['max_ratio'])}
                ratio = smooth = None
                if (tier and own is not None and own >= tier['min_own_snr']
                        and not (route.get('edge_seconds') and near_edge(filterbank_file, dm, tcand, route['edge_seconds']))):
                    try:
                        stretch_data = load_own(filterbank_file, dm, tcand, width, plan, bad_channels, slow_cap)
                        ratio, _ = dispersion_ratio(stretch_data, baseline_seconds=baseline_seconds or 2.0,
                                                    baseline_widths=baseline_widths or 64)
                        if ratio is not None and ratio <= tier['max_ratio'] and route.get('min_smoothness') is not None:
                            smooth = smoothness(stretch_data)
                    except Exception as error:
                        logger.warning("Dispersion not measured at DM=%.2f t=%.3f: %s", dm, tcand, error)
                dispersed = (ratio is not None and ratio <= tier['max_ratio']
                             and (smooth is None or smooth >= route['min_smoothness']))
                if dispersed:
                    counts["dispersed"] += 1
                    logger.info("Dispersed at DM=%.2f t=%.3f: own S/N %.1f, ratio %.2f", dm, tcand, own, ratio)
                    if unjudged:                       # the review plot shows what FETCH would have seen
                        cand, X, Y, time_decimate_factor = fetch_inputs(filterbank_file, dm, tcand, width, snr,
                                                                        bad_channels, time_size, freq_size, dm_size,
                                                                        bowtie=fetch_bowtie, clean=fetch_clean)
                insert_detection(
                    beam_id=beam_id,
                    beam_run_id=beam_run_id,
                    time_seconds=tcand,
                    sample_number=sample_number,
                    candidate_dm=dm,
                    snr=snr,
                    width_samples=width,
                    detection_type="dispersed" if dispersed else "rejected",
                    classification_probability=highest_prob,
                    model_probabilities=fetch_probs,
                    own_snr=own,
                    dispersion_ratio=ratio,
                    dm_galactic=dm_galactic,
                    smoothness=smooth,
                )
                if not dispersed:
                    continue
            else:
                insert_detection(
                    beam_id=beam_id,
                    beam_run_id=beam_run_id,
                    time_seconds=tcand,
                    sample_number=sample_number,
                    candidate_dm=dm,
                    snr=snr,
                    width_samples=width,
                    detection_type="candidate",
                    classification_probability=highest_prob,
                    model_probabilities=fetch_probs,
                    own_snr=own,
                    dispersion_ratio=high_ratio,
                    dm_galactic=dm_galactic,
                    smoothness=high_smooth,
                )

            fil = FilterbankFile(filterbank_file, "read")
            f_start, delta_f, nchan = fil.fch1, fil.foff, fil.nchans
            fil.close()
            frequency_axis = np.flip(f_start + np.arange(nchan) * delta_f)

            # Galactic info
            galactic_info = (
                f"RA: {observation_info['RA (J2000)']}  DEC: {observation_info['DEC (J2000)']} | "
                f"l: {l:.2f} b: {b:.2f}\n"
                f"Max DM NE2001: {dm_ne2001:.1f} | Max DM YMW16: {dm_ymw16:.1f}"
            )

            # Both arrays have been time-decimated and cropped around the event.
            # Display relative time, including the decimation factor, rather than
            # labelling the beginning of the cropped window as the event time.
            plot_tsamp = cand.tsamp * time_decimate_factor
            dm_time_axis = (np.arange(cand.dmt.shape[1]) - (cand.dmt.shape[1]-1)/2) * plot_tsamp
            half_range = getattr(cand, 'fetch_range_dm', FETCH_MIN_RANGE_DM)
            dm_values = np.linspace(dm - half_range, dm + half_range, dm_size)
            # Per-channel robust normalisation for display. A handful of
            # channels carry a persistent offset or several times the typical
            # noise, and on a shared colour scale they stripe the waterfall
            # and take the dynamic range with them. That matters most for a
            # marginal candidate, which is exactly what the plot is for.
            waterfall = np.asarray(cand.dedispersed, dtype=float).T   # (freq, time)
            channel_median = np.median(waterfall, axis=1, keepdims=True)
            channel_scale = np.median(np.abs(waterfall - channel_median),
                                      axis=1, keepdims=True) * 1.4826
            # A flat channel normalises to zero rather than to a division error.
            channel_scale[~np.isfinite(channel_scale) | (channel_scale <= 0)] = np.inf
            waterfall = np.nan_to_num((waterfall - channel_median) / channel_scale)
            # Robust limits, so one surviving spike cannot flatten the rest.
            vmin, vmax = np.percentile(waterfall, [1, 99])

            # Profile measured against its own off-pulse noise, rather than
            # rescaled so its maximum equals the reported S/N. A reader needs
            # to see how far the pulse stands out here, not be told again; and
            # the old scaling silently keyed on whatever the largest sample
            # was, noise spike included.
            time_series = waterfall.sum(axis=0)
            baseline = np.median(time_series)
            spread = np.median(np.abs(time_series - baseline)) * 1.4826
            time_series = (time_series - baseline) / (spread if spread > 0 else 1.)
            time_axis = (np.arange(len(time_series)) - (len(time_series)-1)/2) * plot_tsamp

            # Figure
            fig = plt.figure(figsize=(14,10))
            gs = GridSpec(5,6,figure=fig,
                          width_ratios=[1.7]*4+[3,0.2],
                          height_ratios=[0.35,0.35,1,5,1],
                          wspace=0.3,hspace=0)

            ax_obs = fig.add_subplot(gs[0,0:5])
            ax_obs.axis("off")
            ax_obs.text(0.5,0.5,
                f"{beam_id} DM={dm:.2f} Width={width} S/N={snr:.2f} Time={tcand:.3f}s",
                ha="center",va="center",fontsize=10,family="monospace")

            ax_fetch = fig.add_subplot(gs[1,0:5])
            ax_fetch.axis("off")
            if fetch_probs is None:
                verdict = f"FETCH not asked (wider than it judges) | dispersed: own S/N {own:.1f}, ratio {ratio:.2f}"
            else:
                verdict = ("FETCH: " + " | ".join([f"{k}:{v:.2f}" for k, v in fetch_probs.items()])
                           + (f"   | rejected, but dispersed: own S/N {own:.1f}, ratio {ratio:.2f}"
                              if highest_prob <= 0.5 else ""))
            ax_fetch.text(0.5,0.5,verdict,ha="center",va="center",fontsize=9,family="monospace")

            ax_gal = fig.add_subplot(gs[2,3:5])
            ax_gal.axis("off")
            ax_gal.text(0.5,0.5,galactic_info,ha="center",va="center",fontsize=9,family="monospace")

            ax_ts = fig.add_subplot(gs[2,0:3])
            ax_ts.plot(time_axis,time_series,color="black")
            ax_ts.axhline(0,color="0.7",lw=.5)
            ax_ts.set_ylabel("S/N in window")
            ax_ts.set_xticks([])

            ax_ft = fig.add_subplot(gs[3,0:3])
            # Orientation is unchanged from the original: the transposed array
            # with the default origin puts each channel at its own frequency.
            ax_ft.imshow(waterfall,aspect="auto",cmap="viridis",
                vmin=vmin,vmax=vmax,
                extent=[time_axis.min(),time_axis.max(),frequency_axis.min(),frequency_axis.max()])
            ax_ft.set_xlabel("Time relative to candidate (s)")
            ax_ft.set_ylabel("Freq(MHz)")

            ax_dmt = fig.add_subplot(gs[3,3:5])
            ax_dmt.imshow(cand.dmt,aspect="auto",cmap="viridis",origin="lower",
                extent=[dm_time_axis.min(),dm_time_axis.max(),dm_values.min(),dm_values.max()])
            ax_dmt.set_xlabel("Time relative to candidate (s)")
            ax_dmt.set_ylabel("DM")

            ax_psr = fig.add_subplot(gs[4,0:5])
            ax_psr.axis("off")
            if not catalogue_psrs_df.empty:
                psr_list = "\n".join(
                    [f"{r['PSRJ']} RA:{r['RAJ']} DEC:{r['DECJ']} DM:{r['DM']}"
                     f" sep:{r['separation_deg']:.2f}deg"
                     + ("" if r['separation_deg'] <= VETO_RADIUS_DEG else " (outside veto radius)")
                     for _,r in catalogue_psrs_df.iterrows()]
                )
            else:
                psr_list = "No known pulsars"
            ax_psr.text(0.5,0.5,psr_list,ha="center",va="center",fontsize=9,family="monospace")

            fig.subplots_adjust(left=0.05,right=0.98,top=0.96,bottom=0.07)
            out_path = os.path.join(output_dir,f"DM{dm}_Width{width}_SNR{snr}.png")
            plt.savefig(out_path,dpi=300)
            plt.close()

        # Insert and announce only the best redetections per pulsar
        for psr_name, info in redetections_best.items():
            insert_detection(
                beam_id=beam_id,
                beam_run_id=beam_run_id,
                time_seconds=info["time"],
                sample_number=info["sample"],
                candidate_dm=info["dm"],
                snr=info["snr"],
                width_samples=info["width"],
                detection_type="known_pulsar",
                pulsar_name=psr_name
            )
            logger.info("Redetected %s: DM %.2f, highest S/N %.2f, width %s, %.3f deg away",
                        psr_name, info['dm'], info['snr'], info['width'], info['separation_deg'])
            num_redetections += 1
        counts["known_pulsar"] = num_redetections
        if counts['unclassified'] or counts['unconfirmed']:
            logger.info("Single-pulse review routes: %s", counts)

        update_beam_run(
            row_id=beam_run_id,
            outcome="classified",
            num_candidates=len(candidates_df),
            num_redetections=num_redetections,
            highest_snr=highest_snr
        )

        print("Finished processing.")
        return counts

    except Exception as e:
        update_beam_run(
            row_id=beam_run_id,
            outcome="error",
            error_message=str(e)
        )
        raise
