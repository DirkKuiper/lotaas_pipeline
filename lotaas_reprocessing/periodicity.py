"""Bounded, zero-acceleration periodic search and auditable candidate products.

The statistic is -log10 of the nominal Gamma(H, 1) survival probability for
H noise-normalized Fourier powers. It is NOT a Gaussian S/N or a calibrated
survey false-alarm probability. Local noise estimates, coloured noise and RFI
require empirical validation. Fractional templates use a twice-padded FFT,
with fundamental spacing 1/(2*H*T), keeping every harmonic within 1/4 bin.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import json
import os
import resource
import time

import numpy as np
from scipy import fft
from scipy.signal import find_peaks
from scipy.special import gammainccinv, gammaln

DEFAULTS = {
    "enabled": False,
    "period_min_seconds": 0.016,
    "period_max_seconds": 300.0,
    "harmonics": [1, 2, 4, 8, 16],
    "threshold": 12.0,
    "red_noise_window_bins": 257,
    "rfi_frequencies_hz": [1.0, 50.0, 60.0],
    "rfi_tolerance_bins": 2.0,
    "sift_dm_tolerance": 2.0,
    "sift_period_fraction": 0.001,
    # Retention limits, not coverage limits: every template of every trial is
    # searched. A trial with more peaks keeps its strongest distinct ones, and
    # a beam with more sifts its strongest; both counts are recorded. Aborting
    # instead let one bright pulsar (J0323+3944, up to 1,190 peaks per trial)
    # end the search at DM 25.7 with nothing above it covered.
    "max_candidates_per_trial": 100,
    "max_candidates_per_beam": 20000,
    "max_sift_comparisons": 200000000,
    "max_folds": 16,
    "fold_bins": 64,
    "fold_subintegrations": 32,
    "catalogue_match": True,
    # Trials whose FFT length has a large prime factor fall back to Bluestein's
    # algorithm. Zero-padding each (mean-subtracted) trial up to a 7-smooth
    # length keeps every sample and cut the FFT cost of a beam from 88 s to
    # ~15-25 s; the Fourier grid becomes at most ~2% finer.
    "fft_fast_lengths": True,
}


def resolved_config(config=None):
    result = dict(DEFAULTS)
    result.update(config or {})
    harmonics = result["harmonics"]
    if not harmonics or any(int(h) != h or not 1 <= int(h) <= 32 for h in harmonics):
        raise ValueError("periodicity.harmonics must be integers in 1..32")
    result["harmonics"] = sorted(set(map(int, harmonics)))
    for key in ("period_min_seconds", "period_max_seconds", "threshold", "sift_dm_tolerance",
                "sift_period_fraction", "rfi_tolerance_bins"):
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"periodicity.{key} must be finite and positive")
    if result["period_max_seconds"] <= result["period_min_seconds"] or result["threshold"] > 100:
        raise ValueError("Invalid periodicity period limits or threshold (maximum 100)")
    if result["sift_period_fraction"] > .1:
        raise ValueError("sift_period_fraction must not exceed 0.1")
    for key in ("red_noise_window_bins", "max_candidates_per_trial", "max_candidates_per_beam",
                "max_sift_comparisons", "fold_bins", "fold_subintegrations"):
        if int(result[key]) != result[key] or result[key] < 1:
            raise ValueError(f"periodicity.{key} must be a positive integer")
    if result["red_noise_window_bins"] < 31 or not 8 <= result["fold_bins"] <= 256:
        raise ValueError("Need >=31 noise bins and 8..256 fold bins")
    if int(result["max_folds"]) != result["max_folds"] or not 0 <= result["max_folds"] <= 128:
        raise ValueError("max_folds must be an integer in 0..128")
    if any(not np.isfinite(f) or f <= 0 for f in result["rfi_frequencies_hz"]):
        raise ValueError("RFI frequencies must be finite and positive")
    return result


@lru_cache(maxsize=1)
def _smooth_numbers(limit=1 << 25):
    values = {1}
    for prime in (2, 3, 5, 7):
        values = {v * prime ** k for v in values for k in range(26) if v * prime ** k <= limit}
    return np.array(sorted(values))


def fast_length(n):
    """Smallest length >= n whose doubled (zero-padded) FFT is 7-smooth."""
    smooth = _smooth_numbers()
    if n > smooth[-1]:
        return int(n)
    return int(smooth[np.searchsorted(smooth, n, side="left")])


def thin_candidates(candidates, limit):
    """Keep at most `limit` peaks: strongest first, one per native Fourier bin.

    A bright source puts peaks at the same fundamental in every harmonic sum
    and in the sidelobes beside it; collapsing those first keeps the distinct
    frequencies. Returns the kept rows in their original order and the number
    dropped.
    """
    if len(candidates) <= limit:
        return candidates, 0
    order = sorted(range(len(candidates)), key=lambda i: -candidates[i]["statistic"])
    occupied, kept = set(), []
    for i in order:
        row = candidates[i]
        bin_number = int(round(row["frequency_hz"] / row["frequency_resolution_hz"]))
        if any(b in occupied for b in (bin_number - 1, bin_number, bin_number + 1)):
            continue
        occupied.add(bin_number)
        kept.append(i)
        if len(kept) == limit:
            break
    kept.sort()
    return [candidates[i] for i in kept], len(candidates) - len(kept)


def dm_grid_assessment(plan, tsamp, nu_min, nu_max, max_residual_samples=2.0):
    band_delay_per_dm = abs(float(nu_min) ** -2 - float(nu_max) ** -2) / 2.41e-4
    rows = []
    for entry in plan:
        smear = .5 * float(entry["ddm"]) * band_delay_per_dm
        effective = float(tsamp) * int(entry["downsample"])
        rows.append({"low_dm": float(entry["low_dm"]), "high_dm": float(entry["high_dm"]),
                     "ddm": float(entry["ddm"]), "downsample": int(entry["downsample"]),
                     "effective_sampling_seconds": effective,
                     "nyquist_period_seconds": 2 * effective,
                     "half_step_smear_seconds": smear, "residual_samples": smear / effective,
                     "adequate": smear <= max_residual_samples * effective})
    return rows


def noise_baseline(native_power, width, real_only_last=False):
    """Estimate exponential noise means from independent Fourier-bin medians.

    Small blocks at low frequency track red noise; higher blocks grow to the
    configured width. Correct the exact expectation of a finite-sample median
    of exponentials (not just its asymptotic ln(2)). Interpolation avoids steps.

    The real-only Nyquist bin (real_only_last) is not exponential and is left
    out, and no block is shorter than 31 bins. A final remainder of one bin,
    that Nyquist bin, used to set the noise at the top of the band ~1000x too
    low: whitened power there rose ~100-fold, and trials whose length left
    that remainder reported a false periodicity at their own Nyquist period
    (2 x downsample x tsamp) and its sub-harmonics, filling the fold shortlist.
    """
    centres, estimates = [], []
    end = len(native_power) - (1 if real_only_last and len(native_power) > 32 else 0)
    start, block = 1, 31
    while start < end:
        stop = min(start + block, end)
        if end - stop < 31:
            stop = end
        values = native_power[start:stop]
        m = len(values)
        # E[X_(k)] = sum_{j=m-k+1}^m 1/j for unit exponentials.
        k = (m + 1) // 2
        expectation = np.sum(1. / np.arange(m - k + 1, m + 1))
        if m % 2 == 0:
            expectation += .5 / (m - k)
        centres.append((start + stop - 1) / 2)
        estimates.append(float(np.median(values)) / expectation)
        start = stop
        block = min(int(width), max(31, int(block * 1.5)))
    estimates = np.asarray(estimates)
    positive = estimates[estimates > 0]
    if not positive.size:
        raise ValueError("No positive spectral noise estimate")
    estimates = np.maximum(estimates, np.median(positive) * 1e-12)
    return np.interp(np.arange(len(native_power)), centres, estimates)


def gamma_log10_survival(power, harmonics):
    """Stable integer-shape Gamma survival, including very strong detections."""
    power = np.asarray(power, dtype=float)
    terms = np.zeros_like(power)
    log_power = np.log(np.maximum(power, np.finfo(float).tiny))
    for j in range(1, harmonics):
        terms = np.logaddexp(terms, j * log_power - gammaln(j + 1))
    return np.maximum(0., (power - terms) / np.log(10.))


def _rfi_harmonics(frequency, count, resolution, config):
    flagged = []
    for h in range(1, count + 1):
        f = frequency * h
        for line in config["rfi_frequencies_hz"]:
            multiple = round(f / line)
            if multiple >= 1 and abs(f - multiple * line) <= config["rfi_tolerance_bins"] * resolution:
                flagged.append(h)
                break
    return flagged


def search_trial(data, dt, config, stats=None):
    """Search one valid time series; refuse invalid input, bound what is kept.

    Every template is searched. When a trial yields more peaks than
    max_candidates_per_trial, the strongest distinct frequencies are kept and
    the number dropped is written to `stats` (peaks_found, peaks_dropped).
    """
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 1 or len(values) < 16 or not np.isfinite(values).all():
        raise ValueError("Invalid periodic DM trial: need >=16 finite samples")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Positive finite sampling time required")
    if np.ptp(values) == 0:
        raise ValueError("Constant periodic DM trial has no measurable noise")
    values = values - np.mean(values)  # Never modify a caller's array.
    observed = len(values)
    if config.get("fft_fast_lengths", False):
        # Zeros at the mean add no noise; only the Fourier grid spacing changes.
        values = np.pad(values, (0, fast_length(observed) - observed))
    n = len(values)
    duration, resolution = n * dt, 1. / (n * dt)
    # Zero padding evaluates the complex Fourier transform at half-bin spacing.
    spectrum = fft.rfft(values, n=2*n, workers=1)
    power = np.abs(spectrum) ** 2
    # With an even length the last native bin is the real-only Nyquist term.
    baseline = noise_baseline(power[::2], config["red_noise_window_bins"], real_only_last=n % 2 == 0)
    whitened = power / np.interp(np.arange(len(power)) / 2, np.arange(len(baseline)), baseline)
    whitened[0] = 0.
    min_bin = max(2., duration / config["period_max_seconds"])
    # Exclude the real-only Nyquist bin; its null distribution is different.
    max_bin = min(np.nextafter(n / 2, 0.), duration / config["period_min_seconds"])
    candidates, found = [], 0
    for count in config["harmonics"]:
        # Index the highest harmonic on the half-bin grid, then map all lower
        # harmonics to it. Spacing of the fundamental shrinks with H.
        first = int(np.ceil(min_bin * 2 * count))
        last = min(int(np.floor(max_bin * 2 * count)), n - 1)
        if last < first:
            continue
        high_bins = np.arange(first, last + 1)
        summed = np.zeros(len(high_bins))
        for h in range(1, count + 1):
            indices = np.rint(high_bins * (h / count)).astype(np.int64)
            summed += whitened[indices]
        cutoff = float(gammainccinv(count, 10. ** -config["threshold"]))
        # Padding permits peaks on BOTH limits, including a one-template range.
        peaks = find_peaks(np.r_[-np.inf, summed, -np.inf], height=cutoff)[0] - 1
        found += len(peaks)
        limit = 4 * config["max_candidates_per_trial"]
        if len(peaks) > limit:
            # Only the strongest can survive thinning; skip building the rest.
            peaks = np.sort(peaks[np.argpartition(-summed[peaks], limit)[:limit]])
        scores = gamma_log10_survival(summed[peaks], count)
        for peak, score in zip(peaks, scores):
            bin_number = float(high_bins[peak]) / (2 * count)
            frequency = bin_number * resolution
            flagged = _rfi_harmonics(frequency, count, resolution, config)
            candidates.append({"frequency_hz": frequency, "period_seconds": 1. / frequency,
                "frequency_bin": bin_number, "frequency_resolution_hz": resolution,
                "template_spacing_hz": resolution / (2 * count),
                "statistic": float(score), "statistic_kind": "nominal_minus_log10_p",
                "summed_power": float(summed[peak]), "harmonic_count": int(count),
                "observation_seconds": observed * dt, "fft_samples": int(n),
                "effective_sampling_seconds": float(dt),
                "rfi_like": bool(flagged), "rfi_harmonics": flagged})
    candidates, _ = thin_candidates(candidates, config["max_candidates_per_trial"])
    if stats is not None:
        stats.update(peaks_found=int(found), peaks_dropped=int(found - len(candidates)))
    return candidates


def _harmonic_relation(a, b, tolerance):
    ratio = max(a["period_seconds"], b["period_seconds"]) / min(a["period_seconds"], b["period_seconds"])
    nearest = round(ratio)
    return 2 <= nearest <= 16 and abs(ratio - nearest) <= tolerance * nearest


def sift_candidates(candidates, config):
    """Connected groups, with explicit work limits and edge-based annotations."""
    if len(candidates) > config["max_candidates_per_beam"]:
        raise RuntimeError("Periodicity beam candidate limit exceeded")
    parent = list(range(len(candidates)))
    nearby = [False] * len(candidates)
    harmonic_flags = [False] * len(candidates)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Keep separate indexes for each DM step. A handful of coarse high-DM
    # trials must not inflate every low-DM neighbourhood across the beam.
    dm_tol = config["sift_dm_tolerance"]
    period_tol = config["sift_period_fraction"]
    scale = np.log1p(period_tol)
    indexes = {}
    comparisons = 0
    # Vectorize exact comparisons inside dense buckets. A bright pulsar can
    # produce thousands of harmonic/DM neighbours without being pathological.
    periods = np.asarray([r["period_seconds"] for r in candidates])
    dms = np.asarray([r["dm"] for r in candidates])
    steps = np.asarray([r.get("dm_step", 0.) for r in candidates])
    for i, left in enumerate(candidates):
        dm_bucket = int(np.floor(left["dm"] / dm_tol))
        p = left["period_seconds"]
        targets = [p] + [p*h for h in range(2,17)] + [p/h for h in range(2,17)]
        possible = set()
        for step, index in indexes.items():
            radius = max(dm_tol, 1.01 * steps[i], 1.01 * step)
            if left["dm"] < index["low_dm"] - radius or left["dm"] > index["high_dm"] + radius:
                continue
            first_dm = int(np.floor((left["dm"] - radius) / dm_tol))
            last_dm = int(np.floor((left["dm"] + radius) / dm_tol))
            for d in range(first_dm, last_dm + 1):
                for target in targets:
                    centre = int(np.floor(np.log(target) / scale))
                    # +/-2 covers the asymmetric relative tolerance of a ratio.
                    for b in range(centre - 2, centre + 3):
                        possible.update(index["buckets"].get((d, b), ()))
        comparisons += len(possible)
        if comparisons > config["max_sift_comparisons"]:
            raise RuntimeError("Periodicity sifting work limit exceeded; raw evidence retained")
        indices = np.asarray(sorted(possible), dtype=np.int64)
        if indices.size:
            local_dm_tol = np.maximum(config["sift_dm_tolerance"],
                                      1.01 * np.maximum(steps[i], steps[indices]))
            dm_close = np.abs(dms[i] - dms[indices]) <= local_dm_tol
            same = np.abs(p - periods[indices]) <= period_tol * np.minimum(p, periods[indices])
            ratios = np.maximum(p, periods[indices]) / np.minimum(p, periods[indices])
            nearest = np.rint(ratios)
            harmonic = (nearest >= 2) & (nearest <= 16) & (np.abs(ratios-nearest) <= period_tol*nearest)
            related = dm_close & (same | harmonic)
            for j in indices[related]:
                parent[find(i)] = find(int(j))
            different_dm = indices[related & (dms[i] != dms[indices])]
            if different_dm.size:
                nearby[i] = True
                for j in different_dm:
                    nearby[int(j)] = True
            harmonic_edges = indices[dm_close & harmonic]
            if harmonic_edges.size:
                harmonic_flags[i] = True
                for j in harmonic_edges:
                    harmonic_flags[int(j)] = True
        step = steps[i]
        if step not in indexes:
            indexes[step] = {"low_dm": left["dm"], "high_dm": left["dm"], "buckets": defaultdict(list)}
        index = indexes[step]
        index["low_dm"] = min(index["low_dm"], left["dm"])
        index["high_dm"] = max(index["high_dm"], left["dm"])
        index["buckets"][(dm_bucket, int(np.floor(np.log(p)/scale)))].append(i)
    groups = defaultdict(list)
    for i in range(len(candidates)):
        groups[find(i)].append(i)
    output=[]
    for group, indices in enumerate(sorted(groups.values(),key=min)):
        best=max(indices,key=lambda i:(not candidates[i]["rfi_like"], candidates[i]["statistic"]))
        for i in indices:
            row=dict(candidates[i],sift_group=group,sift_group_size=len(indices),is_sifted_best=i==best,
                     relationships={"nearby_dm":nearby[i],"harmonic":harmonic_flags[i]})
            row["sift_status"]="rfi_like" if row["rfi_like"] else ("best" if i==best else "related")
            output.append(row)
    return output


def sift_bounded(candidates, config):
    """Sift the strongest candidates the work budget allows; never abort.

    Up to max_candidates_per_beam, strongest first, are sifted; if their
    neighbourhoods still exceed the comparison budget, the set is halved until
    it fits. Candidates left out are returned unsifted and labelled, so the raw
    evidence and the reason are both kept. Returns (rows, sift_input_limit).
    """
    order=sorted(range(len(candidates)),key=lambda i:-candidates[i]["statistic"])
    limit=min(len(candidates),config["max_candidates_per_beam"])
    while True:
        chosen=sorted(order[:limit])
        try:
            sifted=sift_candidates([candidates[i] for i in chosen],config)
            break
        except RuntimeError:
            if limit<=1:
                raise
            limit//=2
    left=set(range(len(candidates)))-set(chosen)
    for i in sorted(left):
        sifted.append(dict(candidates[i],sift_group=None,sift_group_size=0,is_sifted_best=False,
                           relationships={},sift_status="not_sifted_beam_limit"))
    return sifted,limit


def atomic_json(path, value):
    path=Path(path)
    partial=path.with_name(path.name+".partial")
    partial.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
    os.replace(partial,path)


def _write_jsonl(path, rows):
    partial=path.with_name(path.name+".partial")
    with partial.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
    os.replace(partial,path)


def valid_trial(path, metadata, dm, downsample):
    """Exclude the tail polluted by circular dedispersion before any analysis."""
    from .matched_filter import wrap_contaminated_samples
    data=np.fromfile(path,dtype="float32")
    if not np.isfinite(data).all():
        raise ValueError(f"Nonfinite periodic trial: {path}")
    if not metadata.get("nu_min") or not metadata.get("nu_max"):
        raise ValueError("Band limits are required to exclude circular dedispersion wrap")
    polluted=wrap_contaminated_samples(dm,metadata["nu_min"],metadata["nu_max"],metadata["tsamp"],downsample)
    valid=len(data)-polluted
    if valid < 16:
        raise ValueError(f"Dispersion sweep leaves fewer than 16 valid samples at DM {dm}")
    return data[:valid],polluted


def run_periodicity_search(trial_dir, output_dir, metadata, config=None):
    from .trials import trial_specs, validate_trials
    config=resolved_config(config if config is not None else metadata.get("periodicity",{}))
    trial_dir,output_dir=Path(trial_dir),Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    # A previous success must not survive a failed rerun.
    summary_path=output_dir/"periodicity_summary.json"
    summary_path.unlink(missing_ok=True)
    plan=metadata.get("periodicity_dm_plan") or metadata["dedispersion_plan"]
    validate_trials(metadata,trial_dir,plan)
    specs=trial_specs(metadata,plan)
    started=time.perf_counter(); cpu_started=time.process_time()
    candidates=[]; coverage=[]
    raw_path=output_dir/"periodicity_raw_candidates.jsonl"
    partial=raw_path.with_name(raw_path.name+".partial")
    with partial.open("w") as raw:
        for name,spec in specs.items():
            data,polluted=valid_trial(trial_dir/name,metadata,spec["dm"],spec["downsample"])
            fft_padded=(fast_length(len(data)) if config["fft_fast_lengths"] else len(data))-len(data)
            dt=float(metadata["tsamp"])*spec["downsample"]
            stats={}
            rows=search_trial(data,dt,config,stats=stats)
            for row in rows:
                row.update(dm=spec["dm"],dm_step=spec["ddm"],trial_file=name,trimmed_samples=polluted)
                raw.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
            raw.flush()
            candidates.extend(rows)
            coverage.append({"dm":spec["dm"],"valid_samples":len(data),"trimmed_samples":polluted,
                "fft_padded_samples":fft_padded,
                "effective_sampling_seconds":dt,"observation_seconds":len(data)*dt,
                "shortest_search_period_seconds":max(config["period_min_seconds"],2*dt),
                "longest_search_period_seconds":min(config["period_max_seconds"],len(data)*dt/2),
                "candidates":len(rows),"peaks_found":stats.get("peaks_found",len(rows)),
                "peaks_dropped":stats.get("peaks_dropped",0)})
    os.replace(partial,raw_path)  # Raw evidence survives a sifting/folding failure.
    search_seconds=time.perf_counter()-started
    sift_start=time.perf_counter()
    sifted,sift_limit=sift_bounded(candidates,config)
    sift_seconds=time.perf_counter()-sift_start
    _write_jsonl(output_dir/"periodicity_candidates.jsonl",sifted)
    atomic_json(output_dir/"periodicity_coverage.json",coverage)
    from .periodicity_folding import fold_candidates
    fold_start=time.perf_counter()
    folds=fold_candidates(sifted,trial_dir,output_dir,metadata,config)
    _write_jsonl(output_dir/"periodicity_folded_candidates.jsonl",folds)
    files=[raw_path,output_dir/"periodicity_candidates.jsonl",output_dir/"periodicity_coverage.json",
           output_dir/"periodicity_folded_candidates.jsonl"]
    for fold in folds:
        files += [output_dir/fold["plot"],output_dir/fold["fold_data"]]
    summary={"schema":"lotaas.periodicity.v2","complete":True,"algorithm":"fractional_fft_harmonic_sum_zero_acceleration",
        "configuration":config,"trials_searched":len(coverage),"raw_candidates":len(candidates),
        "peaks_found":sum(c["peaks_found"] for c in coverage),
        "peaks_dropped_in_crowded_trials":sum(c["peaks_dropped"] for c in coverage),
        "crowded_trials":sum(1 for c in coverage if c["peaks_dropped"]),
        "fft_padded_samples":sum(c["fft_padded_samples"] for c in coverage),
        "sift_input_limit":sift_limit,
        "candidates_not_sifted":sum(1 for r in sifted if r["sift_status"]=="not_sifted_beam_limit"),
        "sifted_candidates":len(sifted),"best_candidates":sum(r["is_sifted_best"] for r in sifted),
        "rfi_like_candidates":sum(r["rfi_like"] for r in candidates),"folded_candidates":len(folds),
        "elapsed_seconds":time.perf_counter()-started,"cpu_seconds":time.process_time()-cpu_started,
        "stage_seconds":{"search":search_seconds,"sift":sift_seconds,"fold":time.perf_counter()-fold_start},
        "process_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "statistic_note":"-log10 nominal Gamma noise tail; not Gaussian S/N or survey significance",
        "period_frame":"topocentric","acceleration_searched":False,"ffa_searched":False,
        "outputs":{str(p.relative_to(output_dir)):p.stat().st_size for p in files}}
    atomic_json(summary_path,summary)
    return summary
