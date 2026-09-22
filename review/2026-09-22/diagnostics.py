"""Small, synthetic review experiments; does not modify survey data or code.

Run from the repository root inside the science container, with PYTHONPATH set.
Results describe individual failure mechanisms, not survey completeness.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import yaml
from scipy import signal

from lotaas_reprocessing.cluster import cluster_candidates, get_ddm
from lotaas_reprocessing.dm_plan import dm_values
from lotaas_reprocessing.matched_filter import (
    _kernel_spectra, generate_boxcar_kernel, run_matched_filtering,
)


def main():
    result = {"synthetic_only": True, "survey_completeness_measurement": False}
    dt = 0.007864319719374176
    nsamp = 457728
    with tempfile.TemporaryDirectory(prefix="lotaas-review-") as directory:
        directory = Path(directory)
        cases = {
            "isolated_events": [(20, 12, 100), (40, 11, 200), (60, 10, 300)],
            "distant_dm_collision": [(50.2, 12, 100), (150.6, 11, 100)],
            "four_second_repeater": [(30, 12, 100), (30, 11, 104), (30, 10, 108)],
            "adjacent_dm_boundary": [(150.5, 12, 100), (150.6, 11, 100)],
        }
        result["clustering"] = {}
        for name, rows in cases.items():
            source, output = directory/f"{name}.cands", directory/f"{name}.tsv"
            source.write_text("".join(f"{dm} {snr} {t} {round(t/dt)} 1\n" for dm, snr, t in rows))
            with contextlib.redirect_stdout(io.StringIO()):
                cluster_candidates(source, output)
            df = pd.read_csv(output, sep="\t")
            result["clustering"][name] = {
                "input_count": len(rows), "output_count": len(df),
                "input_scaled_dm": [dm/get_ddm(dm) for dm, _, _ in rows],
                "output": df.to_dict("records"),
            }

        # Noise-free template geometry, with independent white-noise variance.
        n = 8192
        t = np.arange(n)*dt
        result["kernel_geometry"] = []
        for width in [1, 2, 3, 4, 9, 20]:
            kernel = generate_boxcar_kernel(t, width*dt, 1000)
            pulse = np.zeros(n); pulse[1000:1000+width] = 1
            response = np.fft.irfft(np.fft.rfft(pulse)*np.conj(np.fft.rfft(kernel)), n=n)
            ratio = float(response.max()/np.linalg.norm(kernel)/np.sqrt(width))
            result["kernel_geometry"].append({"width_samples": width, "snr_fraction_of_ideal_boxcar": ratio})

        # Bright pulses contaminate the statistic's own variance estimate.
        rng = np.random.default_rng(390)
        noise = rng.normal(size=nsamp)
        t = np.arange(nsamp)*dt
        widths, spectra = _kernel_spectra(nsamp, dt, 1)
        result["bright_broad_pulses"] = []
        for seconds in [1, 10, 30, 60, 120, 300, 600]:
            pulse = noise.copy()
            active = abs(t-1800) < seconds/2
            pulse[active] += 100
            for detrending in ["mean", "quadratic"]:
                if detrending == "mean":
                    x = pulse - pulse.mean()
                else:
                    x = pulse - np.polynomial.Polynomial.fit(t, pulse, 2)(t)
                fft = np.fft.rfft(x)
                peak = -np.inf
                for kernel in spectra:
                    response = np.fft.irfft(fft*np.conj(kernel), n=nsamp)
                    peak = max(peak, float(response.max()/response.std()))
                result["bright_broad_pulses"].append({
                    "pulse_seconds": seconds, "detrending": detrending,
                    "peak_statistic": peak, "passes_search_5": peak >= 5,
                    "passes_classifier_7": peak > 7,
                    "ideal_white_noise_snr": float(100*np.sqrt(active.sum())),
                })

        # Explicit wrap-around: an event at the end also triggers at time zero.
        x = rng.normal(size=8192).astype("float32")
        x[-10:] += 100
        x -= x.mean()
        source = directory/"edge.dat"; x.tofile(source)
        times, _, strengths, _ = run_matched_filtering(source, dt, 30)
        result["filter_wrap"] = {"injected_last_samples": 10,
                                 "detections_in_first_0_1_seconds": int(np.sum(times < .1)),
                                 "max_first_0_1_seconds_statistic": float(max(strengths[times < .1], default=0))}

        # Host microbenchmark: old circular smoothed filters versus rectangular
        # rolling sums. Different kernels/edges; speed comparison, not parity.
        x = (noise-noise.mean()).astype("float32")
        def old():
            fft = np.fft.rfft(x)
            return [float(np.std(np.fft.irfft(fft*np.conj(k), n=nsamp))) for k in spectra]
        def rolling():
            cumulative = np.empty(nsamp+1, dtype=np.float64); cumulative[0] = 0
            np.cumsum(x, dtype=np.float64, out=cumulative[1:])
            return [float(np.std(cumulative[int(w):]-cumulative[:-int(w)])) for w in widths]
        timings = {}
        for name, function in [("cached_fft", old), ("rectangular_rolling_sum", rolling)]:
            function()
            trials = []
            for _ in range(3):
                start = time.perf_counter(); function(); trials.append(time.perf_counter()-start)
            timings[name] = float(np.median(trials))
        result["filter_microbenchmark_seconds"] = timings

    plans = yaml.safe_load(Path("settings.yaml").read_text())["dedispersion_plan"]
    result["dm_cost"] = [dict(plan, trials=len(dm_values(plan)), samples=len(dm_values(plan))*(nsamp//plan["downsample"])) for plan in plans]
    result["trial_dat_bytes_per_beam"] = 4*sum(p["samples"] for p in result["dm_cost"])
    result["trial_files_per_beam"] = 2*sum(p["trials"] for p in result["dm_cost"])
    low, high = 119.45, 151.04
    sweep = (low**-2-high**-2)/2.41e-4
    result["band_sweep_seconds_per_dm"] = sweep
    result["max_dm_sweep_seconds"] = sweep*10000
    result["channel_smearing_seconds_at_dm100"] = 8.3e3*100*.048828125/135**3
    result["nominal_widths_seconds_low_dm"] = (widths*dt).tolist()
    result["your_min_chunk"] = [{"width_seconds": w, "min_chunk_seconds": 128*w,
                                 "one_float32_waterfall_gb": (128*w/dt)*648*4/1e9}
                                for w in [1, 10, 30, 60, 600]]
    path = Path(__file__).with_name("diagnostics.json")
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
