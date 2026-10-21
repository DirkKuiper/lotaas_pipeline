"""Same-DM repeats: the few pulses an hour of an RRAT, at one DM in one beam (sp_repeats).

An RRAT that gives two to six pulses an hour leaves no periodicity and, alone, no pulse bright enough to stand
out. Its pulses share one DM, while the rest of a beam's clusters spread over thousands. LOTAAS found
J2033+0042 and J1343+6634 this way. This campaign held their pulses (S/N 6-8, in clustered_candidates.txt)
but credited neither, since a pulse counted only with five at its DM or a fold behind it.

Measured on 21 October 2026 over the 244,652 beams searched by then (benchmarks/rrat-repeats-2026-10-21):
- A group: an anchor cluster at S/N >= ANCHOR_SNR and DM >= MIN_DM, brightest first, and the clusters at
  S/N >= MIN_SNR within its window |DM - DM_a| <= max(TOL_MIN, TOL_FRACTION x DM_a), one per moment (clusters
  within MERGE_SECONDS chain into one pulse). Its chance is the Poisson tail of the members beside the anchor,
  against the beam's own pulses per window within a factor BAND in DM, for members at S/N 6 and again at 7.
  log10 FAP adds the beam's anchors as trials.
- Interference spreads over several windows, a source falls in one: only a 'sharp' group counts, with at most
  MAX_ADJACENT pulses in the windows beside it and MAX_SIDE in the ten beyond.
- Against a null with every DM moved within +-3 windows, sharp groups at log10 FAP <= -6 were 1,202 against
  133. They were 415 groups of 75 catalogued pulsars, and 2-bit block steps and interference at one moment
  in many beams; there was 1 unexplained group, and the null had 7.
- J2033+0042 gave 6 pulses at log10 FAP -14.4, J1343+6634 5 at -11.6. J1931+4229's two pulses an hour
  (-0.5) are beyond any count at one DM.
"""
import math

import numpy as np

ANCHOR_SNR = 7.0
MIN_SNR = 6.0
MIN_DM = 3.0
TOL_MIN, TOL_FRACTION = 0.5, 0.02
MERGE_SECONDS = 1.0
SIDE_OFFSETS = (-6, -5, -4, -3, -2, 2, 3, 4, 5, 6)
BAND = 1.5
LAMBDA_FLOOR = 0.002
MAX_ANCHORS = 300
MAX_ADJACENT, MAX_SIDE = 1, 2
# Groups at or below this are kept (sp_repeats); at or below QUEUE_LOG10FAP one counts for a catalogued
# pulsar it belongs to, and one nothing explains reaches the review queue.
STORE_LOG10FAP = -4.0
QUEUE_LOG10FAP = -6.0
# The level steps of the data: LT5_004's 2-bit requantiser every 3072 samples (24.16 s, and its rows of
# 512), the early cycle's rows of 512 or 32 samples. The search sees a step as a cluster at low DM; a train
# of them keeps the step's phase.
LT5_BLOCKS = (3072 * 0.007864319719374176, 512 * 0.007864319719374176)
EC_ROWS = (512 * 0.00786432, 32 * 0.00786432)
BLOCK_TOLERANCE = 0.15
# A known pulsar's group: within this of it, or further (to indexer.known_radius) with every pulse within
# PHASE_TOLERANCE turns of one phase of its rotation.
NEAR_DEG = 1.0
PHASE_TOLERANCE = 0.05
# Other beams of the observation: a pulse there at the same moment and DM is the same pulse; within
# LOCAL_DEG of the group's beam and in at most LOCAL_BEAMS beams, a source between them.
MOMENT_SECONDS = 0.5
LOCAL_DEG, LOCAL_BEAMS = 0.35, 7


def log10_poisson_sf(m, lam):
    """log10 P(X >= m), X ~ Poisson(lam)."""
    if m <= 0:
        return 0.0
    if lam <= 0:
        return -math.inf
    if m <= lam:
        head = sum(math.exp(-lam + i * math.log(lam) - math.lgamma(i + 1)) for i in range(m))
        return math.log10(max(1e-300, 1.0 - head))
    terms = [-lam + i * math.log(lam) - math.lgamma(i + 1) for i in range(m, m + 80)]
    top = max(terms)
    return (top + math.log(sum(math.exp(t - top) for t in terms))) / math.log(10)


def moments(times):
    """How many distinct moments: times more than MERGE_SECONDS apart."""
    if len(times) == 0:
        return 0
    return int(1 + np.sum(np.diff(np.sort(times)) > MERGE_SECONDS))


def distinct(times, snrs):
    """Indices of one event per moment: the brightest of each chain within MERGE_SECONDS."""
    order = np.argsort(times)
    keep, chain = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if times[b] - times[a] <= MERGE_SECONDS:
            chain.append(b)
        else:
            keep.append(max(chain, key=lambda i: snrs[i]))
            chain = [b]
    keep.append(max(chain, key=lambda i: snrs[i]))
    return keep


def groups(events, keep=STORE_LOG10FAP):
    """The sharp groups of one beam at log10 FAP <= keep.

    events: rows (dm, snr, time, width) of its clustered_candidates.txt. Each group: dm (median of its
    pulses), tol, k (pulses), adjacent, side, log10fap and pulses [[time, dm, snr, width]] in time order.
    """
    ev = np.asarray(events, dtype=float).reshape(-1, 4)
    ev = ev[(ev[:, 0] >= MIN_DM - 1) & (ev[:, 1] >= MIN_SNR)]
    if len(ev) < 2:
        return []
    dm, snr, t, w = ev[:, 0], ev[:, 1], ev[:, 2], ev[:, 3]
    used = np.zeros(len(dm), bool)
    anchors = [i for i in np.argsort(-snr, kind='stable') if snr[i] >= ANCHOR_SNR and dm[i] >= MIN_DM][:MAX_ANCHORS]
    trials = math.log10(max(1, len(anchors)))
    out = []
    for a in anchors:
        if used[a]:
            continue
        tol = max(TOL_MIN, TOL_FRACTION * dm[a])
        on = np.flatnonzero((np.abs(dm - dm[a]) <= tol) & ~used)
        members = [on[i] for i in distinct(t[on], snr[on])]
        used[on] = True
        if a not in members or len(members) < 2:
            continue
        d = dm - dm[a]
        adjacent = moments(t[(np.abs(d) > tol) & (np.abs(d) <= 3 * tol)])
        side = sum(moments(t[(d > (2 * o - 1) * tol) & (d <= (2 * o + 1) * tol)]) for o in SIDE_OFFSETS)
        if adjacent > MAX_ADJACENT or side > MAX_SIDE:
            continue
        low, high = max(MIN_DM - 1, dm[a] / BAND), dm[a] * BAND
        band = (dm >= low) & (dm <= high) & (np.abs(d) > 3 * tol)
        span = max(2 * tol, (high - low) - 6 * tol)
        chance = []
        for floor in (6.0, 7.0):
            lam = max(LAMBDA_FLOOR, moments(t[band & (snr >= floor)]) * 2 * tol / span)
            m = sum(1 for i in members if i != a and snr[i] >= floor)
            chance.append(log10_poisson_sf(m, lam))
        log10fap = min(0.0, min(chance) + math.log10(2)) + trials
        if log10fap > keep:
            continue
        out.append({'dm': round(float(np.median(dm[members])), 2), 'tol': round(tol, 2), 'k': len(members),
                    'adjacent': adjacent, 'side': side, 'log10fap': round(log10fap, 2),
                    'pulses': sorted([round(float(t[i]), 3), round(float(dm[i]), 2), round(float(snr[i]), 2),
                                      int(w[i])] for i in members)})
    return out


def phase_spread(times, period):
    """The largest distance in turns of a time from their circular mean phase at this period."""
    phase = (np.asarray(times, dtype=float) / period) % 1.0
    mean = np.angle(np.exp(2j * np.pi * phase).mean()) / (2 * np.pi)
    return float(np.max(np.abs((phase - mean + 0.5) % 1.0 - 0.5)))


def instrumental(times, early_cycle=False):
    """Whether the pulses keep the phase of the data's level steps (LT5_BLOCKS or EC_ROWS)."""
    return any(phase_spread(times, p) * p <= BLOCK_TOLERANCE for p in (EC_ROWS if early_cycle else LT5_BLOCKS))
