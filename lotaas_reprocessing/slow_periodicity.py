"""A second periodicity search for slow pulsars, on data the zero-DM filter has not touched (Slow_DM_trials).

The zero-DM filter (preprocess.zero_dm, since 27 September 2026) subtracts at each sample the mean over
channels. A pulse that lasts longer than its own sweep across the band is mostly that mean, and goes with it:
PSR J0311+1402 (P 40.9 s, DM 1.43, a 0.46 s envelope against a 0.20 s sweep over 119-151 MHz) was found
blind on 23 September at statistic 85 without the filter, and the current search does not find it at all.
Measured on 21 October 2026 (benchmarks/j0311-2026-10-20/zerodm): 40 slow pulsars injected into LT5 beams
(P 5-200 s, DM 1-60, ideal fold S/N 10-60) and J0311+1402's own beam, searched as production does with and
without the filter. Without it the FFT found 21 against 14, 4 against 0 above 60 s, 8 against 3 below DM 5,
and J0311+1402 at statistic 325 at its period; the folds that passed the review page's single-beam checks by
chance were 8 either way.

So the filter stays for the main search (interference, a tied-array beam's broadband wander, the FRB lane)
and this pass searches a copy of the data taken just before it. Its trials need only low DMs and coarse
samples: a slow pulsar nearby enough for the filter to matter sits at low DM, and 31 ms keeps 32 harmonics
of a 2 s period. It has its own folds (a slow pulsar no longer competes with the main search's), the same
cross-beam veto among the pass's own peaks, and products named with PREFIX.

Settings (slow_periodicity, top level), merged over the periodicity block's:
    enabled, zero_dm (whether this pass filters too, false: the point of it), dm_plan, and any periodicity
    key (period_min_seconds, harmonics, threshold, max_folds, ...).
"""
from copy import deepcopy

PREFIX = 'slow_'
TRIALS = 'Slow_DM_trials'
DEFAULTS = {
    'enabled': False,
    'zero_dm': False,
    # 31.5 ms samples; a step of 0.25 smears 14 ms across the band at most, half a sample.
    'dm_plan': [{'low_dm': 0.0, 'high_dm': 100.0, 'ddm': 0.25, 'downsample': 4}],
    'period_min_seconds': 1.0,
    'period_max_seconds': 300.0,
    'harmonics': [1, 2, 4, 8, 16, 32],
    'max_folds': 16,
    # The catalogue folds run once, in the main search.
    'catalogue_fold': False,
}
OWN = ('enabled', 'zero_dm', 'dm_plan')


def settings_of(settings):
    """The pass's settings: DEFAULTS under settings['slow_periodicity']."""
    result = deepcopy(DEFAULTS)
    result.update((settings or {}).get('slow_periodicity') or {})
    return result


def enabled(settings):
    return bool(settings_of(settings)['enabled'])


def preprocessing_of(preprocessing, slow):
    """The main search's preprocessing with this pass's zero-DM choice."""
    return dict(preprocessing or {}, zero_dm=bool(slow.get('zero_dm', False)))


def periodicity_config(main, slow):
    """The periodicity configuration of this pass: the main one's, with the pass's own keys over it."""
    config = dict(main or {})
    config.update({k: v for k, v in slow.items() if k not in OWN})
    return config


def metadata_view(metadata):
    """The beam's metadata as the periodicity search reads it for this pass: its plan, preprocessing and config."""
    slow = metadata['slow_periodicity']
    return dict(metadata, periodicity_dm_plan=slow['dm_plan'], periodicity=metadata['slow_periodicity_config'],
                preprocessing=metadata['slow_preprocessing'])

