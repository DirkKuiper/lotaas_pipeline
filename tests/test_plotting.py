import numpy as np

from lotaas_reprocessing import plotting


def test_a_beam_masked_everywhere_still_gets_its_diagnostic_plot(tmp_path):
    # L597867 SAP001 B037 (22 October 2026): 79% masked, every averaged cell holding a masked sample; the plot's
    # axis limits were NaN and dedispersion of the beam failed on them.
    rng = np.random.default_rng(1)
    data = rng.normal(1.0, 0.1, (64, 4096)).astype(np.float32)
    mask = np.ones_like(data, dtype=bool)
    masked = np.full_like(data, np.nan)
    t = np.arange(data.shape[1]) * 0.00786432
    nu = 151.04 - 0.0488 * np.arange(data.shape[0])
    path = tmp_path / 'plot.png'
    plotting.rfi_diagnostic_plot(masked, data, mask, t, nu, 16, filename='beam', save_path=str(path))
    assert path.stat().st_size > 0
