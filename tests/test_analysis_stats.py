import numpy as np
import pandas as pd

from inragwetrust.analysis import stats


def test_plan_constants():
    assert stats.BOOTSTRAP == 10_000 and stats.SEED == 20261009


def test_holm_textbook_and_nan():
    assert np.allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    out = stats.holm([0.01, np.nan])
    assert np.isclose(out[0], 0.02) and np.isnan(out[1])


def test_mcnemar_exact():
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert np.isclose(stats.mcnemar_exact(10, 0), 2 * 0.5 ** 10)
    assert np.isclose(stats.mcnemar_exact(5, 5), 1.0)


def test_cluster_bootstrap_constant_and_reproducible():
    lo, hi = stats.cluster_bootstrap([2, 4, 6], [4, 8, 12], n_boot=200)
    assert np.isclose(lo, 0.5) and np.isclose(hi, 0.5)
    a = stats.cluster_bootstrap([1, 0, 3, 2], [4, 4, 4, 4], n_boot=500)
    assert a == stats.cluster_bootstrap([1, 0, 3, 2], [4, 4, 4, 4], n_boot=500)


def test_fit_gee_recovers_a_clear_effect():
    rng = np.random.default_rng(0)
    q = np.repeat(np.arange(200), 4)
    x = np.tile([0, 0, 1, 1], 200)
    y = (rng.random(800) < np.where(x == 1, 0.7, 0.2)).astype(int)
    res, cov, note = stats.fit_gee("y ~ x", pd.DataFrame({"y": y, "x": x, "question_id": q}))
    est, lo, hi, p = stats.odds_ratio(res, "x")
    assert cov == "exchangeable" and note == ""
    assert 4 < est < 20 and lo > 1 and p < 0.001


def test_fit_gee_separation_is_not_a_silent_result():
    q = np.repeat(np.arange(50), 2)
    x = np.tile([0, 1], 50)
    res, cov, note = stats.fit_gee("y ~ x", pd.DataFrame({"y": x, "x": x, "question_id": q}))
    assert "D253" in note and (res is None or cov == "independence")
    