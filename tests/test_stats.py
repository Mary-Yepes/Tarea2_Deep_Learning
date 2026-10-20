import numpy as np
import pandas as pd
import pytest

from project.stats import (
    bayesian_signed_rank,
    bootstrap_ci,
    cliffs_delta,
    compare_models,
    corrected_resampled_ttest,
    fit_power_law,
    five_by_two_cv_ftest,
    five_by_two_cv_ttest,
    hierarchical_bootstrap_ci,
    holm_adjust,
    icc_oneway,
    mcnemar_exact,
    nemenyi_cd,
    paired_test,
)


def synthetic_scores(seed=0, n=80):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.6, 0.05, n)
    return pd.DataFrame({"M1": base, "M2": base + 0.05 + rng.normal(0, 0.01, n),
                         "M3": base + rng.normal(0, 0.01, n)})


def test_compare_models_detects_true_difference_only():
    res = compare_models(synthetic_scores())
    assert res["friedman_p"] < 0.05
    pw = res["pairwise"].set_index(["A", "B"])
    assert pw.loc[("M1", "M2"), "reject"] and not pw.loc[("M1", "M3"), "reject"]
    assert res["mean_ranks"].index[0] == "M2"


def test_bootstrap_percentile_and_bca_contain_mean():
    x = synthetic_scores()["M2"].to_numpy()
    for method in ("percentile", "bca"):
        m, lo, hi = bootstrap_ci(x, n_boot=2000, method=method)
        assert lo < m < hi


def test_bootstrap_coverage_close_to_nominal():
    rng = np.random.default_rng(1); hits = 0
    for _ in range(200):
        x = rng.exponential(1.0, 40)
        _, lo, hi = bootstrap_ci(x, n_boot=1000, method="bca", seed=int(rng.integers(1e9)))
        hits += lo <= 1.0 <= hi
    assert 0.85 <= hits / 200 <= 0.99


def test_holm_known_values():
    adj = holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])


def test_cliffs_delta_extremes():
    assert cliffs_delta(np.array([2, 3]), np.array([0, 1])) == 1.0
    assert cliffs_delta(np.array([1, 2]), np.array([1, 2])) == 0.0


def test_corrected_ttest_more_conservative():
    d = np.random.default_rng(0).normal(0.01, 0.01, 10)
    t_corr, p_corr = corrected_resampled_ttest(d, n_train=900, n_test=100)
    t_naive = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    assert abs(t_corr) < abs(t_naive) and 0 <= p_corr <= 1


def test_5x2cv_shapes_and_null():
    rng = np.random.default_rng(0)
    t, p = five_by_two_cv_ttest(rng.normal(0, 0.01, (5, 2)))
    f, pf = five_by_two_cv_ftest(rng.normal(0, 0.01, (5, 2)))
    assert 0 <= p <= 1 and 0 <= pf <= 1


def test_mcnemar():
    y = np.array([1] * 50); a = y.copy(); b = y.copy(); b[:10] = 0
    assert mcnemar_exact(y, a, b) < 0.01
    assert mcnemar_exact(y, a, a) == 1.0


def test_paired_and_nemenyi_and_icc():
    s = synthetic_scores()
    r = paired_test(s["M2"], s["M1"], n_boot=2000)
    assert r["ci_low"] > 0 and r["p_wilcoxon"] < 0.001
    assert nemenyi_cd(3, 80) == pytest.approx(2.343 * np.sqrt(3 * 4 / (6 * 80)))
    g = np.repeat(np.arange(10), 20); v = g * 1.0 + np.random.default_rng(0).normal(0, 0.1, 200)
    assert icc_oneway(v, g) > 0.9


def test_hierarchical_bootstrap_wider_than_naive_with_clusters():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(8), 25); v = rng.normal(0, 1, 8)[g] + rng.normal(0, 0.2, 200)
    _, lo_h, hi_h = hierarchical_bootstrap_ci(v, g, n_boot=1000)
    _, lo_n, hi_n = bootstrap_ci(v, n_boot=1000)
    assert (hi_h - lo_h) > 1.5 * (hi_n - lo_n)


def test_bayesian_signed_rank_and_power_law():
    s = synthetic_scores()
    r = bayesian_signed_rank(s["M2"], s["M1"], rope=0.01, n_samples=3000)
    assert r["p_A_better"] > 0.95
    n = np.array([50, 100, 200, 400, 800.0]); err = 2.0 * n ** -0.5 + 0.1
    p = fit_power_law(n, err)
    assert p["b"] == pytest.approx(0.5, rel=0.05) and p["c"] == pytest.approx(0.1, abs=0.01)
