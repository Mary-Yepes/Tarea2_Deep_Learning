import numpy as np
import pytest

from project.calibration import (
    brier_score,
    conformal_report,
    conformal_sets,
    conformal_threshold,
    coverage_for_accuracy,
    ensemble_uncertainty,
    expected_calibration_error,
    fit_temperature,
    nll,
    risk_coverage_curve,
    softmax,
)
from project.drift import PageHinkley, WindowedDriftMonitor, chi2_categorical, mmd2, mmd_permutation_test, psi


def overconfident(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, (n, 3)); p_true = softmax(z)
    y = (p_true.cumsum(1) > rng.random(n)[:, None]).argmax(1)
    return 3 * z, y


def test_temperature_recovers_scale_and_keeps_argmax():
    logits, y = overconfident()
    T = fit_temperature(logits, y)
    assert 2.0 < T < 4.0
    assert expected_calibration_error(softmax(logits / T), y) < expected_calibration_error(softmax(logits), y)
    assert nll(softmax(logits / T), y) < nll(softmax(logits), y)
    assert (softmax(logits / T).argmax(1) == softmax(logits).argmax(1)).all()


def test_brier_bounds():
    p = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert brier_score(p, np.array([0, 1])) == 0.0 and brier_score(p, np.array([1, 0])) == 2.0


def test_conformal_coverage_holds():
    logits, y = overconfident(1, 6000)
    p = softmax(logits / 3)
    q = conformal_threshold(p[:3000], y[:3000], alpha=0.1)
    rep = conformal_report(conformal_sets(p[3000:], q), y[3000:])
    assert rep["coverage"] >= 0.88


def test_risk_coverage_and_selective():
    conf = np.linspace(1, 0, 100); ok = np.r_[np.ones(80, bool), np.zeros(20, bool)]
    rc = risk_coverage_curve(conf, ok)
    assert rc["risk"][79] == 0.0 and coverage_for_accuracy(conf, ok, 1.0) == pytest.approx(0.8)


def test_ensemble_uncertainty_decomposition():
    mp = np.stack([np.array([[0.9, 0.1]]), np.array([[0.1, 0.9]])])
    u = ensemble_uncertainty(mp)
    assert u["epistemic"][0] > 0.3


def test_psi_mmd_ks_detect_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.05 and psi(ref, rng.normal(1, 1, 5000)) > 0.25
    x, y0, y1 = rng.normal(0, 1, (200, 2)), rng.normal(0, 1, (200, 2)), rng.normal(0.7, 1, (200, 2))
    assert mmd2(x, y1) > mmd2(x, y0)
    _, p = mmd_permutation_test(x[:100], y1[:100], n_perm=100)
    assert p < 0.05


def test_chi2_categorical():
    a = np.array([0] * 50 + [1] * 50); b = np.array([0] * 90 + [1] * 10)
    assert chi2_categorical(a, b)[1] < 0.001 and chi2_categorical(a, a)[1] > 0.5


def test_windowed_monitor_alarm_after_shift():
    rng = np.random.default_rng(0)
    mon = WindowedDriftMonitor(rng.normal(0, 1, (2000, 2)), window=200)
    for _ in range(3):
        assert not mon.update(rng.normal(0, 1, (200, 2)))["alarm"]
    assert mon.update(rng.normal(1.0, 1, (200, 2)))["alarm"]


def test_page_hinkley_detects_error_increase():
    ph = PageHinkley(delta=0.01, lam=5)
    rng = np.random.default_rng(0)
    assert not any(ph.update(float(rng.random() < 0.1)) for _ in range(300))
    assert any(ph.update(float(rng.random() < 0.6)) for _ in range(300))
