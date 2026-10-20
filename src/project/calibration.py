"""Calibración, incertidumbre, predicción conforme y clasificación selectiva."""

from __future__ import annotations

from typing import Any

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 no tiene trapezoid


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(z, float) / T
    z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(probs: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    probs, y = np.asarray(probs), np.asarray(y)
    return float(-np.log(probs[np.arange(len(y)), y] + eps).mean())


def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    """Brier multiclase: media de ||p − onehot(y)||² (rango [0, 2])."""
    probs, y = np.asarray(probs, float), np.asarray(y)
    oh = np.eye(probs.shape[1])[y]
    return float(((probs - oh) ** 2).sum(1).mean())


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15,
                               strategy: str = "uniform") -> float:
    """ECE top-label: sum_b (|B_b|/n) |acc(B_b) − conf(B_b)|.

    ``strategy="quantile"`` usa bins de igual masa (menor varianza con pocas muestras).
    """
    probs, y = np.asarray(probs), np.asarray(y)
    conf, acc = probs.max(1), (probs.argmax(1) == y).astype(float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(conf, np.linspace(0, 1, n_bins + 1)))
        edges[0] = 0.0
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def binary_ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """ECE para probabilidades binarias (p. ej. probabilidad de píxel = célula)."""
    p = np.asarray(p, float).ravel(); y = np.asarray(y).ravel().astype(float)
    edges = np.linspace(0, 1, n_bins + 1); ece = 0.0
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def reliability_table(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict[str, np.ndarray]:
    """Datos del diagrama de confiabilidad (confianza media, exactitud y conteo por bin)."""
    probs, y = np.asarray(probs), np.asarray(y)
    conf, acc = probs.max(1), (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    c, a, n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        c.append(conf[m].mean() if m.any() else np.nan)
        a.append(acc[m].mean() if m.any() else np.nan)
        n.append(int(m.sum()))
    return {"edges": edges, "confidence": np.array(c), "accuracy": np.array(a), "count": np.array(n)}


def plot_reliability(probs: np.ndarray, y: np.ndarray, n_bins: int = 15, ax: Any = None,
                     label: str = "") -> Any:
    import matplotlib.pyplot as plt

    t = reliability_table(probs, y, n_bins)
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    centers = 0.5 * (t["edges"][1:] + t["edges"][:-1])
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.bar(centers, t["accuracy"], width=1 / n_bins, alpha=0.4, edgecolor="k", label=label or None)
    ax.plot(t["confidence"], t["accuracy"], "o-", ms=3)
    ax.set_xlabel("Confianza"); ax.set_ylabel("Exactitud"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"ECE = {expected_calibration_error(probs, y, n_bins):.3f}")
    return ax


def fit_temperature(logits: np.ndarray, y: np.ndarray, bounds: tuple[float, float] = (0.05, 10.0)) -> float:
    """Temperature scaling (Guo et al., 2017): minimiza la NLL **en validación**."""
    from scipy.optimize import minimize_scalar

    logits, y = np.asarray(logits, float), np.asarray(y)
    res = minimize_scalar(lambda T: nll(softmax(logits, T), y), bounds=bounds, method="bounded")
    return float(res.x)


def calibration_summary(probs: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return dict(ece=expected_calibration_error(probs, y), brier=brier_score(probs, y),
                nll=nll(probs, y), accuracy=float((np.asarray(probs).argmax(1) == np.asarray(y)).mean()))


# ------------------------------------------------------------ incertidumbre
def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, float), 1e-12, 1)
    return -(p * np.log(p)).sum(-1)


def ensemble_uncertainty(member_probs: np.ndarray) -> dict[str, np.ndarray]:
    """``member_probs``: (M, n, K) de un ensamble profundo o pasadas MC-Dropout/TTA.

    Descomposición: entropía total = aleatoria (esperada) + epistémica (información mutua).
    """
    mp = np.asarray(member_probs, float)
    mean = mp.mean(0)
    total = predictive_entropy(mean)
    aleatoric = predictive_entropy(mp).mean(0)
    return {"mean": mean, "total": total, "aleatoric": aleatoric, "epistemic": total - aleatoric}


def risk_coverage_curve(confidence: np.ndarray, correct: np.ndarray) -> dict[str, Any]:
    """Curva riesgo–cobertura (predicción selectiva) y su área (AURC)."""
    conf, ok = np.asarray(confidence, float), np.asarray(correct, bool)
    order = np.argsort(-conf)
    ok = ok[order]
    n = len(ok)
    cov = np.arange(1, n + 1) / n
    risk = 1 - np.cumsum(ok) / np.arange(1, n + 1)
    return {"coverage": cov, "risk": risk, "aurc": float(_trapz(risk, cov)),
            "thresholds": conf[order]}


def coverage_for_accuracy(confidence: np.ndarray, correct: np.ndarray, target_acc: float) -> float:
    """Máxima cobertura con exactitud >= objetivo; 1 − cobertura = fracción a revisión humana."""
    rc = risk_coverage_curve(confidence, correct)
    ok = rc["risk"] <= 1 - target_acc
    return float(rc["coverage"][ok].max()) if ok.any() else 0.0


# --------------------------------------------------------- predicción conforme
def conformal_threshold(cal_probs: np.ndarray, cal_y: np.ndarray, alpha: float = 0.1) -> float:
    """Split conformal (LAC, Angelopoulos & Bates 2021): puntaje s = 1 − p_y.

    Garantía marginal: P(y ∈ C(x)) >= 1 − α si calibración y prueba son intercambiables.
    """
    p = np.asarray(cal_probs); y = np.asarray(cal_y)
    s = 1 - p[np.arange(len(y)), y]
    n = len(s)
    q = np.ceil((n + 1) * (1 - alpha)) / n
    return float(np.quantile(s, min(q, 1.0), method="higher"))


def conformal_sets(probs: np.ndarray, qhat: float) -> np.ndarray:
    """Conjuntos de predicción (matriz booleana n x K)."""
    return (1 - np.asarray(probs)) <= qhat


def conformal_report(sets: np.ndarray, y: np.ndarray) -> dict[str, float]:
    y = np.asarray(y)
    return {"coverage": float(sets[np.arange(len(y)), y].mean()),
            "avg_set_size": float(sets.sum(1).mean()),
            "singleton_rate": float((sets.sum(1) == 1).mean())}
