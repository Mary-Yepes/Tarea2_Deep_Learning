"""Detección de deriva: covariables, predicciones y concepto (simulación RQ8)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10, eps: float = 1e-6, smoothing: float = 0.0) -> float:
    """Population Stability Index con cortes por cuantiles de la referencia.

    Reglas usuales: < 0.1 estable, 0.1–0.25 moderado, > 0.25 deriva significativa.
    **Ojo con ventanas pequeñas:** bajo la nula, E[PSI] ≈ (B − 1)(1/n + 1/m), por lo
    que con B = 10 y n = 50 el PSI esperado ya ronda 0.2. Con ``smoothing`` > 0 se
    usa suavizado aditivo (conteo + α) en lugar de recortar a ``eps``, lo que evita
    que un bin vacío dispare log(eps).
    """
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    hr, hc = np.histogram(ref, edges)[0].astype(float), np.histogram(cur, edges)[0].astype(float)
    if smoothing > 0:
        r = (hr + smoothing) / (hr.sum() + smoothing * len(hr))
        c = (hc + smoothing) / (hc.sum() + smoothing * len(hc))
    else:
        r = np.clip(hr / len(ref), eps, None); c = np.clip(hc / len(cur), eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def chi2_categorical(ref: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
    """Prueba chi² de homogeneidad para distribuciones categóricas (p. ej. clases predichas)."""
    cats = np.union1d(np.unique(ref), np.unique(cur))
    t = np.array([[np.sum(ref == c) for c in cats], [np.sum(cur == c) for c in cats]])
    t = t[:, t.sum(0) > 0]
    if t.shape[1] < 2:
        return 0.0, 1.0
    r = stats.chi2_contingency(t)
    return float(r.statistic), float(r.pvalue)


def _rbf(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    d = (x ** 2).sum(1)[:, None] + (y ** 2).sum(1)[None, :] - 2 * x @ y.T
    return np.exp(-gamma * np.maximum(d, 0))


def mmd2(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    """MMD² insesgado con kernel RBF (Gretton et al., 2012). Ancho por mediana si ``gamma`` es None."""
    x = np.atleast_2d(np.asarray(x, float)); y = np.atleast_2d(np.asarray(y, float))
    if x.shape[0] == 1 and x.shape[1] > 1 and y.shape[0] == 1:
        x, y = x.T, y.T
    if gamma is None:
        z = np.vstack([x, y])
        sub = z[np.random.default_rng(0).choice(len(z), min(len(z), 500), replace=False)]
        d = np.sqrt(((sub[:, None] - sub[None]) ** 2).sum(-1))
        med = np.median(d[d > 0]) if (d > 0).any() else 1.0
        gamma = 1.0 / (2 * med ** 2)
    m, n = len(x), len(y)
    kxx, kyy, kxy = _rbf(x, x, gamma), _rbf(y, y, gamma), _rbf(x, y, gamma)
    np.fill_diagonal(kxx, 0); np.fill_diagonal(kyy, 0)
    return float(kxx.sum() / (m * (m - 1)) + kyy.sum() / (n * (n - 1)) - 2 * kxy.mean())


def mmd_permutation_test(x: np.ndarray, y: np.ndarray, n_perm: int = 200, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[:, None] if x.ndim == 1 else x; y = y[:, None] if y.ndim == 1 else y
    z = np.vstack([x, y]); m = len(x)
    stat = mmd2(x, y)
    rng = np.random.default_rng(seed); cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(len(z))
        cnt += mmd2(z[p[:m]], z[p[m:]]) >= stat
    return stat, float((cnt + 1) / (n_perm + 1))


def drift_report(ref: np.ndarray, cur: np.ndarray) -> dict[str, float]:
    ks = stats.ks_2samp(ref, cur)
    return dict(psi=psi(ref, cur), ks_stat=float(ks.statistic), ks_p=float(ks.pvalue))


@dataclass
class WindowedDriftMonitor:
    """Monitor por ventanas: compara cada ventana del flujo con la referencia.

    Alarma si PSI > ``psi_thr`` **o** p(KS) < ``ks_alpha`` en alguna de las
    características (corrección de Bonferroni sobre características). Con
    ``patience`` > 1 exige ventanas consecutivas en alarma (reduce falsas alarmas a
    costa de retardo).
    """

    reference: np.ndarray
    window: int = 200
    psi_thr: float = 0.2
    ks_alpha: float = 0.01
    patience: int = 1
    _buf: deque[np.ndarray] = field(default_factory=deque, init=False, repr=False)
    _streak: int = field(default=0, init=False)
    history: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.reference = np.asarray(self.reference, float)
        if self.reference.ndim == 1:
            self.reference = self.reference[:, None]

    def score_window(self, cur: np.ndarray) -> dict[str, Any]:
        cur = np.asarray(cur, float)
        cur = cur[:, None] if cur.ndim == 1 else cur
        d = self.reference.shape[1]
        psis = [psi(self.reference[:, j], cur[:, j]) for j in range(d)]
        kps = [stats.ks_2samp(self.reference[:, j], cur[:, j]).pvalue for j in range(d)]
        alarm = (max(psis) > self.psi_thr) or (min(kps) < self.ks_alpha / d)
        return {"psi_max": float(max(psis)), "ks_p_min": float(min(kps)), "raw_alarm": bool(alarm)}

    def update(self, x: np.ndarray) -> dict[str, Any] | None:
        """Añade observaciones ``(n, d)`` (o ``(n,)`` si d = 1); devuelve el
        resultado de la última ventana completada, o ``None``."""
        arr = np.asarray(x, float)
        arr = arr.reshape(-1, self.reference.shape[1])
        for row in arr:
            self._buf.append(row)
        res: dict[str, Any] | None = None
        while len(self._buf) >= self.window:
            win = np.array([self._buf.popleft() for _ in range(self.window)])
            res = self.score_window(win)
            self._streak = self._streak + 1 if res["raw_alarm"] else 0
            res["alarm"] = self._streak >= self.patience
            self.history.append(res)
        return res


class PageHinkley:
    """Detector Page–Hinkley para deriva de concepto sobre la pérdida/error en línea."""

    def __init__(self, delta: float = 0.005, lam: float = 20.0, alpha: float = 0.9999):
        self.delta, self.lam, self.alpha = delta, lam, alpha
        self.reset()

    def reset(self) -> None:
        self.mean, self.n, self.cum, self.min_cum = 0.0, 0, 0.0, 0.0

    def update(self, x: float) -> bool:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.cum = self.alpha * self.cum + (x - self.mean - self.delta)
        self.min_cum = min(self.min_cum, self.cum)
        return (self.cum - self.min_cum) > self.lam
