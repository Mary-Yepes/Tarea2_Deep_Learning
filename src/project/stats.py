"""Inferencia estadística para comparar modelos (plan estadístico pre-registrado).

Unidades de análisis: imagen (segmentación) y ejemplo/fold (NLP). Nunca píxeles.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------- bootstrap
def bootstrap_ci(x: np.ndarray, stat: Callable[..., Any] = np.mean, n_boot: int = 10_000,
                 alpha: float = 0.05, seed: int = 0, method: str = "percentile") -> tuple[float, float, float]:
    """IC bootstrap sobre unidades de análisis. Devuelve (estimador, lo, hi).

    ``method``: ``"percentile"`` o ``"bca"`` (sesgo corregido y acelerado, Efron 1987).
    ``stat`` debe aceptar ``axis``.
    """
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = stat(x[idx], axis=1)
    theta = float(stat(x))
    if method == "percentile" or len(x) < 3:
        lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return theta, float(lo), float(hi)
    if method != "bca":
        raise ValueError(method)
    prop = np.clip((boots < theta).mean() + 0.5 * (boots == theta).mean(), 1e-10, 1 - 1e-10)
    z0 = stats.norm.ppf(prop)
    jack = np.array([stat(np.delete(x, i)) for i in range(len(x))])
    d = jack.mean() - jack
    den = 6.0 * (d ** 2).sum() ** 1.5
    a = (d ** 3).sum() / den if den > 0 else 0.0
    zs = stats.norm.ppf([alpha / 2, 1 - alpha / 2])
    adj = stats.norm.cdf(z0 + (z0 + zs) / (1 - a * (z0 + zs)))
    lo, hi = np.percentile(boots, 100 * adj)
    return theta, float(lo), float(hi)


def hierarchical_bootstrap_ci(values: np.ndarray, groups: np.ndarray, n_boot: int = 10_000,
                              alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """IC bootstrap en dos niveles: remuestrea grupos (``sample_id``/placa) y, dentro
    de cada grupo, unidades. Respeta la correlación intra-placa."""
    v, g = np.asarray(values, float), np.asarray(groups)
    ug = np.unique(g)
    by = [v[g == k] for k in ug]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        out[b] = np.concatenate([rng.choice(by[k], len(by[k])) for k in pick]).mean()
    lo, hi = np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def fmt_ci(est: float, lo: float, hi: float, nd: int = 3) -> str:
    """Formato de reporte obligatorio: media [IC 95 %]."""
    return f"{est:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


# --------------------------------------------------------- tamaños de efecto
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (len(a) * len(b)))


def cliffs_magnitude(d: float) -> str:
    """Umbrales de Romano et al. (2006)."""
    d = abs(d)
    return "despreciable" if d < 0.147 else "pequeño" if d < 0.33 else "mediano" if d < 0.474 else "grande"


def cohen_dz(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


# ------------------------------------------------------ comparaciones múltiples
def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float); m = len(p)
    adj = np.empty(m); running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def compare_models(scores: pd.DataFrame, alpha: float = 0.05, delta: float | None = None) -> dict[str, Any]:
    """Filas = unidades pareadas (imágenes/folds), columnas = modelos (>= 3).

    Friedman + Wilcoxon por pares con corrección de Holm + tamaños de efecto. Si se
    da ``delta`` (efecto mínimo relevante), indica si la diferencia media supera δ.
    """
    scores = scores.dropna()
    models = list(scores.columns)
    if len(models) < 3:
        raise ValueError("Friedman requiere >= 3 modelos; use paired_test para 2")
    fr = stats.friedmanchisquare(*[scores[m] for m in models])
    rows = []
    for a, b in combinations(models, 2):
        d = scores[a] - scores[b]
        try:
            p = stats.wilcoxon(scores[a], scores[b]).pvalue
        except ValueError:  # todas las diferencias son cero
            p = 1.0
        cd = cliffs_delta(scores[a].to_numpy(), scores[b].to_numpy())
        rows.append(dict(A=a, B=b, mean_diff=d.mean(), cohen_dz=cohen_dz(scores[a], scores[b]),
                         cliffs_delta=cd, cliffs_mag=cliffs_magnitude(cd), p_raw=p))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p_raw"].to_numpy()); out["reject"] = out["p_holm"] < alpha
    if delta is not None:
        out["practically_relevant"] = out["mean_diff"].abs() >= delta
    ranks = scores.rank(axis=1, ascending=False).mean().sort_values()
    return {"friedman_stat": float(fr.statistic), "friedman_p": float(fr.pvalue),
            "mean_ranks": ranks, "pairwise": out, "n": len(scores),
            "cd_nemenyi": nemenyi_cd(len(models), len(scores), alpha)}


def paired_test(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> dict[str, float]:
    """Dos modelos sobre las mismas unidades: Wilcoxon pareado + IC bootstrap de la diferencia."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    try:
        p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        p = 1.0
    est, lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    return dict(mean_diff=est, ci_low=lo, ci_high=hi, p_wilcoxon=p,
                cohen_dz=cohen_dz(a, b), cliffs_delta=cliffs_delta(a, b))


_Q_NEMENYI_05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
                 9: 3.102, 10: 3.164}


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Diferencia crítica de Nemenyi (Demšar, 2006) para el diagrama CD."""
    if alpha == 0.05 and k in _Q_NEMENYI_05:
        q = _Q_NEMENYI_05[k]
    else:  # q_alpha = rango studentizado / sqrt(2)
        q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


def plot_cd_diagram(mean_ranks: pd.Series, cd: float, ax: Any = None, title: str = "") -> Any:
    """Diagrama de diferencia crítica (estilo Demšar) con barras de grupos no distinguibles."""
    import matplotlib.pyplot as plt

    r = mean_ranks.sort_values(); k = len(r)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.6 + 0.35 * k))
    ax.set_xlim(0.5, k + 0.5); ax.set_ylim(-k * 0.5 - 0.5, 1.5)
    ax.hlines(1, 1, k, color="k"); ax.set_yticks([])
    for i in range(1, k + 1):
        ax.vlines(i, 0.9, 1.1, color="k"); ax.text(i, 1.25, str(i), ha="center")
    for i, (name, rank) in enumerate(r.items()):
        y = -0.4 * (i + 1)
        ax.plot([rank, rank], [1, y], color="0.4", lw=0.8)
        ax.text(rank + (0.05 if i % 2 else -0.05), y, f"{name} ({rank:.2f})",
                ha="left" if i % 2 else "right", va="center", fontsize=8)
    vals = r.to_numpy(); y = 0.6
    for i in range(k):
        j = max(jj for jj in range(i, k) if vals[jj] - vals[i] <= cd)
        if j > i:
            ax.hlines(y, vals[i] - 0.03, vals[j] + 0.03, lw=3, color="tab:red"); y -= 0.12
    ax.hlines(1.45, 1, 1 + cd, lw=2); ax.text(1 + cd / 2, 1.5, f"CD={cd:.2f}", ha="center", fontsize=8)
    for s in ("left", "right", "top", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([]); ax.set_title(title)
    return ax


# ------------------------------------------------------------- NLP / CV
def corrected_resampled_ttest(diffs: np.ndarray, n_train: int, n_test: int) -> tuple[float, float]:
    """t remuestreada corregida (Nadeau & Bengio, 2003) sobre diferencias por repetición."""
    d = np.asarray(diffs, float); k = len(d)
    v = d.var(ddof=1)
    if v == 0:
        return (0.0, 1.0) if d.mean() == 0 else (float(np.sign(d.mean()) * np.inf), 0.0)
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * v)
    return float(t), float(2 * stats.t.sf(abs(t), k - 1))


def five_by_two_cv_ttest(diffs: np.ndarray) -> tuple[float, float]:
    """5×2cv t pareada (Dietterich, 1998). ``diffs`` de forma (5, 2)."""
    p = np.asarray(diffs, float).reshape(5, 2)
    pbar = p.mean(1)
    s2 = ((p - pbar[:, None]) ** 2).sum(1)
    t = p[0, 0] / np.sqrt(s2.mean())
    return float(t), float(2 * stats.t.sf(abs(t), 5))


def five_by_two_cv_ftest(diffs: np.ndarray) -> tuple[float, float]:
    """5×2cv F combinada (Alpaydin, 1999), más robusta que la t de Dietterich."""
    p = np.asarray(diffs, float).reshape(5, 2)
    s2 = ((p - p.mean(1, keepdims=True)) ** 2).sum(1)
    f = (p ** 2).sum() / (2 * s2.sum())
    return float(f), float(stats.f.sf(f, 10, 5))


def mcnemar_exact(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    b01, b10 = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    n = b01 + b10
    return 1.0 if n == 0 else float(stats.binomtest(b01, n, 0.5).pvalue)


# --------------------------------------------------------- estructura y varianza
def mixed_model_check(df: pd.DataFrame, score: str = "score", model: str = "model",
                      group: str = "sample_id") -> Any:
    """Modelo mixto ``score ~ C(model)`` con intercepto aleatorio por ``group``
    (statsmodels.mixedlm). Verifica que la conclusión no se deba a la correlación
    intra-placa. Devuelve el resultado ajustado."""
    import statsmodels.formula.api as smf

    return smf.mixedlm(f"{score} ~ C({model})", df, groups=df[group]).fit(reml=True)


def icc_oneway(values: np.ndarray, groups: np.ndarray) -> float:
    """ICC(1) de una vía (proporción de varianza entre grupos). Útil para la
    homogeneidad entre placas en el EDA y para justificar el modelo mixto."""
    v, g = np.asarray(values, float), np.asarray(groups)
    df = pd.DataFrame({"v": v, "g": g})
    k_i = df.groupby("g")["v"].size().to_numpy()
    a = len(k_i); n = len(v)
    if a < 2:
        return np.nan
    grand = v.mean()
    msb = (k_i * (df.groupby("g")["v"].mean().to_numpy() - grand) ** 2).sum() / (a - 1)
    msw = ((df["v"] - df.groupby("g")["v"].transform("mean")) ** 2).sum() / (n - a)
    k0 = (n - (k_i ** 2).sum() / n) / (a - 1)
    return float((msb - msw) / (msb + (k0 - 1) * msw))


def variance_decomposition(df: pd.DataFrame, score: str = "score",
                           factors: tuple[str, ...] = ("seed", "split")) -> pd.Series:
    """Varianza de la métrica atribuible a cada fuente (Bouthillier et al., 2021):
    varianza de las medias marginales por factor y varianza residual."""
    out = {f: float(df.groupby(f)[score].mean().var(ddof=1)) for f in factors if f in df}
    out["total"] = float(df[score].var(ddof=1))
    return pd.Series(out)


def bayesian_signed_rank(a: np.ndarray, b: np.ndarray, rope: float = 0.01, n_samples: int = 50_000,
                         seed: int = 0) -> dict[str, float]:
    """Prueba bayesiana de rangos con signo con ROPE (Benavoli et al., 2017).

    Devuelve P(A > B), P(ROPE), P(B > A) con prior de Dirichlet (s = 1, z0 = 0).
    """
    d = np.concatenate([[0.0], np.asarray(a, float) - np.asarray(b, float)])
    n = len(d)
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(n), size=n_samples)
    sums = d[:, None] + d[None, :]
    lo_m, hi_m = (sums < -2 * rope).astype(float), (sums > 2 * rope).astype(float)
    # p(left)=sum_ij w_i w_j 1[d_i+d_j < -2r]  (forma cuadrática por muestra)
    p_left = np.einsum("si,ij,sj->s", w, lo_m, w)
    p_right = np.einsum("si,ij,sj->s", w, hi_m, w)
    p_rope = 1 - p_left - p_right
    winner = np.argmax(np.stack([p_left, p_rope, p_right]), axis=0)
    return {"p_B_better": float((winner == 0).mean()), "p_rope": float((winner == 1).mean()),
            "p_A_better": float((winner == 2).mean())}


def fit_power_law(n: np.ndarray, err: np.ndarray) -> dict[str, float]:
    """Ajusta error(n) = a·n^(−b) + c (curvas de eficiencia de datos)."""
    from scipy.optimize import curve_fit

    n, err = np.asarray(n, float), np.asarray(err, float)

    def f(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * np.power(x, -b) + c

    p0 = (max(err.max() - err.min(), 1e-3) * n.min() ** 0.5, 0.5, max(err.min() * 0.9, 0.0))
    (a, b, c), cov = curve_fit(f, n, err, p0=p0, bounds=([0, 0, 0], [np.inf, 5, 1]), maxfev=20_000)
    return dict(a=float(a), b=float(b), c=float(c),
                se_b=float(np.sqrt(np.diag(cov))[1]) if np.isfinite(cov).all() else np.nan)
