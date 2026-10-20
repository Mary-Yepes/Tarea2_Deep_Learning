"""Equidad (contexto de reclutamiento): métricas por subgrupo con IC y pruebas
contrafactuales.

Notas: (i) con tasas base distintas entre grupos no pueden satisfacerse a la vez
calibración por grupo y probabilidades equalizadas (Kleinberg et al., 2016;
Chouldechova, 2017); (ii) los atributos aquí son *proxies* inferidos del texto y
su uso debe justificarse y documentarse en la *model card*.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .stats import bootstrap_ci


def group_rates(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray, positive: Any = 1) -> pd.DataFrame:
    """Tasa de selección, TPR y FPR por grupo (``positive`` = decisión favorable)."""
    yt, yp, g = np.asarray(y_true) == positive, np.asarray(y_pred) == positive, np.asarray(group)
    rows = []
    for k in pd.unique(g):
        m = g == k
        rows.append(dict(group=k, n=int(m.sum()), selection_rate=yp[m].mean(),
                         tpr=yp[m & yt].mean() if (m & yt).any() else np.nan,
                         fpr=yp[m & ~yt].mean() if (m & ~yt).any() else np.nan,
                         base_rate=yt[m].mean()))
    return pd.DataFrame(rows).set_index("group")


def fairness_report(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray, positive: Any = 1,
                    reference: Any = None, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Paridad demográfica (diferencia de tasas de selección), igualdad de
    oportunidades (diferencia de TPR), probabilidades equalizadas (máx. de
    diferencias TPR/FPR) y razón de impacto dispar (regla 4/5), con IC bootstrap
    estratificado por grupo."""
    yt, yp, g = np.asarray(y_true), np.asarray(y_pred), np.asarray(group)
    rates = group_rates(yt, yp, g, positive)
    ref = reference if reference is not None else rates["selection_rate"].idxmax()

    def metrics(idx: np.ndarray) -> dict[str, float]:
        r = group_rates(yt[idx], yp[idx], g[idx], positive)
        sr = r["selection_rate"]
        return dict(dp_diff=float(sr.max() - sr.min()),
                    eo_diff=float(r["tpr"].max() - r["tpr"].min()),
                    eodds=float(max(r["tpr"].max() - r["tpr"].min(), r["fpr"].max() - r["fpr"].min())),
                    di_ratio=float(sr.min() / sr[ref]) if ref in sr and sr[ref] > 0 else np.nan)

    point = metrics(np.arange(len(yt)))
    rng = np.random.default_rng(seed)
    groups = {k: np.where(g == k)[0] for k in pd.unique(g)}
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(v, len(v)) for v in groups.values()])
        boots.append(metrics(idx))
    b = pd.DataFrame(boots)
    ci = {k: (point[k], float(b[k].quantile(0.025)), float(b[k].quantile(0.975))) for k in point}
    return {"rates": rates, "metrics": ci, "four_fifths_ok": bool(point["di_ratio"] >= 0.8)}


# ------------------------------------------------------------ contrafactuales
SWAPS = {
    "he": "she", "she": "he", "him": "her", "his": "her", "her": "his", "man": "woman", "woman": "man",
    "male": "female", "female": "male", "mr": "ms", "ms": "mr", "mrs": "mr", "father": "mother",
    "mother": "father", "husband": "wife", "wife": "husband", "son": "daughter", "daughter": "son",
    "él": "ella", "ella": "él", "hombre": "mujer", "mujer": "hombre", "señor": "señora", "señora": "señor",
}
NAMES = {"male": ["James", "Rahul", "Carlos", "Wei", "Ahmed"], "female": ["Mary", "Priya", "Lucía", "Mei", "Fatima"]}


def gender_swap(text: str) -> str:
    def rep(m: re.Match[str]) -> str:
        w = m.group(0); s = SWAPS.get(w.lower(), w)
        return s.capitalize() if w[0].isupper() else s
    pattern = r"\b(" + "|".join(map(re.escape, SWAPS)) + r")\b"
    return re.sub(pattern, rep, text, flags=re.I)


def age_swap(text: str, new_age: int = 55) -> str:
    return re.sub(r"\b(\d{2})\s*(years? old|años)", lambda m: f"{new_age} {m.group(2)}", text)


def counterfactual_flip_rate(predict_proba: Callable[[Sequence[str]], np.ndarray], texts: Sequence[str],
                             transform: Callable[[str], str], positive_idx: int = 1) -> dict[str, float]:
    """Cambio en la probabilidad de la clase favorable y tasa de cambio de decisión
    al aplicar el contrafactual (solo sobre textos que efectivamente cambian)."""
    changed = [t for t in texts if transform(t) != t]
    if not changed:
        return dict(n_changed=0, mean_abs_delta=0.0, flip_rate=0.0, max_abs_delta=0.0)
    p0 = predict_proba(changed); p1 = predict_proba([transform(t) for t in changed])
    d = p1[:, positive_idx] - p0[:, positive_idx]
    est, lo, hi = bootstrap_ci(np.abs(d), n_boot=2000)
    return dict(n_changed=len(changed), mean_abs_delta=est, ci_low=lo, ci_high=hi,
                max_abs_delta=float(np.abs(d).max()), flip_rate=float((p0.argmax(1) != p1.argmax(1)).mean()))
