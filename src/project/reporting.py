"""Tablas de resultados con incertidumbre y contrastes (insumo del Jupyter Book)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .stats import bootstrap_ci, compare_models, fmt_ci, hierarchical_bootstrap_ci, paired_test

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 no tiene trapezoid


def results_table(df: pd.DataFrame, metrics: Sequence[str], model: str = "modelo", unit: str = "id_imagen",
                  group: str | None = "sample_id", n_boot: int = 10_000, primary: str | None = None) -> pd.DataFrame:
    """Media [IC 95 %] por modelo y métrica. Las semillas se promedian por unidad
    antes de agregar; con ``group`` el IC es jerárquico (placa -> imagen)."""
    agg = df.groupby([model, unit] + ([group] if group else []), as_index=False)[list(metrics)].mean()
    rows = {}
    for m, g in agg.groupby(model):
        r = {}
        for k in metrics:
            v = g[k].replace([np.inf, -np.inf], np.nan)
            ok = v.notna().to_numpy()
            if ok.sum() == 0:
                r[k] = "n/a"; continue
            est = (hierarchical_bootstrap_ci(v[ok].to_numpy(), g[group].to_numpy()[ok], n_boot=min(n_boot, 2000))
                   if group else bootstrap_ci(v[ok].to_numpy(), n_boot=n_boot))
            r[k] = fmt_ci(*est)
        rows[m] = r
    out = pd.DataFrame(rows).T
    if primary:
        piv = agg.pivot_table(index=unit, columns=model, values=primary).dropna()
        out["rank_" + primary] = piv.rank(axis=1, ascending=False).mean().reindex(out.index).round(2)
        out = out.sort_values("rank_" + primary)
    return out


def pairwise_report(df: pd.DataFrame, metric: str, model: str = "modelo", unit: str = "id_imagen",
                    alpha: float = 0.05, delta: float | None = None) -> dict[str, Any]:
    piv = df.groupby([unit, model])[metric].mean().unstack().dropna()
    if piv.shape[1] >= 3:
        return compare_models(piv, alpha, delta)
    a, b = piv.columns[:2]
    return {"pairwise": pd.DataFrame([dict(A=a, B=b, **paired_test(piv[a], piv[b]))])}


def stratified(df: pd.DataFrame, metric: str, by: str, model: str = "modelo") -> pd.DataFrame:
    return df.groupby([by, model])[metric].mean().unstack(model)


def add_terciles(df: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
    per_img = df.groupby("id_imagen")[col].first()
    t = pd.qcut(per_img.rank(method="first"), 3, labels=["bajo", "medio", "alto"])
    return df.merge(t.rename(name), left_on="id_imagen", right_index=True)


def ranking_agreement(df: pd.DataFrame, m1: str, m2: str, model: str = "modelo", n_boot: int = 2000,
                      seed: int = 0) -> dict[str, float]:
    """Spearman entre el orden de modelos por dos métricas, con IC bootstrap
    remuestreando imágenes (H2)."""
    piv1 = df.groupby(["id_imagen", model])[m1].mean().unstack().dropna()
    piv2 = df.groupby(["id_imagen", model])[m2].mean().unstack().reindex(piv1.index)
    rho = stats.spearmanr(piv1.mean(), piv2.mean()).statistic
    rng = np.random.default_rng(seed); bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(piv1), len(piv1))
        bs.append(stats.spearmanr(piv1.iloc[i].mean(), piv2.iloc[i].mean()).statistic)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(rho=float(rho), ci_low=float(lo), ci_high=float(hi))


def simpson_check(df: pd.DataFrame, metric: str, by: str = "tipo_celular", model: str = "modelo") -> pd.DataFrame:
    """Orden de modelos global y por estrato, y τ de Kendall con el global (H3)."""
    glob = df.groupby(model)[metric].mean().rank(ascending=False)
    rows = [dict(estrato="global", orden=" > ".join(glob.sort_values().index), tau=1.0)]
    for k, g in df.groupby(by):
        r = g.groupby(model)[metric].mean().rank(ascending=False).reindex(glob.index)
        rows.append(dict(estrato=k, orden=" > ".join(r.sort_values().index),
                         tau=float(stats.kendalltau(glob, r).statistic)))
    return pd.DataFrame(rows)


def error_profile(df: pd.DataFrame, model: str = "modelo", by: str | None = "tipo_celular") -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("err_")]
    keys = [model] + ([by] if by else [])
    t = df.groupby(keys)[cols + ["n_gt"]].sum()
    out = t[cols].div(t["n_gt"].clip(lower=1), axis=0)
    out.columns = [c.replace("err_", "") + "_por_objeto" for c in cols]
    return out


def best_worst(df: pd.DataFrame, metric: str, k: int = 5, model: str | None = None, seed: int = 42) -> dict[str, list[str]]:
    """IDs para figuras cualitativas: muestra aleatoria (semilla fija) + k mejores + k peores."""
    d = df if model is None else df[df.modelo == model]
    per = d.groupby("id_imagen")[metric].mean().sort_values()
    rnd = per.sample(min(k, len(per)), random_state=seed).index.tolist()
    return {"aleatorias": rnd, "mejores": per.index[-k:].tolist()[::-1], "peores": per.index[:k].tolist()}


def degradation_auc(df: pd.DataFrame, metric: str = "ap", model: str = "modelo") -> pd.DataFrame:
    """Área bajo la curva relativa AP(severidad)/AP(0), por modelo y perturbación."""
    rows = []
    for (m, k), g in df.groupby([model, "perturbacion"]):
        c = g.groupby("severidad")[metric].mean().sort_index()
        rel = c / c.iloc[0] if c.iloc[0] > 0 else c * np.nan
        rows.append(dict(modelo=m, perturbacion=k, auc_relativa=float(_trapz(rel.values, rel.index) / (rel.index.max() or 1)),
                         caida_sev5=float(1 - rel.iloc[-1])))
    return pd.DataFrame(rows)
