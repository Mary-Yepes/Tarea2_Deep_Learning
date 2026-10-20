"""Particiones agrupadas y estratificadas sin fuga de información."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def grouped_train_val_test_split(
    df: pd.DataFrame, group_col: str, strat_col: str, val_frac: float = 0.15,
    test_frac: float = 0.15, seed: int = 42,
) -> pd.Series:
    """Asigna ``train``/``val``/``test`` por **grupo** estratificando por ``strat_col``.

    Se usa ``StratifiedGroupKFold``: el test es un pliegue de ``round(1/test_frac)``
    pliegues; la validación, un pliegue del resto. Las fracciones realizadas son
    aproximadas (los grupos son indivisibles) y deben reportarse
    (``split_summary``). Devuelve una serie alineada con ``df.index``.
    """
    split = pd.Series("train", index=df.index, dtype=object)
    k_test = max(2, round(1 / test_frac))
    sgkf = StratifiedGroupKFold(n_splits=k_test, shuffle=True, random_state=seed)
    _, test_idx = next(sgkf.split(df, df[strat_col], df[group_col]))
    split.iloc[test_idx] = "test"
    rest = df[split == "train"]
    k_val = max(2, round((1 - test_frac) / val_frac))
    sgkf2 = StratifiedGroupKFold(n_splits=k_val, shuffle=True, random_state=seed + 1)
    _, val_idx = next(sgkf2.split(rest, rest[strat_col], rest[group_col]))
    split.loc[rest.index[val_idx]] = "val"
    assert_no_group_leakage(split, df[group_col])
    return split


def assert_no_group_leakage(split: pd.Series, groups: pd.Series) -> None:
    """Falla si algún grupo aparece en más de una partición."""
    n_parts = pd.DataFrame({"s": split.values, "g": groups.values}).groupby("g")["s"].nunique()
    bad = n_parts[n_parts > 1]
    if len(bad):
        raise AssertionError(f"Fuga: {len(bad)} grupos en varias particiones, p. ej. {list(bad.index[:5])}")


def split_summary(df: pd.DataFrame, split: pd.Series, strat_col: str, group_col: str) -> pd.DataFrame:
    t = df.assign(_split=split.values)
    out = t.groupby("_split").agg(n=(strat_col, "size"), n_groups=(group_col, "nunique"))
    out["frac"] = out["n"] / out["n"].sum()
    dist = pd.crosstab(t["_split"], t[strat_col], normalize="index").add_prefix("p_")
    return out.join(dist)


def group_kfold_indices(y: np.ndarray, groups: np.ndarray, k: int = 5, seed: int = 0
                        ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    yield from StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y, groups)


def nested_cv(
    X: Any, y: np.ndarray, groups: np.ndarray, make_model: Callable[[dict[str, Any]], Any],
    grid: list[dict[str, Any]], score: Callable[[np.ndarray, np.ndarray], float],
    k_outer: int = 5, k_inner: int = 3, seed: int = 0,
) -> dict[str, Any]:
    """Validación cruzada anidada con selección por rejilla en el lazo interno.

    Devuelve, por pliegue externo, la mejor puntuación **interna** (estimación
    optimista: el máximo de una búsqueda sobre los mismos datos) y la puntuación
    **externa** de esa configuración reentrenada (estimación honesta). Su
    diferencia media estima el sesgo de selección (fundamento teórico 4).
    """
    y = np.asarray(y); groups = np.asarray(groups)
    rows = []
    for fo, (tr, te) in enumerate(group_kfold_indices(y, groups, k_outer, seed)):
        inner_scores = []
        for params in grid:
            s = []
            for itr, iva in group_kfold_indices(y[tr], groups[tr], k_inner, seed + 1):
                m = make_model(params).fit(_take(X, tr[itr]), y[tr[itr]])
                s.append(score(y[tr[iva]], m.predict(_take(X, tr[iva]))))
            inner_scores.append(float(np.mean(s)))
        best = int(np.argmax(inner_scores))
        m = make_model(grid[best]).fit(_take(X, tr), y[tr])
        rows.append(dict(fold=fo, best_params=grid[best], inner_best=inner_scores[best],
                         outer=score(y[te], m.predict(_take(X, te)))))
    res = pd.DataFrame(rows)
    return {"folds": res, "optimism": float((res["inner_best"] - res["outer"]).mean())}


def _take(X: Any, idx: np.ndarray) -> Any:
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.iloc[idx]
    if isinstance(X, list):
        return [X[i] for i in idx]
    return X[idx]
