"""Eficiencia: latencia (p50/p95/p99), throughput, tamaño, frente de Pareto y CO₂."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def measure_latency(fn: Callable[[], Any], n: int = 50, warmup: int = 5) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); t.append((time.perf_counter() - t0) * 1e3)
    a = np.array(t)
    return dict(p50_ms=float(np.percentile(a, 50)), p95_ms=float(np.percentile(a, 95)),
                p99_ms=float(np.percentile(a, 99)), mean_ms=float(a.mean()), throughput_per_s=float(1e3 / a.mean()))


def file_size_mb(path: str | Path) -> float:
    p = Path(path)
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2 ** 20
    return p.stat().st_size / 2 ** 20


def pareto_front(df: pd.DataFrame, maximize: str, minimize: str) -> pd.DataFrame:
    """Filas no dominadas (mayor ``maximize`` y menor ``minimize``)."""
    d = df.sort_values([minimize, maximize], ascending=[True, False])
    keep, best = [], -np.inf
    for i, r in d.iterrows():
        if r[maximize] > best:
            keep.append(i); best = r[maximize]
    return df.loc[keep]


@contextmanager
def carbon_tracker(project: str = "proyecto-integrador", out_dir: str = "reports") -> Any:
    """``codecarbon`` en modo offline si está instalado; si no, mide solo tiempo."""
    t0 = time.time(); res: dict[str, Any] = {}
    try:
        from codecarbon import OfflineEmissionsTracker

        tr = OfflineEmissionsTracker(project_name=project, output_dir=out_dir, country_iso_code="COL",
                                     log_level="error")
        tr.start()
    except Exception:
        tr = None
    try:
        yield res
    finally:
        res["seconds"] = time.time() - t0
        res["kg_co2eq"] = float(tr.stop()) if tr is not None else None


def multicriteria(df: pd.DataFrame, weights: dict[str, float], higher_better: dict[str, bool]) -> pd.Series:
    """Puntaje multicriterio (min–max por criterio, suma ponderada)."""
    s = pd.Series(0.0, index=df.index)
    for c, w in weights.items():
        x = df[c].astype(float)
        z = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x * 0 + 0.5
        s += w * (z if higher_better[c] else 1 - z)
    return s / sum(weights.values())


def weight_sensitivity(df: pd.DataFrame, weights: dict[str, float], higher_better: dict[str, bool],
                       n: int = 2000, concentration: float = 20.0, seed: int = 0) -> pd.Series:
    """Frecuencia con la que cada alternativa gana al perturbar los pesos (Dirichlet
    centrado en los pesos declarados)."""
    rng = np.random.default_rng(seed)
    keys = list(weights); w0 = np.array([weights[k] for k in keys], float); w0 /= w0.sum()
    wins: dict[Any, int] = dict.fromkeys(df.index, 0)
    for _ in range(n):
        w = rng.dirichlet(w0 * concentration + 1e-3)
        wins[multicriteria(df, dict(zip(keys, w)), higher_better).idxmax()] += 1
    return pd.Series(wins) / n
