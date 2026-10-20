"""Simulación de deriva controlada y evaluación del sistema de monitoreo (RQ8).

Diseño:
1. **Pools precomputados.** Para cada tipo de deriva y severidad se transforma un
   conjunto base (test sellado o, en modo demo, datos sintéticos), se ejecuta el
   modelo desplegado y se guardan características de entrada, salidas y aciertos
   (etiquetas «tardías»). Las ventanas del flujo se remuestrean de esos pools, lo
   que permite cientos de réplicas a costo bajo.
2. **Poder y falsas alarmas.** Para cada severidad se simulan ``R`` ventanas y se
   registra la alarma del monitor (umbral por defecto y umbral calibrado al
   percentil 95 de la distribución nula) y el puntaje continuo (para ROC).
3. **Retardo.** Flujo con deriva gradual que empieza en la ventana ``onset``;
   se mide la primera alarma, las falsas alarmas previas y la caída de la métrica.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .drift import chi2_categorical, psi

SEVERITIES = (0.0, 0.1, 0.2, 0.3, 0.5, 0.8)


# --------------------------------------------------------- transformaciones texto
def drift_vocab(text: str, s: float, rng: np.random.Generator) -> str:
    """Reemplaza ~s/2 de los tokens por variantes no vistas (deriva de vocabulario)."""
    toks = text.split()
    return " ".join(t + "zq" if rng.random() < s * 0.5 else t for t in toks)


def drift_length(text: str, s: float, rng: np.random.Generator) -> str:
    """Acorta el texto conservando ~(1 − 0.8·s) de los tokens (deriva de longitud)."""
    toks = text.split()
    k = max(3, int(round(len(toks) * (1 - 0.8 * s))))
    return " ".join(toks[:k])


TEXT_DRIFTS: dict[str, Callable[[str, float, np.random.Generator], str]] = {
    "vocabulario": drift_vocab, "longitud": drift_length,
}


def text_features(texts: Sequence[str], vocab: set[str]) -> pd.DataFrame:
    rows = []
    for t in texts:
        toks = t.lower().split()
        rows.append(dict(len=len(toks), oov=float(np.mean([w not in vocab for w in toks])) if toks else 0.0))
    return pd.DataFrame(rows)


def build_text_pools(texts: Sequence[str], labels: Sequence[Any], predict_proba: Callable[[list[str]], np.ndarray],
                     classes: Sequence[Any], vocab: set[str], severities: Sequence[float] = SEVERITIES,
                     seed: int = 0) -> dict[tuple[str, float], pd.DataFrame]:
    """Pools por (tipo, severidad). Columnas: len, oov, conf, pred, y, correct.

    La deriva de **tema/prior** («tema») no transforma textos: se simula al
    remuestrear ventanas con proporciones de clase desplazadas (ver ``sample_window``)."""
    rng = np.random.default_rng(seed)
    pools: dict[tuple[str, float], pd.DataFrame] = {}
    y = np.asarray(labels).astype(str); cls = np.asarray(classes).astype(str)
    for kind, fn in TEXT_DRIFTS.items():
        for s in severities:
            if s == 0 and ("base", 0.0) in pools:
                pools[(kind, 0.0)] = pools[("base", 0.0)]
                continue
            tt = [fn(t, s, rng) for t in texts] if s > 0 else list(texts)
            p = predict_proba(tt)
            df = text_features(tt, vocab)
            df["conf"] = p.max(1); df["pred"] = cls[p.argmax(1)]; df["y"] = y
            df["correct"] = (df["pred"] == df["y"]).astype(float)
            pools[(kind, s)] = df
            if s == 0:
                pools[("base", 0.0)] = df
    return pools


# --------------------------------------------------------- muestreo de ventanas
def _draw(idx: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Muestreo sin reemplazo cuando el pool alcanza (evita empates artificiales
    que vuelven anticonservadoras a KS/χ²); con reemplazo solo si no alcanza."""
    return rng.choice(idx, n, replace=n > len(idx))


def sample_window(pool: pd.DataFrame, n: int, rng: np.random.Generator, prior_shift: float = 0.0,
                  target_class: Any = None) -> pd.DataFrame:
    """Ventana de ``n`` observaciones; con ``prior_shift`` > 0 las proporciones de
    clase se desplazan hacia ``target_class``: p = (1 − s)·p_ref + s·e_target."""
    if prior_shift <= 0 or "y" not in pool:
        return pool.iloc[_draw(np.arange(len(pool)), n, rng)]
    cls, cnt = np.unique(pool["y"], return_counts=True)
    p = (1 - prior_shift) * cnt / cnt.sum()
    p[list(cls).index(target_class if target_class is not None else cls[0])] += prior_shift
    ks = rng.multinomial(n, p)
    yv = pool["y"].to_numpy()
    idx = np.concatenate([_draw(np.where(yv == cls[k])[0], int(ks[k]), rng) for k in range(len(cls)) if ks[k]])
    return pool.iloc[idx]


def window_scores(ref: pd.DataFrame, win: pd.DataFrame, features: Sequence[str],
                  categorical: Sequence[str] = ()) -> dict[str, float]:
    """Señales del monitor para una ventana: PSI máximo entre características,
    p mínimo de KS (Bonferroni) y p de χ² para salidas categóricas (deriva de
    predicciones). ``score_comb`` = −log10 del p combinado (Bonferroni) para ROC."""
    nb = int(np.clip(len(win) // 20, 4, 10))            # bins adaptativos al tamaño de ventana
    psis = [psi(ref[f].to_numpy(), win[f].to_numpy(), bins=nb, smoothing=0.5) for f in features]
    kps = [stats.ks_2samp(ref[f], win[f]).pvalue for f in features]
    out: dict[str, float] = dict(psi_max=float(max(psis)), ks_p_min=min(1.0, float(min(kps)) * len(features)))
    ps = [out["ks_p_min"]]
    for c in categorical:
        out[f"chi2_p_{c}"] = chi2_categorical(ref[c].to_numpy(), win[c].to_numpy())[1]
        ps.append(out[f"chi2_p_{c}"])
    out["p_comb"] = min(1.0, min(ps) * len(ps))
    out["score_comb"] = float(-np.log10(max(out["p_comb"], 1e-300)))
    out["metric"] = float(win["correct"].mean()) if "correct" in win else np.nan
    return out


DEFAULT_THR = {"psi": 0.2, "ks": 0.01, "chi2": 0.01}


def alarm(sc: dict[str, float], thr: dict[str, float] = DEFAULT_THR) -> bool:
    """Alarma si PSI > umbral, o p de KS < umbral, o algún p de χ² < umbral."""
    a = sc["psi_max"] > thr["psi"] or sc["ks_p_min"] < thr["ks"]
    a = a or any(v < thr["chi2"] for k, v in sc.items() if k.startswith("chi2_p_"))
    return bool(a)


def apply_alarm(res: pd.DataFrame, thr: dict[str, float]) -> pd.Series:
    return res.apply(lambda r: alarm({k: r[k] for k in r.index if isinstance(k, str)}, thr), axis=1)


def calibrate_union(null: pd.DataFrame, target_fpr: float = 0.05) -> dict[str, float]:
    """Umbrales calibrados por simulación: un mismo cuantil q por señal (PSI en la
    cola superior, p-valores en la inferior) elegido por bisección para que la
    tasa de falsas alarmas de la **unión** bajo la nula sea ``target_fpr``."""
    chi_cols = [c for c in null.columns if str(c).startswith("chi2_p_")]

    def thr_for(q: float) -> dict[str, float]:
        t = {"psi": float(np.quantile(null["psi_max"], 1 - q)), "ks": float(np.quantile(null["ks_p_min"], q))}
        t["chi2"] = float(min(np.quantile(null[c], q) for c in chi_cols)) if chi_cols else 0.0
        return t

    lo, hi = 0.0, target_fpr
    for _ in range(30):
        mid = (lo + hi) / 2
        fpr = apply_alarm(null, thr_for(mid)).mean()
        lo, hi = (mid, hi) if fpr <= target_fpr else (lo, mid)
    return thr_for(lo)


def power_study(ref: pd.DataFrame, pools: dict[tuple[str, float], pd.DataFrame], kinds: Sequence[str],
                features: Sequence[str], categorical: Sequence[str] = (), window: int = 200, reps: int = 200,
                thr: dict[str, float] = DEFAULT_THR, seed: int = 0, prior_kind: str | None = "tema",
                target_class: Any = None, severities: Sequence[float] = SEVERITIES) -> pd.DataFrame:
    """Réplicas de ventanas por (tipo, severidad) con sus señales y la alarma."""
    rng = np.random.default_rng(seed)
    rows = []
    base = pools[("base", 0.0)]
    all_kinds = list(kinds) + ([prior_kind] if prior_kind else [])
    for kind in all_kinds:
        for s in severities:
            for r in range(reps):
                win = (sample_window(base, window, rng, prior_shift=s, target_class=target_class)
                       if kind == prior_kind else sample_window(pools[(kind, s)], window, rng))
                sc = window_scores(ref, win, features, categorical)
                rows.append(dict(tipo=kind, severidad=s, rep=r, **sc, alarma=alarm(sc, thr)))
    return pd.DataFrame(rows)


def calibrate_threshold(null_scores: np.ndarray, fpr: float = 0.05) -> float:
    """Umbral de una sola señal que deja una tasa de falsas alarmas ``fpr`` bajo la nula."""
    return float(np.quantile(null_scores, 1 - fpr))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    p = k / n if n else np.nan
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def power_table(res: pd.DataFrame, col: str = "alarma") -> pd.DataFrame:
    rows = []
    for (k, s), g in res.groupby(["tipo", "severidad"]):
        p, lo, hi = wilson_ci(int(g[col].sum()), len(g))
        rows.append(dict(tipo=k, severidad=s, tasa_alarma=p, ic_low=lo, ic_high=hi, metrica_media=g["metric"].mean()))
    return pd.DataFrame(rows)


def roc_auc_vs_null(res: pd.DataFrame, scores: Sequence[str] = ("psi_max", "score_comb")) -> pd.DataFrame:
    """AUC de cada puntaje del detector para separar ventanas con deriva (severidad s)
    de ventanas sin deriva, por tipo y severidad (curva «ROC vs intensidad»)."""
    from sklearn.metrics import roc_auc_score

    rows = []
    for k, g in res.groupby("tipo"):
        for s, h in g[g.severidad > 0].groupby("severidad"):
            r = dict(tipo=k, severidad=s)
            for sc in scores:
                null = g[g.severidad == 0][sc].to_numpy()
                y = np.r_[np.zeros(len(null)), np.ones(len(h))]
                r[f"auc_{sc}"] = roc_auc_score(y, np.r_[null, h[sc].to_numpy()])
            rows.append(r)
    return pd.DataFrame(rows)


def delay_study(ref: pd.DataFrame, pools: dict[tuple[str, float], pd.DataFrame], kind: str, features: Sequence[str],
                categorical: Sequence[str] = (), window: int = 200, n_windows: int = 30, onset: int = 10,
                ramp: int = 10, reps: int = 100, thr: dict[str, float] = DEFAULT_THR, patience: int = 1,
                seed: int = 0, severities: Sequence[float] = SEVERITIES, prior_kind: str = "tema",
                target_class: Any = None) -> pd.DataFrame:
    """Deriva gradual: severidad 0 hasta ``onset`` y rampa lineal hasta la máxima en
    ``ramp`` ventanas. Devuelve retardo, falsas alarmas previas y caída de métrica."""
    rng = np.random.default_rng(seed)
    sev = np.asarray(severities)
    base = pools[("base", 0.0)]
    rows = []
    for r in range(reps):
        streak, first, false_alarms, metrics = 0, None, 0, []
        for w in range(n_windows):
            target = 0.0 if w < onset else sev[-1] * min(1.0, (w - onset + 1) / ramp)
            s = float(sev[np.argmin(np.abs(sev - target))])
            win = (sample_window(base, window, rng, prior_shift=s, target_class=target_class)
                   if kind == prior_kind else sample_window(pools[(kind, s)], window, rng))
            sc = window_scores(ref, win, features, categorical)
            metrics.append(sc["metric"])
            streak = streak + 1 if alarm(sc, thr) else 0
            if streak >= patience:
                if w < onset:
                    false_alarms += 1; streak = 0
                elif first is None:
                    first = w
        base_m = float(np.mean(metrics[:onset]))
        rows.append(dict(rep=r, retardo=(first - onset) if first is not None else np.nan,
                         detectada=first is not None, falsas_alarmas_previas=false_alarms,
                         caida_metrica_en_deteccion=(base_m - metrics[first]) if first is not None else np.nan,
                         caida_metrica_final=base_m - metrics[-1]))
    return pd.DataFrame(rows)


# --------------------------------------------------------- imágenes
IMAGE_DRIFTS = ("blur", "noise", "illumination")


def image_features(img: np.ndarray) -> dict[str, float]:
    from scipy import ndimage as ndi

    x = img.astype(np.float32)
    return dict(mean=float(x.mean()), std=float(x.std()), lapvar=float(ndi.laplace(x).var()))


def build_image_pools(images: Sequence[np.ndarray], labels: Sequence[np.ndarray], cell_types: Sequence[str],
                      predict_label_map: Callable[[np.ndarray], np.ndarray], n_levels: int = 6,
                      seed: int = 0) -> dict[tuple[str, float], pd.DataFrame]:
    """Pools por (perturbación, severidad) con características de entrada, número
    de instancias predichas (deriva de predicciones) y AP por imagen (métrica con
    etiquetas tardías, en la columna ``correct`` para reutilizar ``window_scores``).
    La deriva de **tipo celular** se simula con ``sample_window(prior_shift=...)``."""
    from .preprocess_seg import perturb
    from .seg_metrics import sartorius_score

    rng = np.random.default_rng(seed)
    pools: dict[tuple[str, float], pd.DataFrame] = {}
    sev = SEVERITIES[:n_levels]
    for kind in IMAGE_DRIFTS:
        for lvl, s in enumerate(sev):
            if lvl == 0 and ("base", 0.0) in pools:
                pools[(kind, 0.0)] = pools[("base", 0.0)]
                continue
            rows = []
            for img, lb, ct in zip(images, labels, cell_types):
                x = perturb(img.astype(np.float32) / 255.0, kind, lvl, rng) if lvl else img.astype(np.float32) / 255.0
                x8 = (x * 255).astype(np.uint8)
                pred = predict_label_map(x8)
                rows.append(dict(**image_features(x8), n_pred=int(pred.max()), y=ct,
                                 correct=sartorius_score(lb, pred)))
            pools[(kind, s)] = pd.DataFrame(rows)
            if lvl == 0:
                pools[("base", 0.0)] = pools[(kind, s)]
    return pools
