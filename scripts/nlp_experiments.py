"""Experimentos de NLP (bloque 2), **reanudables** y con presupuesto declarado.

    python scripts/nlp_experiments.py all          # prepare → hpo → cv → test → preprocess → fractions
    SMOKE=1 python scripts/nlp_experiments.py all  # prueba de humo (minutos)
    python scripts/nlp_experiments.py <etapa>      # prepare | hpo | cv | test | preprocess | fractions

Datos: ``JARVIS_CSV`` (ruta al CSV de entrenamiento) o ``data/raw/jarvis/``; columnas
desde ``configs/nlp.yaml`` (``text_col``/``label_col``) o autodetectadas (se imprimen
para confirmarlas). ``--synthetic`` usa el corpus sintético (salidas en reports/demo).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from project.calibration import fit_temperature, softmax  # noqa: E402
from project.config import CFG, SMOKE, budget, set_global_seed, tables_dir, write_run_manifest  # noqa: E402
from project.nlp_pipeline import classification_metrics, registry  # noqa: E402
from project.splits import group_kfold_indices, grouped_train_val_test_split  # noqa: E402
from project.text_data import clean_minimal, detect_columns, make_synthetic_text, near_duplicate_clusters  # noqa: E402

DEMO = "--synthetic" in sys.argv
TAB = (ROOT / "reports/demo/tables") if DEMO else tables_dir(ROOT); TAB.mkdir(parents=True, exist_ok=True)
SPL = ROOT / ("data/processed/demo_nlp_splits.csv" if DEMO else "data/processed/nlp_splits.csv")
NCFG = yaml.safe_load((ROOT / "configs/nlp.yaml").read_text())
B = budget("nlp", ROOT)
LIGHT = ["Clase mayoritaria", "TF-IDF + LR", "NB-SVM", "TF-IDF + XGBoost", "FastText", "CNN-1D", "Word2Vec + BiLSTM"]
TRANSFORMER = "DistilBERT"
BEST = TAB / "nlp_best_params.json"


def find_csv() -> Path:
    if os.environ.get("JARVIS_CSV"):
        return Path(os.environ["JARVIS_CSV"])
    d = Path(os.environ.get("JARVIS_DIR", ROOT / "data/raw/jarvis"))
    cands = sorted(d.rglob("*.csv"), key=lambda p: ("train" not in p.name.lower(), -p.stat().st_size))
    if not cands:
        raise FileNotFoundError(f"No hay CSV en {d}")
    return cands[0]


def prepare(synthetic: bool) -> pd.DataFrame:
    if SPL.exists() and not synthetic:
        return pd.read_csv(SPL)
    if synthetic:
        df = make_synthetic_text(1500, seed=CFG.seed, shortcut=0.15)
    else:
        path = find_csv(); raw = pd.read_csv(path)
        tc = NCFG.get("text_col") or detect_columns(raw)[0]; yc = NCFG.get("label_col") or detect_columns(raw)[1]
        print(f"[datos] archivo={path.name} columnas={list(raw.columns)} → texto='{tc}', etiqueta='{yc}'")
        df = raw.rename(columns={tc: "text", yc: "label"})[["text", "label"]].dropna()
        if SMOKE:
            df = df.groupby("label").head(60)
    df["text"] = df["text"].astype(str).map(clean_minimal); df["label"] = df["label"].astype(str)
    n0 = len(df)
    df["dup_cluster"] = near_duplicate_clusters(df["text"].tolist(), NCFG["dedup"]["jaccard_threshold"])
    exact = df.duplicated(["text", "label"])
    df = df[~exact].reset_index(drop=True)
    df["split"] = grouped_train_val_test_split(df, "dup_cluster", "label", CFG.val_frac, CFG.test_frac, CFG.seed).values
    SPL.parent.mkdir(parents=True, exist_ok=True); df.to_csv(SPL, index=False)
    # max_length del transformer = p95 de tokens de train (redondeado a 32, máx. 512)
    tr = df[df.split == "train"].text
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(B["transformer"]["model_name"])
        lens = [len(tok(t)["input_ids"]) for t in tr]
    except Exception:
        lens = [int(len(t.split()) * 1.3) + 2 for t in tr]
    p95 = float(np.percentile(lens, 95)); wp95 = float(np.percentile(tr.str.split().str.len(), 95))
    eda = dict(p95_tokens=p95, max_length=int(min(512, max(32, np.ceil(p95 / 32) * 32))),
               p95_palabras=wp95, sequence_length=int(min(512, max(50, 32 * np.ceil(float(tr.str.split().str.len().median()) / 32)))), n=len(df), n_original=n0,
               duplicados_exactos=int(exact.sum()), clases=df.label.value_counts().to_dict())
    (TAB / "nlp_eda_params.json").write_text(json.dumps(eda, indent=2))
    print(f"{n0} textos → {len(df)}; particiones {df.split.value_counts().to_dict()}; {eda}")
    return df


def params_for_registry() -> dict[str, Any]:
    p = json.loads(BEST.read_text()) if BEST.exists() else {}
    eda = json.loads((TAB / "nlp_eda_params.json").read_text()) if (TAB / "nlp_eda_params.json").exists() else {}
    for k in ("cnn1d", "bilstm"):
        p.setdefault(k, {}).setdefault("sequence_length", eda.get("sequence_length", 100))
        p[k]["max_epochs"] = min(int(p[k].get("max_epochs", B["keras_max_epochs"])), B["keras_max_epochs"])
    p["transformer"] = {**{k: v for k, v in B["transformer"].items()},
                        "max_length": int(min(eda.get("max_length", 256), B["transformer"].get("max_length", 512)))}
    p["transformer"]["lr"] = p["transformer"].pop("lr", 3e-5)
    return p


def models(names: list[str]) -> dict[str, Any]:
    reg = registry(params_for_registry())
    return {n: reg[n] for n in names}


def _align(p: np.ndarray, model_classes: Any, classes: np.ndarray) -> np.ndarray:
    mc = list(np.asarray(model_classes).astype(str))
    out = np.zeros((len(p), len(classes)))
    for j, c in enumerate(classes):
        if c in mc:
            out[:, j] = p[:, mc.index(c)]
    s = out.sum(1, keepdims=True); s[s == 0] = 1
    return out / s


def _done(csv: str, **keys: Any) -> bool:
    p = TAB / csv
    if not p.exists():
        return False
    df = pd.read_csv(p); m = np.ones(len(df), bool)
    for k, v in keys.items():
        if k not in df:
            return False
        m &= df[k].astype(str).to_numpy() == str(v)
    return bool(m.any())


def _append(rows: list[dict[str, Any]] | pd.DataFrame, csv: str) -> None:
    df = pd.DataFrame(rows); p = TAB / csv
    df.to_csv(p, mode="a", header=not p.exists(), index=False)


def run_cv(df: pd.DataFrame, names: list[str], seeds: list[int], csv: str, tag: str = "cv",
           fraction: float = 1.0, preprocess: str | None = None) -> None:
    """Repeticiones semilla × fold sobre desarrollo (train+val); el test sigue sellado.
    En cada fold, 10 % del train del fold (por clúster) se reserva para early stopping."""
    dev = df[df.split != "test"].reset_index(drop=True)
    classes = np.unique(dev.label); k = B["k_folds"]
    for seed in seeds:
        for f, (tr, te) in enumerate(group_kfold_indices(dev.label.to_numpy(), dev.dup_cluster.to_numpy(), k, seed)):
            d_tr = dev.iloc[tr]
            inner = grouped_train_val_test_split(d_tr, "dup_cluster", "label", 0.10, 0.10, seed)
            fit_idx, es_idx = d_tr[inner != "val"], d_tr[inner == "val"]
            if fraction < 1:
                fit_idx = fit_idx.groupby("label").sample(frac=fraction, random_state=seed)
            for name, m in models(names).items():
                if preprocess:
                    m.preprocess = preprocess
                key = dict(modelo=name, semilla=seed, fold=f, tag=tag, fraccion=fraction, preprocesamiento=m.preprocess)
                if _done(csv, **key):
                    continue
                set_global_seed(seed); t0 = time.time()
                m.fit(fit_idx.text.tolist(), fit_idx.label.to_numpy(), es_idx.text.tolist(), es_idx.label.to_numpy(), seed)
                p = _align(m.predict_proba(dev.text.iloc[te].tolist()), getattr(m.m_, "classes_", classes), classes)
                r = classification_metrics(dev.label.iloc[te].to_numpy(), p, classes)
                _append([dict(**key, n_train=len(fit_idx), n_test=len(te), segundos=time.time() - t0, **r)], csv)
                print(f"[{tag}] {name} s{seed} f{f} ({m.preprocess}, {fraction:.0%}): F1={r['f1_macro']:.3f}", flush=True)


def run_test(df: pd.DataFrame, names: list[str], seeds: list[int]) -> None:
    """Entrena en train (early stopping/temperatura en val) y evalúa una vez en test."""
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    classes = np.unique(df.label); csv = "nlp_test_preds.csv"
    for name, m in models(names).items():
        for seed in (seeds[:1] if name == "Clase mayoritaria" else seeds):
            if _done(csv, modelo=name, semilla=seed):
                continue
            set_global_seed(seed); t0 = time.time()
            m.fit(tr.text.tolist(), tr.label.to_numpy(), va.text.tolist(), va.label.to_numpy(), seed)
            fit_s = time.time() - t0
            pv = _align(m.predict_proba(va.text.tolist()), getattr(m.m_, "classes_", classes), classes)
            t1 = time.time(); pt = _align(m.predict_proba(te.text.tolist()), getattr(m.m_, "classes_", classes), classes)
            lat = (time.time() - t1) / max(len(te), 1) * 1e3
            zv, zt = np.log(np.clip(pv, 1e-12, 1)), np.log(np.clip(pt, 1e-12, 1))   # log-probs ≡ logits salvo constante
            T = fit_temperature(zv, np.searchsorted(classes, va.label))           # solo validación
            parts = []
            for split_name, P, Z, d in (("val", pv, zv, va), ("test", pt, zt, te)):
                out = pd.DataFrame(P, columns=[f"p_{c}" for c in classes])
                PT = softmax(Z, T); out["p_T_max"] = PT.max(1)
                for j, c in enumerate(classes):
                    out[f"pT_{c}"] = PT[:, j]
                out.insert(0, "idx", d.index.to_numpy()); out.insert(1, "y", d.label.to_numpy())
                out.insert(0, "split", split_name); out.insert(0, "semilla", seed); out.insert(0, "modelo", name)
                out["temperatura"], out["latencia_ms_por_texto"], out["segundos_fit"] = T, lat, fit_s
                parts.append(out)
            _append(pd.concat(parts, ignore_index=True), csv)
            print(f"[test] {name} s{seed}: F1={classification_metrics(te.label.to_numpy(), pt, classes)['f1_macro']:.3f}", flush=True)


def run_hpo(df: pd.DataFrame, names: list[str]) -> None:
    """Optuna/TPE con el MISMO número de trials para todos los modelos con espacio."""
    from project.hpo import NLP_SPACES, importance_report, run_study, suggest

    best = json.loads(BEST.read_text()) if BEST.exists() else {}
    tr, va = df[df.split == "train"], df[df.split == "val"]
    classes = np.unique(df.label)
    for name, (key, space) in NLP_SPACES.items():
        if name not in names or key in best or B["hpo_trials"] <= 0:
            continue

        def objective(trial: Any, name: str = name, space: dict[str, Any] = space) -> float:
            m = registry()[name]; m.params = suggest(trial, space)
            if "max_epochs" in m.params or name in ("CNN-1D", "Word2Vec + BiLSTM"):
                m.params["max_epochs"] = min(int(m.params.get("max_epochs", B["keras_max_epochs"])), B["keras_max_epochs"])
            m.fit(tr.text.tolist(), tr.label.to_numpy(), va.text.tolist(), va.label.to_numpy(), 0)
            p = _align(m.predict_proba(va.text.tolist()), m.m_.classes_, classes)
            return classification_metrics(va.label.to_numpy(), p, classes)["f1_macro"]

        t0 = time.time()
        st = run_study(objective, B["hpo_trials"], seed=0)
        best[key] = st.best_params
        tag = name.replace(" ", "_").replace("+", "")
        st.trials_dataframe().assign(segundos_total=time.time() - t0).to_csv(TAB / f"nlp_hpo_{tag}.csv", index=False)
        try:
            (TAB / f"nlp_hpo_{tag}_importance.json").write_text(json.dumps(importance_report(st), indent=2))
        except Exception as e:      # fANOVA necesita varios trials completos
            print("importancia no disponible:", e)
        BEST.write_text(json.dumps(best, indent=2, default=str))
        print(f"[hpo] {name}: mejor F1 val = {st.best_value:.4f} con {st.best_params}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "hpo", "cv", "test", "preprocess", "fractions", "all"])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--models", nargs="+", default=None, help="por defecto: todos (ligeros + transformer)")
    a = ap.parse_args()
    write_run_manifest(ROOT / f"reports/manifests/nlp_{a.stage}_{int(time.time())}.json", CFG, argv=sys.argv,
                       presupuesto=B, smoke=SMOKE)
    df = prepare(a.synthetic)
    if a.stage == "prepare":
        return
    light = [m for m in (a.models or LIGHT) if m != TRANSFORMER]
    use_tr = a.models is None or TRANSFORMER in a.models
    stages = ["hpo", "cv", "test", "preprocess", "fractions"] if a.stage == "all" else [a.stage]
    for st in stages:
        t0 = time.time()
        if st == "hpo":
            run_hpo(df, light)
        elif st == "cv":
            run_cv(df, light, B["seeds"], "nlp_cv_scores.csv")
            if use_tr:
                run_cv(df, [TRANSFORMER], B["transformer_seeds"], "nlp_cv_scores.csv")
        elif st == "test":
            run_test(df, light, B["seeds"])
            if use_tr:
                run_test(df, [TRANSFORMER], B["transformer_seeds"])
        elif st == "preprocess":     # ablación por familia: clásicos, neuronal y transformer
            fam = [n for n in ("TF-IDF + LR", "NB-SVM", "CNN-1D") if n in light]   # el transformer solo usa «minimal»
            for level in (("minimal", "aggressive") if SMOKE else ("minimal", "standard", "aggressive")):
                for n in fam:
                    run_cv(df, [n], B["seeds"][:3] if n != TRANSFORMER else B["transformer_seeds"][:1],
                           "nlp_preprocess.csv", tag="pre", preprocess=level)
        elif st == "fractions":
            for f in ((0.25, 1.00) if SMOKE else (0.10, 0.25, 0.50, 1.00)):
                fr_models = [n for n in ("TF-IDF + LR", "TF-IDF + XGBoost", "CNN-1D") if n in light]
                run_cv(df, fr_models, B["seeds"][:3], "nlp_fractions.csv", tag="frac", fraction=f)
                if use_tr and f in B.get("transformer_fractions", (0.10, 0.25, 0.50, 1.00)):
                    run_cv(df, [TRANSFORMER], B["transformer_seeds"][:1], "nlp_fractions.csv", tag="frac", fraction=f)
        print(f"== {st} terminada en {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
