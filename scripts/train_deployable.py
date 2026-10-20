"""Entrena y registra los modelos desplegables de cada dominio.

Uso::

    python scripts/train_deployable.py --data real        # datos en data/raw (Kaggle)
    python scripts/train_deployable.py --data synthetic   # humo/CI sin datos

Por defecto despliega los *baselines* (TF-IDF + LR y Otsu + watershed). Tras la
comparación final, ``--text-model``/``--image-model`` seleccionan el ganador
(p. ej. ``--image-model keras --keras-path models/unetpp.keras``).

Genera también el **conjunto dorado** (``data/golden``) y las métricas de
referencia que usa la prueba de regresión de modelo en CI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from project.calibration import calibration_summary, fit_temperature  # noqa: E402
from project.classical import classical_segment  # noqa: E402
from project.config import CFG, git_commit, set_global_seed, sha256_file  # noqa: E402
from project.data_seg import (  # noqa: E402
    build_image_table,
    label_map_for,
    load_annotations,
    make_synthetic_dataset,
    read_image,
)
from project.inference import ModelRegistry  # noqa: E402
from project.models_text import NBSVM, tfidf_logreg  # noqa: E402
from project.seg_metrics import sartorius_score  # noqa: E402
from project.splits import grouped_train_val_test_split  # noqa: E402
from project.text_data import clean_minimal, make_synthetic_text, near_duplicate_clusters  # noqa: E402

SEG_DIR = Path(os.environ.get("SARTORIUS_DIR", ROOT / "data/raw/sartorius"))
NLP_SPLITS = ROOT / "data/processed/nlp_splits.csv"
SEG_SPLITS = ROOT / "data/processed/seg_splits.csv"


def load_text(mode: str) -> tuple[pd.DataFrame, str]:
    """Modo real: usa la partición sin fuga de `nlp_experiments.py prepare` (mismas particiones)."""
    if mode == "real":
        if not NLP_SPLITS.exists():
            raise FileNotFoundError("Ejecute antes `python scripts/nlp_experiments.py prepare`")
        return pd.read_csv(NLP_SPLITS), sha256_file(NLP_SPLITS)
    return make_synthetic_text(1200, seed=CFG.seed), "synthetic"


def best_text_kind() -> str:
    """Mejor modelo desplegable (serializable con joblib) según la F1 macro de la CV repetida."""
    p = ROOT / "reports/tables/nlp_cv_scores.csv"
    if not p.exists():
        return "tfidf_lr"
    cv = pd.read_csv(p); m = cv[cv.modelo.isin(["TF-IDF + LR", "NB-SVM"])].groupby("modelo").f1_macro.mean()
    return "nbsvm" if len(m) and m.idxmax() == "NB-SVM" else "tfidf_lr"


def train_text(mode: str, kind: str, out: Path) -> dict:
    df, data_hash = load_text(mode)
    df["text"] = df["text"].astype(str).map(clean_minimal); df["label"] = df["label"].astype(str)
    if "split" in df:
        split = df["split"]
    else:
        df["group"] = near_duplicate_clusters(df["text"].tolist(), threshold=0.8)
        split = grouped_train_val_test_split(df, "group", "label", CFG.val_frac, CFG.test_frac, CFG.seed)
    tr, va = df[split == "train"], df[split == "val"]
    bp = ROOT / "reports/tables/nlp_best_params.json"
    lr_p = json.loads(bp.read_text()).get("tfidf_lr", {}) if bp.exists() else {}
    ng = lr_p.get("ngram", "1-2"); ng = tuple(int(v) for v in ng.split("-")) if isinstance(ng, str) else tuple(ng)
    model = NBSVM() if kind == "nbsvm" else tfidf_logreg(C=float(lr_p.get("C", 4.0)), ngram=ng)  # type: ignore[arg-type]
    model.fit(tr["text"].tolist(), tr["label"].to_numpy())
    classes = [str(c) for c in model.classes_]
    yv = np.searchsorted(classes, va["label"].to_numpy())
    z = model.decision_function(va["text"].tolist())
    z = np.column_stack([-z / 2, z / 2]) if z.ndim == 1 else z
    T = fit_temperature(z, yv)  # solo validación
    from project.calibration import softmax

    cal = calibration_summary(softmax(z, T), yv)
    from sklearn.metrics import f1_score

    f1 = float(f1_score(yv, softmax(z, T).argmax(1), average="macro"))
    out.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(model, out / "model.joblib")
    golden = va.sample(min(100, len(va)), random_state=0)[["text", "label"]]
    (ROOT / "data/golden").mkdir(parents=True, exist_ok=True)
    golden.to_csv(ROOT / "data/golden/text_golden.csv", index=False)
    gp = model.predict(golden["text"].tolist())
    meta = dict(version=out.name.split("-")[-1], name=out.name, kind=kind, classes=classes, temperature=T,
                trained_on=("real" if mode == "real" else "synthetic — SOLO DEMO, reemplazar"),
                data_hash=data_hash, git_commit=git_commit(), created=time.strftime("%Y-%m-%d"),
                val_metrics=dict(f1_macro=f1, **cal),
                golden_metrics=dict(accuracy=float((gp == golden["label"].to_numpy()).mean())),
                reference=dict(text_len=tr["text"].str.split().str.len().sample(min(2000, len(tr)), random_state=0).tolist(),
                               text_conf=softmax(z, T).max(1).tolist()))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def train_image(mode: str, kind: str, out: Path, keras_path: str | None) -> dict:
    """Registra el segmentador. ``kind='keras'``: U-Net++ entrenada (post-procesamiento
    ajustado en validación, tomado de su JSON); ``'classical'``: baseline con los
    parámetros ajustados en validación. Métricas de referencia y conjunto dorado: validación."""
    if mode == "real":
        ann = load_annotations(SEG_DIR / "train.csv")
        tab = build_image_table(ann, SEG_DIR / "train")
        if SEG_SPLITS.exists():
            tab = tab.merge(pd.read_csv(SEG_SPLITS)[["id_imagen", "split"]], on="id_imagen"); split = tab["split"]
        else:
            split = grouped_train_val_test_split(tab, "sample_id", "tipo_celular", CFG.val_frac, CFG.test_frac, CFG.seed)
        val = tab[split.values == "val"].sample(min(30, int((split == "val").sum())), random_state=0)
        imgs = [read_image(p) for p in val["ruta_imagen"]]
        lbls = [label_map_for(ann, i) for i in val["id_imagen"]]
        data_hash = sha256_file(SEG_DIR / "train.csv")
    else:
        imgs, lbls, _ = make_synthetic_dataset(4, seed=CFG.seed)
        data_hash = "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    arch: dict | None = None
    if kind == "keras":
        import shutil

        from project.inference import ImagePredictor

        assert keras_path, "--keras-path requerido"
        shutil.copy(keras_path, out / "model.weights.h5")
        src_meta = json.loads(Path(str(keras_path).replace(".weights.h5", ".json")).read_text())
        params, arch = src_meta["post"], src_meta["config"]
        tmp_meta = dict(kind="keras", version="tmp", params=params, arch=arch)
        (out / "meta.json").write_text(json.dumps(tmp_meta))
        predictor = ImagePredictor.load(out)
        segment = predictor.predict_label_map
    else:
        tuned = ROOT / "models/seg/classical_tuned.json"
        params = json.loads(tuned.read_text()) if tuned.exists() else dict(method="otsu", min_distance=4, min_size=15)

        def segment(im: np.ndarray) -> np.ndarray:
            return classical_segment(im, **params)
    scores = [sartorius_score(lb, segment(im)) for im, lb in zip(imgs, lbls)]
    gdir = ROOT / "data/golden"; gdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(gdir / "image_golden.npz", **{f"img{i}": im for i, im in enumerate(imgs[:6])},
                        **{f"lbl{i}": lb for i, lb in enumerate(lbls[:6])})
    meta = dict(version=out.name.split("-")[-1], name=out.name, kind=kind, params=params,
                arch=(arch if kind == "keras" else None),
                trained_on=("real" if mode == "real" else "synthetic — SOLO DEMO, reemplazar"),
                data_hash=data_hash, git_commit=git_commit(), created=time.strftime("%Y-%m-%d"),
                val_metrics=dict(ap_sartorius=float(np.mean(scores))),
                golden_metrics=dict(ap_sartorius=float(np.mean(scores[:6]))),
                reference=dict(img_mean=[float(i.mean()) for i in imgs], img_std=[float(i.std()) for i in imgs]))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["real", "synthetic"], default="real")
    ap.add_argument("--text-model", choices=["auto", "tfidf_lr", "nbsvm"], default="auto")
    ap.add_argument("--image-model", choices=["auto", "classical", "keras"], default="auto")
    ap.add_argument("--keras-path", default=None)
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--promote", action="store_true", help="pasar directamente a producción")
    a = ap.parse_args()
    set_global_seed(CFG.seed)
    reg = ModelRegistry(ROOT / "models")
    if a.text_model == "auto":
        a.text_model = best_text_kind() if a.data == "real" else "tfidf_lr"
    if a.image_model == "auto":
        cand = ROOT / "models/seg/unetpp_base_f100_s0.weights.h5"
        a.image_model, a.keras_path = ("keras", str(cand)) if a.data == "real" and cand.exists() else ("classical", None)
    tname, iname = f"text_{a.text_model}-{a.version}", f"image_{a.image_model}-{a.version}"
    mt = train_text(a.data, a.text_model, ROOT / "models" / tname)
    mi = train_image(a.data, a.image_model, ROOT / "models" / iname, a.keras_path)
    from project.tracking import log_run

    for dom, name, m in (("text", tname, mt), ("image", iname, mi)):
        log_run(name, {"kind": m["kind"], "data_hash": m["data_hash"], "trained_on": m["trained_on"]},
                m["val_metrics"], [str(ROOT / "models" / name / "meta.json")], root=ROOT)
        reg.register(dom, name)
        if a.promote:
            reg.promote(dom)
    print(json.dumps({"text": mt["val_metrics"], "image": mi["val_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
