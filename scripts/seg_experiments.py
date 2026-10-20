"""Experimentos de segmentación (bloque 1), **reanudables** y con presupuesto declarado.

Todos los modelos se evalúan con el mismo código (``adapters_seg.evaluate_adapter``)
y escriben métricas **por imagen** en ``reports/tables/seg_*.csv``. Cada
evaluación se identifica por (modelo, semilla, tag, split[, perturbación]) y se
omite si ya existe, de modo que una corrida interrumpida continúa donde quedó.

Uso::

    python scripts/seg_experiments.py main          # 6 paradigmas × semillas → seg_eval_test.csv
    python scripts/seg_experiments.py extras        # fracciones, ablaciones, LOCO, robustez
    python scripts/seg_experiments.py all
    SMOKE=1 python scripts/seg_experiments.py all   # prueba de humo (minutos)
    BUDGET=completo python scripts/seg_experiments.py all

Etapas sueltas: split, eda-params, classical, unetpp, hovernet, maskrcnn, cellpose, sam,
fractions, ablations, loco, robustness, export.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")   # TF y PyTorch comparten la GPU

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from project.adapters_seg import ClassicalAdapter, HoverNetAdapter, UNetPPAdapter, evaluate_adapter  # noqa: E402
from project.config import CFG, SMOKE, budget, set_global_seed, tables_dir, write_run_manifest  # noqa: E402
from project.data_seg import build_image_table, label_map_for, load_annotations, read_image  # noqa: E402
from project.postprocess import tune_watershed, watershed_from_three_class  # noqa: E402
from project.preprocess_seg import PERTURBATIONS, norm_255, norm_percentile, perturb  # noqa: E402
from project.seg_metrics import sartorius_score  # noqa: E402
from project.splits import grouped_train_val_test_split  # noqa: E402

RAW = Path(os.environ.get("SARTORIUS_DIR", ROOT / "data/raw/sartorius"))
SPLITS = ROOT / "data/processed/seg_splits.csv"
TAB = tables_dir(ROOT)
MOD = ROOT / ("models/smoke" if SMOKE else "models/seg"); MOD.mkdir(parents=True, exist_ok=True)
B = budget("seg", ROOT)
NORMS = {"div255": norm_255, "percentile": norm_percentile}
NAMES = {"classical": "Baseline clásico", "unetpp": "U-Net++ + Watershed", "hovernet": "Hover-Net",
         "maskrcnn": "Mask R-CNN (+ Swin)", "cellpose_ft": "Cellpose 2.0 (ajustado)",
         "cellpose_gen": "Cellpose 2.0 (generalista)", "sam_zero": "SAM (cero-shot)", "sam_box": "SAM (prompts de caja)"}


# ------------------------------------------------------------------ datos
_C: dict[str, Any] = {}


def data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if "ann" not in _C:
        ann = load_annotations(RAW / "train.csv")
        tab = build_image_table(ann, RAW / "train")
        sp = pd.read_csv(SPLITS) if SPLITS.exists() else None
        if sp is not None and set(tab.id_imagen) - set(sp.id_imagen):
            print("[aviso] seg_splits.csv no corresponde a estos datos (faltan imágenes): se regenera")
            sp = None
        if sp is not None:
            tab = tab.merge(sp[["id_imagen", "split"]], on="id_imagen")
        else:
            tab["split"] = grouped_train_val_test_split(tab, "sample_id", "tipo_celular", CFG.val_frac,
                                                        CFG.test_frac, CFG.seed).values
            SPLITS.parent.mkdir(parents=True, exist_ok=True)
            tab[["id_imagen", "sample_id", "tipo_celular", "split"]].to_csv(SPLITS, index=False)
        if SMOKE:   # 2 imágenes por tipo y partición
            tab = tab.groupby(["split", "tipo_celular"], group_keys=False).head(2).reset_index(drop=True)
        _C["ann"], _C["tab"] = ann, tab
    return _C["ann"], _C["tab"]


def load_split(split: str, fraction: float = 1.0, seed: int = 0, types: list[str] | None = None,
               exclude_types: list[str] | None = None, limit: int | None = None
               ) -> tuple[list[np.ndarray], list[np.ndarray], pd.DataFrame]:
    """Imágenes y etiquetas de una partición; la fracción se submuestrea por grupo
    (sample_id), estratificada por tipo celular y con semilla."""
    key = (split, fraction, seed, tuple(types or ()), tuple(exclude_types or ()), limit)
    if key in _C:
        return _C[key]
    ann, tab = data()
    t = tab[tab.split == split]
    if types:
        t = t[t.tipo_celular.isin(types)]
    if exclude_types:
        t = t[~t.tipo_celular.isin(exclude_types)]
    if fraction < 1.0:
        rng = np.random.default_rng(seed); keep = []
        for _, g in t.groupby("tipo_celular"):
            groups = g.sample_id.unique(); rng.shuffle(groups)
            keep.append(g[g.sample_id.isin(groups[:max(1, int(round(len(groups) * fraction)))])])
        t = pd.concat(keep)
    if limit:
        t = t.sample(min(limit, len(t)), random_state=0)
    t = t.reset_index(drop=True)
    out = ([read_image(p) for p in t.ruta_imagen], [label_map_for(ann, i) for i in t.id_imagen], t)
    _C[key] = out
    return out


def eda_params() -> dict[str, Any]:
    """Parámetros derivados del EDA (solo con train): anchors y máximo de detecciones."""
    p = MOD / "eda_params.json"
    if p.exists():
        return json.loads(p.read_text())
    _, Y, _ = load_split("train")
    counts = [int(y.max()) for y in Y]
    diam = np.concatenate([2 * np.sqrt(np.bincount(y.ravel())[1:] / np.pi) for y in Y if y.max() > 0])
    q = np.quantile(diam, [0.05, 0.25, 0.5, 0.75, 0.95])
    # un tamaño por nivel FPN (strides 4..64): el menor es la potencia de 2 bajo el p5 del diámetro
    base = int(2 ** np.floor(np.log2(max(q[0], 4))))
    anchors = [[base * 2 ** i] for i in range(5)]
    prm = dict(anchor_sizes=anchors, max_det=int(max(100, 1.2 * np.percentile(counts, 99))),
               diam_quantiles=q.round(1).tolist(), count_p99=float(np.percentile(counts, 99)))
    p.write_text(json.dumps(prm, indent=2))
    return prm


# ------------------------------------------------------------ registro y reanudación
def done(csv: str, **keys: Any) -> bool:
    p = TAB / csv
    if not p.exists():
        return False
    df = pd.read_csv(p)
    m = np.ones(len(df), bool)
    for k, v in keys.items():
        if k not in df:
            return False
        m &= df[k].astype(str).to_numpy() == str(v)
    return bool(m.any())


def append(df: pd.DataFrame, csv: str) -> None:
    p = TAB / csv
    if p.exists():
        old = pd.read_csv(p); df = pd.concat([old, df], ignore_index=True)
    df.to_csv(p, index=False)


def evaluate(adapter: Any, split: str, seed: int, tag: str, csv: str, types: list[str] | None = None,
             perturbation: tuple[str, int] | None = None, **extra: Any) -> pd.DataFrame | None:
    keys = dict(modelo=adapter.name, semilla=seed, tag=tag, split=split)
    if perturbation:
        keys.update(perturbacion=perturbation[0], severidad=perturbation[1])
    if done(csv, **keys):
        print("ya evaluado:", keys); return None
    X, Y, t = load_split(split, types=types)
    if perturbation:
        rng = np.random.default_rng(seed)
        X = [(perturb(x.astype(np.float32) / 255.0, perturbation[0], perturbation[1], rng) * 255).astype(np.uint8)
             for x in X]
    df = pd.DataFrame(evaluate_adapter(adapter, X, Y, t.id_imagen.tolist()))
    df = df.merge(t[["id_imagen", "tipo_celular", "sample_id"]], on="id_imagen")
    for k, v in {**keys, **extra}.items():
        df[k] = v
    append(df, csv)
    print(f"[{csv}] {adapter.name} s{seed} {tag}: AP = {df.ap.mean():.4f}", flush=True)
    return df


def log(name: str, params: dict[str, Any], metrics: dict[str, Any], paths: list[str]) -> None:
    from project.tracking import log_run

    log_run(name, params, {k: v for k, v in metrics.items() if isinstance(v, (int, float))}, paths, root=ROOT)


# ------------------------------------------------------------ modelos Keras
def train_keras(kind: str, seed: int, fraction: float = 1.0, tag: str = "base", over: dict[str, Any] | None = None,
                exclude_types: list[str] | None = None) -> tuple[Path, dict[str, Any]]:
    """U-Net++ (``kind='unetpp'``) o Hover-Net (``kind='hovernet'``), implementación propia."""
    name = f"{kind}_{tag}_f{int(fraction * 100)}_s{seed}"
    path, mpath = MOD / f"{name}.weights.h5", MOD / f"{name}.json"
    if path.exists() and mpath.exists():
        return path, json.loads(mpath.read_text())
    from tensorflow import keras

    from project.models_seg import build_hovernet, build_unetpp, compile_hovernet, compile_unetpp, make_generator

    a = {**dict(base_filters=32, depth=4, deep_supervision=True, dropout=0.1, encoder="scratch", loss="bce_dice",
                lr=1e-3, optimizer="adam", scheduler="cosine", augment="geom+photo+elastic", normalize="percentile",
                resolution="native"), **B[kind]}
    a.update(over or {})
    set_global_seed(seed)
    Xtr, Ytr, _ = load_split("train", fraction, seed, exclude_types=exclude_types)
    Xva, Yva, _ = load_split("val", exclude_types=exclude_types)
    if str(a["resolution"]) in ("256", "512"):
        from project.preprocess_seg import resize_pair

        r = int(a["resolution"])

        def rs(ims: list[np.ndarray], lbs: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
            o = [resize_pair(i, lb, (max(32, int(r * i.shape[0] / max(i.shape)) // 32 * 32), r)) for i, lb in zip(ims, lbs)]
            return [x[0] for x in o], [x[1] for x in o]
        Xtr, Ytr = rs(Xtr, Ytr); Xva, Yva = rs(Xva, Yva)
    norm = NORMS[a["normalize"]]
    if kind == "unetpp":
        model = build_unetpp((None, None, 1), base_filters=a["base_filters"], depth=a["depth"],
                             deep_supervision=a["deep_supervision"], dropout=a["dropout"], encoder=a["encoder"])
        n_out = len(model.outputs)
        gtr = make_generator(Xtr, Ytr, a["batch_size"], a["crop"], a["augment"], n_out, norm, seed)
        gva = make_generator(Xva, Yva, a["batch_size"], a["crop"], "none", n_out, norm, seed, shuffle=False)
        compile_unetpp(model, a["loss"], a["lr"], a["max_epochs"], len(gtr), a["scheduler"], a["optimizer"])
    else:
        model = build_hovernet((None, None, 1), base_filters=a["base_filters"], dropout=a["dropout"])
        gtr = make_generator(Xtr, Ytr, a["batch_size"], a["crop"], a["augment"], 2, norm, seed, target="hovernet")
        gva = make_generator(Xva, Yva, a["batch_size"], a["crop"], "none", 2, norm, seed, shuffle=False, target="hovernet")
        compile_hovernet(model, a["lr"], a["max_epochs"], len(gtr))
    cbs = [keras.callbacks.EarlyStopping("val_loss", patience=a["patience"], restore_best_weights=True),
           keras.callbacks.CSVLogger(str(TAB / f"history_{name}.csv"))]
    t0 = time.time()
    model.fit(gtr, validation_data=gva, epochs=a["max_epochs"], callbacks=cbs, verbose=2)
    secs = time.time() - t0
    model.save_weights(path)
    # post-procesamiento ajustado SOLO en validación
    Xv, Yv = Xva[:30], Yva[:30]
    if kind == "unetpp":
        ad = UNetPPAdapter(model, normalize=norm, resolution=str(a["resolution"]))
        items = [(ad.probs(x), y) for x, y in zip(Xv, Yv)]
        post, s = tune_watershed(items, watershed_from_three_class,
                                 {"fg_thr": [0.4, 0.5, 0.6], "h": [0.05, 0.1, 0.2], "min_size": [10, 20]}, sartorius_score)
    else:
        from project.adapters_seg import hovernet_postprocess
        from project.models_seg import hovernet_infer_fn

        f = hovernet_infer_fn(model, norm)
        items = [(f(x), y) for x, y in zip(Xv, Yv)]
        post, s = tune_watershed(items, lambda o, **k: hovernet_postprocess(o[0], o[1], **k),
                                 {"k": [0.3, 0.4, 0.5], "sobel_ksize": [5, 11, 21]}, sartorius_score)
    meta = dict(name=name, kind=kind, seed=seed, fraction=fraction, config=a, post=post, val_ap=s,
                train_seconds=secs, params=int(model.count_params()), n_train=len(Xtr))
    mpath.write_text(json.dumps(meta, indent=2, default=str))
    log(name, {**a, "seed": seed, "fraction": fraction}, meta, [str(path)])
    return path, meta


def keras_adapter(path: Path, meta: dict[str, Any], tta: bool = False) -> Any:
    from project.models_seg import load_seg_model

    model = load_seg_model(str(path), meta)
    norm = NORMS[meta["config"].get("normalize", "percentile")]
    if meta["kind"] == "hovernet":
        from project.models_seg import hovernet_infer_fn

        return HoverNetAdapter(hovernet_infer_fn(model, norm), post=meta["post"], name=NAMES["hovernet"])
    return UNetPPAdapter(model, post=meta["post"], tta=tta, resolution=str(meta["config"].get("resolution", "native")),
                         normalize=norm, name=NAMES["unetpp"])


# ------------------------------------------------------------ modelos externos
def train_maskrcnn_run(seed: int) -> tuple[Path, dict[str, Any]]:
    path, mpath = MOD / f"maskrcnn_s{seed}.pt", MOD / f"maskrcnn_s{seed}.json"
    if path.exists() and mpath.exists():
        return path, json.loads(mpath.read_text())
    from project.maskrcnn_torch import train_maskrcnn

    e = eda_params(); c = B["maskrcnn"]
    Xtr, Ytr, _ = load_split("train"); Xva, Yva, _ = load_split("val", limit=30)
    res = train_maskrcnn(Xtr, Ytr, Xva, Yva, str(path), backbone=c["backbone"], epochs=c["epochs"], lr=c["lr"],
                         batch_size=c["batch_size"], patience=c["patience"], seed=seed, max_det=e["max_det"],
                         anchor_sizes=e["anchor_sizes"], time_budget_h=c["time_budget_h"])
    # umbral de puntuación ajustado en validación
    from project.adapters_seg import MaskRCNNTorchvisionAdapter

    best, bs = 0.5, -1.0
    for thr in (0.3, 0.5, 0.7):
        ad = MaskRCNNTorchvisionAdapter(str(path), c["backbone"], score_thr=thr, detections_per_img=e["max_det"])
        s = float(np.mean([sartorius_score(y, ad.predict(x)[0]) for x, y in zip(Xva, Yva)]))
        best, bs = (thr, s) if s > bs else (best, bs)
    meta = dict(res, history=res["history"], score_thr=best, val_ap_tuned=bs, backbone=c["backbone"], max_det=e["max_det"])
    mpath.write_text(json.dumps(meta, indent=2, default=str))
    log(f"maskrcnn_s{seed}", c, meta, [str(path)])
    return path, meta


def maskrcnn_adapter(seed: int) -> Any:
    from project.adapters_seg import MaskRCNNTorchvisionAdapter

    path, meta = train_maskrcnn_run(seed)
    return MaskRCNNTorchvisionAdapter(str(path), meta["backbone"], score_thr=meta["score_thr"],
                                      detections_per_img=meta["max_det"], name=NAMES["maskrcnn"])


def train_cellpose_run(seed: int, fraction: float = 1.0) -> str:
    """Ajuste fino de Cellpose desde ``cyto2`` con la CLI (mismas particiones)."""
    tag = f"cellpose_f{int(fraction * 100)}_s{seed}"
    mp = MOD / f"{tag}.path"
    if mp.exists():
        return mp.read_text().strip()
    from project.export_formats import export_cellpose

    base = ROOT / "data/processed/cellpose" / tag
    if base.exists():
        shutil.rmtree(base)
    X, Y, t = load_split("train", fraction, seed); export_cellpose(zip(t.id_imagen, X, Y), base / "train")
    Xv, Yv, tv = load_split("val", limit=30); export_cellpose(zip(tv.id_imagen, Xv, Yv), base / "val")
    c = B["cellpose"]
    cmd = [sys.executable, "-m", "cellpose", "--train", "--dir", str(base / "train"), "--test_dir", str(base / "val"),
           "--pretrained_model", "cyto2", "--chan", "0", "--chan2", "0", "--n_epochs", str(c["n_epochs"]),
           "--learning_rate", str(c["learning_rate"]), "--weight_decay", str(c["weight_decay"]),
           "--mask_filter", "_masks", "--min_train_masks", "1", "--model_name_out", tag, "--verbose"]
    try:
        import torch
        if torch.cuda.is_available():
            cmd.append("--use_gpu")
    except ImportError:
        pass
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONHASHSEED": str(seed)})
    cands = sorted((base / "train" / "models").glob("*"), key=lambda p: p.stat().st_mtime)
    path = str(cands[-1])
    mp.write_text(path)
    return path


def cellpose_adapter(model: str, seed: int = 0) -> Any:
    """Adaptador con ``cellprob_threshold`` ajustado en validación."""
    from project.adapters_seg import CellposeAdapter

    key = MOD / f"cellpose_post_{Path(model).name}.json"
    if key.exists():
        thr = json.loads(key.read_text())["cellprob_threshold"]
    else:
        Xv, Yv, _ = load_split("val", limit=20); best, bs = 0.0, -1.0
        for th in (-1.0, 0.0, 1.0):
            ad = CellposeAdapter(model_type=model, cellprob_threshold=th)
            s = float(np.mean([sartorius_score(y, ad.predict(x)[0]) for x, y in zip(Xv, Yv)]))
            best, bs = (th, s) if s > bs else (best, bs)
        thr = best; key.write_text(json.dumps({"cellprob_threshold": best, "val_ap": bs}))
    nm = NAMES["cellpose_gen"] if model == "cyto2" else NAMES["cellpose_ft"]
    return CellposeAdapter(model_type=model, cellprob_threshold=thr, name=nm)


def sam_checkpoint() -> str:
    p = ROOT / "models/external/sam_vit_b_01ec64.pth"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request

        urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", p)
    return str(p)


# ------------------------------------------------------------------ etapas
ONLY: set[str] = set()


def want(kind: str) -> bool:
    return not ONLY or kind in ONLY


def stage_main() -> None:
    """Comparación principal: 6 paradigmas en el test sellado. Con ``--only`` se
    ejecuta un subconjunto (útil para correr Keras y PyTorch en procesos separados)."""
    csv = "seg_eval_test.csv"
    # 0. baseline clásico (rejilla en validación)
    if want("classical"):
        p = MOD / "classical_tuned.json"
        if not p.exists():
            from project.classical import classical_segment

            Xv, Yv, _ = load_split("val", limit=30); best, bs = {}, -1.0
            for m, md, bg in product(["otsu", "adaptive"], [3, 4, 6, 8], [21, 31, 51]):
                prm = dict(method=m, min_distance=md, bg_size=bg)
                s = float(np.mean([sartorius_score(y, classical_segment(x, **prm)) for x, y in zip(Xv, Yv)]))
                best, bs = (prm, s) if s > bs else (best, bs)
            p.write_text(json.dumps(best))
        evaluate(ClassicalAdapter(json.loads(p.read_text()), name=NAMES["classical"]), "test", 0, "final", csv)
    for seed in B["seeds"]:
        for kind in ("unetpp", "hovernet"):                                 # 3. U-Net++ y 2. Hover-Net (propios)
            if want(kind):
                path, meta = train_keras(kind, seed)
                evaluate(keras_adapter(path, meta), "test", seed, "final", csv)
        if want("maskrcnn"):
            evaluate(maskrcnn_adapter(seed), "test", seed, "final", csv)    # 1. Mask R-CNN + Swin
        if want("cellpose"):
            evaluate(cellpose_adapter(train_cellpose_run(seed), seed), "test", seed, "final", csv)   # 4. Cellpose ajustado
    if want("cellpose"):
        evaluate(cellpose_adapter("cyto2"), "test", 0, "final", csv)        # 4b. Cellpose generalista
    if want("sam"):
        from project.adapters_seg import SAMAutoAdapter, SAMBoxAdapter  # 5. SAM
        ck = sam_checkpoint()
        evaluate(SAMAutoAdapter(ck, name=NAMES["sam_zero"]), "test", 0, "final", csv)
        det = maskrcnn_adapter(B["seeds"][0])
        evaluate(SAMBoxAdapter(ck, det.boxes, name=NAMES["sam_box"]), "test", 0, "final", csv)


def stage_fractions() -> None:
    for f, seed in product(B["fractions"], B["seeds"]):
        if "unetpp" in B["fraction_models"]:
            path, meta = train_keras("unetpp", seed, f, "base" if f == 1.0 else "frac")
            evaluate(keras_adapter(path, meta), "test", seed, f"frac{f}", "seg_fractions.csv", fraccion=f, n_train=meta["n_train"])
        if "cellpose" in B["fraction_models"]:
            evaluate(cellpose_adapter(train_cellpose_run(seed, f), seed), "test", seed, f"frac{f}", "seg_fractions.csv", fraccion=f)


def stage_ablations() -> None:
    """Un factor a la vez respecto de la configuración base; evaluación en VALIDACIÓN."""
    base = dict(resolution="native", augment="geom+photo+elastic", loss="bce_dice", encoder="scratch", normalize="percentile")
    over_ep = dict(max_epochs=B["ablation"]["max_epochs"], patience=B["ablation"]["patience"])
    for factor, levels in B["ablation_levels"].items():
        for lvl, seed in product(levels, B["seeds"]):
            cfg = {**base, factor: lvl, **over_ep}
            train_res = "native" if str(lvl).startswith("tiling") else cfg["resolution"]
            tag = "abl_base" if all(cfg[k] == base[k] for k in base) or (factor == "resolution" and lvl == "tiling_256_ov64") \
                else f"abl_{factor}_{lvl}".replace("+", "-")
            path, meta = train_keras("unetpp", seed, 1.0, tag, {**cfg, "resolution": train_res})
            ad = keras_adapter(path, meta)
            if factor == "resolution":
                ad.resolution = str(lvl)
            evaluate(ad, "val", seed, f"{factor}={lvl}", "seg_ablations.csv", factor=factor, nivel=str(lvl))


def stage_loco() -> None:
    _, tab = data()
    for held, seed in product(sorted(tab.tipo_celular.unique()), B["loco_seeds"]):
        path, meta = train_keras("unetpp", seed, 1.0, f"loco_{held}", exclude_types=[held])
        evaluate(keras_adapter(path, meta), "test", seed, f"loco_{held}", "seg_loco.csv", types=[held], excluido=held)


def stage_robustness() -> None:
    path, meta = train_keras("unetpp", B["seeds"][0])
    sevs = list(B.get("robustness_severities", range(6)))
    tta_sevs = list(B.get("robustness_tta_severities", [0, 3]))     # TTA es más caro: solo en algunas severidades
    for tta in (False, True):
        ad = keras_adapter(path, meta, tta=tta)
        for kind in PERTURBATIONS:
            for sev in sevs:
                if tta and sev not in tta_sevs:
                    continue
                evaluate(ad, "test", B["seeds"][0], f"rob_tta{int(tta)}", "seg_robustness.csv",
                         perturbation=(kind, sev), tta=tta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["split", "eda-params", "main", "extras", "all", "fractions", "ablations",
                                      "loco", "robustness", "export"])
    ap.add_argument("--only", nargs="+", default=[], help="classical unetpp hovernet maskrcnn cellpose sam")
    a = ap.parse_args()
    ONLY.update(a.only)
    write_run_manifest(ROOT / f"reports/manifests/seg_{a.stage}_{int(time.time())}.json", CFG, argv=sys.argv,
                       presupuesto=B, smoke=SMOKE)
    t0 = time.time()
    if a.stage == "split":
        print(data()[1].groupby(["split", "tipo_celular"]).size().unstack())
    elif a.stage == "eda-params":
        print(eda_params())
    elif a.stage == "export":
        from project.export_formats import export_coco, export_hovernet

        for sp in ("train", "val", "test"):
            X, Y, t = load_split(sp)
            export_coco(zip(t.id_imagen, Y, [Path(p).name for p in t.ruta_imagen]), ROOT / f"data/processed/coco/annotations_{sp}.json")
            export_hovernet(zip(t.id_imagen, X, Y), ROOT / f"data/processed/hovernet/{sp}")
    else:
        stages = {"main": [stage_main], "fractions": [stage_fractions], "ablations": [stage_ablations],
                  "loco": [stage_loco], "robustness": [stage_robustness],
                  "extras": [stage_robustness, stage_loco, stage_fractions, stage_ablations],
                  "all": [stage_main, stage_robustness, stage_loco, stage_fractions, stage_ablations]}[a.stage]
        for st in stages:
            s = time.time(); st()
            print(f"== {st.__name__} terminada en {(time.time() - s) / 60:.1f} min", flush=True)
    print(f"Total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
