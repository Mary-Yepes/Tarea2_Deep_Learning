"""Prueba de regresión de modelo: las métricas sobre el conjunto dorado no pueden
caer más de ε respecto de las registradas para el modelo en producción."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from project.inference import ImagePredictor, ModelRegistry, TextPredictor
from project.seg_metrics import sartorius_score

ROOT = Path(__file__).resolve().parents[1]
EPS = {"text_accuracy": 0.01, "image_ap": 0.01}


def registry():
    if not (ROOT / "models/registry.json").exists():
        pytest.skip("Sin modelos registrados")
    return ModelRegistry(ROOT / "models")


def test_text_golden_regression():
    d = registry().production_dir("text")
    m = TextPredictor.load(d)
    g = pd.read_csv(ROOT / "data/golden/text_golden.csv")
    acc = float((m.model.predict(g["text"].tolist()) == g["label"].astype(str).to_numpy()).mean())
    ref = json.loads((d / "meta.json").read_text())["golden_metrics"]["accuracy"]
    assert acc >= ref - EPS["text_accuracy"], f"regresión: {acc:.4f} < {ref:.4f} − ε"


def test_image_golden_regression():
    d = registry().production_dir("image")
    m = ImagePredictor.load(d)
    z = np.load(ROOT / "data/golden/image_golden.npz")
    n = len([k for k in z.files if k.startswith("img")])
    ap = float(np.mean([sartorius_score(z[f"lbl{i}"], m.predict_label_map(z[f"img{i}"])) for i in range(n)]))
    ref = json.loads((d / "meta.json").read_text())["golden_metrics"]["ap_sartorius"]
    assert ap >= ref - EPS["image_ap"], f"regresión: {ap:.4f} < {ref:.4f} − ε"


def test_temperature_does_not_change_argmax():
    m = TextPredictor.load(registry().production_dir("text"))
    g = pd.read_csv(ROOT / "data/golden/text_golden.csv")["text"].tolist()[:30]
    p_t = m.predict_proba_batch(g)
    assert (np.asarray(m.classes)[p_t.argmax(1)] == m.model.predict(g)).all()
