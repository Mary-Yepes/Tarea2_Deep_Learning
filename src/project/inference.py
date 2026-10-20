"""Interfaz de inferencia común y registro de modelos con etapas y *rollback*.

``models/registry.json``::

    {"text":  {"production": "text_tfidf_lr-1.0.0", "staging": null, "history": [...]},
     "image": {"production": "image_classical-1.0.0", ...}}

Cada versión vive en ``models/<nombre>/`` con ``model.joblib`` (o ``model.keras``)
y ``meta.json`` (versión, hash de datos, métricas de validación, clases...).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .rle import label_map_to_rles


# --------------------------------------------------------------- registro
class ModelRegistry:
    def __init__(self, root: str | Path = "models"):
        self.root = Path(root)
        self.path = self.root / "registry.json"
        self.data: dict[str, Any] = json.loads(self.path.read_text()) if self.path.exists() else {}

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def register(self, domain: str, name: str, stage: str = "staging") -> None:
        d = self.data.setdefault(domain, {"production": None, "staging": None, "history": []})
        d[stage] = name
        d["history"].append({"stage": stage, "name": name})
        self.save()

    def promote(self, domain: str) -> None:
        """staging -> production (la versión anterior queda en el historial para rollback)."""
        d = self.data[domain]
        if not d.get("staging"):
            raise ValueError("No hay versión en staging")
        d["history"].append({"stage": "production", "name": d["staging"], "previous": d.get("production")})
        d["production"], d["staging"] = d["staging"], None
        self.save()

    def rollback(self, domain: str) -> str:
        d = self.data[domain]
        prev = [h for h in d["history"] if h["stage"] == "production" and h.get("previous")]
        if not prev:
            raise ValueError("No hay versión previa")
        d["production"] = prev[-1]["previous"]
        d["history"].append({"stage": "rollback", "name": d["production"]})
        self.save()
        return str(d["production"])

    def production_dir(self, domain: str) -> Path:
        name = self.data.get(domain, {}).get("production")
        if not name:
            raise FileNotFoundError(f"No hay modelo en producción para '{domain}'")
        return self.root / name


# ------------------------------------------------------------- predictores
@dataclass
class TextPredictor:
    model: Any
    classes: list[str]
    version: str
    meta: dict[str, Any] = field(default_factory=dict)
    temperature: float = 1.0

    @classmethod
    def load(cls, path: str | Path) -> TextPredictor:
        import joblib

        p = Path(path)
        meta = json.loads((p / "meta.json").read_text())
        return cls(joblib.load(p / "model.joblib"), list(map(str, meta["classes"])), meta["version"], meta,
                   float(meta.get("temperature", 1.0)))

    def predict_proba_batch(self, texts: list[str]) -> np.ndarray:
        if self.temperature != 1.0 and hasattr(self.model, "decision_function"):
            z = np.asarray(self.model.decision_function(texts), float)
            z = z[:, None] * np.array([-0.5, 0.5]) if z.ndim == 1 else z
            z = z / self.temperature
            e = np.exp(z - z.max(1, keepdims=True))
            return e / e.sum(1, keepdims=True)
        return np.asarray(self.model.predict_proba(texts))

    def predict_proba(self, text: str) -> dict[str, float]:
        p = self.predict_proba_batch([text])[0]
        return {c: float(v) for c, v in zip(self.classes, p)}


@dataclass
class ImagePredictor:
    kind: str                      # "classical" | "keras"
    version: str
    params: dict[str, Any] = field(default_factory=dict)
    model: Any = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> ImagePredictor:
        p = Path(path)
        meta = json.loads((p / "meta.json").read_text())
        model = None
        if meta["kind"] == "keras":
            from .models_seg import load_seg_model

            model = load_seg_model(str(p / "model.weights.h5"), {"kind": "unetpp", "config": meta["arch"]})
        return cls(meta["kind"], meta["version"], meta.get("params", {}), model, meta)

    def predict_label_map(self, img: np.ndarray) -> np.ndarray:
        if self.kind == "classical":
            from .classical import classical_segment

            return classical_segment(img, **self.params)
        from .models_seg import predict_probs
        from .postprocess import watershed_from_three_class

        probs = predict_probs(self.model, img)
        assert isinstance(probs, np.ndarray)
        return watershed_from_three_class(probs, **self.params)

    def predict(self, img: np.ndarray) -> dict[str, Any]:
        lbl = self.predict_label_map(img)
        return {"n_instances": int(lbl.max()), "rles": label_map_to_rles(lbl), "shape": list(lbl.shape)}


def decode_image_bytes(data: bytes) -> np.ndarray:
    """PNG/JPEG/TIFF en bytes -> arreglo (escala de grises si los canales son iguales)."""
    import cv2

    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError("No se pudo decodificar la imagen")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr[..., :3], cv2.COLOR_BGR2GRAY)
    return arr
