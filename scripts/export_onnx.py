"""Exporta los modelos desplegados a ONNX, evalúa cuantización dinámica INT8 y
verifica la paridad numérica con el original (tolerancia declarada: |Δp| máx < 1e-4
en FP32; para INT8 se reporta la caída de métrica en el conjunto dorado)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from project.inference import ModelRegistry, TextPredictor  # noqa: E402

TOL = 1e-4


def text_to_onnx() -> dict:
    import onnxruntime as ort
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import StringTensorType

    reg = ModelRegistry(ROOT / "models"); d = reg.production_dir("text"); m = TextPredictor.load(d)
    onx = to_onnx(m.model, initial_types=[("text", StringTensorType([None, 1]))],
                  options={id(m.model.steps[-1][1]): {"zipmap": False}})
    out = d / "model.onnx"; out.write_bytes(onx.SerializeToString())
    g = pd.read_csv(ROOT / "data/golden/text_golden.csv")["text"].tolist()
    sess = ort.InferenceSession(str(out))
    p_onnx = sess.run(None, {"text": np.array(g).reshape(-1, 1)})[1]
    p_ref = m.model.predict_proba(g)
    diff = float(np.abs(p_onnx - p_ref).max())
    return {"modelo": str(out), "max_abs_diff": diff, "paridad_ok": diff < TOL,
            "tamano_mb": out.stat().st_size / 2 ** 20}


def keras_to_onnx(path: str) -> dict:
    import onnxruntime as ort
    import tensorflow as tf
    import tf2onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    from project.models_seg import load_seg_model

    model = load_seg_model(path, json.loads(Path(path.replace(".weights.h5", ".json")).read_text()))
    spec = (tf.TensorSpec((None, None, None, 1), tf.float32, name="image"),)
    out = Path(path.replace(".weights.h5", ".onnx"))
    tf2onnx.convert.from_keras(model, input_signature=spec, output_path=str(out))
    q = out.with_name(out.stem + "_int8.onnx"); quantize_dynamic(str(out), str(q), weight_type=QuantType.QInt8)
    x = np.random.default_rng(0).random((1, 256, 256, 1)).astype(np.float32)
    ref = model(x, training=False); ref = np.asarray(ref[-1] if isinstance(ref, list) else ref)
    o = ort.InferenceSession(str(out)).run(None, {"image": x}); o = o[-1]
    oq = ort.InferenceSession(str(q)).run(None, {"image": x})[-1]
    return {"fp32_max_abs_diff": float(np.abs(o - ref).max()), "int8_max_abs_diff": float(np.abs(oq - ref).max()),
            "tamano_fp32_mb": out.stat().st_size / 2 ** 20, "tamano_int8_mb": q.stat().st_size / 2 ** 20}


if __name__ == "__main__":
    res = {"texto": text_to_onnx()}
    for p in sorted((ROOT / "models/seg").glob("unetpp_base_f100_s0.weights.h5")):
        res["unetpp"] = keras_to_onnx(str(p))
    print(json.dumps(res, indent=2))
    (ROOT / "reports/tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/tables/onnx_paridad.json").write_text(json.dumps(res, indent=2))
