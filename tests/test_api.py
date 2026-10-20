"""Esquema de la API, request-id, validación, timeouts y prueba de humo en proceso."""
import io
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MODELS_DIR", str(ROOT / "models"))

from fastapi.testclient import TestClient  # noqa: E402

import app.main as main  # noqa: E402


@pytest.fixture
def client():
    if not (ROOT / "models/registry.json").exists():
        pytest.skip("Sin modelos registrados: ejecute `make train-demo` o `make train`")
    with TestClient(main.app) as c:
        yield c


def png_bytes(img):
    import cv2

    ok, buf = cv2.imencode(".png", img)
    return io.BytesIO(buf.tobytes())


def test_health_ready_and_request_id(client):
    r = client.get("/health", headers={"x-request-id": "abc"})
    assert r.status_code == 200 and r.json() == {"status": "ok"} and r.headers["x-request-id"] == "abc"
    assert client.get("/ready").status_code == 200


def test_predict_text_schema(client):
    r = client.post("/predict_text", json={"text": "python sql pipeline deploy"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"label", "probabilities", "model_version", "request_id", "latency_ms"}
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-6 and body["label"] in body["probabilities"]


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "x" * 5001}, {}, {"txt": "a"}])
def test_predict_text_validation(client, payload):
    assert client.post("/predict_text", json=payload).status_code == 422


def test_predict_image_returns_rle_and_count(client):
    from project.data_seg import make_synthetic_image
    from project.rle import rle_decode

    img, _ = make_synthetic_image((128, 160), "cort", seed=0)
    r = client.post("/predict_image", files={"file": ("x.png", png_bytes(img), "image/png")})
    assert r.status_code == 200
    b = r.json()
    assert b["n_instances"] == len(b["rles"]) and b["shape"] == [128, 160]
    if b["rles"]:
        assert rle_decode(b["rles"][0], (128, 160)).sum() > 0


def test_predict_image_rejects_garbage_and_tiny(client):
    assert client.post("/predict_image", files={"file": ("x.png", io.BytesIO(b"nope"), "image/png")}).status_code == 422
    tiny = np.zeros((8, 8), np.uint8)
    assert client.post("/predict_image", files={"file": ("t.png", png_bytes(tiny), "image/png")}).status_code == 422


def test_model_info_metrics_and_drift(client):
    info = client.get("/model_info").json()
    assert "version" in info["text"] and "data_hash" in info["text"] and "val_metrics" in info["image"]
    client.get("/health")
    assert "http_requests_total" in client.get("/metrics").text
    assert isinstance(client.get("/drift").json(), dict)


def test_timeout_returns_504(client, monkeypatch):
    import time

    monkeypatch.setattr(main, "TIMEOUT_S", 0.05)
    monkeypatch.setattr(client.app.state.text_model, "predict_proba", lambda t: time.sleep(0.5) or {"a": 1.0})
    assert client.post("/predict_text", json={"text": "hola"}).status_code == 504
