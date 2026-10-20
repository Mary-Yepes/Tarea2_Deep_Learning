"""API de inferencia (FastAPI).

Endpoints: ``/health``, ``/ready``, ``/predict_text``, ``/predict_image``,
``/model_info``, ``/metrics`` (formato Prometheus) y ``/drift`` (monitoreo en línea).

Variables de entorno: ``MODELS_DIR`` (por defecto ``models``), ``PREDICT_TIMEOUT_S``
(por defecto 10), ``MAX_IMAGE_MB`` (por defecto 10).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from project.drift import psi
from project.inference import ImagePredictor, ModelRegistry, TextPredictor, decode_image_bytes

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
TIMEOUT_S = float(os.getenv("PREDICT_TIMEOUT_S", "10"))
MAX_IMAGE_MB = float(os.getenv("MAX_IMAGE_MB", "10"))


# ------------------------------------------------------------ logging JSON
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {"ts": round(time.time(), 3), "level": record.levelname, "msg": record.getMessage()}
        for k in ("request_id", "path", "status", "latency_ms", "label", "n_instances", "model_version"):
            if hasattr(record, k):
                base[k] = getattr(record, k)
        return json.dumps(base)


log = logging.getLogger("inference")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout); h.setFormatter(JsonFormatter())
    log.addHandler(h); log.setLevel(logging.INFO)


# ----------------------------------------------------------------- métricas
class Metrics:
    BUCKETS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)

    def __init__(self) -> None:
        self.requests: dict[tuple[str, int], int] = {}
        self.lat: dict[str, list[int]] = {}
        self.lat_sum: dict[str, float] = {}

    def observe(self, path: str, status: int, ms: float) -> None:
        self.requests[(path, status)] = self.requests.get((path, status), 0) + 1
        b = self.lat.setdefault(path, [0] * (len(self.BUCKETS) + 1))
        for i, ub in enumerate(self.BUCKETS):
            if ms <= ub:
                b[i] += 1
        b[-1] += 1
        self.lat_sum[path] = self.lat_sum.get(path, 0.0) + ms

    def render(self) -> str:
        out = ["# TYPE http_requests_total counter"]
        out += [f'http_requests_total{{path="{p}",status="{s}"}} {n}' for (p, s), n in self.requests.items()]
        out.append("# TYPE http_request_latency_ms histogram")
        for p, b in self.lat.items():
            out += [f'http_request_latency_ms_bucket{{path="{p}",le="{ub}"}} {b[i]}' for i, ub in enumerate(self.BUCKETS)]
            out += [f'http_request_latency_ms_bucket{{path="{p}",le="+Inf"}} {b[-1]}',
                    f'http_request_latency_ms_count{{path="{p}"}} {b[-1]}',
                    f'http_request_latency_ms_sum{{path="{p}"}} {self.lat_sum[p]:.3f}']
        return "\n".join(out) + "\n"


class OnlineMonitor:
    """Buffer deslizante de características de entrada y confianza para deriva."""

    def __init__(self, reference: dict[str, list[float]], maxlen: int = 500):
        self.ref = {k: np.asarray(v, float) for k, v in reference.items()}
        self.buf: dict[str, deque[float]] = {k: deque(maxlen=maxlen) for k in self.ref}

    def add(self, **feats: float) -> None:
        for k, v in feats.items():
            if k in self.buf:
                self.buf[k].append(float(v))

    def report(self, min_n: int = 50, thr: float = 0.2) -> dict[str, Any]:
        rep: dict[str, Any] = {}
        for k, b in self.buf.items():
            if len(b) >= min_n and len(self.ref[k]) >= min_n:
                v = psi(self.ref[k], np.asarray(b))
                rep[k] = {"psi": round(v, 4), "n": len(b), "alarm": v > thr}
            else:
                rep[k] = {"psi": None, "n": len(b), "alarm": False}
        return rep


# -------------------------------------------------------------- esquemas
class TextIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class TextOut(BaseModel):
    label: str
    probabilities: dict[str, float]
    model_version: str
    request_id: str
    latency_ms: float


class ImageOut(BaseModel):
    n_instances: int
    rles: list[str]
    shape: list[int]
    model_version: str
    request_id: str
    latency_ms: float


# --------------------------------------------------------------- ciclo de vida
def _load(app: FastAPI) -> None:
    app.state.text_model = app.state.image_model = None
    app.state.load_errors = {}
    reg = ModelRegistry(MODELS_DIR)
    app.state.registry = reg
    try:
        app.state.text_model = TextPredictor.load(reg.production_dir("text"))
    except Exception as e:  # el servicio arranca, pero /ready lo refleja
        app.state.load_errors["text"] = repr(e)
    try:
        app.state.image_model = ImagePredictor.load(reg.production_dir("image"))
    except Exception as e:
        app.state.load_errors["image"] = repr(e)
    tref = (app.state.text_model.meta.get("reference", {}) if app.state.text_model else {})
    iref = (app.state.image_model.meta.get("reference", {}) if app.state.image_model else {})
    app.state.monitor = OnlineMonitor({**tref, **iref})


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    app.state.metrics = Metrics()
    _load(app)  # carga única al arrancar; versión desde el registro
    yield


app = FastAPI(title="Inference API — proyecto integrador", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = rid
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
    finally:
        ms = (time.perf_counter() - t0) * 1e3
        request.app.state.metrics.observe(request.url.path, status, ms)
        log.info("request", extra={"request_id": rid, "path": request.url.path, "status": status,
                                   "latency_ms": round(ms, 2)})
    response.headers["x-request-id"] = rid
    return response


async def _with_timeout(fn: Any, *args: Any) -> Any:
    try:
        return await asyncio.wait_for(run_in_threadpool(fn, *args), timeout=TIMEOUT_S)
    except TimeoutError as e:
        raise HTTPException(504, f"Inferencia excedió {TIMEOUT_S}s") from e


# ------------------------------------------------------------------ rutas
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request) -> Any:
    st = request.app.state
    ok = st.text_model is not None and st.image_model is not None
    body = {"ready": ok, "text": st.text_model is not None, "image": st.image_model is not None,
            "errors": st.load_errors}
    if not ok:
        raise HTTPException(503, body)
    return body


@app.post("/predict_text", response_model=TextOut)
async def predict_text(payload: TextIn, request: Request) -> TextOut:
    m: TextPredictor | None = request.app.state.text_model
    if m is None:
        raise HTTPException(503, "Modelo de texto no disponible")
    t0, rid = time.perf_counter(), request.state.request_id
    probs = await _with_timeout(m.predict_proba, payload.text)
    label = max(probs, key=probs.get)
    request.app.state.monitor.add(text_len=len(payload.text.split()), text_conf=max(probs.values()))
    log.info("predict_text", extra={"request_id": rid, "label": label, "model_version": m.version})
    return TextOut(label=label, probabilities=probs, model_version=m.version, request_id=rid,
                   latency_ms=(time.perf_counter() - t0) * 1e3)


@app.post("/predict_image", response_model=ImageOut)
async def predict_image(request: Request, file: UploadFile = File(...)) -> ImageOut:
    m: ImagePredictor | None = request.app.state.image_model
    if m is None:
        raise HTTPException(503, "Modelo de imagen no disponible")
    data = await file.read()
    if len(data) > MAX_IMAGE_MB * 2 ** 20:
        raise HTTPException(413, f"Imagen mayor a {MAX_IMAGE_MB} MB")
    try:
        img = decode_image_bytes(data)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    if img.ndim != 2 or min(img.shape) < 32 or max(img.shape) > 4096:
        raise HTTPException(422, f"Resolución no soportada: {img.shape}")
    t0, rid = time.perf_counter(), request.state.request_id
    out = await _with_timeout(m.predict, img)
    request.app.state.monitor.add(img_mean=float(img.mean()), img_std=float(img.std()),
                                  n_instances=out["n_instances"])
    log.info("predict_image", extra={"request_id": rid, "n_instances": out["n_instances"],
                                     "model_version": m.version})
    return ImageOut(**out, model_version=m.version, request_id=rid, latency_ms=(time.perf_counter() - t0) * 1e3)


@app.get("/model_info")
def model_info(request: Request) -> dict[str, Any]:
    st = request.app.state
    keep = ("version", "trained_on", "data_hash", "val_metrics", "classes", "created", "git_commit", "kind")
    return {"text": {k: v for k, v in (st.text_model.meta if st.text_model else {}).items() if k in keep},
            "image": {k: v for k, v in (st.image_model.meta if st.image_model else {}).items() if k in keep},
            "registry": st.registry.data}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics(request: Request) -> str:
    return request.app.state.metrics.render()


@app.get("/drift")
def drift(request: Request) -> dict[str, Any]:
    return request.app.state.monitor.report()
