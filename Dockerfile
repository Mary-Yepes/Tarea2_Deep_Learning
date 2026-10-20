# Imagen de inferencia: multi-etapa, usuario no root, dependencias bloqueadas,
# HEALTHCHECK y SIN datos de entrenamiento (solo app/, src/ y models/).
FROM python:3.12-slim AS builder
WORKDIR /w
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements-serve.lock .
RUN pip install --no-cache-dir --prefix=/install -r requirements-serve.lock
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir --no-deps --prefix=/install .

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* && useradd -m -u 10001 appuser
COPY --from=builder /install /usr/local
WORKDIR /srv
COPY app/ app/
COPY models/ models/
ENV MODELS_DIR=/srv/models PYTHONUNBUFFERED=1 PREDICT_TIMEOUT_S=10
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
