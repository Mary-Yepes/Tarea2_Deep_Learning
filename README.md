# Proyecto integrador: segmentación de instancias celulares y clasificación de texto

Este repositorio es una investigación empírica completa con un mismo estándar metodológico en dos
dominios: **Sartorius Cell Instance Segmentation** (imágenes) y **Jarvis Calling Hiring Contest** (texto,
reclutamiento). Incluye pre-registro, auditoría de datos, comparación con presupuesto equitativo e
inferencia estadística, ablaciones, robustez, calibración, incertidumbre, explicabilidad y equidad, y
un despliegue reproducible con monitoreo de deriva. El producto final es un **Jupyter Book con
estructura de artículo** (`book/`) y un paquete instalable (`src/project`).

> **Estado.** El código, las pruebas, los fundamentos teóricos (resueltos y verificados), el
> pre-registro, la API, el Docker, la CI, las *model cards*, las *datasheets* y el experimento de deriva
> están listos. **Las cifras empíricas de los modelos requieren los datos de Kaggle y GPU**: se generan
> con `make reproduce`. Sin datos, los notebooks corren en **modo demo** (sintético), lo anuncian en su
> primera salida y escriben en `reports/demo/`. **Las cifras demo no se reportan.**

## Estructura

```
data/                 raw/ (Kaggle, versionado con DVC), processed/ (particiones), golden/ (conjunto dorado de CI)
notebooks/            00–07 (fuentes en notebooks/src/*.py, formato percent; `make notebooks` los ejecuta)
src/project/          paquete: rle, seg_metrics, stats, calibration, drift, drift_sim, splits, data_seg,
                      preprocess_seg, postprocess, classical, losses, models_seg (U-Net++ propia),
                      adapters_seg (interfaz común de los 6 paradigmas), text_data, models_text
                      (BiLSTM/CNN-1D propias, NB-SVM), nlp_pipeline, transformer_text, fairness,
                      robustness_text, hpo, efficiency, inference (registro con rollback), reporting, tracking
app/                  API FastAPI (/health, /ready, /predict_text, /predict_image, /model_info, /metrics, /drift)
configs/              prereg.yaml (protocolo congelado) + YAML por experimento
tests/                pytest: unitarias, datos (pandera), fuga, determinismo, API, regresión de modelo
models/               registry.json + versiones (modelos demo incluidos solo para humo/CI)
reports/              model_cards/, datasheets/, tables/, figures/, demo/
book/                 Jupyter Book (_config.yml, _toc.yml, capítulos y anexos)
scripts/              seg_experiments.py, nlp_experiments.py, train_deployable.py, build_notebooks.py,
                      reproduce.py, smoke_test.sh, load_test.py, export_onnx.py, external/README.md
.github/workflows/    ci.yml: ruff → mypy → pytest → pandera → regresión → build → humo → pip-audit/Trivy
Makefile, Dockerfile, dvc.yaml, requirements.lock, requirements-serve.lock, pyproject.toml
```

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.12
make setup                                             # requirements.lock + pip install -e .
# Solo servicio/pruebas:  pip install -r requirements-serve.lock && pip install -e ".[dev]"
```

## Reproducción (comando único)

```bash
# 0) Congelar el pre-registro ANTES de entrenar
git add configs/prereg.yaml && git commit -m "Pre-registro" && git tag prereg
# 1) Datos (requiere ~/.kaggle/kaggle.json y aceptar las reglas de ambas competiciones)
make data
# 2) Todo: pruebas → segmentación → NLP → modelos desplegables → notebooks → libro (+ tiempo y CO₂)
make reproduce
```

Los pasos también se pueden ejecutar por separado: `make seg`, `make nlp`, `make train`,
`make notebooks` y `make book`. Los modelos externos (Mask R-CNN+Swin, Hover-Net, Cellpose y SAM) se
entrenan con las **mismas particiones** siguiendo `scripts/external/README.md` y se evalúan con
`scripts/seg_experiments.py evaluate --model ...`.

## Pruebas y calidad

```bash
make lint typecheck test       # ruff, mypy, pytest con cobertura
make test-data                 # esquema pandera
make notebooks-demo            # ejecuta todos los notebooks en modo demo (sin kernel ni datos)
```

## Despliegue

```bash
make train-demo                # (sin datos) registra modelos de humo; con datos: make train
make serve                     # http://localhost:8000/docs
make build && make smoke       # imagen multi-etapa no-root con HEALTHCHECK + prueba de humo
make load-test                 # p50/p95/p99, throughput, punto de saturación
make onnx                      # exportación ONNX + paridad numérica + cuantización INT8
```

Ejemplo:

```bash
curl -X POST localhost:8000/predict_text -H 'content-type: application/json' -d '{"text":"python sql pipeline"}'
curl -X POST localhost:8000/predict_image -F file=@imagen.png
```

Para promover un modelo nuevo se usa `scripts/train_deployable.py` (registra en *staging*);
`ModelRegistry.promote` lo pasa a producción y `ModelRegistry.rollback` vuelve a la versión anterior.

## Documentos clave

* Pre-registro: `configs/prereg.yaml` · Fundamentos teóricos: `book/fundamentos_teoricos.ipynb`
* Model cards: `reports/model_cards/` · Datasheets: `reports/datasheets/`
* Experimento de deriva (RQ8): `notebooks/07_drift_simulation.ipynb` y `book/mlops.md`
* Autoevaluación tipo revisión por pares: `book/anexos/revision_por_pares.md`
* Declaración de uso de IA: `book/anexos/declaracion_ia.md` (**completar antes de entregar**)
