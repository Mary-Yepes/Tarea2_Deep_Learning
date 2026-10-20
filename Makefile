# Uso: make <objetivo>. `make reproduce` regenera TODAS las tablas y figuras desde los datos crudos.
SHELL := /bin/bash
PY ?= python
export PYTHONHASHSEED := 42
export PYTHONPATH := src:.
IMAGE ?= proyecto-integrador:latest

.PHONY: help setup lock data prereg-check lint typecheck test test-data coverage train-demo train \
        seg nlp drift notebooks notebooks-demo book reproduce build serve smoke load-test audit onnx clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "%-16s %s\n",$$1,$$2}'

setup: ## Instala el entorno completo bloqueado + paquete editable
	$(PY) -m pip install -r requirements.lock && $(PY) -m pip install -e . --no-deps

lock: ## Regenera los lockfiles con uv
	uv pip compile pyproject.toml --extra serve -o requirements-serve.lock
	uv pip compile pyproject.toml --all-extras -o requirements.lock

data: ## Descarga los datos de Kaggle y los versiona con DVC (requiere credenciales de Kaggle)
	mkdir -p data/raw/sartorius data/raw/jarvis
	kaggle competitions download -c sartorius-cell-instance-segmentation -p data/raw/sartorius && cd data/raw/sartorius && unzip -qo *.zip
	kaggle competitions download -c jarvis-calling-hiring-contest -p data/raw/jarvis && cd data/raw/jarvis && unzip -qo *.zip
	dvc add data/raw/sartorius data/raw/jarvis && git add data/raw/*.dvc && echo "Hashes registrados en data/raw/*.dvc"

prereg-check: ## Falla si configs/prereg.yaml cambió desde la etiqueta `prereg`
	@git rev-parse -q --verify prereg >/dev/null || (echo "Falta la etiqueta 'prereg' (congelar el pre-registro)"; exit 1)
	@git diff --quiet prereg -- configs/prereg.yaml || (echo "prereg.yaml cambió: registre la desviación"; exit 1)

lint: ## ruff
	ruff check src app tests scripts

typecheck: ## mypy
	mypy

test: ## Pruebas unitarias, de API y de regresión de modelo
	pytest --cov --cov-report=term-missing --cov-report=xml

test-data: ## Pruebas de datos (pandera)
	pytest tests/test_data_schema.py

train-demo: ## Modelos de humo (sintéticos) para CI y para probar la API sin datos
	$(PY) scripts/train_deployable.py --data synthetic --promote

train: ## Modelos desplegables con datos reales (tras la comparación final, ver README)
	$(PY) scripts/train_deployable.py --data real --promote

seg: ## Todos los experimentos de segmentación (GPU recomendada)
	$(PY) scripts/seg_experiments.py split
	$(PY) scripts/seg_experiments.py tune-classical
	$(PY) scripts/seg_experiments.py hpo --trials 30
	for s in 0 1 2; do $(PY) scripts/seg_experiments.py train --seed $$s --tag base; done
	$(PY) scripts/seg_experiments.py evaluate --model classical --split test
	for s in 0 1 2; do $(PY) scripts/seg_experiments.py evaluate --model unetpp:models/seg/unetpp_base_f100_s$$s.keras --seed $$s --split test; done
	$(PY) scripts/seg_experiments.py fractions
	$(PY) scripts/seg_experiments.py ablations
	$(PY) scripts/seg_experiments.py loco
	$(PY) scripts/seg_experiments.py robustness --model unetpp:models/seg/unetpp_base_f100_s0.keras
	@echo "Modelos externos (Mask R-CNN, Hover-Net, Cellpose, SAM): ver scripts/external/README.md"

nlp: ## Todos los experimentos de NLP
	$(PY) scripts/nlp_experiments.py prepare
	$(PY) scripts/nlp_experiments.py hpo --model "Word2Vec + BiLSTM" --trials 30
	$(PY) scripts/nlp_experiments.py hpo --model "TF-IDF + XGBoost" --trials 30
	$(PY) scripts/nlp_experiments.py cv --models all
	$(PY) scripts/nlp_experiments.py preprocess --models all
	$(PY) scripts/nlp_experiments.py fractions --models all
	$(PY) scripts/nlp_experiments.py test --models all

notebooks: ## Construye y ejecuta los notebooks (kernel de Jupyter)
	$(PY) scripts/build_notebooks.py --execute

notebooks-demo: ## Ejecuta los notebooks en modo demo sin kernel (salida en reports/demo/notebooks)
	$(PY) scripts/build_notebooks.py --execute --fallback --out reports/demo/notebooks

book: ## Copia los notebooks ejecutados al libro y lo construye (book/_build/html)
	mkdir -p book/resultados && cp notebooks/0*.ipynb book/resultados/
	jupyter-book build book

reproduce: ## Comando único: experimentos → modelos → notebooks → libro (tiempo y CO2 en reports/reproduccion.json)
	$(PY) scripts/reproduce.py

build: ## Imagen Docker
	docker build -t $(IMAGE) .

serve: ## API local
	uvicorn app.main:app --reload --port 8000

smoke: ## Prueba de humo del contenedor
	bash scripts/smoke_test.sh $(IMAGE)

load-test: ## Pruebas de carga (latencia p50/p95/p99, saturación)
	$(PY) scripts/load_test.py --url http://localhost:8000 --concurrency 1 2 4 8 16 32 --requests 400

audit: ## Vulnerabilidades de dependencias
	pip-audit -r requirements-serve.lock

onnx: ## Exporta a ONNX y verifica paridad numérica
	$(PY) scripts/export_onnx.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache book/_build coverage.xml
