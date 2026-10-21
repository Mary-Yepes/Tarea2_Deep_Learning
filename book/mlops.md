# MLOps y gobernanza

**Objetivo:** desplegar el mejor modelo de cada dominio como un servicio verificable, reproducible y
observable, y **demostrar experimentalmente** que el sistema detecta la deriva de datos (RQ8).

## Servicio de inferencia (FastAPI)

`app/main.py` expone estos endpoints:

| Endpoint | Función |
|---|---|
| `GET /health` | Vivacidad |
| `GET /ready` | Listo solo si ambos modelos del registro cargaron (503 en caso contrario, con el error) |
| `POST /predict_text` | `{"text": str[1..5000]}` → etiqueta, probabilidades (calibradas con T), versión, `request_id`, latencia |
| `POST /predict_image` | Archivo PNG/JPEG/TIFF (≤ 10 MB, 32–4096 px) → RLE por instancia, `n_instances`, forma, versión |
| `GET /model_info` | Versión, hash de datos, commit, métricas de validación y estado del registro |
| `GET /metrics` | Formato Prometheus: contador por ruta y estado, histograma de latencia |
| `GET /drift` | PSI en línea de las características de entrada y de salida frente a la referencia guardada en el modelo |

La validación usa Pydantic. Hay `x-request-id` de extremo a extremo, *logging* JSON estructurado, un
*timeout* configurable (504) y carga única de los modelos en `lifespan`. Las pruebas
(`tests/test_api.py`) cubren el esquema, las validaciones 422, el request-id, `/metrics`, `/drift` y el
*timeout*.

## Contenedor y CI/CD

* **Docker.** Imagen multi-etapa `python:3.12-slim`, usuario no root (UID 10001), dependencias
  bloqueadas (`requirements-serve.lock`), `HEALTHCHECK` contra `/health` y **sin datos de entrenamiento**
  (`.dockerignore` excluye `data/`, `reports/` y `notebooks/`).
* **GitHub Actions** (`.github/workflows/ci.yml`), en este orden: ruff → mypy → pytest (unitarias:
  RLE, métricas con casos conocidos, fuga entre particiones, determinismo) → pandera → **regresión de
  modelo** (las métricas del conjunto dorado no pueden caer más de ε = 0.01 frente a las registradas) →
  esquema de API y *timeouts* → *build* → **prueba de humo** del contenedor (`/ready`, `/predict_*`) →
  pip-audit y Trivy. Con etiquetas `v*`, la imagen se publica en GHCR.

## Seguimiento, registro y exportación

* MLflow (o, si no está, `reports/runs.jsonl`) registra commit, configuración, semilla, métricas,
  artefactos y tiempo (`project.tracking.log_run`).
* El registro de modelos con etapas *staging* → *production* y *rollback* está en
  `project.inference.ModelRegistry` y se prueba en `test_registry_promote_and_rollback`.
* DVC versiona los datos y artefactos pesados (`dvc.yaml`).
* ONNX con paridad numérica (tolerancia declarada |Δp| < 10⁻⁴ en FP32) y cuantización dinámica INT8:
  `scripts/export_onnx.py`.

## Pruebas de rendimiento

`scripts/load_test.py` (asíncrono con httpx; alternativa: `scripts/locustfile.py`) mide p50, p95 y p99,
throughput y tasa de error para concurrencias de 1 a 32 en ambos endpoints, y detecta el **punto de
saturación**: el primer nivel en que el throughput crece menos del 10 % o la p95 se duplica. En esta entrega **no se ejecutó** la prueba de carga (no se generó `reports/tables/load_test.json`); queda como trabajo pendiente.

## Monitoreo implementado (simulación)

| Dimensión | Señal | Implementación |
|---|---|---|
| Deriva de covariables | PSI (bins adaptativos y suavizado), KS, MMD | `drift.psi`, `drift_sim.window_scores`, `drift.mmd_permutation_test` |
| Deriva de predicciones | Clase predicha (χ²), confianza, número de instancias por imagen | `drift.chi2_categorical`, características `conf` y `n_pred` |
| Deriva de concepto | Métrica con etiquetas tardías por ventana; Page–Hinkley sobre el error | `drift.PageHinkley`, columna `metric` |
| Operacional | Latencia, tasa de errores | `/metrics` (Prometheus + Grafana) |
| Calidad de entrada | Esquema, tamaño, resolución, longitud del texto | Validación en `app/main.py` |

## Experimento de deriva (RQ8)

El notebook {doc}`resultados/07_drift_simulation` implementa el experimento obligatorio: deriva
controlada y gradual sobre un flujo simulado, con poder (curva ROC frente a intensidad) y falsas alarmas,
retardo de detección y caída de la métrica asociada, y la política de alerta y reentrenamiento. Las
tablas están en `reports/tables/drift_*.csv`, o en `reports/demo/tables/` en modo demo.

```{admonition} Resultados de la versión demo (validación del sistema, no del proyecto)
:class: note
Se ejecutó con los modelos demo (TF-IDF + LR y baseline clásico), un corpus sintético de 1 000 textos y
150 imágenes sintéticas por pool, ventanas de 200 textos o 50 imágenes, y 200 réplicas por condición.
Lo que **sí** se puede afirmar a partir de esta corrida, porque son propiedades del sistema de monitoreo
y no del dataset, es lo siguiente:

* **Los umbrales por defecto no controlan las falsas alarmas con ventanas pequeñas.** En imágenes
  (ventanas de 50), la tasa de falsas alarmas fue 0.13–0.17 (IC de Wilson ≈ [0.09, 0.23]), muy por
  encima del 5 % pre-registrado. Con ventanas de 200 textos fue ≤ 0.01. Esto concuerda con la
  aproximación E[PSI] ≈ (B − 1)(1/n + 1/m) bajo la nula. Por eso los umbrales operativos se **calibran
  por simulación** (unión de señales con FPR = 5 % en una corrida nula independiente). En la evaluación
  independiente, la FPR calibrada quedó en 0.055–0.075.
* **Las derivas de entrada se detectan antes de que la métrica caiga.** Con vocabulario, desenfoque y
  ruido, el poder es ≈ 1 desde la severidad más baja, con un retardo mediano de 0 ventanas (paciencia
  1) o de 1 ventana (paciencia 2), mientras la exactitud o el AP apenas han cambiado.
* **Las derivas de mezcla (tema/prior, tipo celular) son las difíciles.** No cambian las marginales de
  las características de entrada. En texto se detectan sobre todo por la deriva de predicciones (χ² de
  la clase predicha): poder 0.78 con umbral por defecto y 0.91 calibrado a severidad 0.2. En imágenes,
  la deriva de tipo celular tuvo un poder de solo 0.375 [0.31, 0.44] con umbral calibrado a severidad
  0.2, y alcanzó ≥ 0.98 desde severidad 0.5. Su retardo mediano fue de 2 ventanas (paciencia 1) o 4
  (paciencia 2), con una caída de AP ya visible al detectarse (≈ 0.03–0.04).
* **Paciencia 2.** Reduce las falsas alarmas previas al inicio de la deriva de ≈ 0.5–0.6 a ≈ 0.0–0.05
  por flujo (10 ventanas), a cambio de 1–2 ventanas de retardo.
* **H6 (demo).** Se refuta para la deriva de longitud e iluminación leves y para la de tipo celular a
  severidad 2. Queda apoyada o no concluyente para vocabulario, desenfoque y ruido. La conclusión
  definitiva requiere la corrida con los modelos y datos reales.
```

### Resultados con los modelos desplegados

```{include} _generado/deriva.md
```

### Política de alerta y despliegue

* **Aviso.** Una ventana en alarma con umbrales calibrados lleva a registrar el evento y etiquetar una
  muestra prioritaria.
* **Alerta.** Dos ventanas consecutivas, o Page–Hinkley sobre el error con etiquetas tardías, llevan a
  revisión humana y a poner el modelo candidato en **shadow**.
* **Crítico.** Una caída de la métrica > 2δ o errores operacionales > 1 % provocan **rollback** desde
  el registro y reentrenamiento.

Un modelo reentrenado pasa por la regresión del conjunto dorado en CI, luego por *shadow* durante al
menos 2 ventanas, luego por *canary* al 10 % con *rollback* automático por latencia o errores, y solo
entonces se promueve.

## Gobernanza

*Model cards* ({doc}`anexos/model_cards`), *datasheets* ({doc}`anexos/datasheets`), declaración de
reproducibilidad con un comando único y huella de carbono ({doc}`anexos/reproducibilidad`).
