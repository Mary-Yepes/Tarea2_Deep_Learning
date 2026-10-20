# Anexo C · Reporte de reproducibilidad

## Comando único

```bash
make setup         # entorno bloqueado (requirements.lock) + paquete editable
make data          # descarga Kaggle + registro DVC (hashes en data/raw/*.dvc)
make reproduce     # pruebas → experimentos de segmentación y NLP → modelos desplegables → notebooks → libro
```

`make reproduce` escribe `reports/reproduccion.json` con el commit, las semillas, el entorno (versiones de
Python y bibliotecas, CPU y GPU), la duración de cada paso, el tiempo total y las emisiones de CO₂
(`codecarbon`, offline, país COL). Cada corrida individual deja su manifiesto en `reports/manifests/`.

## Semillas y determinismo

`PYTHONHASHSEED=42` (exportado por el `Makefile`), `set_global_seed` (Python, NumPy, TF con
`enable_op_determinism`, PyTorch con `use_deterministic_algorithms`). La prueba
`tests/test_determinism_config.py` verifica que dos corridas con la misma semilla dan el mismo resultado.
Algunas operaciones de GPU (por ejemplo, ciertos *kernels* de atención o el *scatter-add* de
`torchvision`) no son deterministas aun con estas banderas. La varianza residual se cuantifica con varias
semillas.

## Entorno, hardware y tiempos

| Elemento | Valor |
|---|---|
| Python / SO | *[completar desde reports/reproduccion.json]* |
| GPU / CPU / RAM | *[completar]* |
| Tiempo total de `make reproduce` | *[completar]* |
| Horas-GPU por modelo (presupuesto pre-registrado: 6 h segmentación, 2 h transformer) | *[completar]* |
| Emisiones totales (kg CO₂eq) | *[completar]* |

## Estado verificado en el entorno de desarrollo

* 71 pruebas aprobadas (RLE, métricas, estadística, calibración, deriva, particiones sin fuga,
  modelos y baselines de texto, determinismo, registro de modelos y regresión de modelo). Las pruebas de
  API y `pandera` se ejecutan en CI, donde esas dependencias están instaladas.
* `ruff` y `mypy` sin errores.
* Todos los notebooks se ejecutan de arriba abajo en modo demo (`make notebooks-demo`).
