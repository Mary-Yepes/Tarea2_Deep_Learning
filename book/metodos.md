# Métodos

## Protocolo pre-registrado

El protocolo completo se congeló en `configs/prereg.yaml` (etiqueta git `prereg`) **antes** de
entrenar cualquier modelo final. La tabla de pre-registro, su huella SHA-256 y el registro de
desviaciones están en {doc}`anexos/preregistro`. El objetivo `make prereg-check` falla si el archivo
cambió después de la etiqueta. Los análisis que no estaban pre-registrados se rotulan como
**exploratorios**.

## Particiones y prevención de fuga

* **Segmentación.** 70/15/15 agrupado por `sample_id` con `StratifiedGroupKFold` estratificado por tipo
  celular (`splits.grouped_train_val_test_split`). Nunca se divide por instancia. Hay una prueba
  automática de ausencia de fuga (`tests/test_splits_leakage.py`).
* **NLP.** Primero se eliminan los duplicados exactos. Los casi-duplicados (MinHash/LSH implementado
  desde cero en `text_data.MinHashLSH`) se agrupan como **grupo** de partición, para que nunca crucen
  train, val y test.
* **Test sellado.** Es un 15 % que se evalúa **una sola vez por modelo final**. Umbrales, temperatura,
  parámetros de watershed y NMS, cuantil conforme y early stopping se ajustan solo con validación.
* **Transformaciones ajustadas solo con train.** TF-IDF, vocabulario, Word2Vec/FastText y estadísticos
  de normalización (`TrainStatsNormalizer`).
* **Validación.** Anidada completa para los modelos ligeros (`splits.nested_cv`). Para los pesados, HPO
  sobre un único split de validación y luego 5-fold con configuración fija; el sesgo optimista que esto
  introduce se declara. Su magnitud se estima en el fundamento 4.

## Modelos

```{list-table} Segmentación (todos producen un mapa de etiquetas y se evalúan con el mismo código)
:header-rows: 1
* - Pos
  - Modelo
  - Grado de implementación
* - 0
  - Umbral Otsu/adaptativo sobre desviación al fondo + watershed por distancia
  - Propio (`classical.py`)
* - 1
  - Mask R-CNN + Swin-T (anchors del EDA, Soft-NMS, `max_per_img` del conteo)
  - Reutilizado (MMDetection), adaptador propio
* - 2
  - Hover-Net (ramas NP + HV; BCE+Dice y MSE+MSGE)
  - Reutilizado; post-procesamiento HV reimplementado y probado
* - 3
  - **U-Net++** con supervisión profunda, objetivo de 3 clases, watershed con h-maxima
  - **Desde cero en Keras** (`models_seg.py`, `losses.py`, generador con augmentation)
* - 4
  - Cellpose 2.0 (generalista vs ajustado; estimación de diámetro)
  - Reutilizado (`cellpose` 3.x)
* - 5
  - SAM/MedSAM (cero-shot, prompts de caja, LoRA del decodificador)
  - Reutilizado (`segment-anything`)
```

En NLP se comparan baselines (clase mayoritaria, TF-IDF + LR y NB-SVM propio), DistilBERT/RoBERTa
(*warmup*, decaimiento lineal, LLRD, fp16, ≥ 5 semillas), **Word2Vec + BiLSTM propia** (pooling último,
max o atención, con máscara de padding), **CNN-1D propia** (Kim, 2014), TF-IDF + XGBoost (Optuna, early
stopping) y FastText supervisado.

**Interfaz común TF–PyTorch.** Los modelos PyTorch (Mask R-CNN, SAM, Cellpose y transformers) solo se
comunican con el resto mediante arreglos NumPy (`adapters_seg.SegModel`, `nlp_pipeline.TextModel`). Así
se reutilizan implementaciones oficiales sin portar pesos y la evaluación es idéntica para todos.

## Optimización de hiperparámetros

Se usa búsqueda bayesiana (Optuna/TPE) con poda por mediana, con **el mismo número de trials (30) y las
mismas horas-GPU por modelo comparable** y semilla fija del *sampler* (`hpo.run_study`). La importancia
se estima con fANOVA y la dependencia parcial con medias marginales. La rejilla completa del BiLSTM
(576 configuraciones) se reemplaza por 30 trials. Las curvas de aprendizaje usan 10, 25, 50 y 100 % de
los datos con ≥ 3 semillas, y se ajusta la ley de potencia error(n) = a·n^(−b) + c
(`stats.fit_power_law`).

## Métricas

* **Segmentación.** La primaria es AP@[0.5:0.95] estilo Sartorius: TP/(TP+FP+FN) promediado sobre
  umbrales de IoU de 0.50 a 0.95, por imagen. Secundarias: AP50, AP75, PQ = SQ·RQ, AJI, P/R/F1 por
  instancia, MAE y RMSE del conteo, Bland–Altman, Dice, IoU, HD95, F1 de borde, y AUC-ROC, AUC-PR y BA
  de píxel con el mapa de probabilidad definido por modelo (`adapters_seg.instance_prob_map`). La
  taxonomía de errores (omitida, falsa, fusión, división, desplazada) se define a priori en
  `seg_metrics.error_taxonomy`. Todo se calcula **por imagen** y luego se agrega.
* **NLP.** Accuracy, P/R/F1 (macro, micro y ponderado), MCC, BA, ROC-AUC y PR-AUC (OvR), log-loss,
  Brier y ECE (`nlp_pipeline.classification_metrics`).

Cada métrica está implementada y probada en `src/project` con casos conocidos
(`tests/test_seg_metrics.py` y siguientes). Los supuestos y casos borde (objetos disjuntos, convención
RLE, imágenes vacías y HD95 infinito) están documentados en cada *docstring*.

## Plan estadístico

| Situación | Procedimiento | Implementación |
|---|---|---|
| IC de cualquier métrica | Bootstrap BCa, 10 000 remuestreos sobre unidades; jerárquico placa → imagen | `stats.bootstrap_ci`, `hierarchical_bootstrap_ci` |
| 2 modelos (segmentación) | Wilcoxon pareado por imagen + IC de la diferencia | `stats.paired_test` |
| 2 modelos (NLP) | McNemar exacto (test) y t corregida de Nadeau–Bengio o 5×2cv | `mcnemar_exact`, `corrected_resampled_ttest`, `five_by_two_cv_*` |
| ≥ 3 modelos | Friedman + Wilcoxon por pares con Holm + diagrama CD (Nemenyi) | `compare_models`, `plot_cd_diagram` |
| Tamaño de efecto | d de Cohen pareado, δ de Cliff (umbrales de Romano), relevancia frente a δ | `cohen_dz`, `cliffs_delta` |
| Estructura jerárquica | Modelo mixto con intercepto aleatorio por `sample_id`; ICC | `mixed_model_check`, `icc_oneway` |
| Varianza | Por semilla, por split y total | `variance_decomposition` |
| Opcional | Rangos con signo bayesiano con ROPE = δ | `bayesian_signed_rank` |

La cobertura del bootstrap BCa se verificó por simulación: entre el 85 y el 99 % para una nominal del
95 % con n = 40 de una exponencial (`tests/test_stats.py`).

## Reproducibilidad y seguimiento

Se fijan semillas en Python, NumPy, TF y PyTorch con operaciones deterministas (`config.set_global_seed`).
Las versiones están bloqueadas (`requirements.lock`) y la configuración está en YAML (`configs/`). Cada
corrida escribe un manifiesto (commit, configuración, entorno y hardware) y se registra en MLflow o, si
no está, en `reports/runs.jsonl`. Los datos se versionan con DVC (`dvc.yaml`, hashes en
`data/raw/*.dvc`). `make reproduce` regenera todo y registra tiempo y CO₂ ({doc}`anexos/reproducibilidad`).

## Reglas de reporte

Toda cifra se reporta como media [IC 95 %]. Las figuras cualitativas combinan una muestra aleatoria con
semilla fija y los 5 mejores y 5 peores casos (`reporting.best_worst`). Toda métrica remite a su
definición y a su implementación.
