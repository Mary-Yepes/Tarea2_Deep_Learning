# Datos y auditoría

## Fuentes

* **Segmentación.** *Sartorius – Cell Instance Segmentation* (Kaggle): imágenes de contraste de fase con
  máscaras en RLE, tres tipos celulares (SH-SY5Y, astrocitos y neuronas corticales), metadatos de placa
  y muestra, e imágenes sin anotar para extensiones semi-supervisadas. Literatura de contexto: LIVECell
  (Edlund et al., 2021).
* **NLP.** *Jarvis Calling Hiring Contest* (Kaggle): textos etiquetados en contexto de reclutamiento,
  analizados según la guía de EDA de texto. Por tratarse de **selección de personal**, el análisis de
  equidad y riesgo es obligatorio ({doc}`resultados/06_nlp_robustness_fairness`).

## Fichas de datos (*datasheets*)

Antes de modelar, cada dataset se documenta con su origen y licencia, el proceso de recolección y
anotación, su composición, los usos previstos y no previstos, sus limitaciones y los riesgos de
privacidad. **El esquema real de los archivos se verifica en código** y cada discrepancia con lo
supuesto queda registrada. Las fichas completas están en {doc}`anexos/datasheets`.

## Auditoría: criterio de entrada al modelado

| Auditoría | Segmentación (notebook 01) | NLP (notebook 04) |
|---|---|---|
| Integridad y esquema | `check_integrity` + `pandera` (nulos, rangos, RLE dentro de los límites, IDs únicos, metadatos consistentes) | `detect_columns` + revisión manual; codificación y *mojibake* |
| Fuga | dHash entre placas (Hamming ≤ 4); particiones por `sample_id` | Hash exacto + MinHash/LSH (Jaccard ≥ 0.8) como grupos; 0 clústeres que crucen particiones |
| Ruido de anotación | Áreas extremas, máscaras fragmentadas y solapes; revisión manual de 50 imágenes | *Confident learning* fuera de muestra + revisión manual de sospechosos |
| Sesgos y atajos | Efecto de lote: clasificador de dominio (AUC) + permutación; ICC de intensidad por placa | Log-odds con prior de Dirichlet (IC), PII y proxies de atributos protegidos, diversidad léxica |
| Desbalance | Por tipo, densidad y tamaño; Gini y CV del conteo | Razón de desbalance; truncamiento por clase |

## Implicaciones de diseño derivadas del EDA

Las decisiones de modelado salen de cifras del EDA, no de valores por defecto. Las siete principales para
segmentación están al final del notebook 01: resolución, anchors y diámetro, `max_per_img`, objetivo de
3 clases, normalización, particiones por placa y estratificación del reporte. En NLP se derivan la
métrica primaria, `max_length` y `sequence_length` (a partir del p95 de tokens), el preprocesamiento por
familia y el plan de verificación de atajos ({doc}`resultados/04_nlp_eda_audit`, §3).
