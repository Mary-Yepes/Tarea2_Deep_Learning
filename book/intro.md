# Resumen

**Contexto.** Comparar modelos de *deep learning* con una sola ejecución y un *ranking* de una métrica
suele llevar a conclusiones frágiles. Este proyecto estudia dos dominios con un mismo estándar
metodológico: la segmentación de instancias celulares en microscopía de contraste de fase (Sartorius,
tres tipos celulares) y la clasificación de texto en un contexto de selección de personal (Jarvis
Calling Hiring Contest).

**Métodos.** Antes de entrenar se congeló un protocolo **pre-registrado** (`configs/prereg.yaml`,
etiqueta git `prereg`). Ese protocolo fija las hipótesis falsables H1–H6, las métricas primarias
(AP@[0.5:0.95] y F1 macro), las particiones agrupadas sin fuga (por `sample_id` y por clúster de
casi-duplicados MinHash/LSH), el presupuesto de cómputo equitativo (30 *trials* de HPO por modelo), las
semillas (≥ 5 en modelos ligeros y ≥ 3 en pesados) y el plan estadístico: IC bootstrap BCa/jerárquico,
Friedman + Wilcoxon–Holm, t corregida de Nadeau–Bengio, McNemar, δ de Cliff y modelo mixto por placa.

Se comparan seis paradigmas de segmentación: baseline clásico, Mask R-CNN+Swin, Hover-Net, U-Net++
propia, Cellpose y SAM. En NLP se comparan baselines, DistilBERT/RoBERTa, BiLSTM propia, CNN-1D propia,
TF-IDF+XGBoost y FastText. Todos se evalúan con un único código de métricas. Además se cuantifican la
robustez (perturbaciones y *leave-one-cell-type-out*), la calibración (ECE y *temperature scaling*), la
incertidumbre (ensambles, MC-Dropout y predicción conforme), la explicabilidad, la equidad y el costo. El
mejor modelo de cada dominio se despliega con FastAPI, Docker, CI/CD y un sistema de monitoreo cuya
capacidad de detectar deriva se mide experimentalmente (RQ8).

**Resultados.** Los fundamentos teóricos (cap. {doc}`fundamentos_teoricos`) se demuestran y verifican
numéricamente. Entre otras cosas, se muestra que el ranking por Dice medio puede invertir el ranking por
IoU medio, que la validación no anidada infla la exactitud estimada (≈ 0.07 en un escenario sin señal) y
que la temperatura no altera la exactitud pero sí la NLL. Las cifras empíricas de los modelos se generan
con `make reproduce` sobre los datos reales.

```{admonition} Estado de los resultados empíricos
:class: warning
Este libro se construye a partir de los notebooks ejecutados. Mientras los datos de Kaggle no estén en
`data/raw/` y los modelos no se hayan entrenado (GPU), cada notebook corre en **modo demo** con datos
sintéticos y lo indica explícitamente. **Las cifras del modo demo validan el código; no son resultados
del proyecto y no deben reportarse.** Los capítulos de resultados y discusión contienen el análisis y
los criterios de decisión ya fijados. Las frases marcadas como *[completar con resultados]* se
reemplazan después de ejecutar `make reproduce`.
```

**Guía de lectura.** {doc}`introduccion` (objetivos, preguntas e hipótesis) → {doc}`datos` (fichas y
auditoría) → {doc}`metodos` (protocolo) → {doc}`fundamentos_teoricos` → resultados por dominio →
{doc}`mlops` (despliegue y deriva) → {doc}`discusion` → {doc}`conclusiones`. Los anexos incluyen el
pre-registro, la autoevaluación tipo revisión por pares, el reporte de reproducibilidad, las *model
cards*, las *datasheets* y la declaración de uso de asistentes de IA.
