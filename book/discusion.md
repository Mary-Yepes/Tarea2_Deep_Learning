# Discusión

Este capítulo responde las preguntas de análisis crítico del enunciado. Las respuestas se apoyan en la
evidencia de los notebooks. **El criterio de interpretación quedó fijado antes de ver los datos**, para
que la discusión no se acomode a los resultados. Las cifras provienen de la corrida real en Kaggle con el
perfil de presupuesto «urgente» (una semilla en segmentación, entrenamientos cortos; ver la desviación
declarada en `configs/prereg.yaml`). **Lectura de las cifras:** son preliminares. Las diferencias
pequeñas no se deben tomar como efectos establecidos.

## Segmentación

**¿Qué modelo funciona mejor, para qué tipo celular y régimen, y por qué?** Se responde con la tabla
estratificada (modelo × tipo celular × tercil de densidad y de tamaño) y con la taxonomía de errores. La
explicación tiene que ser un **mecanismo verificable**, no el ranking. Ejemplos:

* Si U-Net++ gana en `cort` (células redondas, poco densas) pero pierde en `shsy5y`, y en `shsy5y`
  dominan las *fusiones*, el mecanismo propuesto es que el watershed no separa células alargadas en
  contacto cuando la clase borde-contacto tiene baja probabilidad. Se verifica con la ablación del
  parámetro `h` y del ancho del borde, y con el cambio en la tasa de fusiones por objeto.
* Si Cellpose ajustado gana en los terciles de tamaño grande pero no en los pequeños, se contrasta con la
  dependencia del diámetro (estimación global vs por tipo).

**¿Coinciden los rankings de píxel e instancia?** H2 predice que no. El fundamento 3 muestra que AUC y
BA son invariantes a la prevalencia y saturan, mientras que la precisión y la AP se desploman. En
segmentación, un modelo que **fusiona** dos células obtiene un Dice de píxel casi perfecto y pierde dos
TP de instancia. Para el uso práctico (conteo y morfología por célula), la métrica relevante es la de
instancia: AP@[.5:.95], PQ y MAE del conteo.

**¿Qué errores dominan?** Se espera, y se contrasta, que las fusiones crezcan con la densidad (Spearman
cobertura–contacto del EDA) y que las omisiones dominen en los objetos menores que el *stride* efectivo.
Las cifras están en {doc}`resultados_segmentacion`.

```{admonition} Qué mostró la corrida real (segmentación)
:class: note
* **Mejor modelo.** Cellpose ajustado: AP@[.5:.95] = 0.236 [0.196, 0.275] en el test sellado (86 imágenes),
  rango medio 1.09 y primero en los tres tipos celulares (astro 0.097, cort 0.329, shsy5y 0.176). Le siguen
  U-Net++ propia (0.161) y Hover-Net (0.145). Los modelos de base (clásico, Mask R-CNN, SAM sin ajuste,
  Cellpose generalista) quedan por debajo de 0.07.
* **Mecanismo (hipótesis apoyada por la taxonomía de errores, no verificada con una ablación).** El error
  dominante de Cellpose ajustado es la **omisión** (0.38 por objeto en astro, 0.14 en cort, 0.19 en shsy5y)
  con pocas falsas detecciones (≤ 0.10): es conservador (precisión 0.77, recall 0.65). U-Net++ y Hover-Net
  detectan más células, pero **fusionan** en shsy5y (0.27 y 0.28 por objeto, frente a 0.15 de Cellpose), el
  tipo más denso (mediana de 324 células por imagen). En astro, Hover-Net divide células grandes
  (0.36 divisiones por objeto).
* **Densidad.** El AP baja al aumentar la densidad en todos los modelos principales (Cellpose 0.33 → 0.16;
  U-Net++ 0.23 → 0.09; Hover-Net 0.22 → 0.09, de baja a alta densidad).
* **Mask R-CNN y SAM con prompts.** Mask R-CNN recibió una sola época por el presupuesto (AP 0.038); su
  resultado refleja el presupuesto y no la capacidad de la arquitectura. SAM con cajas (0.070) usa las cajas
  que produce Mask R-CNN, así que queda limitado por ese modelo.
```

**Preprocesamiento (ablación).** Se reportan solo los efectos con IC que excluye 0 **y** magnitud
≥ δ = 0.02. Los factores sin efecto se declaran **irrelevantes**, que es un resultado útil porque
simplifica el pipeline. Las cifras están en el capítulo de resultados correspondiente (tablas generadas por `scripts/fill_report.py`).

```{admonition} Qué mostró la corrida real (ablaciones, U-Net++ en validación, una semilla)
:class: note
* **Pérdida:** BCE + Dice supera a BCE en 0.026 de AP [0.018, 0.035]. Es el único factor con IC que
  excluye 0 **y** magnitud ≥ δ = 0.02, así que es el único efecto práctico.
* **Codificador preentrenado:** ResNet50-ImageNet supera al entrenamiento desde cero en 0.015
  [0.008, 0.022]. El IC excluye 0, pero la magnitud queda por debajo de δ: efecto estadístico, no práctico.
* **Aumento de datos:** +0.016 [0.006, 0.027] (p = 0.141): no se distingue de cero con la prueba usada.
* **Limitación:** una semilla, entrenamientos de 6 épocas y solo dos niveles por factor. Las ablaciones de
  resolución y de normalización no se corrieron en el perfil urgente.
```

**Despliegue con otro microscopio o protocolo.** Se estima con la caída *leave-one-cell-type-out* (cambio
de dominio biológico) y con las curvas de degradación (cambio de adquisición: desenfoque, ruido,
iluminación, compresión). Si la caída supera δ, el modelo **no** debe usarse en otro protocolo sin
reajuste. El monitor de deriva (cap. MLOps) detecta ese cambio en producción.

```{admonition} Qué mostró la corrida real (cambio de dominio y robustez, U-Net++)
:class: note
* **Cambio de tipo celular (LOCO):** al excluir un tipo del entrenamiento, el AP cae un 55 % (astro),
  93 % (cort) y 58 % (shsy5y) respecto al modelo que sí lo vio. **No** debe usarse con un tipo celular
  nuevo sin reajuste.
* **Cambio de adquisición:** el desenfoque es lo más dañino (a severidad 5 el AP cae un 91 %; AUC relativa
  0.64). El ruido (0.85) y la compresión JPEG (0.86) degradan de forma moderada. El brillo y el contraste
  no afectan. La iluminación degrada poco (0.96).
* **TTA:** mejora el AP a severidad 3 en 0.011 [0.008, 0.014]. Es una ganancia pequeña pero consistente.
```

## NLP

**¿Transformers vs clásicos?** Se decide con H4 (Nadeau–Bengio + Holm + TOST con δ = 0.02) y con el frente
de Pareto F1–latencia–costo. Si el IC de ΔF1 cae dentro de [−δ, δ], un TF-IDF + XGBoost/LR, varios
órdenes de magnitud más barato, es preferible. Las cifras están en el capítulo de resultados correspondiente (tablas generadas por `scripts/fill_report.py`).

```{admonition} Qué mostró la corrida real (NLP)
:class: note
* **H4 no concluyente.** DistilBERT obtiene F1 macro = 0.712 [0.671, 0.751] y TF-IDF + XGBoost, 0.708
  [0.654, 0.753] en el test sellado. ΔF1 = +0.017, IC 90 % corregido [−0.019, 0.054]: ni se demuestra una
  diferencia relevante ni una equivalencia dentro de ±0.02, con solo 5 repeticiones. En la práctica,
  XGBoost cuesta mucho menos (1.8 ms por texto frente a 10.0 ms de DistilBERT, unas 5.7 veces menos).
* **Clásicos frente a redes.** TF-IDF + XGBoost supera a la CNN-1D (ΔF1 = 0.05, p_Holm = 0.024) y a
  Word2Vec + BiLSTM (0.12, p_Holm = 0.002). **TF-IDF + LR fue el peor de los modelos competitivos
  (F1 = 0.604)**, algo inesperado para un modelo lineal sobre texto. FastText (0.437) quedó
  claramente entrenado de menos y su resultado no es representativo del método.
* **Datos.** Con el 25 % de los textos, DistilBERT ya alcanza F1 ≈ 0.68 (0.74 con el 100 % en CV), frente a
  0.62 de XGBoost y 0.52 de TF-IDF + LR. Con pocos datos, el transformer saca ventaja.
* **Calibración (H5 refutada en sus dos partes).** DistilBERT con T = 1 ya está **mejor calibrado** que
  TF-IDF + LR (ECE 0.082 frente a 0.312; la diferencia tiene IC [−0.28, −0.16], es decir, DistilBERT
  menor), al contrario de lo esperado. El temperature scaling lo mejora, pero solo en un 30 % (bootstrap,
  semilla 0) o un 45 % (estimación puntual del cuaderno 05), por debajo del 50 % pre-registrado.
* **Preprocesamiento.** El efecto es pequeño: CNN-1D con preprocesamiento mínimo obtiene 0.686 frente a
  0.663 con el agresivo; para TF-IDF + LR y NB-SVM las diferencias son ≤ 0.01.
* **Atajos y equidad (TF-IDF + LR, notebook 06).** La auditoría solo marcó una palabra sospechosa
  (`3d`); enmascararla no cambia el F1 (0.595). Los cambios de género y edad en el texto alteran la
  probabilidad en promedio en menos de 0.001 (máximo 0.003) y no cambian ninguna predicción. Los
  subgrupos por proxy de género tienen menos de 30 currículos, así que no se reportan métricas por
  subgrupo: esa parte de la auditoría no es concluyente y se documenta como limitación.
```

**Impacto del preprocesamiento por familia.** Se espera que la normalización agresiva perjudique al
transformer y que sea neutra o favorable para TF-IDF. Se contrasta con pruebas pareadas por familia
(notebook 05, §5).

**Errores sistemáticos.** La taxonomía anotada por dos personas (κ de Cohen) separa los errores del
modelo (negación, dominio) de los errores del dato (etiqueta dudosa, ambigüedad). Si predominan estos
últimos, el techo de desempeño lo fija el etiquetado, no el modelo.

## Amenazas a la validez

| Tipo | Amenaza | Mitigación aplicada | Riesgo residual |
|---|---|---|---|
| Interna | Fuga entre particiones (placas, casi-duplicados) | Particiones agrupadas + pruebas automáticas + dHash/MinHash | Fuga por variables no observadas, como el mismo operador en placas distintas |
| Interna | Selección optimista de hiperparámetros | Anidada en modelos ligeros; test sellado | Sesgo optimista en pesados (HPO en un split), declarado |
| Interna | Presupuesto desigual | 30 trials y horas-GPU iguales | Los modelos preentrenados aprovechan cómputo previo no contabilizado |
| Externa | Un laboratorio, un microscopio, tres tipos celulares; un único corpus de reclutamiento | LOCO, perturbaciones, OOD simulado | Generalización a otros centros no demostrada |
| De constructo | AP mide la coincidencia con anotaciones ruidosas, no con la «verdad» biológica; en NLP no se sabe qué mide la etiqueta | Cota de ruido por revisión manual; *confident learning* | El techo de la métrica es desconocido |
| De conclusión estadística | Muchas comparaciones; dependencia intra-placa | Holm, IC jerárquico, modelo mixto, tamaños de efecto y δ | Potencia limitada en estratos pequeños |

## Comparación final y decisión

**Análisis multicriterio.** Las alternativas se puntúan con `efficiency.multicriteria` (min–max por
criterio y suma ponderada). Pesos declarados:

| Criterio | Peso | Justificación |
|---|---|---|
| Desempeño (métrica primaria) | 0.35 | Objetivo principal |
| Robustez (AUC relativa de degradación / OOD) | 0.20 | Un cambio de protocolo o de población es probable |
| Calibración (ECE tras T*) | 0.10 | Necesaria para derivar casos a revisión humana |
| Latencia p95 | 0.10 | Restricción operativa |
| Costo (entrenamiento + inferencia) | 0.10 | Costo total de propiedad |
| Equidad (NLP: Δ contrafactual y paridad) / Conteo (imágenes: MAE) | 0.15 | Riesgo regulatorio en reclutamiento / utilidad científica |

**Sensibilidad a los pesos.** `efficiency.weight_sensitivity` perturba los pesos con una Dirichlet
centrada en los declarados (2000 réplicas) y reporta en qué fracción gana cada alternativa. La decisión
se considera **robusta** si la ganadora supera el 70 % de las réplicas. Si no lo hace, se reporta el
empate y el criterio que lo rompe.

**Costo total de propiedad (plantilla).**
TCO = horas-GPU de entrenamiento × costo/hora
+ (solicitudes/mes ÷ throughput) × costo de cómputo de inferencia
+ horas-persona de mantenimiento (monitoreo, revisión de alertas)
+ reentrenamientos/año × costo de cada reentrenamiento.
Las horas-GPU y la energía se miden con `codecarbon`; el throughput, con `scripts/load_test.py`.
En esta entrega **no** se midió el costo ni la prueba de carga: los tiempos por etapa están en los
registros de Kaggle y en `reports/manifests/`, y la prueba de carga queda como trabajo pendiente.
