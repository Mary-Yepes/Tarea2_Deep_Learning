# Discusión

Este capítulo responde las preguntas de análisis crítico del enunciado. Las respuestas se apoyan en la
evidencia de los notebooks. Donde la cifra depende de la corrida real se indica *[completar con
resultados]*. **El criterio de interpretación quedó fijado antes de ver los datos**, para que la
discusión no se acomode a los resultados.

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

**Preprocesamiento (ablación).** Se reportan solo los efectos con IC que excluye 0 **y** magnitud
≥ δ = 0.02. Los factores sin efecto se declaran **irrelevantes**, que es un resultado útil porque
simplifica el pipeline. Las cifras están en el capítulo de resultados correspondiente (tablas generadas por `scripts/fill_report.py`).

**Despliegue con otro microscopio o protocolo.** Se estima con la caída *leave-one-cell-type-out* (cambio
de dominio biológico) y con las curvas de degradación (cambio de adquisición: desenfoque, ruido,
iluminación, compresión). Si la caída supera δ, el modelo **no** debe usarse en otro protocolo sin
reajuste. El monitor de deriva (cap. MLOps) detecta ese cambio en producción.

## NLP

**¿Transformers vs clásicos?** Se decide con H4 (Nadeau–Bengio + Holm + TOST con δ = 0.02) y con el frente
de Pareto F1–latencia–costo. Si el IC de ΔF1 cae dentro de [−δ, δ], un TF-IDF + XGBoost/LR, varios
órdenes de magnitud más barato, es preferible. Las cifras están en el capítulo de resultados correspondiente (tablas generadas por `scripts/fill_report.py`).

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
Las cifras están en el capítulo de resultados correspondiente (tablas generadas por `scripts/fill_report.py`).
