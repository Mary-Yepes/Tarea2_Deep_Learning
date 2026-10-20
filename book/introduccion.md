# Introducción

## Motivación

En segmentación celular y en clasificación de texto es habitual declarar un «mejor modelo» a partir de
una sola ejecución, una sola métrica y un conjunto de prueba contaminado por la selección de
hiperparámetros. Tres problemas hacen que esas conclusiones sean frágiles:

1. **Fuga de información.** En Sartorius, las imágenes de una misma placa comparten condiciones
   técnicas. En los textos de reclutamiento, los casi-duplicados y las plantillas cruzan las particiones.
2. **Variabilidad no reportada** entre semillas, *splits* y datos (Bouthillier et al., 2021).
3. **Métricas que no miden lo que importa.** Las métricas de píxel saturan con desbalance (fundamento 3)
   y el ranking puede cambiar entre estratos (paradoja de Simpson).

Este trabajo se plantea como una **investigación empírica**, no como una colección de modelos: interesa
saber *cuándo, por qué, con qué incertidumbre, a qué costo y con qué riesgos* falla cada enfoque.

## Objetivo general

Diseñar, implementar y comparar con validez estadística modelos de *deep learning* y clásicos para
(i) segmentación de instancias celulares y (ii) clasificación de texto, cuantificando no solo su
desempeño sino también cuándo, por qué, con qué incertidumbre, a qué costo y con qué riesgos fallan.

## Objetivos específicos

| ID | Objetivo | Dónde se cumple |
|---|---|---|
| O1 | Auditar los datos: sesgos, fuga y ruido de anotación | Notebooks 01 y 04; {doc}`datos` |
| O2 | Implementar desde cero un modelo por dominio (U-Net++ y BiLSTM/CNN-1D en Keras) con pérdidas y métricas propias | `src/project/models_seg.py`, `models_text.py`, `losses.py`, `seg_metrics.py` |
| O3 | Comparar bajo un protocolo controlado con inferencia estadística | {doc}`metodos`; notebooks 02 y 05 |
| O4 | Identificar con ablaciones qué componentes explican el rendimiento | Notebooks 03 y 05 |
| O5 | Evaluar robustez, calibración, incertidumbre y equidad | Notebooks 03 y 06 |
| O6 | Empaquetar y desplegar con pruebas, CI/CD y un experimento de deriva | {doc}`mlops`; notebook 07 |

## Preguntas de investigación

| ID | Pregunta | Dominio | Hipótesis |
|---|---|---|---|
| RQ1 | ¿Cómo varía el desempeño de los paradigmas según tipo celular, densidad y tamaño de objeto, y qué diferencias son estadística **y** prácticamente relevantes? | Imágenes | H3 |
| RQ2 | ¿Cuál es la contribución marginal de resolución, *augmentation*, pérdida y preentrenamiento? ¿Cómo escala el rendimiento con la fracción de datos? | Imágenes | H1 |
| RQ3 | ¿Son consistentes las métricas de píxel con las de instancia? ¿Cuándo se contradicen? | Imágenes | H2 |
| RQ4 | ¿Qué tan robustos y calibrados son los modelos bajo cambio de dominio y perturbaciones? | Imágenes | — |
| RQ5 | Controlando presupuesto y semillas, ¿justifica su costo un *transformer* frente a baselines clásicos? | NLP | H4 |
| RQ6 | ¿Qué proporción del rendimiento se debe a atajos léxicos o a ruido de etiquetas? | NLP | — |
| RQ7 | ¿Están calibradas las probabilidades y hay disparidades entre subgrupos? | NLP | H5 |
| RQ8 | ¿Cómo se degrada el rendimiento ante deriva y con qué rapidez y poder la detecta el monitoreo? | MLOps | H6 |

## Hipótesis falsables (refinadas y pre-registradas)

El texto completo, con la prueba y el criterio numérico de refutación, está en {doc}`anexos/preregistro`.

* **H1.** El preentrenamiento supera a U-Net++ desde cero con ≤ 25 % de los datos y la brecha se reduce
  con el 100 %. *Se refuta si el IC 95 % de Δ25 incluye 0 o si la brecha no se reduce.*
* **H2.** AUC y BA de píxel saturan (> 0.95) sin discriminar entre modelos; AP sí discrimina. *Se refuta
  si la ρ de Spearman entre rankings es > 0.9.*
* **H3.** El ranking global cambia al estratificar por tipo celular. *Se refuta si el orden es idéntico
  en los tres tipos.*
* **H4 (nula explícita).** La diferencia de F1 entre el mejor *transformer* y TF-IDF + XGBoost no es
  significativa tras Holm y varianza entre semillas. Se contrasta además la equivalencia práctica
  (TOST, δ = 0.02).
* **H5.** El *fine-tuning* empeora la ECE y el *temperature scaling* la recupera sin cambiar la
  exactitud. La segunda parte está garantizada por el fundamento 5; lo empírico es la magnitud.
* **H6.** El monitor detecta deriva de severidad ≥ 2 con poder ≥ 0.8 y falsas alarmas ≤ 5 %.

```{admonition} Valor de los resultados negativos
Refutar una hipótesis con evidencia sólida es un resultado tan valioso como confirmarla. Por eso todas las
hipótesis se declararon **antes** de ejecutar los experimentos, y las refutaciones se reportan con el
mismo detalle que las confirmaciones.
```
