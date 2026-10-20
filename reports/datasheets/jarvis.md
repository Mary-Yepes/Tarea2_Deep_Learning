# Datasheet — Jarvis Calling Hiring Contest (texto, reclutamiento)

Formato: Gebru et al. (2021). Al momento de escribir esta ficha no fue posible inspeccionar los archivos
desde el entorno de desarrollo, y la página pública de la competición solo describe una evaluación de
contratación en NLP y GenAI. **Todo el esquema debe verificarse** con `notebooks/04_nlp_eda_audit.ipynb`
(primera celda: `dtypes`, columnas y ejemplos) y registrarse abajo. El código no supone nombres de
columna: `text_data.detect_columns` propone (texto, etiqueta), la propuesta se confirma a mano y se fija
en `configs/nlp.yaml` (`text_col`, `label_col`).

## Motivación
* **Propósito.** Competición de Kaggle usada como prueba de selección de candidatos en NLP. Los textos
  están etiquetados en un **contexto de reclutamiento**.

## Composición (completar con el EDA)
| Elemento | Valor real | Fuente |
|---|---|---|
| Archivos | *[completar]* | `ls data/raw/jarvis` |
| Filas train / test | *[completar]* | notebook 04 |
| Columna de texto y de etiqueta | *[completar]* | notebook 04, `detect_columns` + revisión |
| Clases y proporciones | *[completar]* (razón de desbalance) | notebook 04, §1 |
| Idioma(s) y codificación | *[completar]* (no-ASCII, *mojibake*) | notebook 04, §2.5 |
| Longitud (p50/p95/p99 tokens) | *[completar]* | notebook 04, §1 |
| Duplicados exactos y casi-duplicados | *[completar]* | notebook 04, §2.1 |
| Error de etiqueta estimado (*confident learning*) | *[completar]* (+ revisión manual) | notebook 04, §2.2 |
| ¿Estructura temporal o por autor? | *[completar]* (si existe, la partición es temporal o por grupo) | — |

## Proceso de recolección y anotación
*[Completar con la descripción de la competición: origen de los textos, quién etiquetó, criterios.]* Si
no está documentado, se registra como **desconocido**. Esto es una limitación para la validez de
constructo: no se sabe qué mide exactamente la etiqueta.

## Usos
* **Previstos.** Evaluación metodológica de clasificadores de texto y estudio de atajos, calibración y
  equidad.
* **No previstos / desaconsejados.** **Decidir automáticamente sobre candidatos reales.** En la UE es
  un uso de **alto riesgo** según el AI Act (anexo III) y exige supervisión humana, gestión de riesgos y
  transparencia. En Colombia aplica la Ley 1581 de 2012 de protección de datos personales.

## Datos personales y atributos protegidos
* Se buscan automáticamente patrones de PII (correos, teléfonos, URL, fechas, edades) y proxies de
  atributos protegidos (términos de género, nombres propios, nacionalidad e instituciones):
  `text_data.pii_scan`.
* **Política.** No se exportan ejemplos con PII a reportes ni figuras. Si hay PII, se enmascara antes de
  publicar cualquier muestra cualitativa.
* No hay atributos protegidos auto-declarados. Las métricas por subgrupo se calculan solo con proxies
  justificables y con ≥ 30 ejemplos por grupo, y su carácter de proxy se declara.

## Distribución y licencia
Reglas de la competición en Kaggle **[verificar la pestaña *Rules*]**. Los datos no se redistribuyen y
se versionan con DVC.

## Limitaciones y riesgos
Posibles plantillas, firmas o nombres que actúen como atajos (se verifica con log-odds y con la ablación
de atajos). Ruido de etiquetas. Muestra posiblemente pequeña. Origen y criterio de etiquetado
desconocidos. Riesgo de sesgo histórico propio de los datos de contratación.

## Discrepancias con el enunciado (registro)
| Fecha | Supuesto | Observado | Acción |
|---|---|---|---|
| *[completar]* | | | |
