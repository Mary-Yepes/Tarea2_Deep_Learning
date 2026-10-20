# Model card — Clasificador de texto desplegado (`text_*` en `models/registry.json`)

Formato: Mitchell et al. (2019). Los campos marcados con *[completar]* se rellenan con la versión
promovida a producción tras la comparación final. El modelo inicial desplegado es el **baseline TF-IDF +
Regresión Logística** con temperatura ajustada en validación.

## Detalles del modelo
* **Versión / registro:** `models/registry.json` → `text.production`. Hash de datos y commit en
  `meta.json` (`data_hash`, `git_commit`).
* **Tipo:** *[TF-IDF + LR | NB-SVM | TF-IDF + XGBoost | DistilBERT]*. Preprocesamiento:
  *[mínimo/estándar]*, el mejor según la ablación por familia.
* **Calibración:** *temperature scaling* (T en `meta.json`), ajustada **solo en validación**.
* **Licencia del código:** la del repositorio. Licencia de pesos preentrenados (si hay transformer):
  Apache-2.0 (DistilBERT/RoBERTa) **[verificar]**.

## Uso previsto
Herramienta de **apoyo** que prioriza o deriva textos de candidatos para revisión humana en un proceso de
selección. **No** debe rechazar candidatos de forma automática.

## Usos no previstos
Decisiones automáticas sobre personas, uso en otros idiomas o dominios sin revalidar, e inferencia de
atributos personales.

## Factores
Subgrupos por proxies textuales de género (pronombres, términos) y edad (menciones explícitas),
longitud del texto, presencia de plantillas e idioma.

## Métricas (test sellado, media [IC 95 %])
| Métrica | Valor |
|---|---|
| F1 macro (primaria) | *[completar]* |
| MCC, balanced accuracy, ROC-AUC, PR-AUC | *[completar]* |
| ECE (T=1 → T*), Brier | *[completar]* |
| Cobertura conforme (α = 0.1) | *[completar]* |
| Latencia p50/p95 (CPU) | *[completar]* (`scripts/load_test.py`) |

## Datos de evaluación y de entrenamiento
Ver `reports/datasheets/jarvis.md`. Partición 70/15/15 estratificada y agrupada por clúster de
casi-duplicados; test sellado.

## Análisis cuantitativos por subgrupo y contrafactuales
| Prueba | Resultado |
|---|---|
| Δp(favorable) al intercambiar género | *[completar]* (notebook 06, §6) |
| Δp(favorable) al cambiar la edad | *[completar]* |
| Paridad demográfica / igualdad de oportunidades / razón de impacto dispar | *[completar o «no evaluable: subgrupos insuficientes»]* |
| Tasa de fallo de invarianza (mayúsculas, typos, sinónimos, nombres) | *[completar]* |
| Caída de F1 al enmascarar atajos | *[completar]* |

## Consideraciones éticas y riesgos
* **Riesgos.** Sesgo histórico en las etiquetas, atajos léxicos (plantillas, nombres o instituciones)
  que actúan como proxies de atributos protegidos, y exceso de confianza.
* **Mitigaciones.** Supervisión humana obligatoria; derivación de casos con confianza baja o conjunto
  conforme de tamaño > 1; auditoría periódica por subgrupo con datos auto-declarados (con
  consentimiento); monitoreo de deriva (notebook 07) y *rollback* desde el registro; explicaciones y vía
  de impugnación para el candidato.

## Advertencias
Un modelo de demostración entrenado con datos sintéticos (`trained_on: synthetic — SOLO DEMO`) solo
sirve para pruebas de humo y CI, nunca para uso real.
