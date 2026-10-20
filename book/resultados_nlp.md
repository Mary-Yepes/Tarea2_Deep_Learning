# Resultados: clasificación de texto

Las tablas se generan automáticamente con `python scripts/fill_report.py`. El detalle está en los
notebooks:

1. {doc}`resultados/04_nlp_eda_audit`: EDA y auditoría (duplicados, ruido de etiquetas, atajos, PII).
2. {doc}`resultados/05_nlp_models_eval`: comparación con repeticiones, H4, calibración (H5), HPO y
   preprocesamiento por familia.
3. {doc}`resultados/06_nlp_robustness_fairness`: CheckList, atajos, predicción conforme, explicabilidad
   y equidad.

## Datos

```{include} _generado/eda_nlp.md
```

## Resultados en el test sellado, comparación estadística, H4 y H5

```{include} _generado/tabla_nlp.md
```
