# Resultados: segmentación de instancias

Las tablas de este capítulo se **generan automáticamente** a partir de `reports/tables/` con
`python scripts/fill_report.py`. Los veredictos de las hipótesis aplican el criterio pre-registrado. El
detalle, las figuras y el código están en los notebooks:

1. {doc}`resultados/01_seg_eda_audit`: EDA, auditoría y decisiones de diseño.
2. {doc}`resultados/02_seg_models_eval`: comparación de modelos e inferencia estadística.
3. {doc}`resultados/03_seg_robustness_ablations`: ablaciones, robustez, calibración, explicabilidad y
   eficiencia.

## Datos (cifras verificadas en los archivos)

```{include} _generado/eda_seg.md
```

## Tabla de resultados en el test sellado

```{include} _generado/tabla_seg.md
```

## Estratificación por tipo celular y densidad (RQ1, H3)

```{include} _generado/seg_estratificado.md
```

## Métricas de píxel vs instancia (RQ3, H2)

```{include} _generado/seg_contrastes.md
```

## Taxonomía de errores

```{include} _generado/seg_errores.md
```

## Eficiencia de datos (RQ2, H1)

```{include} _generado/seg_fracciones.md
```

## Ablaciones (RQ2)

```{include} _generado/seg_ablaciones.md
```

## Robustez y cambio de dominio (RQ4)

```{include} _generado/seg_robustez.md
```

```{include} _generado/seg_loco.md
```
