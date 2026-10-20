# Model card — Segmentador de instancias desplegado (`image_*` en `models/registry.json`)

Formato: Mitchell et al. (2019). El modelo inicial desplegado es el **baseline clásico** (Otsu +
watershed, parámetros ajustados en validación). Tras la comparación final, se registra el ganador
(previsiblemente U-Net++ o Cellpose ajustado) con `scripts/train_deployable.py --image-model keras`.

## Detalles del modelo
* **Versión / registro:** `image.production`. `meta.json` contiene el tipo, los parámetros de
  post-procesamiento, `data_hash`, `git_commit` y las métricas de validación.
* **Salida:** mapa de instancias y lista de RLE (1-indexado, orden C) más el número de instancias.
* **Componentes reutilizados (si aplica):** versión, commit y licencia en `scripts/external/README.md`.

## Uso previsto
Conteo y segmentación de células en imágenes de contraste de fase **del mismo protocolo** que Sartorius
(SH-SY5Y, astrocitos y neuronas corticales), para investigación.

## Usos no previstos
Diagnóstico clínico. Otros microscopios, aumentos, tinciones o tipos celulares sin revalidar: la caída
*leave-one-cell-type-out* y las curvas de degradación están en el notebook 03.

## Factores
Tipo celular, densidad (terciles), tamaño de objeto (terciles), calidad de imagen (enfoque, iluminación)
y placa (`sample_id`).

## Métricas (test sellado, media [IC 95 %], bootstrap jerárquico por placa)
| Métrica | Global | shsy5y | astro | cort |
|---|---|---|---|---|
| AP@[0.5:0.95] (primaria) | *[completar]* | | | |
| AP50 / PQ / AJI | *[completar]* | | | |
| MAE del conteo, sesgo Bland–Altman | *[completar]* | | | |
| Errores dominantes (fusión, división, omisión…) | *[completar]* | | | |
| Latencia p50/p95 por imagen (CPU/GPU) | *[completar]* | | | |

## Robustez, calibración e incertidumbre
AUC relativa de degradación por perturbación, efecto de TTA, ECE de píxel y correlación entre
incertidumbre y error: *[completar con el notebook 03]*.

## Consideraciones éticas
No hay datos personales. Los riesgos son científicos: conclusiones biológicas erróneas por errores
sistemáticos, sobre todo fusiones en imágenes densas que sesgan el conteo hacia abajo. Mitigación:
reportar el conteo con su IC y monitorear la deriva de `n_instances` y de la calidad de entrada.

## Advertencias
El modelo de demostración (`trained_on: synthetic`) solo sirve para pruebas de humo.
