AP medio por tipo celular:

| modelo | astro | cort | shsy5y |
|---|---|---|---|
| Baseline clásico | 0.01 | 0.006 | 0.016 |
| Cellpose 2.0 (ajustado) | 0.097 | 0.329 | 0.176 |
| Cellpose 2.0 (generalista) | 0.072 | 0.008 | 0.021 |
| Hover-Net | 0.025 | 0.224 | 0.098 |
| Mask R-CNN (+ Swin) | 0.003 | 0.066 | 0.015 |
| SAM (cero-shot) | 0.023 | 0.026 | 0.017 |
| SAM (prompts de caja) | 0.007 | 0.118 | 0.03 |
| U-Net++ + Watershed | 0.038 | 0.242 | 0.112 |

AP medio por tercil de densidad (número de células):

| modelo | baja | media | alta |
|---|---|---|---|
| Baseline clásico | 0.007 | 0.009 | 0.013 |
| Cellpose 2.0 (ajustado) | 0.33 | 0.216 | 0.16 |
| Cellpose 2.0 (generalista) | 0.008 | 0.04 | 0.03 |
| Hover-Net | 0.215 | 0.133 | 0.089 |
| Mask R-CNN (+ Swin) | 0.056 | 0.044 | 0.014 |
| SAM (cero-shot) | 0.024 | 0.026 | 0.019 |
| SAM (prompts de caja) | 0.106 | 0.076 | 0.029 |
| U-Net++ + Watershed | 0.226 | 0.165 | 0.09 |

Orden por estrato:

| estrato | orden | tau |
|---|---|---|
| global | Cellpose 2.0 (ajustado) > U-Net++ + Watershed > Hover-Net > SAM (prompts de caja) > Mask R-CNN (+ Swin) > Cellpose 2.0 (generalista) > SAM (cero-shot) > Baseline clásico | 1 |
| astro | Cellpose 2.0 (ajustado) > Cellpose 2.0 (generalista) > U-Net++ + Watershed > Hover-Net > SAM (cero-shot) > Baseline clásico > SAM (prompts de caja) > Mask R-CNN (+ Swin) | 0.4286 |
| cort | Cellpose 2.0 (ajustado) > U-Net++ + Watershed > Hover-Net > SAM (prompts de caja) > Mask R-CNN (+ Swin) > SAM (cero-shot) > Cellpose 2.0 (generalista) > Baseline clásico | 0.9286 |
| shsy5y | Cellpose 2.0 (ajustado) > U-Net++ + Watershed > Hover-Net > SAM (prompts de caja) > Cellpose 2.0 (generalista) > SAM (cero-shot) > Baseline clásico > Mask R-CNN (+ Swin) | 0.7857 |

**H3.** **Apoyada**: el orden de los modelos cambia al estratificar por tipo celular (τ de Kendall con el global: astro 0.43, cort 0.93, shsy5y 0.79).

**Mejor modelo por tipo:** astro: Cellpose 2.0 (ajustado); cort: Cellpose 2.0 (ajustado); shsy5y: Cellpose 2.0 (ajustado).
