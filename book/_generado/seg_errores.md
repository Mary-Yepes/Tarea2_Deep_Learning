Errores por objeto de referencia (taxonomía a priori):

| modelo | tipo_celular | omitida_por_objeto | falsa_por_objeto | fusion_por_objeto | division_por_objeto | desplazada_por_objeto |
|---|---|---|---|---|---|---|
| Baseline clásico | astro | 0.096 | 0.627 | 0.06 | 0.108 | 0.108 |
| Baseline clásico | cort | 0 | 0 | 0.067 | 0 | 0 |
| Baseline clásico | shsy5y | 0 | 0 | 0.528 | 0.011 | 0.082 |
| Cellpose 2.0 (ajustado) | astro | 0.096 | 0.747 | 0.06 | 0.145 | 0.084 |
| Cellpose 2.0 (ajustado) | cort | 0 | 0.011 | 0.022 | 0 | 0.022 |
| Cellpose 2.0 (ajustado) | shsy5y | 0 | 0.004 | 0.457 | 0.026 | 0.112 |
| Hover-Net | astro | 0.096 | 0.41 | 0.145 | 0.06 | 0.145 |
| Hover-Net | cort | 0 | 0 | 0.178 | 0 | 0 |
| Hover-Net | shsy5y | 0 | 0 | 0.688 | 0 | 0.019 |
| U-Net++ + Watershed | astro | 0.096 | 1.072 | 0.096 | 0.133 | 0.145 |
| U-Net++ + Watershed | cort | 0 | 0.356 | 0 | 0 | 0.022 |
| U-Net++ + Watershed | shsy5y | 0.078 | 0.335 | 0.097 | 0.026 | 0.271 |

**Error dominante:** Baseline clásico / astro: falsa; Baseline clásico / cort: fusion; Baseline clásico / shsy5y: fusion; Cellpose 2.0 (ajustado) / astro: falsa; Cellpose 2.0 (ajustado) / cort: fusion; Cellpose 2.0 (ajustado) / shsy5y: fusion; Hover-Net / astro: falsa; Hover-Net / cort: fusion; Hover-Net / shsy5y: fusion; U-Net++ + Watershed / astro: falsa; U-Net++ + Watershed / cort: falsa; U-Net++ + Watershed / shsy5y: falsa.
