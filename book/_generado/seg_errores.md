Errores por objeto de referencia (taxonomía a priori):

| modelo | tipo_celular | omitida_por_objeto | falsa_por_objeto | fusion_por_objeto | division_por_objeto | desplazada_por_objeto |
|---|---|---|---|---|---|---|
| Baseline clásico | astro | 0.382 | 0.578 | 0.04 | 0.327 | 0.164 |
| Baseline clásico | cort | 0.045 | 10.26 | 0.159 | 0.16 | 0.404 |
| Baseline clásico | shsy5y | 0.146 | 0.652 | 0.198 | 0.151 | 0.384 |
| Cellpose 2.0 (ajustado) | astro | 0.384 | 0.093 | 0.016 | 0.027 | 0.215 |
| Cellpose 2.0 (ajustado) | cort | 0.143 | 0.101 | 0.047 | 0.004 | 0.045 |
| Cellpose 2.0 (ajustado) | shsy5y | 0.192 | 0.038 | 0.15 | 0.019 | 0.095 |
| Cellpose 2.0 (generalista) | astro | 0.604 | 0.038 | 0.065 | 0.004 | 0.113 |
| Cellpose 2.0 (generalista) | cort | 0.929 | 2.042 | 0.016 | 0 | 0.019 |
| Cellpose 2.0 (generalista) | shsy5y | 0.747 | 0.18 | 0.13 | 0 | 0.07 |
| Hover-Net | astro | 0.073 | 0.342 | 0.018 | 0.358 | 0.324 |
| Hover-Net | cort | 0.088 | 0.241 | 0.06 | 0.017 | 0.104 |
| Hover-Net | shsy5y | 0.04 | 0.16 | 0.277 | 0.053 | 0.164 |
| Mask R-CNN (+ Swin) | astro | 0.771 | 0.091 | 0.001 | 0.055 | 0.153 |
| Mask R-CNN (+ Swin) | cort | 0.269 | 0.627 | 0.019 | 0.114 | 0.108 |
| Mask R-CNN (+ Swin) | shsy5y | 0.672 | 0.044 | 0.037 | 0.033 | 0.177 |
| SAM (cero-shot) | astro | 0.737 | 0.33 | 0.124 | 0.006 | 0.058 |
| SAM (cero-shot) | cort | 0.69 | 2.357 | 0.07 | 0.001 | 0.051 |
| SAM (cero-shot) | shsy5y | 0.867 | 0.316 | 0.058 | 0 | 0.04 |
| SAM (prompts de caja) | astro | 0.775 | 0.085 | 0 | 0.053 | 0.142 |
| SAM (prompts de caja) | cort | 0.268 | 0.565 | 0.024 | 0.061 | 0.08 |
| SAM (prompts de caja) | shsy5y | 0.683 | 0.059 | 0.048 | 0.027 | 0.125 |
| U-Net++ + Watershed | astro | 0.042 | 0.415 | 0.117 | 0.288 | 0.258 |
| U-Net++ + Watershed | cort | 0.157 | 0.145 | 0.04 | 0.014 | 0.091 |
| U-Net++ + Watershed | shsy5y | 0.041 | 0.228 | 0.265 | 0.048 | 0.128 |

**Error dominante:** Baseline clásico / astro: falsa; Baseline clásico / cort: falsa; Baseline clásico / shsy5y: falsa; Cellpose 2.0 (ajustado) / astro: omitida; Cellpose 2.0 (ajustado) / cort: omitida; Cellpose 2.0 (ajustado) / shsy5y: omitida; Cellpose 2.0 (generalista) / astro: omitida; Cellpose 2.0 (generalista) / cort: falsa; Cellpose 2.0 (generalista) / shsy5y: omitida; Hover-Net / astro: division; Hover-Net / cort: falsa; Hover-Net / shsy5y: fusion; Mask R-CNN (+ Swin) / astro: omitida; Mask R-CNN (+ Swin) / cort: falsa; Mask R-CNN (+ Swin) / shsy5y: omitida; SAM (cero-shot) / astro: omitida; SAM (cero-shot) / cort: falsa; SAM (cero-shot) / shsy5y: omitida; SAM (prompts de caja) / astro: omitida; SAM (prompts de caja) / cort: falsa; SAM (prompts de caja) / shsy5y: omitida; U-Net++ + Watershed / astro: falsa; U-Net++ + Watershed / cort: omitida; U-Net++ + Watershed / shsy5y: fusion.
