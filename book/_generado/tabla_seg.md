Media [IC 95 %] por imagen del test sellado (bootstrap jerárquico placa → imagen; semillas promediadas por imagen). `rank_ap` = rango medio por imagen en la métrica primaria (menor = mejor).

| Modelo | AP@[.5:.95] | AP50 | PQ | Dice | IoU | Precision | Recall | AUC | HD95 | Bal. Acc | rank_ap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline clásico | 0.009 [0.007, 0.012] | 0.032 [0.026, 0.040] | 0.038 [0.030, 0.046] | 0.357 [0.318, 0.398] | 0.227 [0.197, 0.259] | 0.405 [0.337, 0.477] | 0.545 [0.479, 0.605] | 0.920 [0.909, 0.929] | 79.646 [62.464, 99.084] | 0.742 [0.713, 0.770] | 6.89 |
| Mask R-CNN (+ Swin) | 0.038 [0.028, 0.048] | 0.126 [0.097, 0.156] | 0.130 [0.101, 0.159] | 0.395 [0.325, 0.463] | 0.279 [0.222, 0.336] | 0.696 [0.639, 0.746] | 0.360 [0.283, 0.437] | 0.677 [0.638, 0.716] | 100.873 [77.309, 126.999] | 0.677 [0.638, 0.715] | 6.01 |
| Hover-Net | 0.145 [0.113, 0.181] | 0.374 [0.304, 0.445] | 0.339 [0.285, 0.393] | 0.674 [0.621, 0.715] | 0.523 [0.474, 0.565] | 0.683 [0.625, 0.737] | 0.744 [0.696, 0.790] | 0.969 [0.960, 0.976] | 35.055 [21.507, 57.030] | 0.855 [0.831, 0.877] | 3.06 |
| U-Net++ + Watershed | 0.161 [0.126, 0.199] | 0.385 [0.314, 0.455] | 0.356 [0.298, 0.412] | 0.691 [0.626, 0.740] | 0.547 [0.490, 0.594] | 0.719 [0.657, 0.771] | 0.719 [0.650, 0.778] | 0.971 [0.963, 0.978] | 36.374 [21.957, 57.350] | 0.842 [0.809, 0.870] | 2.68 |
| Cellpose 2.0 (ajustado) | 0.236 [0.196, 0.275] | 0.515 [0.446, 0.581] | 0.466 [0.414, 0.514] | 0.665 [0.606, 0.713] | 0.520 [0.465, 0.568] | 0.766 [0.706, 0.818] | 0.650 [0.593, 0.700] | 0.978 [0.971, 0.984] | 43.129 [28.895, 63.370] | 0.816 [0.787, 0.841] | 1.09 |
| Cellpose 2.0 (generalista) | 0.026 [0.018, 0.036] | 0.068 [0.048, 0.092] | 0.078 [0.056, 0.104] | 0.192 [0.140, 0.246] | 0.120 [0.086, 0.157] | 0.265 [0.189, 0.349] | 0.193 [0.148, 0.242] | 0.701 [0.638, 0.760] | 121.915 [99.809, 145.377] | 0.539 [0.509, 0.567] | 5.8 |
| SAM (cero-shot) | 0.023 [0.019, 0.028] | 0.050 [0.041, 0.060] | 0.066 [0.055, 0.078] | 0.210 [0.181, 0.243] | 0.123 [0.104, 0.145] | 0.225 [0.185, 0.268] | 0.247 [0.211, 0.282] | 0.566 [0.533, 0.597] | 97.945 [83.031, 114.929] | 0.568 [0.536, 0.598] | 5.74 |
| SAM (prompts de caja) | 0.070 [0.054, 0.086] | 0.172 [0.137, 0.209] | 0.186 [0.148, 0.222] | 0.411 [0.337, 0.481] | 0.295 [0.235, 0.355] | 0.713 [0.656, 0.763] | 0.379 [0.298, 0.461] | 0.687 [0.647, 0.728] | 102.214 [78.333, 129.108] | 0.687 [0.646, 0.728] | 4.73 |

**Friedman:** χ² = 408.15, p = 4.26e-84 (n = 86 imágenes); diferencia crítica de Nemenyi = 1.13 rangos.

**Wilcoxon por pares con Holm** (δ = 0.02):

| A | B | mean_diff | cliffs_delta | cliffs_mag | p_holm | reject | practically_relevant |
|---|---|---|---|---|---|---|---|
| Baseline clásico | Cellpose 2.0 (ajustado) | -0.2264 | -0.9862 | grande | 0 | True | True |
| Baseline clásico | Cellpose 2.0 (generalista) | -0.0168 | -0.2034 | pequeño | 0.0003 | True | False |
| Baseline clásico | Hover-Net | -0.1361 | -0.9159 | grande | 0 | True | True |
| Baseline clásico | Mask R-CNN (+ Swin) | -0.0285 | -0.4365 | mediano | 0 | True | True |
| Baseline clásico | SAM (cero-shot) | -0.0137 | -0.4873 | grande | 0 | True | False |
| Baseline clásico | SAM (prompts de caja) | -0.0603 | -0.6103 | grande | 0 | True | True |
| Baseline clásico | U-Net++ + Watershed | -0.1518 | -0.9048 | grande | 0 | True | True |
| Cellpose 2.0 (ajustado) | Cellpose 2.0 (generalista) | 0.2095 | 0.9264 | grande | 0 | True | True |
| Cellpose 2.0 (ajustado) | Hover-Net | 0.0903 | 0.401 | mediano | 0 | True | True |
| Cellpose 2.0 (ajustado) | Mask R-CNN (+ Swin) | 0.1978 | 0.8848 | grande | 0 | True | True |
| Cellpose 2.0 (ajustado) | SAM (cero-shot) | 0.2127 | 0.9511 | grande | 0 | True | True |
| Cellpose 2.0 (ajustado) | SAM (prompts de caja) | 0.166 | 0.7361 | grande | 0 | True | True |
| Cellpose 2.0 (ajustado) | U-Net++ + Watershed | 0.0746 | 0.3364 | mediano | 0 | True | True |
| Cellpose 2.0 (generalista) | Hover-Net | -0.1192 | -0.7531 | grande | 0 | True | True |
| Cellpose 2.0 (generalista) | Mask R-CNN (+ Swin) | -0.0117 | -0.2025 | pequeño | 0.104 | False | False |
| Cellpose 2.0 (generalista) | SAM (cero-shot) | 0.0032 | -0.1555 | pequeño | 0.5035 | False | False |
| Cellpose 2.0 (generalista) | SAM (prompts de caja) | -0.0435 | -0.4292 | mediano | 0.0002 | True | True |
| Cellpose 2.0 (generalista) | U-Net++ + Watershed | -0.135 | -0.7756 | grande | 0 | True | True |
| Hover-Net | Mask R-CNN (+ Swin) | 0.1075 | 0.6436 | grande | 0 | True | True |
| Hover-Net | SAM (cero-shot) | 0.1224 | 0.7528 | grande | 0 | True | True |
| Hover-Net | SAM (prompts de caja) | 0.0757 | 0.3972 | mediano | 0 | True | True |
| Hover-Net | U-Net++ + Watershed | -0.0157 | -0.0738 | despreciable | 0.0001 | True | False |
| Mask R-CNN (+ Swin) | SAM (cero-shot) | 0.0148 | 0.1228 | despreciable | 0.0247 | True | False |
| Mask R-CNN (+ Swin) | SAM (prompts de caja) | -0.0318 | -0.2761 | pequeño | 0 | True | True |
| Mask R-CNN (+ Swin) | U-Net++ + Watershed | -0.1233 | -0.6823 | grande | 0 | True | True |
| SAM (cero-shot) | SAM (prompts de caja) | -0.0467 | -0.4043 | mediano | 0 | True | True |
| SAM (cero-shot) | U-Net++ + Watershed | -0.1381 | -0.7861 | grande | 0 | True | True |
| SAM (prompts de caja) | U-Net++ + Watershed | -0.0915 | -0.4531 | mediano | 0 | True | True |
