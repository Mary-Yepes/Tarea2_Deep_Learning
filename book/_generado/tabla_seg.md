Media [IC 95 %] por imagen del test sellado (bootstrap jerárquico placa → imagen; semillas promediadas por imagen). `rank_ap` = rango medio por imagen en la métrica primaria (menor = mejor).

| Modelo | AP@[.5:.95] | AP50 | PQ | Dice | IoU | Precision | Recall | AUC | HD95 | Bal. Acc | rank_ap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline clásico | 0.156 [0.092, 0.223] | 0.577 [0.363, 0.798] | 0.416 [0.295, 0.534] | 0.754 [0.713, 0.788] | 0.610 [0.559, 0.652] | 0.640 [0.619, 0.666] | 0.936 [0.835, 0.999] | 0.871 [0.787, 0.943] | 5.402 [3.866, 7.477] | 0.876 [0.812, 0.929] | 2.08 |
| Hover-Net | 0.141 [0.082, 0.206] | 0.499 [0.311, 0.694] | 0.382 [0.268, 0.496] | 0.754 [0.713, 0.788] | 0.610 [0.559, 0.652] | 0.640 [0.619, 0.666] | 0.937 [0.835, 0.999] | 0.871 [0.787, 0.943] | 5.401 [3.866, 7.477] | 0.876 [0.812, 0.929] | 2.69 |
| U-Net++ + Watershed | 0.108 [0.070, 0.151] | 0.434 [0.283, 0.601] | 0.343 [0.250, 0.435] | 0.710 [0.670, 0.750] | 0.554 [0.505, 0.604] | 0.625 [0.598, 0.652] | 0.842 [0.739, 0.935] | 0.871 [0.787, 0.943] | 7.619 [3.801, 13.547] | 0.838 [0.769, 0.904] | 2.78 |
| Cellpose 2.0 (ajustado) | 0.153 [0.091, 0.216] | 0.568 [0.362, 0.779] | 0.411 [0.290, 0.526] | 0.755 [0.713, 0.788] | 0.610 [0.559, 0.652] | 0.640 [0.619, 0.667] | 0.936 [0.835, 0.999] | 0.871 [0.787, 0.943] | 5.403 [3.866, 7.477] | 0.876 [0.812, 0.929] | 2.44 |

**Friedman:** χ² = 3.44, p = 3.29e-01 (n = 18 imágenes); diferencia crítica de Nemenyi = 1.11 rangos.

**Wilcoxon por pares con Holm** (δ = 0.02):

| A | B | mean_diff | cliffs_delta | cliffs_mag | p_holm | reject | practically_relevant |
|---|---|---|---|---|---|---|---|
| Baseline clásico | Cellpose 2.0 (ajustado) | 0.0032 | 0.0432 | despreciable | 0.5675 | False | False |
| Baseline clásico | Hover-Net | 0.015 | 0.071 | despreciable | 0.2355 | False | False |
| Baseline clásico | U-Net++ + Watershed | 0.0477 | 0.216 | pequeño | 0.2997 | False | True |
| Cellpose 2.0 (ajustado) | Hover-Net | 0.0119 | 0.0741 | despreciable | 0.4012 | False | False |
| Cellpose 2.0 (ajustado) | U-Net++ + Watershed | 0.0446 | 0.216 | pequeño | 0.2997 | False | True |
| Hover-Net | U-Net++ + Watershed | 0.0327 | 0.1173 | despreciable | 0.5675 | False | True |
