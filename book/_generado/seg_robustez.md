Degradación del AP con severidad creciente (sin TTA):

| modelo | perturbacion | auc_relativa | caida_sev5 |
|---|---|---|---|
| U-Net++ + Watershed | blur | 0.64 | 0.912 |
| U-Net++ + Watershed | brightness | 1 | 0.002 |
| U-Net++ + Watershed | contrast | 1.001 | -0.011 |
| U-Net++ + Watershed | illumination | 0.956 | 0.117 |
| U-Net++ + Watershed | jpeg | 0.859 | 0.49 |
| U-Net++ + Watershed | noise | 0.85 | 0.446 |

Efecto de TTA (AP medio):

| perturbacion | severidad | False | True |
|---|---|---|---|
| blur | 0 | 0.1612 | — |
| blur | 3 | 0.1039 | 0.1111 |
| brightness | 0 | 0.1612 | — |
| brightness | 3 | 0.1612 | 0.1687 |
| contrast | 0 | 0.1612 | — |
| contrast | 3 | 0.1609 | 0.1698 |
| illumination | 0 | 0.1612 | — |
| illumination | 3 | 0.1545 | 0.1664 |
| jpeg | 0 | 0.1612 | — |
| jpeg | 3 | 0.1474 | 0.1593 |
| noise | 0 | 0.1612 | — |
| noise | 3 | 0.1414 | 0.16 |
