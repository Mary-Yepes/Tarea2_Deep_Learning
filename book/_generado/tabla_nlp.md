Test sellado, media [IC 95 %] (bootstrap sobre ejemplos; probabilidades promediadas entre semillas).

| Modelo | accuracy | precision_macro | recall_macro | f1_macro | mcc | roc_auc | pr_auc | ece | latencia (ms/texto) |
|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + LR | 0.991 [0.977, 1.000] | 0.991 [0.978, 1.000] | 0.989 [0.973, 1.000] | 0.990 [0.975, 1.000] | 0.986 [0.965, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.062 [0.050, 0.074] | 0.049 |
| NB-SVM | 0.986 [0.972, 1.000] | 0.987 [0.971, 1.000] | 0.985 [0.968, 1.000] | 0.986 [0.970, 1.000] | 0.979 [0.957, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.010 [0.002, 0.020] | 0.069 |
| Clase mayoritaria | 0.367 [0.307, 0.433] | 0.122 [0.102, 0.144] | 0.333 [0.333, 0.333] | 0.179 [0.157, 0.201] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.333 [0.333, 0.333] | 0.633 [0.567, 0.693] | 0.011 |

F1 macro en CV repetida (media, DE entre semillas×folds, n):

| modelo | mean | std | count |
|---|---|---|---|
| TF-IDF + LR | 0.9875 | 0.002 | 2 |
| NB-SVM | 0.9835 | 0.0058 | 2 |
| Clase mayoritaria | 0.1783 | 0.0003 | 2 |

**H4.** pendiente (requiere transformer y TF-IDF + XGBoost en CV)

**H5.** pendiente (requiere el transformer en test)
