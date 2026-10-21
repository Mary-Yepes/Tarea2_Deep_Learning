Test sellado, media [IC 95 %] (bootstrap sobre ejemplos; probabilidades promediadas entre semillas).

| Modelo | accuracy | precision_macro | recall_macro | f1_macro | mcc | roc_auc | pr_auc | ece | latencia (ms/texto) |
|---|---|---|---|---|---|---|---|---|---|
| DistilBERT | 0.785 [0.742, 0.827] | 0.727 [0.685, 0.775] | 0.721 [0.687, 0.762] | 0.712 [0.671, 0.751] | 0.776 [0.731, 0.820] | 0.960 [0.945, 0.974] | 0.784 [0.752, 0.832] | 0.082 [0.061, 0.122] | 10.020 |
| TF-IDF + XGBoost | 0.762 [0.720, 0.810] | 0.731 [0.678, 0.787] | 0.712 [0.668, 0.759] | 0.708 [0.654, 0.753] | 0.752 [0.708, 0.801] | 0.974 [0.965, 0.984] | 0.801 [0.767, 0.856] | 0.058 [0.050, 0.107] | 1.760 |
| CNN-1D | 0.759 [0.714, 0.802] | 0.710 [0.656, 0.757] | 0.699 [0.658, 0.745] | 0.686 [0.636, 0.729] | 0.749 [0.702, 0.793] | 0.973 [0.961, 0.985] | 0.813 [0.771, 0.865] | 0.069 [0.062, 0.123] | 1.334 |
| NB-SVM | 0.722 [0.680, 0.768] | 0.701 [0.630, 0.757] | 0.670 [0.628, 0.726] | 0.664 [0.610, 0.714] | 0.710 [0.666, 0.757] | 0.952 [0.932, 0.969] | 0.729 [0.693, 0.792] | 0.125 [0.108, 0.176] | 1.835 |
| Word2Vec + BiLSTM | 0.731 [0.685, 0.776] | 0.653 [0.603, 0.707] | 0.671 [0.634, 0.712] | 0.652 [0.605, 0.695] | 0.719 [0.671, 0.766] | 0.940 [0.921, 0.957] | 0.713 [0.678, 0.762] | 0.069 [0.064, 0.126] | 1.504 |
| TF-IDF + LR | 0.646 [0.598, 0.691] | 0.640 [0.576, 0.696] | 0.612 [0.561, 0.664] | 0.604 [0.544, 0.647] | 0.631 [0.580, 0.678] | 0.945 [0.923, 0.964] | 0.656 [0.622, 0.723] | 0.312 [0.270, 0.353] | 1.174 |
| FastText | 0.487 [0.433, 0.541] | 0.448 [0.400, 0.501] | 0.447 [0.404, 0.493] | 0.437 [0.389, 0.477] | 0.464 [0.408, 0.519] | 0.876 [0.849, 0.899] | 0.471 [0.440, 0.540] | 0.132 [0.103, 0.186] | 0.773 |
| Clase mayoritaria | 0.048 [0.028, 0.068] | 0.002 [0.001, 0.003] | 0.042 [0.042, 0.043] | 0.004 [0.002, 0.006] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.042 [0.042, 0.042] | 0.952 [0.932, 0.972] | 0.716 |

Comparación por pares (t remuestreada corregida de Nadeau–Bengio sobre semilla×fold, Holm):

| A | B | dF1 | n_rep | p_NB | p_holm |
|---|---|---|---|---|---|
| CNN-1D | Clase mayoritaria | 0.6633 | 15 | 0 | 0 |
| CNN-1D | FastText | 0.3262 | 15 | 0 | 0 |
| CNN-1D | NB-SVM | -0.0001 | 15 | 0.9943 | 1 |
| CNN-1D | TF-IDF + LR | 0.0814 | 15 | 0 | 0 |
| CNN-1D | TF-IDF + XGBoost | -0.0495 | 15 | 0.0041 | 0.0244 |
| CNN-1D | Word2Vec + BiLSTM | 0.0726 | 15 | 0.0117 | 0.041 |
| Clase mayoritaria | FastText | -0.3371 | 15 | 0 | 0 |
| Clase mayoritaria | NB-SVM | -0.6634 | 15 | 0 | 0 |
| Clase mayoritaria | TF-IDF + LR | -0.5819 | 15 | 0 | 0 |
| Clase mayoritaria | TF-IDF + XGBoost | -0.7128 | 15 | 0 | 0 |
| Clase mayoritaria | Word2Vec + BiLSTM | -0.5907 | 15 | 0 | 0 |
| FastText | NB-SVM | -0.3263 | 15 | 0 | 0 |
| FastText | TF-IDF + LR | -0.2448 | 15 | 0 | 0 |
| FastText | TF-IDF + XGBoost | -0.3757 | 15 | 0 | 0 |
| FastText | Word2Vec + BiLSTM | -0.2536 | 15 | 0 | 0 |
| NB-SVM | TF-IDF + LR | 0.0815 | 15 | 0 | 0.0002 |
| NB-SVM | TF-IDF + XGBoost | -0.0494 | 15 | 0.0047 | 0.0244 |
| NB-SVM | Word2Vec + BiLSTM | 0.0727 | 15 | 0.0102 | 0.041 |
| TF-IDF + LR | TF-IDF + XGBoost | -0.1309 | 15 | 0 | 0 |
| TF-IDF + LR | Word2Vec + BiLSTM | -0.0088 | 15 | 0.7165 | 1 |
| TF-IDF + XGBoost | Word2Vec + BiLSTM | 0.1221 | 15 | 0.0002 | 0.0016 |

F1 macro en CV repetida (media, DE entre semillas×folds, n):

| modelo | mean | std | count |
|---|---|---|---|
| DistilBERT | 0.7362 | 0.0144 | 5 |
| TF-IDF + XGBoost | 0.7165 | 0.0208 | 15 |
| NB-SVM | 0.6671 | 0.027 | 15 |
| CNN-1D | 0.667 | 0.0214 | 15 |
| Word2Vec + BiLSTM | 0.5944 | 0.0447 | 15 |
| TF-IDF + LR | 0.5856 | 0.0201 | 15 |
| FastText | 0.3408 | 0.0159 | 15 |
| Clase mayoritaria | 0.0037 | 0.0002 | 15 |

**H4.** **No concluyente** (ni diferencia relevante demostrada ni equivalencia): ΔF1 (DistilBERT − TF-IDF + XGBoost) = +0.0174, IC 90 % corregido [-0.0190, 0.0538], p_Holm = nan, 5 repeticiones.

**H5.** Parte 1 refutada: ECE(DistilBERT, T=1) − ECE(TF-IDF+LR) IC 95 % [-0.2755, -0.1615]. Parte 2 refutada: reducción de ECE con T* = 30%, ECE(T*) = 0.0645 vs TF-IDF+LR 0.3122. La exactitud no cambia (fundamento 5).
