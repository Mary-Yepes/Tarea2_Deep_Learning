| Hipótesis | Veredicto según el criterio pre-registrado |
|---|---|
| H1 | **Apoyada**: Δ25 (Cellpose 2.0 (ajustado) − U-Net++ + Watershed) = 0.218 [0.188, 0.247]; Δ100 = 0.075 [0.065, 0.085]; Δ25 − Δ100 = 0.142 [0.115, 0.171]. |
| H2 | **Refutada**: ρ de Spearman entre rankings por AUC de píxel y por AP = 0.89 [0.79, 1.00]; AUC medias 0.566–0.978 (saturación no). |
| H3 | **Apoyada**: el orden de los modelos cambia al estratificar por tipo celular (τ de Kendall con el global: astro 0.43, cort 0.93, shsy5y 0.79). |
| H4 | **No concluyente** (ni diferencia relevante demostrada ni equivalencia): ΔF1 (DistilBERT − TF-IDF + XGBoost) = +0.0174, IC 90 % corregido [-0.0190, 0.0538], p_Holm = nan, 5 repeticiones. |
| H5 | Parte 1 refutada: ECE(DistilBERT, T=1) − ECE(TF-IDF+LR) IC 95 % [-0.2755, -0.1615]. Parte 2 refutada: reducción de ECE con T* = 30%, ECE(T*) = 0.0645 vs TF-IDF+LR 0.3122. La exactitud no cambia (fundamento 5). |
| H6 | **Refutada** con los umbrales pre-registrados (por tipo de deriva: {'refutada': 4, 'apoyada': 2, 'no concluyente': 1}); ver la tabla para los umbrales calibrados. |
