| Hipótesis | Veredicto según el criterio pre-registrado |
|---|---|
| H1 | **Refutada**: Δ25 (Cellpose 2.0 (ajustado) − U-Net++ + Watershed) = 0.036 [0.011, 0.063]; Δ100 = 0.045 [0.014, 0.079]; Δ25 − Δ100 = -0.009 [-0.016, -0.003]. |
| H2 | no evaluable: las AUC medias no varían entre modelos |
| H3 | **Apoyada**: el orden de los modelos cambia al estratificar por tipo celular (τ de Kendall con el global: astro 0.67, cort 0.67, shsy5y -0.33). |
| H4 | pendiente (requiere transformer y TF-IDF + XGBoost en CV) |
| H5 | pendiente (requiere el transformer en test) |
| H6 | **Refutada** con los umbrales pre-registrados (por tipo de deriva: {'refutada': 5, 'no concluyente': 1, 'apoyada': 1}); ver la tabla para los umbrales calibrados. |
