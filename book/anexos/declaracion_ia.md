# Anexo F · Declaración de uso de asistentes de IA

El enunciado (sección «Restricciones») exige declarar de forma transparente el uso de asistentes de IA:
**qué** se generó, **dónde** y **cómo se verificó**. El autor o la autora responde íntegramente por la
corrección de todo lo entregado.

**Asistente:** Claude (Anthropic), usado en octubre de 2026 a través de la aplicación Claude.
**Autoras:** Daniela Hernández, Alejandra Meneses Gómez y Mariangel Yepes.    
**Revisión humana:** las tres autoras, 19–20 de octubre de 2026 (ejecución de los notebooks en Kaggle, lectura de los registros y de las tablas de resultados; ver el alcance exacto más abajo).

| Qué generó el asistente | Dónde | Cómo se verificó |
|---|---|---|
| Borrador del paquete `src/project`: métricas, estadística, calibración, deriva, particiones, U-Net++, Hover-Net, BiLSTM y CNN-1D en Keras, Mask R-CNN+Swin, adaptadores | `src/project/*.py` | Pruebas unitarias con casos conocidos (`tests/`), ruff y mypy sin errores; prueba de humo de cada modelo con datos reales en Kaggle antes de la corrida larga; *[revisión línea a línea por …]* |
| Demostraciones de los 6 fundamentos teóricos | `book/fundamentos_teoricos.ipynb` | Cada resultado verificado numéricamente en una celda (diferencias finitas, simulación, contraejemplos); *[revisión de cada paso algebraico por …]* |
| Propuesta de pre-registro (H1–H6, δ, presupuesto) | `configs/prereg.yaml` | Revisado y aceptado por el autor **antes** de congelarlo (etiqueta git `prereg`, huella en `reports/prereg_congelado.json`) |
| Scripts de experimentos y notebooks de Kaggle | `scripts/`, `kaggle/` | Ejecución completa; manifiestos de cada corrida en `reports/manifests/` |
| Texto de capítulos, *model cards*, *datasheets* y autoevaluación | `book/`, `reports/` | Las tablas y los veredictos se generan desde los resultados (`scripts/fill_report.py`); *[el texto fue leído y ajustado por el autor]* |
| Infraestructura (API, Dockerfile, CI, Makefile) | `app/`, `.github/`, raíz | Pruebas de API en CI; *[enlace a la ejecución de CI, si se subió a GitHub]* |

**Ningún resultado numérico fue escrito por el asistente.** Todas las cifras de resultados provienen de
la ejecución del código sobre los datos; las cifras de los fundamentos teóricos, de simulaciones
reproducibles con semilla fija.
