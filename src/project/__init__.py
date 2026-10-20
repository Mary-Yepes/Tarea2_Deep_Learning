"""Proyecto integrador: segmentación de instancias celulares y clasificación de texto.

Paquete instalable con las piezas metodológicas del proyecto (métricas, inferencia
estadística, calibración, deriva, particiones sin fuga, modelos propios y utilidades
de datos). Las dependencias pesadas (TensorFlow, XGBoost, statsmodels, pandera,
optuna) se importan de forma perezosa dentro de las funciones que las necesitan,
de modo que el núcleo (NumPy/SciPy/pandas/scikit-learn) se puede probar sin GPU.
"""

__version__ = "1.0.0"
