# %% [markdown]
# # 1A · Segmentación: comparación de los 6 paradigmas (test sellado)
#
# **Antes de ejecutar** (panel derecho del editor de Kaggle):
# 1. *Add Input* → **Competition** `sartorius-cell-instance-segmentation` y **Dataset** `proyecto-integrador`
#    (el zip del proyecto que subiste).
# 2. *Settings* → **Accelerator: GPU** (T4 o P100) e **Internet: On**.
# 3. *Save Version* → **Save & Run All (Commit)**. Corre sola (hasta 12 h) aunque cierres el navegador.
#
# Duración estimada: 15 min de prueba de humo + ≈ 2.5–3.5 h con el perfil «urgente». Si algo falla, el notebook se detiene con el
# error en la pestaña *Logs*: cópialo y envíalo para corregirlo. Lo ya terminado no se repite (el código
# es reanudable), así que basta con volver a ejecutar.

# %%
import os, sys, glob
sys.path.insert(0, next(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/kaggle/comun.py", recursive=True)))
from comun import WORK, freeze_prereg, pack, setup, sh
os.environ["BUDGET"] = "urgente"   # "rapido" (≈10–14 h GPU) o "completo" si tienes más tiempo
setup('"cellpose>=3,<4" "git+https://github.com/facebookresearch/segment-anything.git"')
freeze_prereg()

# %% [markdown]
# ## Prueba de humo (≈ 10–15 min): ejecuta TODO con 2 imágenes por tipo y 1 época para detectar errores

# %%
SMOKE = {"SMOKE": "1"}
for only in ("classical", "unetpp hovernet", "maskrcnn", "cellpose", "sam"):
    sh(f"python scripts/seg_experiments.py main --only {only}", env=SMOKE)
sh("python scripts/seg_experiments.py extras", env=SMOKE)
print("PRUEBA DE HUMO OK")

# %% [markdown]
# ## Corrida real (cada modelo en un proceso separado: TensorFlow y PyTorch no compiten por la GPU)

# %%
sh("python scripts/seg_experiments.py split")
sh("python scripts/seg_experiments.py eda-params")
sh("python scripts/seg_experiments.py main --only classical")
sh("python scripts/seg_experiments.py main --only unetpp hovernet")

# %%
sh("python scripts/seg_experiments.py main --only maskrcnn")

# %%
sh("python scripts/seg_experiments.py main --only cellpose")
sh("python scripts/seg_experiments.py main --only sam")

# %%
sh("python scripts/seg_experiments.py robustness")
sh("python scripts/seg_experiments.py loco")
import pandas as pd
print(pd.read_csv(WORK / "reports/tables/seg_eval_test.csv").groupby("modelo").ap.agg(["mean", "count"]))
print("Listo. Las salidas quedan en la pestaña Output de este notebook.")
