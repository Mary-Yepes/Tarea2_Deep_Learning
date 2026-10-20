# %% [markdown]
# # 1B · Segmentación: fracciones de datos (H1) y ablaciones (RQ2)
#
# **Antes de ejecutar:** las mismas entradas que 1A (competición Sartorius + dataset `proyecto-integrador`),
# **GPU** e **Internet: On**. Puede correr **en paralelo** con 1A. Duración estimada ≈ 1.5–2 h con el perfil «urgente».
# Después: *Save & Run All (Commit)*.

# %%
import os, sys, glob
sys.path.insert(0, next(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/kaggle/comun.py", recursive=True)))
from comun import WORK, freeze_prereg, pack, setup, sh
os.environ["BUDGET"] = "urgente"   # "rapido" (≈10–14 h GPU) o "completo" si tienes más tiempo
setup('"cellpose>=3,<4"')
freeze_prereg()

# %%
sh("python scripts/seg_experiments.py fractions", env={"SMOKE": "1"})      # humo (minutos)
sh("python scripts/seg_experiments.py ablations", env={"SMOKE": "1"})
print("PRUEBA DE HUMO OK")

# %%
sh("python scripts/seg_experiments.py split")
sh("python scripts/seg_experiments.py fractions")

# %%
sh("python scripts/seg_experiments.py ablations")
print("Listo. Las salidas quedan en la pestaña Output de este notebook.")
