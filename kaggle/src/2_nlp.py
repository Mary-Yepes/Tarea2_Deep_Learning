# %% [markdown]
# # 2 · NLP: HPO, CV repetida, test sellado, preprocesamiento y curvas de datos
#
# **Antes de ejecutar:** *Add Input* → **Competition** `jarvis-calling-hiring-contest` (únete a la
# competición si Kaggle lo pide) y **Dataset** `proyecto-integrador`; **GPU** e **Internet: On**.
# Duración estimada ≈ 30–90 min con el perfil «urgente» (según el tamaño del corpus). *Save & Run All (Commit)*.
#
# **Importante:** la primera celda imprime las columnas del CSV y las que se detectaron como texto y
# etiqueta. Si no son correctas, escríbelas en `configs/nlp.yaml` (`text_col`, `label_col`), sube de
# nuevo el dataset y vuelve a ejecutar.

# %%
import os, sys, glob
sys.path.insert(0, next(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/kaggle/comun.py", recursive=True)))
from comun import WORK, freeze_prereg, pack, setup, sh
os.environ["BUDGET"] = "urgente"   # "rapido" (≈10–14 h GPU) o "completo" si tienes más tiempo
setup("gensim fasttext-wheel")
freeze_prereg()
import glob as g
print("Archivos del dataset Jarvis:", g.glob(os.environ.get("JARVIS_DIR", "/kaggle/input/jarvis*") + "/**", recursive=True)[:20])

# %%
sh("python scripts/nlp_experiments.py all", env={"SMOKE": "1"})   # humo con 60 textos por clase
sh("rm -f data/processed/nlp_splits.csv")                        # el humo no deja partición
print("PRUEBA DE HUMO OK")

# %%
sh("python scripts/nlp_experiments.py prepare")
sh("python scripts/nlp_experiments.py hpo")
sh("python scripts/nlp_experiments.py cv")

# %%
sh("python scripts/nlp_experiments.py test")
sh("python scripts/nlp_experiments.py preprocess")
sh("python scripts/nlp_experiments.py fractions")
print("Listo. Las salidas quedan en la pestaña Output de este notebook.")
