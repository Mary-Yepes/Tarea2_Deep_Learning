# %% [markdown]
# # 3 · Reporte final: modelos desplegables, notebooks 00–07, deriva, libro y paquete de entrega
#
# **Antes de ejecutar:** *Add Input* → las **salidas** de los notebooks 1A, 1B y 2 (pestaña *Notebooks*
# → *Your Work*), la competición Sartorius, la competición Jarvis y el dataset `proyecto-integrador`.
# **GPU** e **Internet: On**. Duración ≈ 1–1.5 h. Al final, descarga `ENTREGA.zip` de la pestaña *Output*.

# %%
import os, sys, glob
sys.path.insert(0, next(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/kaggle/comun.py", recursive=True)))
from comun import WORK, freeze_prereg, pack, restore_from_inputs, setup, sh
os.environ["BUDGET"] = "urgente"   # "rapido" (≈10–14 h GPU) o "completo" si tienes más tiempo
setup('fastapi httpx python-multipart pandera pytest pytest-timeout "jupyter-book>=1,<2" nbclient ipykernel')
restore_from_inputs()
freeze_prereg()
sh("ls reports/tables")

# %% [markdown]
# ## Modelos desplegables (mejor modelo serializable de cada dominio) y pruebas

# %%
sh("python scripts/train_deployable.py --data real --version 1.1.0 --promote")
sh("python -m pytest -q --timeout=600 -p no:cacheprovider", check=False)

# %% [markdown]
# ## Notebooks 00–07 con datos reales (de arriba abajo) y libro

# %%
sh("python scripts/build_notebooks.py --execute")
sh("python scripts/fill_report.py")
sh("mkdir -p book/resultados && cp notebooks/0*.ipynb book/resultados/")
sh("jupyter-book build book", check=False)
sh("python scripts/export_onnx.py", check=False)

# %% [markdown]
# ## Paquete de entrega

# %%
sh("grep -rl 'DEMO — cifras no reportables' book/resultados || echo 'OK: ningún notebook en modo demo'", check=False)
sh("cat book/_generado/hipotesis.md", check=False)
sh("rm -rf models/smoke reports/smoke reports/demo", check=False)
pack("ENTREGA", ["."], exclude=("models/seg/*abl*", "models/seg/*frac*", "models/seg/*loco*", "models/seg/hpo*",
                                 "models/seg/maskrcnn_s1.pt", "models/seg/maskrcnn_s2.pt", "data/processed/cellpose/*",
                                 "data/processed/coco/*", "data/processed/hovernet/*", "book/_build/.doctrees/*",
                                 "*/.ipynb_checkpoints/*", "*__pycache__*", "models/external/*"))
