# %% [markdown]
# # 00 · Protocolo experimental pre-registrado
#
# Este notebook carga `configs/prereg.yaml`, comprueba que todos los elementos de la tabla de
# pre-registro estén completos, calcula su huella SHA-256 y exporta la tabla que se reproduce en el
# Jupyter Book. **Congelamiento:**
#
# ```bash
# git add configs/prereg.yaml && git commit -m "Pre-registro del protocolo" && git tag prereg
# ```
#
# Cualquier cambio posterior se agrega a `desviaciones` en el YAML y se declara en el reporte.
# Para verificar que el archivo no cambió desde la etiqueta, se usa
# `git diff prereg -- configs/prereg.yaml`.

# %%
import hashlib
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from project.config import CFG

ROOT = Path("..").resolve()
P = ROOT / "configs/prereg.yaml"
PREREG = yaml.safe_load(P.read_text(encoding="utf-8"))
print("SHA-256 del pre-registro:", hashlib.sha256(P.read_bytes()).hexdigest())
try:
    tag = subprocess.run(["git", "rev-list", "-n", "1", "prereg"], capture_output=True, text=True, cwd=ROOT)
    print("Commit de la etiqueta `prereg`:", tag.stdout.strip() or "(aún no creada: congelar antes de entrenar)")
except FileNotFoundError:
    print("git no disponible")

# %% [markdown]
# ## Verificación de completitud
#
# Elementos exigidos por la tabla de pre-registro: hipótesis con criterio numérico de refutación, métrica
# primaria, secundarias, unidad de análisis, particiones, pruebas, δ, semillas, presupuesto y criterios de
# parada.

# %%
REQUIRED = ["hipotesis", "metrica_primaria", "metricas_secundarias", "unidad_de_analisis", "particiones",
            "pruebas_estadisticas", "efecto_minimo_relevante", "semillas", "presupuesto", "criterios_de_parada"]
missing = [k for k in REQUIRED if not PREREG.get(k)]
for h, d in PREREG["hipotesis"].items():
    for f in ("enunciado", "contraste", "prueba", "refutacion"):
        if not d.get(f):
            missing.append(f"{h}.{f}")
assert len(PREREG["semillas"]["modelos_ligeros"]) >= 5 and len(PREREG["semillas"]["modelos_pesados"]) >= 3
assert PREREG["pruebas_estadisticas"]["alpha"] == CFG.alpha
print("Elementos faltantes:", missing or "ninguno")

# %%
tabla = pd.DataFrame([
    ("Hipótesis", "; ".join(f"{k}: {v['refutacion'].strip()}" for k, v in PREREG["hipotesis"].items())),
    ("Métrica primaria", f"Imágenes: {PREREG['metrica_primaria']['imagenes']}. NLP: {PREREG['metrica_primaria']['nlp']}"),
    ("Métricas secundarias", "Imágenes: " + ", ".join(PREREG["metricas_secundarias"]["imagenes"]) + ". NLP: " + ", ".join(PREREG["metricas_secundarias"]["nlp"])),
    ("Unidad de análisis", f"{PREREG['unidad_de_analisis']['imagenes']} / {PREREG['unidad_de_analisis']['nlp']}"),
    ("Diseño de particiones", PREREG["particiones"]["segmentacion"].strip() + " | " + PREREG["particiones"]["nlp"].strip()),
    ("Pruebas estadísticas", f"α = {PREREG['pruebas_estadisticas']['alpha']}, corrección {PREREG['pruebas_estadisticas']['correccion']}; "
     + PREREG["pruebas_estadisticas"]["tres_o_mas_modelos"]),
    ("Efecto mínimo relevante (δ)", f"AP: {PREREG['efecto_minimo_relevante']['imagenes_AP']}; F1: {PREREG['efecto_minimo_relevante']['nlp_F1']}; ECE: {PREREG['efecto_minimo_relevante']['ECE']}"),
    ("Semillas", f"ligeros {PREREG['semillas']['modelos_ligeros']}, pesados {PREREG['semillas']['modelos_pesados']}"),
    ("Presupuesto", f"{PREREG['presupuesto']['n_trials_hpo']} trials HPO por modelo; horas-GPU {PREREG['presupuesto']['horas_gpu_por_modelo']}"),
    ("Criterios de parada", "; ".join(f"{k}: {v}" for k, v in PREREG["criterios_de_parada"].items())),
], columns=["Elemento", "Declaración"])
pd.set_option("display.max_colwidth", 200)
(ROOT / "reports/tables").mkdir(parents=True, exist_ok=True)
tabla.to_csv(ROOT / "reports/tables/prereg_tabla.csv", index=False)
tabla

# %% [markdown]
# ## Registro de desviaciones
#
# Toda desviación respecto de este protocolo se añade a `desviaciones` con fecha, sección, motivo e
# impacto esperado. En el reporte se distingue entre análisis **confirmatorios** (pre-registrados) y
# **exploratorios**.

# %%
pd.DataFrame(PREREG.get("desviaciones") or [], columns=["fecha", "seccion", "descripcion", "motivo", "impacto_esperado"])
