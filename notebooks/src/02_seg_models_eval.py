# %% [markdown]
# # 02 · Segmentación: modelos, HPO y evaluación con inferencia estadística
#
# **Entrada:** `reports/tables/seg_eval_test.csv` (métricas por imagen de cada modelo y semilla en el test
# sellado). Ese archivo lo genera `scripts/seg_experiments.py evaluate ...` para los seis paradigmas,
# todos con el mismo código de evaluación (`project.adapters_seg.evaluate_adapter`).
#
# | Pos | Modelo | Implementación | Comando |
# |---|---|---|---|
# | 0 | Otsu/adaptativo + watershed | propia (`classical.py`) | `evaluate --model classical` |
# | 1 | Mask R-CNN + Swin | MMDetection (config en `configs/seg_maskrcnn.yaml`) | `evaluate --model maskrcnn_mmdet:cfg,ckpt` |
# | 2 | Hover-Net (NP + HV) | repo oficial + post-proceso propio | `evaluate --model hovernet:mod:fn` |
# | 3 | U-Net++ + watershed | **propia en Keras** (`models_seg.py`) | `train` + `evaluate --model unetpp:ruta` |
# | 4 | Cellpose 2.0 (generalista / ajustado) | `cellpose` 3.x | `evaluate --model cellpose:cyto2` |
# | 5 | SAM / MedSAM (cero-shot, cajas, LoRA) | `segment-anything` | `evaluate --model sam_auto:ckpt` |
#
# > **Modo demo.** Si el CSV no existe, el notebook evalúa en datos sintéticos el baseline clásico y dos
# > variantes declaradas (umbral adaptativo y `min_distance` = 8). Así se ejercita **todo** el pipeline
# > estadístico, pero esas cifras no son resultados del proyecto.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

import yaml

from project.config import CFG, set_global_seed
from project.reporting import (add_terciles, best_worst, error_profile, pairwise_report, ranking_agreement,
                               results_table, simpson_check, stratified)
from project.seg_metrics import THRESHOLDS
from project.stats import bootstrap_ci, fmt_ci, mixed_model_check, plot_cd_diagram

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*constant.*")
set_global_seed(CFG.seed)
ROOT = Path("..").resolve()
PREREG = yaml.safe_load((ROOT / "configs/prereg.yaml").read_text())
DELTA = PREREG["efecto_minimo_relevante"]["imagenes_AP"]
SRC = ROOT / "reports/tables/seg_eval_test.csv"
REAL = SRC.exists()
from project.config import report_dirs
TAB, FIG = report_dirs(ROOT, REAL)
if REAL:
    df = pd.read_csv(SRC)
else:
    from project.adapters_seg import ClassicalAdapter, evaluate_adapter
    from project.data_seg import make_synthetic_dataset
    X, Y, meta = make_synthetic_dataset(10, shape=(128, 160), seed=777)
    rows = []
    for ad in (ClassicalAdapter(name="Baseline Otsu"), ClassicalAdapter({"method": "adaptive"}, name="Variante adaptativa"),
               ClassicalAdapter({"min_distance": 8}, name="Variante min_dist=8")):
        rows += evaluate_adapter(ad, X, Y, meta.id_imagen.tolist())
    df = pd.DataFrame(rows).merge(meta[["id_imagen", "tipo_celular", "sample_id"]], on="id_imagen")
    df["semilla"] = 0
print("Modo:", "REAL" if REAL else "DEMO — cifras no reportables", "|", df.modelo.unique().tolist())
df["densidad"] = df["n_gt"]

# %% [markdown]
# ## 1. Tabla de resultados (test sellado)
#
# Se reporta la media [IC 95 %] con bootstrap jerárquico (placa → imagen). Las semillas se promedian por
# imagen y los modelos se ordenan por el rango medio en la métrica primaria.

# %%
METRICS = ["ap", "ap50", "pq", "aji", "dice", "iou", "precision", "recall", "auc_roc", "hd95", "balanced_accuracy"]
METRICS = [m for m in METRICS if m in df]
tabla = results_table(df, METRICS, primary="ap")
tabla.to_csv(TAB / "seg_tabla_resultados.csv"); tabla

# %% [markdown]
# ## 2. Inferencia estadística (métrica primaria)
#
# Se aplica Friedman y luego Wilcoxon pareado por imagen con corrección de Holm. Se reportan d de Cohen
# pareado, δ de Cliff, relevancia práctica respecto a δ = 0.02, el diagrama de diferencia crítica y la
# verificación con un modelo mixto (intercepto aleatorio por `sample_id`).

# %%
rep = pairwise_report(df, "ap", delta=DELTA)
if "friedman_p" in rep:
    print(f"Friedman χ² = {rep['friedman_stat']:.2f}, p = {rep['friedman_p']:.2e}; CD Nemenyi = {rep['cd_nemenyi']:.2f}")
    plot_cd_diagram(rep["mean_ranks"], rep["cd_nemenyi"], title="Diferencia crítica — AP@[.5:.95]")
    plt.savefig(FIG / "seg_cd_diagram.png", dpi=120, bbox_inches="tight")
rep["pairwise"].round(4)

# %%
try:
    mm = mixed_model_check(df.groupby(["modelo", "id_imagen", "sample_id"], as_index=False).ap.mean(),
                           score="ap", model="modelo", group="sample_id")
    print(mm.summary().tables[1])
except ImportError:
    print("statsmodels no instalado: `pip install statsmodels` para el modelo mixto")

# %% [markdown]
# ## 3. Estratificación: tipo celular, densidad y tamaño (RQ1, H3)

# %%
df = add_terciles(df, "densidad", "tercil_densidad")
heat = stratified(df, "ap", "tipo_celular")
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
im = ax[0].imshow(heat.values, cmap="viridis"); ax[0].set_xticks(range(heat.shape[1])); ax[0].set_xticklabels(heat.columns, rotation=30, ha="right", fontsize=8)
ax[0].set_yticks(range(heat.shape[0])); ax[0].set_yticklabels(heat.index); plt.colorbar(im, ax=ax[0]); ax[0].set_title("AP: modelo × tipo celular")
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        ax[0].text(j, i, f"{heat.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
stratified(df, "ap", "tercil_densidad").loc[["bajo", "medio", "alto"]].plot(ax=ax[1], marker="o")
ax[1].set_title("AP por tercil de densidad"); plt.tight_layout(); fig.savefig(FIG / "seg_estratificado.png", dpi=120)
h3 = simpson_check(df, "ap")
print("H3:", "REFUTADA (orden idéntico en los tres tipos)" if h3.iloc[1:].orden.nunique() == 1 and h3.iloc[1:].orden.iloc[0] == h3.orden.iloc[0]
      else "APOYADA (el orden cambia al estratificar)")
h3

# %% [markdown]
# ## 4. Métricas de píxel vs instancia (RQ3, H2)

# %%
if "auc_roc" in df and df.auc_roc.notna().any():
    ag = ranking_agreement(df, "auc_roc", "ap")
    means = df.groupby("modelo").auc_roc.mean()
    sat = bool((means > 0.95).all() and (means.max() - means.min()) < 0.02)
    print(f"Spearman(rank AUC píxel, rank AP) = {ag['rho']:.2f} [{ag['ci_low']:.2f}, {ag['ci_high']:.2f}]; saturación AUC: {sat}")
    if np.isnan(ag["rho"]):
        print("H2: no evaluable (las medias de AUC no varían entre modelos; p. ej. mismo mapa de probabilidad)")
    else:
        print("H2:", "REFUTADA" if ag["rho"] > 0.9 or not (means > 0.95).all() else "APOYADA")
    ag_ba = ranking_agreement(df, "balanced_accuracy", "ap")
    print(f"Spearman(rank BA píxel, rank AP) = {ag_ba['rho']:.2f}")
# Curva AP vs umbral de IoU (columnas ap_t50 … ap_t95 de evaluate_image)
cols = [c for c in df.columns if c.startswith("ap_t")]
if cols:
    curve = df.groupby("modelo")[cols].mean().T; curve.index = [int(c[4:]) / 100 for c in cols]
    curve.plot(marker="o", title="AP vs umbral de IoU", xlabel="umbral IoU", ylabel="TP/(TP+FP+FN)", figsize=(6, 3.5))
    plt.savefig(FIG / "seg_ap_vs_iou.png", dpi=120, bbox_inches="tight")

# %% [markdown]
# ## 5. Taxonomía de errores
#
# Se clasifican automáticamente en omitida, falsa, fusión, división y desplazada (definiciones a priori en
# `seg_metrics.error_taxonomy`). Las tasas se expresan por objeto de referencia, se desglosan por tipo
# celular y se relacionan con la densidad.

# %%
ep = error_profile(df)
ep_d = error_profile(df, by="tercil_densidad")
ep.to_csv(TAB / "seg_errores.csv")
ax = ep_d.unstack("tercil_densidad").filter(like="fusion").plot(kind="bar", figsize=(7, 3), title="Fusiones por objeto según tercil de densidad")
plt.tight_layout()
ep.round(3)

# %% [markdown]
# ## 6. Conteo: utilidad práctica (Bland–Altman)

# %%
from project.seg_metrics import count_metrics
cm = pd.DataFrame({m: count_metrics(g.n_gt, g.n_pred) for m, g in df.groupby("modelo")}).T
fig, ax = plt.subplots(1, len(cm), figsize=(4 * len(cm), 3), squeeze=False)
for a, (m, g) in zip(ax[0], df.groupby("modelo")):
    mean, diff = (g.n_gt + g.n_pred) / 2, g.n_pred - g.n_gt
    a.scatter(mean, diff, s=8); a.axhline(diff.mean(), c="k"); a.axhline(cm.loc[m, "loa_low"], ls="--", c="r"); a.axhline(cm.loc[m, "loa_high"], ls="--", c="r")
    a.set(title=m, xlabel="media del conteo", ylabel="pred − ref")
plt.tight_layout()
cm.round(2)

# %% [markdown]
# ## 7. Distribuciones y casos cualitativos
#
# Se muestran un boxplot por métrica, una muestra aleatoria con semilla fija y los 5 mejores y 5 peores
# casos (los IDs se guardan para las figuras de error del notebook 03).

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
for a, m in zip(ax, ["ap", "pq", "dice"]):
    a.boxplot([g[m].dropna() for _, g in df.groupby("modelo")], showmeans=True)
    a.set_xticks(range(1, df.modelo.nunique() + 1)); a.set_xticklabels(list(df.groupby("modelo").groups))
    a.set_title(m); a.tick_params(axis="x", rotation=30, labelsize=7)
plt.tight_layout(); fig.savefig(FIG / "seg_boxplots.png", dpi=120)
casos = best_worst(df, "ap"); pd.Series(casos)

# %% [markdown]
# ## 8. Contraste de H1 (eficiencia de datos)
#
# Usa `reports/tables/seg_fractions.csv` (`scripts/seg_experiments.py fractions`). Se calcula Δ25 y Δ100 por
# imagen, (mejor preentrenado) − (U-Net++ desde cero), con IC bootstrap. Además se ajusta la ley de
# potencia error(n) = a·n^(−b) + c.

# %%
FR = TAB / "seg_fractions.csv"
if FR.exists():
    from project.stats import fit_power_law
    fr = pd.read_csv(FR)
    for m, g in fr.groupby("modelo"):
        c = g.groupby("fraccion").ap.mean()
        print(m, "ley de potencia:", {k: round(v, 3) for k, v in fit_power_law(c.index.to_numpy() * 100, 1 - c.to_numpy()).items()})
    piv = fr.groupby(["fraccion", "id_imagen", "modelo"]).ap.mean().unstack()
    scratch = [c for c in piv.columns if "U-Net++" in c][0]
    pre = [c for c in piv.columns if c != scratch]
    best_pre = piv.loc[0.25, pre].mean().idxmax()
    d25 = (piv.loc[0.25, best_pre] - piv.loc[0.25, scratch]).dropna(); d100 = (piv.loc[1.0, best_pre] - piv.loc[1.0, scratch]).dropna()
    print("Δ25 =", fmt_ci(*bootstrap_ci(d25.to_numpy())), "| Δ100 =", fmt_ci(*bootstrap_ci(d100.to_numpy())),
          "| Δ25−Δ100 =", fmt_ci(*bootstrap_ci((d25 - d100).dropna().to_numpy())))
else:
    print("Pendiente: ejecutar `python scripts/seg_experiments.py fractions` (requiere datos y GPU).")

# %% [markdown]
# ## 9. HPO: importancia de hiperparámetros y dependencia parcial

# %%
HP = TAB / "seg_hpo_trials.csv"
if HP.exists():
    tr = pd.read_csv(HP)
    pcols = [c for c in tr.columns if c.startswith("params_")]
    fig, ax = plt.subplots(1, len(pcols), figsize=(3 * len(pcols), 2.8))
    for a, c in zip(np.atleast_1d(ax), pcols):
        tr.groupby(c)["value"].mean().plot(ax=a, marker="o"); a.set_title(c.replace("params_", ""), fontsize=8)
    plt.tight_layout()
    imp = TAB / "seg_hpo_importance.json"
    print(imp.read_text() if imp.exists() else "")
else:
    print("Pendiente: `python scripts/seg_experiments.py hpo --trials 30`.")
