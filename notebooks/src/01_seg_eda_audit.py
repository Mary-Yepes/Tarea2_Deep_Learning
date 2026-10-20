# %% [markdown]
# # 01 · Segmentación: análisis exploratorio y auditoría de datos
#
# **Objetivo.** Entender la estructura, los sesgos y las fuentes de error del dataset Sartorius **antes**
# de modelar, y derivar de ese análisis decisiones de diseño justificadas con cifras: resolución, anchors
# o diámetro, post-procesamiento y estratificación. Cada análisis se reporta **globalmente y por tipo
# celular**.
#
# **Datos.** Va en `data/raw/sartorius/` (`train.csv`, `train/*.png`, `train_semi_supervised/`,
# `LIVECell_dataset_2021/`) y se obtiene con
# `kaggle competitions download -c sartorius-cell-instance-segmentation`. Si los datos no están, el
# notebook corre en **modo demo** con imágenes sintéticas para verificar el código. **Las cifras del
# modo demo no describen el dataset real.**

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from project.config import CFG, set_global_seed
from project.data_seg import (build_image_table, check_integrity, density_contact, dhash, image_quality,
                              instance_table, label_map_for, load_annotations, make_synthetic_dataset,
                              near_duplicate_pairs, read_image)
from project.rle import instances_to_label_map, rle_encode
from project.splits import grouped_train_val_test_split, split_summary
from project.stats import bootstrap_ci, fmt_ci, icc_oneway

set_global_seed(CFG.seed)
ROOT = Path("..").resolve()
import os
RAW = Path(os.environ.get("SARTORIUS_DIR", ROOT / "data/raw/sartorius"))
REAL = (RAW / "train.csv").exists()
from project.config import report_dirs
TAB, FIG = report_dirs(ROOT, REAL)
print("Modo:", "REAL" if REAL else "DEMO (sintético) — cifras no reportables")

# %% [markdown]
# ## 1. Carga y dataset estructurado
#
# Comprobamos los canales y la profundidad de bits reales, decodificamos RLE a mapa de etiquetas y
# construimos la tabla por imagen y la tabla por instancia.

# %%
if REAL:
    ann = load_annotations(RAW / "train.csv")
    imgs_tab = build_image_table(ann, RAW / "train")
    sample = read_image(imgs_tab.ruta_imagen.iloc[0])
    print("Columnas reales:", list(ann.columns))
    print(f"Forma/dtype de una imagen: {sample.shape} {sample.dtype}; rango [{sample.min()}, {sample.max()}]")
    def get(i):
        r = imgs_tab.iloc[i]
        return read_image(r.ruta_imagen), label_map_for(ann, r.id_imagen)
else:
    _imgs, _lbls, imgs_tab = make_synthetic_dataset(12, shape=(260, 352), seed=CFG.seed)
    imgs_tab["alto"], imgs_tab["ancho"] = 260, 352
    ann = pd.DataFrame([dict(id=r.id_imagen, annotation=rle_encode(lb == k), width=352, height=260,
                             cell_type=r.tipo_celular, sample_id=r.sample_id)
                        for (_, r), lb in zip(imgs_tab.iterrows(), _lbls) for k in range(1, lb.max() + 1)])
    def get(i):
        return _imgs[i], _lbls[i]
print(f"{len(imgs_tab)} imágenes, {len(ann)} instancias, {imgs_tab.sample_id.nunique()} grupos sample_id")
imgs_tab.head()

# %% [markdown]
# ### Pruebas de integridad
#
# Revisamos ausencia de nulos, rangos válidos, que los RLE queden dentro de los límites, IDs únicos y
# metadatos consistentes por imagen. En CI se ejecuta además el esquema `pandera`
# (`tests/test_data_schema.py`).

# %%
integ = check_integrity(ann)
try:
    from project.data_seg import pandera_schema
    pandera_schema().validate(ann); integ["pandera"] = "OK"
except ImportError:
    integ["pandera"] = "pandera no instalado"
except Exception as e:  # se registra la discrepancia en la datasheet
    integ["pandera"] = f"FALLA: {str(e)[:200]}"
integ

# %%
inst, extra = [], []
for i in range(len(imgs_tab)):
    img, lbl = get(i)
    r = imgs_tab.iloc[i]
    t = instance_table(lbl, r.id_imagen); t["tipo_celular"] = r.tipo_celular; inst.append(t)
    lbl_o, ov = instances_to_label_map(ann[ann.id == r.id_imagen].annotation.tolist(), lbl.shape, return_overlap=True)
    extra.append(dict(id_imagen=r.id_imagen, pix_solapados=ov, **image_quality(img), **density_contact(lbl), dhash=dhash(img)))
inst = pd.concat(inst, ignore_index=True)
imgs_tab = imgs_tab.merge(pd.DataFrame(extra), on="id_imagen")
inst.to_csv(TAB / "seg_instancias.csv", index=False); imgs_tab.to_csv(TAB / "seg_imagenes.csv", index=False)
inst.describe().T.round(2)

# %% [markdown]
# ## 2. Análisis obligatorios (globales y por tipo celular)

# %%
types = sorted(imgs_tab.tipo_celular.unique())
fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
ax[0, 0].scatter(imgs_tab.ancho, imgs_tab.alto, c=pd.Categorical(imgs_tab.tipo_celular).codes, cmap="tab10")
ax[0, 0].set(title="Tamaño de imagen", xlabel="ancho", ylabel="alto")
for t in types:
    ax[0, 1].hist(imgs_tab[imgs_tab.tipo_celular == t].n_instancias, bins=20, alpha=0.5, label=t)
ax[0, 1].set(title="Células por imagen", xlabel="n instancias"); ax[0, 1].legend()
ax[0, 2].violinplot([imgs_tab[imgs_tab.tipo_celular == t].n_instancias for t in types], showmedians=True)
ax[0, 2].set_xticks(range(1, len(types) + 1)); ax[0, 2].set_xticklabels(types); ax[0, 2].set_title("Conteo por tipo")
bins = np.logspace(np.log10(max(inst.area.min(), 1)), np.log10(inst.area.max()), 40)
for t in types:
    ax[1, 0].hist(inst[inst.tipo_celular == t].area, bins=bins, alpha=0.5, label=t)
ax[1, 0].set(xscale="log", title="Área de células (escala log)", xlabel="píxeles"); ax[1, 0].legend()
ax[1, 1].boxplot([inst[inst.tipo_celular == t].area for t in types]); ax[1, 1].set(yscale="log", title="Área por tipo")
ax[1, 1].set_xticks(range(1, len(types) + 1)); ax[1, 1].set_xticklabels(types)
for t in types:
    v = imgs_tab[imgs_tab.tipo_celular == t]
    ax[1, 2].scatter(v["mean"], v["contrast_rms"], label=t, s=12)
ax[1, 2].set(title="Intensidad media vs contraste RMS", xlabel="media", ylabel="contraste"); ax[1, 2].legend()
plt.tight_layout(); fig.savefig(FIG / "seg_eda_overview.png", dpi=110)

# %% [markdown]
# ### Ajuste de la distribución de áreas (log-normal vs gamma)
#
# Se ajusta por máxima verosimilitud y se evalúa la bondad de ajuste con KS. Con miles de instancias, KS
# rechaza casi cualquier modelo, así que se reporta también el AIC para comparar las dos familias.

# %%
rows = []
for t in ["global"] + types:
    a = inst.area.to_numpy() if t == "global" else inst[inst.tipo_celular == t].area.to_numpy()
    for name, dist in (("lognorm", stats.lognorm), ("gamma", stats.gamma)):
        p = dist.fit(a, floc=0)
        ll = dist.logpdf(a, *p).sum()
        rows.append(dict(tipo=t, dist=name, ks=stats.kstest(a, dist.cdf, args=p).statistic, aic=2 * (len(p) - 1) - 2 * ll))
fit = pd.DataFrame(rows); fit["mejor_aic"] = fit.groupby("tipo").aic.transform("min") == fit.aic
fit.round(3)

# %% [markdown]
# ### Cualitativo: muestra aleatoria con semilla fija y casos extremos
#
# Se muestra una muestra aleatoria (semilla 42) y los casos extremos: más y menos células, y peor SNR.

# %%
rng = np.random.default_rng(CFG.seed)
pick = list(rng.choice(len(imgs_tab), 3, replace=False))
pick += [int(imgs_tab.n_instancias.idxmax()), int(imgs_tab.n_instancias.idxmin()), int(imgs_tab.snr.idxmin())]
names = ["aleatoria", "aleatoria", "aleatoria", "más células", "menos células", "menor SNR"]
fig, ax = plt.subplots(2, 6, figsize=(18, 5.5))
from skimage.segmentation import find_boundaries
for j, (i, nm) in enumerate(zip(pick, names)):
    img, lbl = get(i)
    ax[0, j].imshow(img, cmap="gray"); ax[0, j].set_title(f"{nm}\n{imgs_tab.tipo_celular.iloc[i]}", fontsize=9)
    ov = np.dstack([img / max(img.max(), 1)] * 3); ov[find_boundaries(lbl)] = [1, 0.2, 0.2]
    ax[1, j].imshow(ov); ax[1, j].set_title(f"{lbl.max()} instancias", fontsize=9)
for a in ax.ravel(): a.axis("off")
plt.tight_layout(); fig.savefig(FIG / "seg_eda_cualitativo.png", dpi=110)

# %% [markdown]
# **Validación visual del RLE.** Los contornos rojos deben coincidir con las células. Si aparecen
# desplazados o transpuestos, el orden de aplanado es incorrecto (`order="F"`) y debe corregirse antes de
# continuar.
#
# ## 3. Análisis adicionales (criterio de entrada al modelado)
#
# ### 3.1 Densidad y contacto

# %%
dens = imgs_tab.groupby("tipo_celular")[["cobertura", "frac_celulas_en_contacto", "pares_contacto"]].agg(["mean", "median"])
print(dens.round(3))
r = stats.spearmanr(imgs_tab.cobertura, imgs_tab.frac_celulas_en_contacto)
print(f"Spearman(cobertura, fracción en contacto) = {r.statistic:.2f} (p = {r.pvalue:.1e}): "
      "las imágenes densas son las de mayor riesgo de fusiones en el post-procesamiento.")

# %% [markdown]
# ### 3.2 Tamaños y cola pesada respecto al stride y al campo receptivo
#
# Se calcula qué fracción de las instancias tiene un diámetro equivalente menor que el stride de cada
# nivel del backbone (4, 8, 16, 32 px), y cuántas desaparecerían o quedarían por debajo de 10 px al
# reescalar a 256 o 512.

# %%
from project.preprocess_seg import instances_lost_by_resize
q = inst.groupby("tipo_celular").area.quantile([0.01, 0.05, 0.5, 0.95, 0.99]).unstack().round(0)
print("Cuantiles de área (px):\n", q)
for s in (4, 8, 16, 32):
    print(f"  % instancias con diámetro equivalente < stride {s:>2}: {100 * (inst.diametro_eq < s).mean():5.1f} %")
loss = []
for i in range(min(len(imgs_tab), 60)):
    img, lbl = get(i)
    for size in (256, 512):
        h, w = lbl.shape; sc = size / max(h, w)
        loss.append(dict(tamano=size, **instances_lost_by_resize(lbl, (max(int(h * sc), 1), max(int(w * sc), 1)), 10)))
pd.DataFrame(loss).groupby("tamano")[["frac_perdidas"]].mean()

# %% [markdown]
# ### 3.3 Calidad de imagen
#
# Criterios explícitos y **declarados antes** de mirar los resultados:
# * Desenfoque: varianza del Laplaciano < percentil 5 global.
# * Iluminación no uniforme: rango del fondo ajustado > 0.15 de la media.
# * Saturación: > 1 % de píxeles en 0 o en el máximo.

# %%
crit = pd.DataFrame({
    "desenfoque": imgs_tab.laplacian_var < imgs_tab.laplacian_var.quantile(0.05),
    "iluminacion": imgs_tab.illum_range > 0.15,
    "saturacion": (imgs_tab.sat_low + imgs_tab.sat_high) > 0.01,
})
crit["alguno"] = crit.any(axis=1)
out = {c: fmt_ci(*bootstrap_ci(crit[c].astype(float).to_numpy(), n_boot=10_000)) for c in crit}
print("Proporción de imágenes afectadas [IC 95 %]:", out)
crit.groupby(imgs_tab.tipo_celular).mean().round(3)

# %% [markdown]
# ### 3.4 Efectos de lote (batch effects)
#
# Proyectamos embeddings de las imágenes coloreando por tipo celular y por `sample_id`. En modo real
# conviene usar un codificador preentrenado (p. ej. ResNet50/ImageNet, `include_top=False` + *global
# pooling*). Por defecto, y para no depender de TF, se usan descriptores de textura e intensidad
# (histograma de intensidad, LBP y estadísticos). Para cuantificar el efecto, un **clasificador de
# dominio** predice la placa (AUC macro OvR con validación cruzada) y una **prueba de permutación**
# compara ese AUC contra placas barajadas.

# %%
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def descriptor(img):
    x = img.astype(np.float32); x = (x - x.min()) / max(x.max() - x.min(), 1e-6)
    hist = np.histogram(x, 32, (0, 1), density=True)[0]
    lbp = np.histogram(local_binary_pattern((x * 255).astype(np.uint8), 8, 1, "uniform"), 10, (0, 10), density=True)[0]
    return np.r_[hist, lbp, x.mean(), x.std(), stats.skew(x.ravel()), stats.kurtosis(x.ravel())]

E = StandardScaler().fit_transform(np.array([descriptor(get(i)[0]) for i in range(len(imgs_tab))]))
Z = PCA(2, random_state=0).fit_transform(E)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for t in types:
    m = (imgs_tab.tipo_celular == t).to_numpy(); ax[0].scatter(Z[m, 0], Z[m, 1], s=12, label=t)
ax[0].legend(); ax[0].set_title("PCA de descriptores — tipo celular")
ax[1].scatter(Z[:, 0], Z[:, 1], c=pd.Categorical(imgs_tab.sample_id).codes, cmap="tab20", s=12); ax[1].set_title("— sample_id")
plt.tight_layout()

def domain_auc(groups, seed=0):
    g = pd.Categorical(groups).codes
    keep = np.isin(g, np.where(np.bincount(g) >= 3)[0])
    gk = pd.Categorical(g[keep]).codes
    if len(np.unique(gk)) < 2:
        return np.nan
    p = cross_val_predict(LogisticRegression(max_iter=2000), E[keep], gk, cv=StratifiedKFold(3, shuffle=True, random_state=seed), method="predict_proba")
    return roc_auc_score(gk, p, multi_class="ovr") if len(np.unique(gk)) > 2 else roc_auc_score(gk, p[:, 1])

for col in ("tipo_celular", "sample_id"):
    if imgs_tab[col].nunique() < 2:
        continue
    obs = domain_auc(imgs_tab[col].to_numpy())
    perm = [domain_auc(np.random.default_rng(s).permutation(imgs_tab[col].to_numpy()), s) for s in range(30)]
    print(f"AUC clasificador de dominio ({col}) = {obs:.3f}; nula por permutación: media {np.mean(perm):.3f}, "
          f"p ≈ {(1 + sum(p >= obs for p in perm)) / 31:.3f}")
# Dentro de un mismo tipo celular: ¿son distinguibles las placas? (efecto de lote puro)
for t in types:
    sub = (imgs_tab.tipo_celular == t).to_numpy()
    if imgs_tab[sub].sample_id.nunique() >= 2:
        Eb = E; E = E[sub]
        print(f"  [{t}] AUC placa dentro del tipo = {domain_auc(imgs_tab[sub].sample_id.to_numpy()):.3f}")
        E = Eb

# %% [markdown]
# ### 3.5 Calidad de anotación
#
# Se revisan instancias con área extrema (< 1 % del cuantil), máscaras fragmentadas (más de una
# componente conexa en el RLE original) y solapes. Para la **cota de ruido de anotación** se revisa a
# mano una muestra aleatoria de 50 imágenes (semilla 42), contando células visibles sin anotar y máscaras
# erróneas. El resultado se registra en `reports/tables/seg_revision_manual.csv`, con columnas
# `id_imagen, celulas_visibles, sin_anotar, mascaras_erroneas`.

# %%
from scipy import ndimage as ndi
frag = []
for rle, w, h in zip(ann.annotation.head(5000), ann.width.head(5000), ann.height.head(5000)):
    from project.rle import rle_decode
    frag.append(ndi.label(rle_decode(rle, (h, w)))[1] > 1)
print(f"Instancias fragmentadas (muestra de {len(frag)}): {100 * np.mean(frag):.2f} %")
print(f"Instancias diminutas (< 30 px): {100 * (inst.area < 30).mean():.2f} %")
print(f"Imágenes con píxeles solapados entre instancias: {(imgs_tab.pix_solapados > 0).mean():.2%}")
manual = TAB / "seg_revision_manual.csv"
if manual.exists():
    mr = pd.read_csv(manual)
    rate = (mr.sin_anotar + mr.mascaras_erroneas) / mr.celulas_visibles
    print("Tasa de error de anotación estimada:", fmt_ci(*bootstrap_ci(rate.to_numpy())))
else:
    print("Pendiente: revisión manual de 50 imágenes →", manual.name)
    pd.DataFrame({"id_imagen": imgs_tab.sample(min(50, len(imgs_tab)), random_state=CFG.seed).id_imagen,
                  "celulas_visibles": np.nan, "sin_anotar": np.nan, "mascaras_erroneas": np.nan}).to_csv(TAB / "seg_revision_manual_PLANTILLA.csv", index=False)

# %% [markdown]
# ### 3.6 Auditoría de fuga: imágenes casi idénticas entre placas y particiones por grupo

# %%
dups = near_duplicate_pairs(imgs_tab.set_index("id_imagen").dhash, imgs_tab.set_index("id_imagen").sample_id, max_dist=4)
print(f"Pares casi idénticos entre sample_id distintos (Hamming dHash ≤ 4): {len(dups)}")
split = grouped_train_val_test_split(imgs_tab, "sample_id", "tipo_celular", CFG.val_frac, CFG.test_frac, CFG.seed)
imgs_tab["split"] = split.values
if REAL:
    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    imgs_tab[["id_imagen", "sample_id", "tipo_celular", "split"]].to_csv(ROOT / "data/processed/seg_splits.csv", index=False)
if len(dups):
    s = imgs_tab.set_index("id_imagen").split
    print("Pares casi idénticos que cruzan particiones:", int((s[dups.a].values != s[dups.b].values).sum()))
split_summary(imgs_tab, split, "tipo_celular", "sample_id").round(3)

# %% [markdown]
# ### 3.7 Desbalance y su efecto esperado sobre la métrica global
#
# AP@[0.5:0.95] se promedia **por imagen**. El peso de cada tipo celular en la métrica global es su
# proporción de imágenes, no de células. Si un tipo tiene AP sistemáticamente menor (por ejemplo, las
# neuronas SH-SY5Y, alargadas y densas), su proporción en test determina buena parte de la cifra global.
# Por eso se reporta siempre estratificado.

# %%
counts = imgs_tab.groupby("tipo_celular").agg(imagenes=("id_imagen", "size"), celulas=("n_instancias", "sum"))
counts["prop_imagenes"] = counts.imagenes / counts.imagenes.sum(); counts["prop_celulas"] = counts.celulas / counts.celulas.sum()
gini = lambda x: np.abs(np.subtract.outer(x, x)).mean() / (2 * x.mean())
print(f"CV del conteo por imagen = {imgs_tab.n_instancias.std() / imgs_tab.n_instancias.mean():.2f}; Gini = {gini(imgs_tab.n_instancias.to_numpy()):.2f}")
counts.round(3)

# %% [markdown]
# ## 4. Interpretación cuantitativa obligatoria
#
# Las respuestas se calculan abajo con cifras e intervalos. Las frases en *cursiva* son plantillas que se
# completan al ejecutar con los datos reales.

# %%
icc_int = icc_oneway(imgs_tab["mean"].to_numpy(), imgs_tab.sample_id.to_numpy())
icc_type = icc_oneway(imgs_tab["mean"].to_numpy(), imgs_tab.tipo_celular.to_numpy())
cv_plate = imgs_tab.groupby("sample_id")["mean"].mean().std() / imgs_tab["mean"].mean()
kw = stats.kruskal(*[imgs_tab[imgs_tab.tipo_celular == t]["mean"] for t in types])
print(f"¿Homogéneas?  ICC(intensidad | placa) = {icc_int:.2f}; ICC(| tipo) = {icc_type:.2f}; "
      f"CV entre placas = {cv_plate:.3f}; Kruskal–Wallis por tipo p = {kw.pvalue:.1e}")
print(f"¿Desbalance de conteo?  Gini = {gini(imgs_tab.n_instancias.to_numpy()):.2f}; proporción de imágenes por tipo = "
      f"{counts.prop_imagenes.round(2).to_dict()}")
print(f"¿Pequeñas o grandes?  mediana de área = {inst.area.median():.0f} px, p5 = {inst.area.quantile(.05):.0f} px; "
      f"{100 * (inst.diametro_eq < 16).mean():.1f} % con diámetro < stride 16")
print(f"¿Ruido/artefactos?  imágenes con algún criterio = {fmt_ci(*bootstrap_ci(crit.alguno.astype(float).to_numpy()))}")

# %% [markdown]
# ### Implicaciones de diseño derivadas del EDA
#
# 1. **Resolución.** Si un porcentaje no despreciable de instancias queda por debajo de 10 px al reescalar
#    a 256 (tabla 3.2), se descarta 256 como resolución principal. Se usa la nativa con padding a múltiplos
#    de 32, o *tiling* con solape de 64 px.
# 2. **Anchors / diámetro.** Los tamaños de anchor de Mask R-CNN se fijan en los cuantiles 5–95 del
#    diámetro equivalente, y el diámetro de Cellpose se estima **por tipo celular** con la mediana de 3.2.
# 3. **Máximo de detecciones.** `max_per_img` ≈ 1.2 × p99 del conteo por imagen. Con un valor por defecto
#    de 100 se truncan las imágenes densas.
# 4. **Separación de células en contacto.** La alta fracción de células que se tocan en los tipos densos
#    justifica el objetivo de 3 clases (borde-contacto) y la búsqueda de `h` de h-maxima en validación.
# 5. **Normalización.** Si la intensidad depende de la placa (ICC alto) o hay gradiente de iluminación, se
#    prefiere la normalización por percentiles por imagen frente a `/255`, y se verifica en la ablación.
# 6. **Particiones.** Cuando las placas son distinguibles (AUC del clasificador de dominio > 0.5 dentro de
#    un tipo), la partición por `sample_id` es obligatoria. Toda métrica se acompaña de un modelo mixto con
#    intercepto aleatorio por placa.
# 7. **Estratificación del reporte.** El desbalance por tipo y por densidad obliga a reportar por tipo
#    celular y por terciles de densidad y tamaño (H3).
