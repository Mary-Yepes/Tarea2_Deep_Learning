"""Datos de segmentación (Sartorius Cell Instance Segmentation): carga, tablas,
integridad, calidad de imagen, densidad/contacto, hash perceptual y un generador
sintético para pruebas de humo y CI.

Esquema esperado de ``train.csv`` (verificar contra los archivos reales y
registrar discrepancias en la *datasheet*): ``id, annotation, width, height,
cell_type, plate_time, sample_date, sample_id, elapsed_timedelta``; una fila por
instancia; imágenes PNG en ``train/<id>.png``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .rle import instances_to_label_map, rle_area

EXPECTED_COLUMNS = ["id", "annotation", "width", "height", "cell_type", "plate_time",
                    "sample_date", "sample_id", "elapsed_timedelta"]


# ------------------------------------------------------------------ carga
def read_image(path: str | Path) -> np.ndarray:
    """Lee una imagen conservando profundidad de bits; si es RGB con canales
    idénticos (gris replicado) devuelve un solo canal y lo indica en el EDA."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3 and np.all(img[..., 0] == img[..., 1]) and np.all(img[..., 1] == img[..., 2]):
        img = img[..., 0]
    return img


def load_annotations(csv_path: str | Path) -> pd.DataFrame:
    ann = pd.read_csv(csv_path)
    missing = set(EXPECTED_COLUMNS) - set(ann.columns)
    if missing:
        print(f"[datasheet] columnas esperadas ausentes: {sorted(missing)}")
    return ann


def build_image_table(ann: pd.DataFrame, img_dir: str | Path | None = None) -> pd.DataFrame:
    """Una fila por imagen con metadatos y conteo de instancias."""
    agg = {c: "first" for c in ["width", "height", "cell_type", "plate_time", "sample_date",
                                 "sample_id", "elapsed_timedelta"] if c in ann}
    df = ann.groupby("id").agg(agg)
    df["n_instancias"] = ann.groupby("id").size()
    df = df.rename(columns={"width": "ancho", "height": "alto", "cell_type": "tipo_celular"})
    if img_dir is not None:
        df["ruta_imagen"] = [str(Path(img_dir) / f"{i}.png") for i in df.index]
    return df.reset_index().rename(columns={"id": "id_imagen"})


def label_map_for(ann: pd.DataFrame, image_id: str) -> np.ndarray:
    a = ann[ann["id"] == image_id]
    h, w = int(a["height"].iloc[0]), int(a["width"].iloc[0])
    return instances_to_label_map(a["annotation"].tolist(), (h, w))  # type: ignore[return-value]


def instance_table(lbl: np.ndarray, image_id: str = "") -> pd.DataFrame:
    """Morfología por instancia: área, perímetro, excentricidad, solidez."""
    from skimage.measure import regionprops_table

    props = regionprops_table(lbl, properties=("label", "area", "perimeter", "eccentricity",
                                               "solidity", "equivalent_diameter_area"))
    t = pd.DataFrame(props).rename(columns={"area": "area", "perimeter": "perimetro",
                                            "eccentricity": "excentricidad", "solidity": "solidez",
                                            "equivalent_diameter_area": "diametro_eq"})
    t.insert(0, "id_imagen", image_id)
    return t


# ------------------------------------------------------------ integridad
def check_integrity(ann: pd.DataFrame) -> dict[str, Any]:
    """Pruebas de integridad sin dependencias (se complementan con pandera)."""
    rep: dict[str, Any] = {}
    rep["n_filas"] = len(ann)
    rep["nulos"] = ann.isna().sum().to_dict()
    rep["dims_validas"] = bool(((ann["width"] > 0) & (ann["height"] > 0)).all())
    ends = []
    for r, w, h in zip(ann["annotation"], ann["width"], ann["height"]):
        s = np.asarray(str(r).split(), dtype=np.int64)
        ends.append(bool((s[0::2] - 1 + s[1::2] <= w * h).all()) if s.size else True)
    rep["rle_en_limites"] = bool(np.all(ends))
    rep["areas_nulas"] = int((ann["annotation"].map(rle_area) == 0).sum())
    rep["anotaciones_duplicadas"] = int(ann.duplicated(["id", "annotation"]).sum())
    per_img = ann.groupby("id")[["width", "height", "cell_type"]].nunique()
    rep["metadatos_consistentes_por_imagen"] = bool((per_img <= 1).all().all())
    return rep


def pandera_schema() -> Any:
    """Esquema pandera de ``train.csv`` (usado en CI: etapa «pruebas de datos»)."""
    try:
        import pandera.pandas as pa  # pandera >= 0.24
    except ImportError:
        import pandera as pa  # versiones anteriores

    return pa.DataFrameSchema({
        "id": pa.Column(str, nullable=False),
        "annotation": pa.Column(str, pa.Check.str_matches(r"^\d+( \d+)*$"), nullable=False),
        "width": pa.Column(int, pa.Check.gt(0)),
        "height": pa.Column(int, pa.Check.gt(0)),
        "cell_type": pa.Column(str, pa.Check.isin(["shsy5y", "astro", "cort"])),
        "sample_id": pa.Column(str, nullable=False),
    }, strict=False)


# ----------------------------------------------------------- calidad
def image_quality(img: np.ndarray) -> dict[str, float]:
    """Enfoque (varianza del Laplaciano), gradiente de iluminación (ajuste
    polinómico de grado 2 al fondo suavizado), saturación, contraste y SNR."""
    x = img.astype(np.float64)
    lap = ndi.laplace(x)
    h, w = x.shape
    small = ndi.zoom(ndi.gaussian_filter(x, 15), 64 / max(h, w), order=1)
    yy, xx = np.mgrid[0:small.shape[0], 0:small.shape[1]] / 64.0
    A = np.stack([np.ones_like(xx), xx, yy, xx * yy, xx ** 2, yy ** 2], -1).reshape(-1, 6)
    coef, *_ = np.linalg.lstsq(A, small.ravel(), rcond=None)
    bg = A @ coef
    maxv = 255.0 if img.dtype == np.uint8 else float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else 1.0
    noise = float(np.std(x - ndi.median_filter(x, 3)))
    return dict(
        laplacian_var=float(lap.var()),
        illum_range=float((bg.max() - bg.min()) / max(x.mean(), 1e-9)),
        sat_low=float((x <= 0).mean()), sat_high=float((x >= maxv).mean()),
        contrast_rms=float(x.std() / max(x.mean(), 1e-9)),
        p1=float(np.percentile(x, 1)), p99=float(np.percentile(x, 99)), mean=float(x.mean()),
        snr=float(x.mean() / max(noise, 1e-9)),
    )


def density_contact(lbl: np.ndarray) -> dict[str, float]:
    """Fracción cubierta y pares de células que se tocan (adyacencia 8-conexa)."""
    cover = float((lbl > 0).mean())
    n = int(lbl.max())
    if n == 0:
        return dict(cobertura=cover, pares_contacto=0, frac_celulas_en_contacto=0.0)
    pairs: set[tuple[int, int]] = set()
    h, w = lbl.shape
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = lbl[0:h - dy, max(0, -dx):w - max(0, dx)]          # píxel (y, x)
        b = lbl[dy:h, max(0, dx):w - max(0, -dx)]              # vecino (y+dy, x+dx)
        m = (a > 0) & (b > 0) & (a != b)
        for u, v in zip(a[m], b[m]):
            pairs.add((min(u, v), max(u, v)))
    touching = {i for p in pairs for i in p}
    return dict(cobertura=cover, pares_contacto=len(pairs), frac_celulas_en_contacto=len(touching) / n)


def dhash(img: np.ndarray, size: int = 8) -> int:
    """Hash perceptual por diferencias (dHash) de 64 bits."""
    import cv2

    small = cv2.resize(img.astype(np.float32), (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def near_duplicate_pairs(hashes: pd.Series, groups: pd.Series, max_dist: int = 4) -> pd.DataFrame:
    """Pares de imágenes casi idénticas **entre grupos distintos** (auditoría de fuga)."""
    ids, hs, gs = hashes.index.to_numpy(), hashes.to_numpy(), groups.reindex(hashes.index).to_numpy()
    rows = []
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            if gs[i] != gs[j]:
                d = hamming(int(hs[i]), int(hs[j]))
                if d <= max_dist:
                    rows.append((ids[i], ids[j], gs[i], gs[j], d))
    return pd.DataFrame(rows, columns=["a", "b", "grupo_a", "grupo_b", "hamming"])


# --------------------------------------------------------- sintético
CELL_TYPES_SYN: dict[str, dict[str, Any]] = {
    # radio medio (px), excentricidad máx., número de células por imagen
    "shsy5y": dict(r=(5, 9), ecc=0.85, n=(60, 140)),
    "astro": dict(r=(10, 18), ecc=0.6, n=(15, 40)),
    "cort": dict(r=(6, 9), ecc=0.4, n=(20, 60)),
}


def make_synthetic_image(shape: tuple[int, int] = (128, 160), cell_type: str = "cort",
                         seed: int = 0, noise: float = 0.04, blur: float = 0.0,
                         illum: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Imagen sintética tipo contraste de fase (interior oscuro + halo brillante) y
    su mapa de etiquetas. **Solo** para pruebas de humo, CI y simulaciones; nunca
    para reportar resultados del proyecto."""
    rng = np.random.default_rng(seed)
    p = CELL_TYPES_SYN[cell_type]
    h, w = shape
    scale = (h * w) / (520 * 704)
    n = max(1, int(rng.integers(*p["n"]) * max(scale * 8, 0.2)))
    lbl = np.zeros(shape, np.int32)
    yy, xx = np.mgrid[0:h, 0:w]
    k = 0
    for _ in range(n * 3):
        if k >= n:
            break
        r = rng.uniform(*p["r"]); e = rng.uniform(0, p["ecc"])
        a, b = r, r * np.sqrt(1 - e ** 2); th = rng.uniform(0, np.pi)
        cy, cx = rng.uniform(r, h - r), rng.uniform(r, w - r)
        X, Y = xx - cx, yy - cy
        u = (X * np.cos(th) + Y * np.sin(th)) / a; v = (-X * np.sin(th) + Y * np.cos(th)) / b
        m = (u ** 2 + v ** 2) <= 1
        if (lbl[m] > 0).mean() > 0.15:
            continue
        k += 1
        lbl[m & (lbl == 0)] = k
    fg = lbl > 0
    halo = ndi.binary_dilation(fg, iterations=2) & ~fg
    img = 0.5 + 0.0 * xx
    img = img - 0.18 * fg + 0.25 * halo
    img = ndi.gaussian_filter(img, 0.8 + blur)
    if illum:
        img = img + illum * (xx / w - 0.5)
    img = img + rng.normal(0, noise, shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8), lbl


def make_synthetic_dataset(n_per_type: int = 6, shape: tuple[int, int] = (128, 160), seed: int = 0
                           ) -> tuple[list[np.ndarray], list[np.ndarray], pd.DataFrame]:
    imgs, lbls, rows = [], [], []
    for t_i, t in enumerate(CELL_TYPES_SYN):
        for j in range(n_per_type):
            s = seed * 10_000 + t_i * 1000 + j
            im, lb = make_synthetic_image(shape, t, seed=s)
            imgs.append(im); lbls.append(lb)
            rows.append(dict(id_imagen=f"syn_{t}_{j:03d}", tipo_celular=t,
                             sample_id=f"{t}_plate{j // 2}", n_instancias=int(lb.max())))
    return imgs, lbls, pd.DataFrame(rows)
