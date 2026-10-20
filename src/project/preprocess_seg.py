"""Preprocesamiento, objetivos por modelo y aumento de datos para segmentación.

Todas las transformaciones geométricas se aplican **idénticas** a imagen y
máscara; las máscaras se remuestrean con vecino más próximo.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi


# -------------------------------------------------------------- normalización
def norm_255(img: np.ndarray) -> np.ndarray:
    """Baseline: img / 255."""
    return img.astype(np.float32) / 255.0


def norm_percentile(img: np.ndarray, lo: float = 1, hi: float = 99) -> np.ndarray:
    """Normalización por percentiles **por imagen** (robusta a iluminación)."""
    a, b = np.percentile(img, [lo, hi])
    return np.clip((img.astype(np.float32) - a) / max(b - a, 1e-6), 0, 1)


class TrainStatsNormalizer:
    """Estandarización con media/DE **ajustadas solo con train**."""

    def __init__(self) -> None:
        self.mean: float | None = None
        self.std: float | None = None

    def fit(self, imgs: list[np.ndarray]) -> TrainStatsNormalizer:
        v = np.concatenate([i.ravel().astype(np.float64) for i in imgs])
        self.mean, self.std = float(v.mean()), float(v.std())
        return self

    def __call__(self, img: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None, "llame a fit() con train"
        return ((img.astype(np.float32) - self.mean) / max(self.std, 1e-6)).astype(np.float32)


# ----------------------------------------------------------------- resolución
def pad_to_multiple(x: np.ndarray, m: int = 32, mode: str = "reflect") -> tuple[np.ndarray, tuple[int, int]]:
    """Rellena alto/ancho hasta múltiplos de ``m`` (U-Net++ con 5 niveles: 32)."""
    h, w = x.shape[:2]
    ph, pw = (-h) % m, (-w) % m
    pad = [(0, ph), (0, pw)] + [(0, 0)] * (x.ndim - 2)
    return np.pad(x, pad, mode=mode), (h, w)  # use mode="constant" para mapas de etiquetas


def unpad(x: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    return x[: hw[0], : hw[1]]


def tile_starts(n: int, tile: int, overlap: int) -> list[int]:
    if n <= tile:
        return [0]
    step = tile - overlap
    s = list(range(0, n - tile, step)) + [n - tile]
    return sorted(set(s))


def tile_image(x: np.ndarray, tile: int = 256, overlap: int = 64) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    tiles, pos = [], []
    for y in tile_starts(x.shape[0], tile, overlap):
        for z in tile_starts(x.shape[1], tile, overlap):
            tiles.append(x[y:y + tile, z:z + tile]); pos.append((y, z))
    return tiles, pos


def stitch_tiles(tiles: list[np.ndarray], pos: list[tuple[int, int]], shape: tuple[int, ...]) -> np.ndarray:
    """Recompone predicciones continuas (probabilidades) promediando con pesos
    que decrecen hacia el borde del tile (evita costuras)."""
    out = np.zeros(shape, np.float64); wsum = np.zeros(shape[:2], np.float64)
    th, tw = tiles[0].shape[:2]
    wy = np.minimum(np.arange(th) + 1, np.arange(th)[::-1] + 1).astype(float)
    wx = np.minimum(np.arange(tw) + 1, np.arange(tw)[::-1] + 1).astype(float)
    wt = np.outer(wy, wx)
    for t, (y, z) in zip(tiles, pos):
        h, w = t.shape[:2]
        ww = wt[:h, :w]
        out[y:y + h, z:z + w] += t * (ww[..., None] if t.ndim == 3 else ww)
        wsum[y:y + h, z:z + w] += ww
    return out / (wsum[..., None] if out.ndim == 3 else wsum)


def resize_pair(img: np.ndarray, lbl: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    im = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    lb = cv2.resize(lbl.astype(np.int32).astype(np.float32), (size[1], size[0]),
                    interpolation=cv2.INTER_NEAREST).astype(np.int32)
    return im, lb


def instances_lost_by_resize(lbl: np.ndarray, size: tuple[int, int], min_pixels: int = 10) -> dict[str, float]:
    """¿Cuántas instancias desaparecen o quedan por debajo de ``min_pixels`` al reescalar?"""
    _, lb = resize_pair(np.zeros(lbl.shape, np.uint8), lbl, size)
    before = np.unique(lbl[lbl > 0])
    areas = np.bincount(lb.ravel(), minlength=int(lbl.max()) + 1)
    after = areas[before] if len(before) else np.array([])
    return dict(n=len(before), desaparecidas=int((after == 0).sum()),
                bajo_min=int(((after > 0) & (after < min_pixels)).sum()),
                frac_perdidas=float((after < min_pixels).mean()) if len(before) else 0.0)


# ------------------------------------------------------------------- objetivos
def three_class_target(lbl: np.ndarray, border_width: int = 2) -> np.ndarray:
    """0 = fondo, 1 = interior de célula, 2 = borde/contacto entre células.

    El borde incluye el contorno de cada instancia; las zonas de contacto quedan
    marcadas como clase 2, lo que permite separar células que se tocan con
    watershed (Caicedo et al., 2019)."""
    from skimage.segmentation import find_boundaries

    b = find_boundaries(lbl, mode="inner")
    if border_width > 1:
        b = ndi.binary_dilation(b, iterations=border_width - 1) & (lbl > 0)
    tgt = (lbl > 0).astype(np.uint8)
    tgt[b] = 2
    return tgt


def distance_target(lbl: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Mapa de distancia al borde por instancia (normalizado a [0, 1] por instancia)."""
    out = np.zeros(lbl.shape, np.float32)
    for sl_i, sl in enumerate(ndi.find_objects(lbl), start=1):
        if sl is None:
            continue
        m = lbl[sl] == sl_i
        d = ndi.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]
        if normalize and d.max() > 0:
            d = d / d.max()
        out[sl][m] = d[m]
    return out


def hv_target(lbl: np.ndarray) -> np.ndarray:
    """Mapas horizontal/vertical de Hover-Net: coordenadas normalizadas a [-1, 1]
    respecto al centro de masa de cada instancia. Salida (H, W, 2)."""
    out = np.zeros(lbl.shape + (2,), np.float32)
    for i, sl in enumerate(ndi.find_objects(lbl), start=1):
        if sl is None:
            continue
        m = lbl[sl] == i
        yy, xx = np.nonzero(m)
        cy, cx = yy.mean(), xx.mean()
        h = xx - cx; v = yy - cy
        h = h / max(np.abs(h).max(), 1e-6); v = v / max(np.abs(v).max(), 1e-6)
        sub = out[sl]
        sub[yy, xx, 0], sub[yy, xx, 1] = h, v
    return out


# ------------------------------------------------------------- augmentation
def augment_pair(img: np.ndarray, lbl: np.ndarray, rng: np.random.Generator, level: str = "geom+photo+elastic"
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Aumento de datos en NumPy (equivalente a la receta de albumentations).

    ``level`` en {"none", "geom", "geom+photo", "geom+photo+elastic"} (ablación).
    ``img`` float32 en [0, 1]; ``lbl`` mapa de etiquetas entero.
    """
    if level == "none":
        return img, lbl
    k = int(rng.integers(4))
    img, lbl = np.rot90(img, k), np.rot90(lbl, k)
    if rng.random() < 0.5:
        img, lbl = img[:, ::-1], lbl[:, ::-1]
    if rng.random() < 0.5:
        img, lbl = img[::-1], lbl[::-1]
    if rng.random() < 0.3:  # zoom aleatorio con recorte/relleno a la forma original
        z = rng.uniform(0.85, 1.25)
        h, w = img.shape[:2]
        im2 = ndi.zoom(img, z, order=1); lb2 = ndi.zoom(lbl, z, order=0)
        img, lbl = _center_fit(im2, (h, w)), _center_fit(lb2, (h, w))
    img = np.ascontiguousarray(img, dtype=np.float32); lbl = np.ascontiguousarray(lbl)
    if "elastic" in level and rng.random() < 0.3:
        img, lbl = elastic_pair(img, lbl, rng)
    if "photo" in level:
        img = img * rng.uniform(0.8, 1.2) + rng.uniform(-0.1, 0.1)          # contraste/brillo
        if rng.random() < 0.3:
            img = ndi.gaussian_filter(img, rng.uniform(0.3, 1.2))            # desenfoque
        if rng.random() < 0.3:
            img = img + rng.normal(0, rng.uniform(0.01, 0.05), img.shape)    # ruido
        img = np.clip(img, 0, 1).astype(np.float32)
    return img, lbl


def _center_fit(x: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    H, W = x.shape[:2]
    if H >= h and W >= w:
        y0, x0 = (H - h) // 2, (W - w) // 2
        return x[y0:y0 + h, x0:x0 + w]
    out = np.zeros((h, w) + x.shape[2:], x.dtype)
    y0, x0 = (h - H) // 2, (w - W) // 2
    out[max(y0, 0):max(y0, 0) + min(H, h), max(x0, 0):max(x0, 0) + min(W, w)] = \
        x[max(-y0, 0):max(-y0, 0) + min(H, h), max(-x0, 0):max(-x0, 0) + min(W, w)]
    return out


def elastic_pair(img: np.ndarray, lbl: np.ndarray, rng: np.random.Generator,
                 alpha: float = 30.0, sigma: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Deformación elástica (Simard et al., 2003); orden 1 en imagen y 0 en máscara."""
    h, w = img.shape[:2]
    dx = ndi.gaussian_filter(rng.uniform(-1, 1, (h, w)), sigma) * alpha
    dy = ndi.gaussian_filter(rng.uniform(-1, 1, (h, w)), sigma) * alpha
    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.array([yy + dy, xx + dx])
    im = ndi.map_coordinates(img, coords, order=1, mode="reflect").astype(np.float32)
    lb = ndi.map_coordinates(lbl, coords, order=0, mode="constant", cval=0).astype(lbl.dtype)
    return im, lb


def alignment_check_grid(shape: tuple[int, int] = (128, 128), step: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Imagen de cuadrícula y su máscara idéntica para verificar alineación tras augment."""
    g = np.zeros(shape, np.float32)
    g[::step, :] = 1; g[:, ::step] = 1
    return g, (g > 0).astype(np.int32)


def tta_predict(predict: Any, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TTA con las 8 simetrías del grupo diedral; devuelve media y varianza de las
    probabilidades re-alineadas (la varianza sirve como incertidumbre)."""
    outs = []
    for k in range(4):
        for flip in (False, True):
            x = np.rot90(img, k)
            x = x[:, ::-1] if flip else x
            p = predict(np.ascontiguousarray(x))
            p = p[:, ::-1] if flip else p
            outs.append(np.rot90(p, -k))
    o = np.stack(outs)
    return o.mean(0), o.var(0)


# ------------------------------------------------------------- perturbaciones
def perturb(img: np.ndarray, kind: str, severity: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Perturbaciones sintéticas de severidad 1–5 sobre imagen float [0, 1]
    (robustez, RQ4, y deriva de imágenes, RQ8)."""
    rng = rng or np.random.default_rng(0)
    s = int(severity)
    if s == 0:
        return img
    x = img.astype(np.float32)
    if kind == "noise":
        x = x + rng.normal(0, [0.02, 0.04, 0.06, 0.09, 0.12][s - 1], x.shape)
    elif kind == "blur":
        x = ndi.gaussian_filter(x, [0.5, 1.0, 1.5, 2.0, 3.0][s - 1])
    elif kind == "contrast":
        c = [0.8, 0.65, 0.5, 0.35, 0.2][s - 1]
        x = (x - x.mean()) * c + x.mean()
    elif kind == "brightness":
        x = x + [0.05, 0.1, 0.15, 0.2, 0.3][s - 1]
    elif kind == "illumination":
        h, w = x.shape[:2]
        x = x + [0.05, 0.1, 0.2, 0.3, 0.4][s - 1] * (np.linspace(-0.5, 0.5, w)[None, :])
    elif kind == "jpeg":
        import cv2

        q = [80, 60, 40, 20, 10][s - 1]
        ok, buf = cv2.imencode(".jpg", (np.clip(x, 0, 1) * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
        x = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    else:
        raise ValueError(kind)
    return np.clip(x, 0, 1).astype(np.float32)


PERTURBATIONS = ("noise", "blur", "contrast", "brightness", "illumination", "jpeg")
