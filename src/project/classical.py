"""Baseline clásico (posición 0): umbralización Otsu/adaptativa + watershed.

Sin aprendizaje: fija el piso de referencia. En contraste de fase las células
pueden ser más oscuras o más claras que el fondo según el foco, por lo que se
umbraliza la **desviación absoluta respecto al fondo** estimado (filtro de mediana
grande), lo que además reduce la sensibilidad a gradientes de iluminación.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .postprocess import remove_small


def estimate_background(x: np.ndarray, size: int, factor: int = 4) -> np.ndarray:
    """Fondo por filtro de mediana de ventana ``size`` (píxeles originales) calculado sobre la imagen
    reducida ``factor`` veces y reescalada: ~50x más rápido que la mediana a resolución completa
    (la mediana de 51x51 tarda ~9 s por imagen) y casi idéntico, pues el fondo varía lentamente."""
    import cv2

    h, w = x.shape
    if min(h, w) < 256:           # imágenes pequeñas: ya es rápido y la reducción perdería detalle
        return np.asarray(ndi.median_filter(x, size=size), dtype=np.float32)
    small = cv2.resize(x, (max(w // factor, 1), max(h // factor, 1)), interpolation=cv2.INTER_AREA)
    k = max(3, int(round(size / factor)) | 1)
    bg = ndi.median_filter(small, size=k, mode="reflect")
    return np.asarray(cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR), dtype=np.float32)


def classical_segment(img: np.ndarray, method: str = "otsu", bg_size: int = 31, smooth: float = 1.0,
                      min_distance: int = 4, min_size: int = 15, block_size: int = 51) -> np.ndarray:
    """Imagen (cualquier dtype) -> mapa de etiquetas de instancias."""
    from skimage.feature import peak_local_max
    from skimage.filters import threshold_local, threshold_otsu
    from skimage.segmentation import watershed

    x = img.astype(np.float32)
    x = (x - x.min()) / max(float(x.max() - x.min()), 1e-6)
    bg = estimate_background(x, bg_size)
    dev = ndi.gaussian_filter(np.abs(x - bg), smooth)
    if method == "otsu":
        fg = dev > threshold_otsu(dev)
    elif method == "adaptive":
        fg = dev > threshold_local(dev, block_size, offset=-0.01)
    else:
        raise ValueError(method)
    fg = ndi.binary_fill_holes(ndi.binary_opening(fg, iterations=1))
    dist = ndi.distance_transform_edt(fg)
    peaks = peak_local_max(dist, min_distance=min_distance, labels=ndi.label(fg)[0], exclude_border=False)
    markers = np.zeros(fg.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    lbl = watershed(-dist, markers, mask=fg)
    return remove_small(lbl.astype(np.int32), min_size)


def classical_prob_map(img: np.ndarray, bg_size: int = 31, smooth: float = 1.0) -> np.ndarray:
    """Mapa pseudo-probabilístico (desviación al fondo reescalada a [0, 1]) para
    poder calcular AUC/AP de píxel también en el baseline."""
    x = img.astype(np.float32)
    x = (x - x.min()) / max(float(x.max() - x.min()), 1e-6)
    dev = ndi.gaussian_filter(np.abs(x - estimate_background(x, bg_size)), smooth)
    return dev / max(float(dev.max()), 1e-6)
