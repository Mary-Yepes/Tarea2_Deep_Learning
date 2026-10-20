"""Post-procesamiento a mapa de etiquetas de instancias (interfaz común de evaluación).

Los parámetros (umbral de primer plano, ``h`` de h-maxima, tamaño mínimo) se
ajustan **solo en validación** (``tune_watershed``).
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
from scipy import ndimage as ndi


def remove_small(lbl: np.ndarray, min_size: int = 15) -> np.ndarray:
    from .rle import relabel_sequential

    if lbl.max() == 0:
        return lbl
    areas = np.bincount(lbl.ravel())
    small = np.where(areas < min_size)[0]
    out = lbl.copy()
    out[np.isin(out, small[small > 0])] = 0
    return relabel_sequential(out)


def watershed_from_three_class(probs: np.ndarray, fg_thr: float = 0.5, h: float = 0.1,
                               min_size: int = 15) -> np.ndarray:
    """Instancias desde probabilidades de 3 clases (fondo, interior, borde).

    Energía = p(interior) − p(borde); marcadores por **h-maxima** de la energía
    suavizada dentro del primer plano; watershed sobre −energía restringido a
    p(interior)+p(borde) > ``fg_thr``.
    """
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed

    p_in, p_b = probs[..., 1], probs[..., 2]
    fg = (p_in + p_b) > fg_thr
    energy = ndi.gaussian_filter(p_in - p_b, 1.0)
    markers = ndi.label(h_maxima(np.where(fg, energy, energy.min()), h) & fg & (p_in > p_b))[0]
    lbl = watershed(-energy, markers, mask=fg)
    return remove_small(lbl.astype(np.int32), min_size)


def watershed_from_distance(dist: np.ndarray, fg: np.ndarray, h: float = 0.1, min_size: int = 15) -> np.ndarray:
    """Instancias desde un mapa de distancia regresado y una máscara de primer plano."""
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed

    d = ndi.gaussian_filter(dist.astype(np.float64), 1.0)
    markers = ndi.label(h_maxima(np.where(fg, d, 0), h) & fg)[0]
    return remove_small(watershed(-d, markers, mask=fg).astype(np.int32), min_size)


def tune_watershed(val_items: list[tuple[np.ndarray, np.ndarray]], fn: Any,
                   grid: dict[str, list[float]], metric: Any) -> tuple[dict[str, float], float]:
    """Búsqueda en rejilla de parámetros de post-procesamiento **sobre validación**.

    ``val_items``: lista de (salida del modelo, mapa de etiquetas de referencia).
    """
    keys = list(grid)
    best, best_s = {}, -np.inf
    for vals in product(*[grid[k] for k in keys]):
        params = dict(zip(keys, vals))
        s = float(np.mean([metric(gt, fn(out, **params)) for out, gt in val_items]))
        if s > best_s:
            best, best_s = params, s
    return best, best_s
