"""Codificación RLE y mapas de etiquetas de instancias.

Convención Sartorius (verificada contra el *notebook* oficial de la competición):
RLE **1-indexado**, pares ``"inicio longitud"`` sobre la imagen aplanada en orden
**de filas** (C / row-major). Otras competiciones (p. ej. HuBMAP) usan orden de
columnas (Fortran); por eso ``order`` es un parámetro explícito y el EDA debe
validar visualmente la decodificación superponiendo máscara e imagen.

Supuestos y casos borde documentados:

* Cadena vacía, ``None`` o ``NaN`` -> máscara vacía.
* Runs fuera de rango -> ``ValueError`` (en lugar de truncar en silencio).
* Runs solapados o desordenados dentro de un RLE -> se aceptan (unión).
* Solapes **entre** instancias al construir el mapa de etiquetas: prevalece la
  primera instancia; los píxeles en conflicto se cuentan y se reportan, porque
  las métricas de instancia asumen objetos disjuntos.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np

Order = Literal["C", "F"]


def _is_empty(rle: object) -> bool:
    if rle is None:
        return True
    if isinstance(rle, float) and np.isnan(rle):
        return True
    return not (isinstance(rle, str) and rle.strip())


def rle_decode(rle: str | None, shape: tuple[int, int], order: Order = "C") -> np.ndarray:
    """RLE -> máscara binaria ``uint8`` de forma ``(alto, ancho)``."""
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros(h * w, dtype=np.uint8)
    if _is_empty(rle):
        return mask.reshape((h, w), order=order)
    assert isinstance(rle, str)
    s = np.asarray(rle.split(), dtype=np.int64)
    if s.size % 2:
        raise ValueError("RLE con número impar de elementos")
    starts, lengths = s[0::2] - 1, s[1::2]
    if (starts < 0).any() or (lengths < 0).any() or (starts + lengths > h * w).any():
        raise ValueError("RLE fuera de los límites de la imagen")
    for a, n in zip(starts, lengths):
        mask[a:a + n] = 1
    return mask.reshape((h, w), order=order)


def rle_encode(mask: np.ndarray, order: Order = "C") -> str:
    """Máscara binaria -> RLE 1-indexado (cadena vacía si la máscara está vacía)."""
    pixels = np.concatenate([[0], (np.asarray(mask) > 0).flatten(order=order).astype(np.int8), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs))


def rle_area(rle: str | None) -> int:
    """Área (píxeles) de un RLE sin decodificarlo (asume runs no solapados)."""
    if _is_empty(rle):
        return 0
    assert isinstance(rle, str)
    return int(np.asarray(rle.split(), dtype=np.int64)[1::2].sum())


def instances_to_label_map(
    rles: Iterable[str | None], shape: tuple[int, int], order: Order = "C",
    return_overlap: bool = False,
) -> np.ndarray | tuple[np.ndarray, int]:
    """Lista de RLE -> mapa de etiquetas ``int32`` (0 = fondo; 1..n instancias).

    Solapes: prevalece la primera instancia. Con ``return_overlap=True`` devuelve
    además el número de píxeles en conflicto (para la auditoría de anotación).
    """
    lbl = np.zeros(shape, dtype=np.int32)
    overlap = 0
    for i, r in enumerate(rles, start=1):
        m = rle_decode(r, shape, order).astype(bool)
        overlap += int((m & (lbl > 0)).sum())
        lbl[m & (lbl == 0)] = i
    return (lbl, overlap) if return_overlap else lbl


def label_map_to_rles(lbl: np.ndarray, order: Order = "C") -> list[str]:
    """Mapa de etiquetas -> lista de RLE (una por instancia, en orden de etiqueta)."""
    return [rle_encode(lbl == k, order) for k in np.unique(lbl) if k != 0]


def relabel_sequential(lbl: np.ndarray) -> np.ndarray:
    """Reetiqueta a 0..n consecutivos conservando el fondo en 0."""
    u, inv = np.unique(lbl, return_inverse=True)
    inv = inv.reshape(lbl.shape).astype(np.int32)
    return inv if u[0] == 0 else inv + 1


def masks_to_label_map(masks: Sequence[np.ndarray] | np.ndarray, scores: Sequence[float] | None = None) -> np.ndarray:
    """Máscaras por instancia (posiblemente solapadas, p. ej. Mask R-CNN o SAM)
    -> mapa de etiquetas disjunto. Si hay puntuaciones, gana la instancia de
    mayor puntuación en los píxeles en conflicto (interfaz común de evaluación).
    """
    masks = list(masks)
    if not masks:
        raise ValueError("Se necesita al menos una máscara para inferir la forma")
    order = np.argsort(-np.asarray(scores)) if scores is not None else np.arange(len(masks))
    lbl = np.zeros(np.asarray(masks[0]).shape, dtype=np.int32)
    k = 0
    for i in order:
        m = np.asarray(masks[i]).astype(bool) & (lbl == 0)
        if m.any():
            k += 1
            lbl[m] = k
    return lbl
