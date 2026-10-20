"""Exportación de las particiones a los formatos de los repositorios reutilizados,
**respetando las mismas particiones** (``data/processed/seg_splits.csv``).

* COCO (MMDetection / Detectron2 / torchvision): ``annotations_<split>.json``.
* Cellpose: ``<split>/<id>.png`` + ``<split>/<id>_masks.tif``.
* Hover-Net: parches ``.npy`` de 5 canales (img RGB, inst_map, type_map = 0).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def label_map_to_coco_anns(lbl: np.ndarray, image_id: int, start_ann_id: int = 1) -> list[dict]:
    """Instancias -> anotaciones COCO con RLE comprimido (pycocotools) o polígonos."""
    anns = []
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        mask_utils = None
    k = start_ann_id
    for v in np.unique(lbl):
        if v == 0:
            continue
        m = (lbl == v)
        ys, xs = np.nonzero(m)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
        if mask_utils is not None:
            rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode()
            seg: object = rle
        else:
            import cv2

            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            seg = [c.ravel().astype(float).tolist() for c in cs if len(c) >= 3]
        anns.append(dict(id=k, image_id=image_id, category_id=1, iscrowd=0, area=int(m.sum()), bbox=bbox,
                         segmentation=seg))
        k += 1
    return anns


def export_coco(items: Iterable[tuple[str, np.ndarray, str]], out_json: str | Path) -> None:
    """``items``: (id_imagen, mapa de etiquetas, nombre de archivo)."""
    images, anns, k = [], [], 1
    for i, (iid, lbl, fname) in enumerate(items, start=1):
        images.append(dict(id=i, file_name=fname, height=int(lbl.shape[0]), width=int(lbl.shape[1]), sartorius_id=iid))
        a = label_map_to_coco_anns(lbl, i, k); anns += a; k += len(a)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(dict(images=images, annotations=anns,
                                              categories=[dict(id=1, name="cell")])))


def export_cellpose(items: Iterable[tuple[str, np.ndarray, np.ndarray]], out_dir: str | Path) -> None:
    """``items``: (id_imagen, imagen, mapa de etiquetas)."""
    import cv2
    import tifffile

    d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
    for iid, img, lbl in items:
        cv2.imwrite(str(d / f"{iid}.png"), img)
        tifffile.imwrite(d / f"{iid}_masks.tif", lbl.astype(np.uint16))


def export_hovernet(items: Iterable[tuple[str, np.ndarray, np.ndarray]], out_dir: str | Path,
                    patch: int = 256, step: int = 128) -> None:
    from .preprocess_seg import tile_image

    d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
    for iid, img, lbl in items:
        rgb = np.stack([img] * 3, -1).astype(np.float32)
        stack = np.dstack([rgb, lbl[..., None].astype(np.float32), np.zeros_like(lbl, np.float32)[..., None]])
        tiles, pos = tile_image(stack, patch, patch - step)
        for t, (y, x) in zip(tiles, pos):
            if t.shape[:2] == (patch, patch):
                np.save(d / f"{iid}_{y}_{x}.npy", t)
