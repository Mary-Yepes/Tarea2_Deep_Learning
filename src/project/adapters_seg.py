"""Interfaz común de evaluación para los seis paradigmas de segmentación.

Cada adaptador implementa ``predict(img) -> (label_map, prob_map | None)``, donde
``label_map`` es un mapa de instancias disjuntas (int32, 0 = fondo) y
``prob_map`` es la probabilidad de píxel «célula» definida explícitamente para
cada modelo (necesaria para AUC/AP de píxel; ``None`` = no aplicable).

Modelos reutilizados (dependencias opcionales; documentar versión/commit/licencia
en ``reports/model_cards`` y ``book/metodos``):

* Cellpose 2.0/3.x — ``pip install "cellpose>=3,<4"`` (BSD-3). En Cellpose 4 la
  API cambió (modelo ``cpsam``): fije la versión.
* Segment Anything — ``pip install git+https://github.com/facebookresearch/segment-anything``
  (Apache-2.0) + checkpoint ``sam_vit_b_01ec64.pth``. MedSAM usa la misma
  arquitectura con otro checkpoint.
* Mask R-CNN — torchvision (BSD-3; ResNet50-FPN v2) o MMDetection (Apache-2.0;
  ``mask-rcnn_swin-t-p4-w7_fpn_1x_coco``) para el backbone Swin del enunciado.
* Hover-Net — repositorio oficial ``vqdang/hover_net`` (MIT); ramas NP + HV sin
  rama de tipo. Aquí se incluye el post-procesamiento HV propio para que la
  salida de cualquier implementación (mapas NP y HV) se evalúe igual.

Los adaptadores PyTorch conviven con el flujo TensorFlow solo a través de esta
interfaz NumPy (justificación: reutilizar implementaciones oficiales sin portar
pesos; la evaluación es idéntica para todos).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from scipy import ndimage as ndi

from .postprocess import remove_small, watershed_from_three_class
from .rle import masks_to_label_map


class SegModel(Protocol):
    name: str

    def predict(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]: ...


def _to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99])
    x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    x = (x * 255).astype(np.uint8)
    return np.stack([x] * 3, -1) if x.ndim == 2 else x


def instance_prob_map(masks: list[np.ndarray], scores: list[float], shape: tuple[int, int]) -> np.ndarray:
    """Mapa de probabilidad de píxel para modelos de instancias: máximo, por
    píxel, de la puntuación de las instancias que lo cubren (definición declarada)."""
    p = np.zeros(shape, np.float32)
    for m, s in zip(masks, scores):
        p = np.maximum(p, np.asarray(m, bool) * float(s))
    return p


# ----------------------------------------------------------- 0. clásico
@dataclass
class ClassicalAdapter:
    params: dict[str, Any] = field(default_factory=dict)
    name: str = "Baseline clásico"

    def predict(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        from .classical import classical_prob_map, classical_segment

        return classical_segment(img, **self.params), classical_prob_map(img)


# ----------------------------------------------------------- 3. U-Net++
@dataclass
class UNetPPAdapter:
    model: Any
    post: dict[str, Any] = field(default_factory=lambda: dict(fg_thr=0.5, h=0.1, min_size=15))
    tta: bool = False
    resolution: str = "native"         # "native" (padding a 32) | "tiling_256_ov64" | "256" | "512"
    normalize: Any = None              # None = percentiles por imagen (la misma que en entrenamiento)
    name: str = "U-Net++ + Watershed"

    def _single(self, img: np.ndarray) -> np.ndarray:
        import cv2

        from .models_seg import predict_probs as _pp
        from .preprocess_seg import norm_percentile, stitch_tiles, tile_image

        norm = self.normalize or norm_percentile

        def predict_probs(model: Any, x: np.ndarray) -> Any:
            return _pp(model, x, normalize=norm)

        if self.resolution.startswith("tiling"):
            tiles, pos = tile_image(img, 256, 64)
            outs = [predict_probs(self.model, t) for t in tiles]
            return stitch_tiles(outs, pos, img.shape[:2] + (3,)).astype(np.float32)  # type: ignore[arg-type]
        if self.resolution in ("256", "512"):
            h, w = img.shape[:2]; tw = int(self.resolution); th = max(32, int(tw * h / w) // 32 * 32)
            p = predict_probs(self.model, cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA))
            return cv2.resize(np.asarray(p, np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        out = predict_probs(self.model, img)
        assert isinstance(out, np.ndarray)
        return out

    def probs(self, img: np.ndarray) -> np.ndarray:
        from .preprocess_seg import tta_predict

        if self.tta:
            return tta_predict(self._single, img)[0]
        return self._single(img)

    def predict(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        p = self.probs(img)
        return watershed_from_three_class(p, **self.post), p[..., 1] + p[..., 2]


# ----------------------------------------------------------- 4. Cellpose
@dataclass
class CellposeAdapter:
    model_type: str = "cyto2"          # generalista; o ruta a un modelo ajustado (fine-tuned)
    diameter: float | None = None      # None = estimación automática del diámetro
    gpu: bool = True
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    name: str = "Cellpose 2.0"
    _m: Any = None

    def _model(self) -> Any:
        if self._m is None:
            from cellpose import models

            kw = ({"pretrained_model": self.model_type} if "/" in self.model_type or self.model_type.endswith(".pth")
                  else {"model_type": self.model_type})
            self._m = models.CellposeModel(gpu=self.gpu, **kw)
        return self._m

    def predict(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        masks, flows, _ = self._model().eval(img, diameter=self.diameter, channels=[0, 0],
                                             flow_threshold=self.flow_threshold,
                                             cellprob_threshold=self.cellprob_threshold)
        cellprob = 1 / (1 + np.exp(-np.asarray(flows[2], float)))   # logits de probabilidad de célula
        return np.asarray(masks, np.int32), cellprob.astype(np.float32)


# ----------------------------------------------------------- 5. SAM / MedSAM
@dataclass
class SAMAutoAdapter:
    """(a) Cero-shot con el generador automático de máscaras."""

    checkpoint: str
    model_type: str = "vit_b"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.86
    stability_score_thresh: float = 0.92
    max_area_frac: float = 0.05        # descarta máscaras gigantes (fondo/regiones)
    min_area: int = 15
    device: str = "cuda"
    name: str = "SAM (cero-shot)"
    _g: Any = None

    def _gen(self) -> Any:
        if self._g is None:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint).to(self.device)
            self._g = SamAutomaticMaskGenerator(sam, points_per_side=self.points_per_side,
                                                pred_iou_thresh=self.pred_iou_thresh,
                                                stability_score_thresh=self.stability_score_thresh,
                                                min_mask_region_area=self.min_area)
        return self._g

    def predict(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        out = self._gen().generate(_to_rgb_uint8(img))
        h, w = img.shape[:2]
        keep = [o for o in out if self.min_area <= o["area"] <= self.max_area_frac * h * w]
        if not keep:
            return np.zeros((h, w), np.int32), np.zeros((h, w), np.float32)
        masks = [o["segmentation"] for o in keep]; scores = [float(o["predicted_iou"]) for o in keep]
        return masks_to_label_map(masks, scores), instance_prob_map(masks, scores, (h, w))


@dataclass
class SAMBoxAdapter:
    """(b) SAM con prompts de caja generados por un detector (p. ej. Mask R-CNN)."""

    checkpoint: str
    box_fn: Any                        # img -> (N, 4) cajas xyxy y (N,) puntuaciones
    model_type: str = "vit_b"
    device: str = "cuda"
    name: str = "SAM (prompts de caja)"
    _p: Any = None

    def _pred(self) -> Any:
        if self._p is None:
            from segment_anything import SamPredictor, sam_model_registry

            self._p = SamPredictor(sam_model_registry[self.model_type](checkpoint=self.checkpoint).to(self.device))
        return self._p

    def predict(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        boxes, scores = self.box_fn(img)
        h, w = img.shape[:2]
        if len(boxes) == 0:
            return np.zeros((h, w), np.int32), np.zeros((h, w), np.float32)
        import torch

        p = self._pred(); rgb = _to_rgb_uint8(img); p.set_image(rgb)
        masks = []
        bx = torch.as_tensor(np.asarray(boxes, np.float32), device=p.device)
        for i in range(0, len(bx), 64):                      # prompts en lotes (cientos de células por imagen)
            tb = p.transform.apply_boxes_torch(bx[i:i + 64], rgb.shape[:2])
            m, _, _ = p.predict_torch(point_coords=None, point_labels=None, boxes=tb, multimask_output=False)
            masks += list(m[:, 0].cpu().numpy())
        return masks_to_label_map(masks, list(scores)), instance_prob_map(masks, list(scores), (h, w))


# ----------------------------------------------------------- 1. Mask R-CNN
@dataclass
class MaskRCNNTorchvisionAdapter:
    """Mask R-CNN (torchvision) con backbone Swin-T + FPN (``maskrcnn_torch.build_maskrcnn``)
    o ResNet50-FPN v2. ``detections_per_img`` se fija con el conteo del EDA."""

    weights_path: str | None = None
    backbone: str = "swin_t"
    score_thr: float = 0.5
    mask_thr: float = 0.5
    detections_per_img: int = 600
    device: str = "cuda"
    name: str = "Mask R-CNN (+ Swin)"
    _m: Any = None

    def _model(self) -> Any:
        if self._m is None:
            import torch

            from .maskrcnn_torch import build_maskrcnn

            dev = self.device if torch.cuda.is_available() else "cpu"
            m = build_maskrcnn(self.backbone, max_det=self.detections_per_img, pretrained=False)
            if self.weights_path:
                m.load_state_dict(torch.load(self.weights_path, map_location="cpu"))
            self._m = m.eval().to(dev); self.device = dev
        return self._m

    def boxes(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cajas xyxy y puntuaciones (prompts para SAM)."""
        import torch

        from .maskrcnn_torch import _to_tensor

        with torch.no_grad():
            out = self._model()([_to_tensor(img).to(self.device)])[0]
        keep = out["scores"] >= self.score_thr
        return out["boxes"][keep].cpu().numpy(), out["scores"][keep].cpu().numpy()

    def predict(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        import torch

        from .maskrcnn_torch import _to_tensor

        with torch.no_grad():
            out = self._model()([_to_tensor(img).to(self.device)])[0]
        keep = out["scores"] >= self.score_thr
        masks = (out["masks"][keep, 0] >= self.mask_thr).cpu().numpy()
        scores = out["scores"][keep].cpu().numpy().tolist()
        h, w = img.shape[:2]
        if len(masks) == 0:
            return np.zeros((h, w), np.int32), np.zeros((h, w), np.float32)
        return masks_to_label_map(list(masks), scores), instance_prob_map(list(masks), scores, (h, w))


@dataclass
class MaskRCNNMMDetAdapter:
    config: str                         # p. ej. configs/mmdet/mask-rcnn_swin-t_cells.py
    checkpoint: str
    score_thr: float = 0.5
    device: str = "cuda:0"
    name: str = "Mask R-CNN (+ Swin)"
    _m: Any = None

    def predict(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        from mmdet.apis import inference_detector, init_detector

        if self._m is None:
            self._m = init_detector(self.config, self.checkpoint, device=self.device)
        res = inference_detector(self._m, _to_rgb_uint8(img)[..., ::-1])   # BGR
        inst = res.pred_instances
        keep = inst.scores.cpu().numpy() >= self.score_thr
        masks = list(inst.masks.cpu().numpy()[keep]); scores = inst.scores.cpu().numpy()[keep].tolist()
        h, w = img.shape[:2]
        if not masks:
            return np.zeros((h, w), np.int32), np.zeros((h, w), np.float32)
        return masks_to_label_map(masks, scores), instance_prob_map(masks, scores, (h, w))


# ----------------------------------------------------------- 2. Hover-Net
def hovernet_postprocess(np_prob: np.ndarray, hv: np.ndarray, fg_thr: float = 0.5, k: float = 0.4,
                         sobel_ksize: int = 21, min_size: int = 10) -> np.ndarray:
    """Post-procesamiento de Hover-Net (Graham et al., 2019, sec. 3.4), fiel a
    ``__proc_np_hv`` del repositorio oficial: gradientes Sobel de los mapas H y V
    normalizados; la energía es alta en las fronteras entre instancias; los
    marcadores son el primer plano menos la frontera (energía >= ``k``); watershed
    sobre la distancia suavizada. ``sobel_ksize`` debe escalar con el tamaño de
    célula (21 en histología a 40×; menor en células pequeñas: se ajusta en validación).
    """
    import cv2
    from skimage.segmentation import watershed

    blb = ndi.label(np_prob >= fg_thr)[0]
    blb = (remove_small(blb.astype(np.int32), min_size) > 0).astype(np.float32)
    h = cv2.normalize(hv[..., 0].astype(np.float32), None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v = cv2.normalize(hv[..., 1].astype(np.float32), None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    sh = 1 - cv2.normalize(cv2.Sobel(h, cv2.CV_64F, 1, 0, ksize=sobel_ksize), None, 0, 1, cv2.NORM_MINMAX)
    sv = 1 - cv2.normalize(cv2.Sobel(v, cv2.CV_64F, 0, 1, ksize=sobel_ksize), None, 0, 1, cv2.NORM_MINMAX)
    overall = np.maximum(sh, sv) - (1 - blb)
    overall[overall < 0] = 0
    dist = -cv2.GaussianBlur(((1.0 - overall) * blb).astype(np.float32), (3, 3), 0)
    marker = blb - (overall >= k).astype(np.float32)
    marker[marker < 0] = 0
    marker = ndi.binary_fill_holes(marker).astype(np.uint8)
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    marker = remove_small(ndi.label(marker)[0].astype(np.int32), min_size)
    lbl = watershed(dist, markers=marker, mask=blb > 0)
    return remove_small(lbl.astype(np.int32), min_size)


@dataclass
class HoverNetAdapter:
    """``infer_fn(img) -> (np_prob (H, W), hv (H, W, 2))`` envuelve la red
    (repositorio oficial o reimplementación); el post-procesamiento es común."""

    infer_fn: Any
    post: dict[str, Any] = field(default_factory=dict)
    name: str = "Hover-Net"

    def predict(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        npp, hv = self.infer_fn(img)
        return hovernet_postprocess(npp, hv, **self.post), npp.astype(np.float32)


def evaluate_adapter(adapter: SegModel, images: list[np.ndarray], labels: list[np.ndarray],
                     ids: list[str]) -> list[dict[str, Any]]:
    """Evalúa cualquier adaptador con el mismo código de métricas (por imagen)."""
    import time

    from .seg_metrics import evaluate_image

    rows = []
    for img, gt, i in zip(images, labels, ids):
        t0 = time.perf_counter()
        lbl, prob = adapter.predict(img)
        dt = (time.perf_counter() - t0) * 1e3
        rows.append(dict(id_imagen=i, modelo=adapter.name, latency_ms=dt, **evaluate_image(gt, lbl, prob)))
    return rows
