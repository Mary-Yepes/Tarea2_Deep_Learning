"""Mask R-CNN con backbone **Swin-T** preentrenado (ImageNet) + FPN, en torchvision.

Se construye a mano porque torchvision no trae un ``maskrcnn_swin``. Los anchors
salen del EDA (cuantiles del diámetro equivalente) y ``box_detections_per_img``
del conteo máximo. El entrenamiento usa las **mismas particiones**
(``data/processed/seg_splits.csv``) y la selección del mejor *checkpoint* se hace
con AP@[.5:.95] en validación.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np

DEFAULT_ANCHORS = ((8,), (16,), (32,), (64,), (128,))


def build_maskrcnn(backbone: str = "swin_t", num_classes: int = 2, anchor_sizes: Sequence[Sequence[int]] = DEFAULT_ANCHORS,
                   max_det: int = 600, pretrained: bool = True, min_size: int = 520, max_size: int = 1024) -> Any:
    import torch.nn as nn
    import torchvision
    from torchvision.models.detection import MaskRCNN
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.ops import FeaturePyramidNetwork
    from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

    anchors = AnchorGenerator(tuple(tuple(a) for a in anchor_sizes), ((0.5, 1.0, 2.0),) * len(anchor_sizes))
    if backbone == "resnet50":     # alternativa (desviación declarada si se usa)
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

        m = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT" if pretrained else None,
                                                                   box_detections_per_img=max_det,
                                                                   min_size=min_size, max_size=max_size)
        m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, num_classes)
        m.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, num_classes)
        m.rpn.anchor_generator = anchors
        return m

    class SwinFPN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            sw = torchvision.models.swin_t(weights="IMAGENET1K_V1" if pretrained else None)
            self.body = sw.features               # salidas NHWC; etapas en índices 1, 3, 5, 7 (strides 4..32)
            self.fpn = FeaturePyramidNetwork([96, 192, 384, 768], 256, extra_blocks=LastLevelMaxPool())
            self.out_channels = 256

        def forward(self, x: Any) -> Any:
            feats: OrderedDict[str, Any] = OrderedDict()
            for i, mod in enumerate(self.body):
                x = mod(x)
                if i in (1, 3, 5, 7):
                    feats[str(len(feats))] = x.permute(0, 3, 1, 2).contiguous()
            return self.fpn(feats)

    return MaskRCNN(SwinFPN(), num_classes=num_classes, rpn_anchor_generator=anchors,
                    box_detections_per_img=max_det, min_size=min_size, max_size=max_size,
                    rpn_pre_nms_top_n_test=3000, rpn_post_nms_top_n_test=2000,      # imágenes densas
                    rpn_pre_nms_top_n_train=3000, rpn_post_nms_top_n_train=2000)


def _to_tensor(img: np.ndarray) -> Any:
    import torch

    x = img.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99]); x = np.clip((x - lo) / max(float(hi - lo), 1e-6), 0, 1).astype(np.float32)
    return torch.from_numpy(np.stack([x] * 3, 0).astype(np.float32))   # float32: con NumPy 2 la división daba float64


def _target(lbl: np.ndarray) -> dict[str, Any]:
    import torch

    ids = [i for i in np.unique(lbl) if i != 0]
    masks = np.stack([lbl == i for i in ids]).astype(np.uint8) if ids else np.zeros((0, *lbl.shape), np.uint8)
    boxes = []
    for m in masks:
        ys, xs = np.nonzero(m); boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones(len(ids), dtype=torch.int64), "masks": torch.from_numpy(masks)}


def train_maskrcnn(train_imgs: Sequence[np.ndarray], train_lbls: Sequence[np.ndarray], val_imgs: Sequence[np.ndarray],
                   val_lbls: Sequence[np.ndarray], out_path: str, backbone: str = "swin_t", epochs: int = 20,
                   lr: float = 1e-4, batch_size: int = 2, patience: int = 4, seed: int = 0, max_det: int = 600,
                   anchor_sizes: Sequence[Sequence[int]] = DEFAULT_ANCHORS, time_budget_h: float | None = None) -> dict[str, Any]:
    """Ajuste fino con AdamW + coseno, precisión mixta, flips/rotaciones y selección
    por AP en validación. Respeta un presupuesto de horas si se indica."""
    import torch

    from .config import set_global_seed
    from .rle import masks_to_label_map
    from .seg_metrics import sartorius_score

    set_global_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_maskrcnn(backbone, max_det=max_det, anchor_sizes=anchor_sizes).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(train_imgs) / batch_size))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    rng = np.random.default_rng(seed)
    best, bad, hist, t0 = -1.0, 0, [], time.time()
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(train_imgs)); losses = []
        for b in range(0, len(order), batch_size):
            xs, ts = [], []
            for i in order[b:b + batch_size]:
                img, lbl = train_imgs[i], train_lbls[i]
                k = int(rng.integers(4)); img, lbl = np.rot90(img, k), np.rot90(lbl, k)
                if rng.random() < 0.5:
                    img, lbl = img[:, ::-1], lbl[:, ::-1]
                img, lbl = np.ascontiguousarray(img), np.ascontiguousarray(lbl)
                t = _target(lbl)
                if len(t["boxes"]) == 0:
                    continue
                xs.append(_to_tensor(img).to(dev)); ts.append({k2: v.to(dev) for k2, v in t.items()})
            if not xs:
                continue
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = sum(model(xs, ts).values())
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
            losses.append(float(loss))
        model.eval(); aps = []
        with torch.no_grad():
            for img, lbl in zip(val_imgs, val_lbls):
                o = model([_to_tensor(img).to(dev)])[0]
                keep = o["scores"] >= 0.5
                ms = list((o["masks"][keep, 0] >= 0.5).cpu().numpy())
                pred = masks_to_label_map(ms, o["scores"][keep].cpu().numpy().tolist()) if ms else np.zeros_like(lbl)
                aps.append(sartorius_score(lbl, pred))
        val_ap = float(np.mean(aps)); hist.append(dict(epoch=ep, loss=float(np.mean(losses)) if losses else np.nan, val_ap=val_ap))
        print(f"[maskrcnn {backbone}] época {ep}: pérdida {hist[-1]['loss']:.3f}, AP val {val_ap:.4f}", flush=True)
        if val_ap > best:
            best, bad = val_ap, 0; torch.save(model.state_dict(), out_path)
        else:
            bad += 1
        if bad >= patience or (time_budget_h and (time.time() - t0) / 3600 > time_budget_h):
            break
    return {"best_val_ap": best, "history": hist, "seconds": time.time() - t0, "path": out_path,
            "params": int(sum(p.numel() for p in model.parameters()))}
