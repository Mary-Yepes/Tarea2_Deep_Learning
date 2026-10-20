"""Métricas de segmentación: píxel, borde, instancia, conteo y taxonomía de errores.

Todas las funciones operan **por imagen**; la agregación entre imágenes se hace
después con IC bootstrap (``project.stats``), nunca mezclando píxeles de imágenes
distintas.

Supuesto central: los mapas de etiquetas contienen **instancias disjuntas**
(0 = fondo). Con esa condición, y para umbrales de IoU >= 0.5, cada instancia
de referencia empareja como mucho con una predicción (dos predicciones disjuntas
no pueden tener ambas IoU > 0.5 con el mismo objeto), por lo que contar
``iou > t`` equivale al emparejamiento húngaro. Para umbrales < 0.5 se usa
emparejamiento óptimo explícito (``match_instances``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

from .rle import relabel_sequential

THRESHOLDS = np.round(np.arange(0.5, 1.0, 0.05), 2)


# --------------------------------------------------------------------- píxel
def dice_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> tuple[float, float]:
    """Dice e IoU binarios. Con ambas máscaras vacías devuelve (1, 1) por convención."""
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float((2 * inter + eps) / (p.sum() + g.sum() + eps)), float((inter + eps) / (union + eps))


def pixel_confusion_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    p, g = pred.astype(bool), gt.astype(bool)
    tp = int(np.logical_and(p, g).sum()); fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum()); tn = int(np.logical_and(~p, ~g).sum())
    recall = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return dict(precision=prec, recall=recall, specificity=spec,
                accuracy=(tp + tn) / max(tp + tn + fp + fn, 1),
                balanced_accuracy=0.5 * (recall + spec),
                f1=2 * prec * recall / max(prec + recall, 1e-12))


def pixel_prob_metrics(prob: np.ndarray, gt: np.ndarray, max_pixels: int | None = 200_000,
                       seed: int = 0) -> dict[str, float]:
    """AUC-ROC y AUC-PR (average precision) de un mapa de probabilidad de píxel.

    Se submuestrean píxeles de forma estratificada para acotar el costo; el
    resultado es por imagen. Si la imagen solo tiene una clase, las AUC no están
    definidas y se devuelven NaN (se reporta el número de imágenes excluidas).
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = gt.astype(bool).ravel(); s = np.asarray(prob, float).ravel()
    if y.all() or (~y).all():
        return dict(auc_roc=np.nan, auc_pr=np.nan, prevalence=float(y.mean()))
    if max_pixels and y.size > max_pixels:
        rng = np.random.default_rng(seed)
        idx = rng.choice(y.size, max_pixels, replace=False)
        y, s = y[idx], s[idx]
    return dict(auc_roc=float(roc_auc_score(y, s)), auc_pr=float(average_precision_score(y, s)),
                prevalence=float(y.mean()))


# --------------------------------------------------------------------- borde
def _boundary(m: np.ndarray) -> np.ndarray:
    m = m.astype(bool)
    return m ^ ndi.binary_erosion(m, border_value=0)


def hausdorff95(a: np.ndarray, b: np.ndarray) -> float:
    """Distancia de Hausdorff al percentil 95 entre contornos (píxeles).

    Ambos vacíos -> 0; exactamente uno vacío -> ``inf`` (debe reportarse aparte,
    no promediarse: se recomienda la mediana o un tope declarado).
    """
    a, b = a.astype(bool), b.astype(bool)
    if not a.any() and not b.any():
        return 0.0
    if not a.any() or not b.any():
        return float("inf")
    sa, sb = _boundary(a), _boundary(b)
    da = ndi.distance_transform_edt(~sb)[sa]
    db = ndi.distance_transform_edt(~sa)[sb]
    return float(np.percentile(np.concatenate([da, db]), 95))


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tol: float = 2.0) -> float:
    """F1 de borde (BF score, Csurka et al., 2013) con tolerancia ``tol`` píxeles."""
    sp, sg = _boundary(pred), _boundary(gt)
    if not sp.any() and not sg.any():
        return 1.0
    if not sp.any() or not sg.any():
        return 0.0
    prec = float((ndi.distance_transform_edt(~sg)[sp] <= tol).mean())
    rec = float((ndi.distance_transform_edt(~sp)[sg] <= tol).mean())
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


# ------------------------------------------------------------------ instancia
def _overlap_tables(gt_lbl: np.ndarray, pr_lbl: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersecciones (n_gt+1 x n_pr+1, incluye fondo) y áreas por instancia."""
    gt_lbl, pr_lbl = relabel_sequential(gt_lbl), relabel_sequential(pr_lbl)
    n_gt, n_pr = int(gt_lbl.max()), int(pr_lbl.max())
    idx = gt_lbl.ravel().astype(np.int64) * (n_pr + 1) + pr_lbl.ravel()
    inter = np.bincount(idx, minlength=(n_gt + 1) * (n_pr + 1)).reshape(n_gt + 1, n_pr + 1)
    return inter, inter.sum(1), inter.sum(0)


def instance_iou_matrix(gt_lbl: np.ndarray, pr_lbl: np.ndarray) -> np.ndarray:
    """Matriz IoU (n_gt x n_pred) entre mapas de etiquetas."""
    inter, a_gt, a_pr = _overlap_tables(gt_lbl, pr_lbl)
    union = a_gt[:, None] + a_pr[None, :] - inter
    return (inter / np.maximum(union, 1))[1:, 1:]


def match_instances(iou: np.ndarray, thr: float = 0.5) -> list[tuple[int, int, float]]:
    """Emparejamiento uno-a-uno óptimo (húngaro) con IoU > ``thr``."""
    if iou.size == 0:
        return []
    r, c = linear_sum_assignment(-iou)
    return [(int(i), int(j), float(iou[i, j])) for i, j in zip(r, c) if iou[i, j] > thr]


def _tp_fp_fn(iou: np.ndarray, t: float) -> tuple[int, int, int]:
    n_gt, n_pr = iou.shape
    tp = int((iou > t).sum()) if t >= 0.5 else len(match_instances(iou, t))
    return tp, n_pr - tp, n_gt - tp


def sartorius_score(gt_lbl: np.ndarray, pr_lbl: np.ndarray, thresholds: np.ndarray = THRESHOLDS) -> float:
    """Métrica Sartorius: media sobre umbrales de IoU de TP/(TP+FP+FN).

    Imagen sin objetos en referencia ni en predicción -> 1 (convención Kaggle).
    """
    return float(np.mean(list(ap_per_threshold(gt_lbl, pr_lbl, thresholds).values())))


def ap_per_threshold(gt_lbl: np.ndarray, pr_lbl: np.ndarray,
                     thresholds: np.ndarray = THRESHOLDS) -> dict[float, float]:
    """TP/(TP+FP+FN) para cada umbral (permite AP50, AP75 y curvas AP vs IoU)."""
    iou = instance_iou_matrix(gt_lbl, pr_lbl)
    out = {}
    for t in thresholds:
        tp, fp, fn = _tp_fp_fn(iou, float(t))
        d = tp + fp + fn
        out[float(t)] = tp / d if d else 1.0
    return out


def instance_prf(gt_lbl: np.ndarray, pr_lbl: np.ndarray, thr: float = 0.5) -> dict[str, float]:
    tp, fp, fn = _tp_fp_fn(instance_iou_matrix(gt_lbl, pr_lbl), thr)
    p = tp / max(tp + fp, 1) if (tp + fp) else 1.0
    r = tp / max(tp + fn, 1) if (tp + fn) else 1.0
    return dict(inst_precision=p, inst_recall=r, inst_f1=0.0 if p + r == 0 else 2 * p * r / (p + r))


def panoptic_quality(gt_lbl: np.ndarray, pr_lbl: np.ndarray) -> tuple[float, float, float]:
    """PQ = SQ * RQ (Kirillov et al., 2019). Devuelve (PQ, SQ, RQ)."""
    iou = instance_iou_matrix(gt_lbl, pr_lbl)
    n_gt, n_pr = iou.shape
    matched = iou > 0.5
    tp = int(matched.sum()); fp = n_pr - tp; fn = n_gt - tp
    if tp + fp + fn == 0:
        return 1.0, 1.0, 1.0
    sq = float(iou[matched].mean()) if tp else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn)
    return sq * rq, sq, rq


def aggregated_jaccard_index(gt_lbl: np.ndarray, pr_lbl: np.ndarray) -> float:
    """AJI (Kumar et al., 2017).

    Para cada objeto de referencia se toma la predicción de mayor IoU; se acumulan
    intersección y unión; las predicciones nunca usadas suman su área a la unión.
    (Variante original: una predicción puede reutilizarse; véase AJI+ para la
    versión uno-a-uno.)
    """
    inter, a_gt, a_pr = _overlap_tables(gt_lbl, pr_lbl)
    inter, a_gt, a_pr = inter[1:, 1:], a_gt[1:], a_pr[1:]
    if a_gt.size == 0 and a_pr.size == 0:
        return 1.0
    if a_gt.size == 0 or a_pr.size == 0:
        return 0.0
    union = a_gt[:, None] + a_pr[None, :] - inter
    iou = inter / np.maximum(union, 1)
    best = iou.argmax(1)
    c = inter[np.arange(len(a_gt)), best].sum()
    u = union[np.arange(len(a_gt)), best].sum()
    unused = np.setdiff1d(np.arange(len(a_pr)), best)
    u += a_pr[unused].sum()
    return float(c / max(u, 1))


# ------------------------------------------------------------- taxonomía
ERROR_TYPES = ("omitida", "falsa", "fusion", "division", "desplazada")


def error_taxonomy(gt_lbl: np.ndarray, pr_lbl: np.ndarray, match_thr: float = 0.5,
                   cover_thr: float = 0.5, touch_thr: float = 0.1) -> dict[str, int]:
    """Clasifica automáticamente los errores de instancia.

    Definiciones (a priori, declaradas en el pre-registro):

    * TP: par con IoU > ``match_thr``.
    * **fusión** (merge): predicción no emparejada que cubre >= ``cover_thr`` del
      área de **dos o más** objetos de referencia. Se cuentan los objetos fusionados.
    * **división** (split): objeto de referencia no emparejado del que >= 2
      predicciones tienen >= ``cover_thr`` de su propia área dentro de él. Se
      cuentan los objetos divididos.
    * **desplazada / borde impreciso**: objeto no emparejado (ni fusión ni
      división) cuya mejor IoU está en (``touch_thr``, ``match_thr``].
    * **omitida**: objeto de referencia sin predicción con IoU > ``touch_thr``.
    * **falsa**: predicción no emparejada, que no participa en fusión/división y
      cuya mejor IoU es <= ``touch_thr``.
    """
    inter, a_gt, a_pr = _overlap_tables(gt_lbl, pr_lbl)
    inter, a_gt, a_pr = inter[1:, 1:].astype(float), a_gt[1:], a_pr[1:]
    n_gt, n_pr = len(a_gt), len(a_pr)
    out = dict.fromkeys(ERROR_TYPES, 0) | {"tp": 0, "n_gt": n_gt, "n_pred": n_pr}
    if n_gt == 0:
        out["falsa"] = n_pr
        return out
    if n_pr == 0:
        out["omitida"] = n_gt
        return out
    union = a_gt[:, None] + a_pr[None, :] - inter
    iou = inter / np.maximum(union, 1)
    matched_gt = (iou > match_thr).any(1); matched_pr = (iou > match_thr).any(0)
    out["tp"] = int((iou > match_thr).sum())
    cov_gt = inter / np.maximum(a_gt[:, None], 1)    # fracción del objeto cubierta por p
    cov_pr = inter / np.maximum(a_pr[None, :], 1)    # fracción de p dentro del objeto
    gt_done = matched_gt.copy(); pr_done = matched_pr.copy()
    # fusiones
    for j in np.where(~matched_pr)[0]:
        g = np.where((cov_gt[:, j] >= cover_thr) & ~matched_gt)[0]
        if len(g) >= 2:
            out["fusion"] += len(g); gt_done[g] = True; pr_done[j] = True
    # divisiones
    for i in np.where(~gt_done)[0]:
        p = np.where((cov_pr[i, :] >= cover_thr) & ~matched_pr)[0]
        if len(p) >= 2:
            out["division"] += 1; gt_done[i] = True; pr_done[p] = True
    best_gt = iou.max(1); best_pr = iou.max(0)
    for i in np.where(~gt_done)[0]:
        out["desplazada" if best_gt[i] > touch_thr else "omitida"] += 1
    out["falsa"] = int(((~pr_done) & (best_pr <= touch_thr)).sum())
    return out


# --------------------------------------------------------------------- conteo
def count_metrics(n_true: np.ndarray, n_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, correlación de Pearson y Bland–Altman del conteo por imagen."""
    t, p = np.asarray(n_true, float), np.asarray(n_pred, float)
    d = p - t
    md, sd = float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else 0.0
    r = float(np.corrcoef(t, p)[0, 1]) if len(t) > 1 and t.std() > 0 and p.std() > 0 else np.nan
    return dict(mae=float(np.abs(d).mean()), rmse=float(np.sqrt((d ** 2).mean())), pearson_r=r,
                bias=md, loa_low=md - 1.96 * sd, loa_high=md + 1.96 * sd)


# ------------------------------------------------------------------- todo junto
def evaluate_image(gt_lbl: np.ndarray, pr_lbl: np.ndarray, prob: np.ndarray | None = None) -> dict[str, Any]:
    """Todas las métricas para un par (referencia, predicción) de una imagen."""
    gt_b, pr_b = gt_lbl > 0, pr_lbl > 0
    d, j = dice_iou(pr_b, gt_b)
    ap = ap_per_threshold(gt_lbl, pr_lbl)
    pq, sq, rq = panoptic_quality(gt_lbl, pr_lbl)
    row: dict[str, Any] = dict(
        ap=float(np.mean(list(ap.values()))), ap50=ap[0.5], ap75=ap[0.75],
        pq=pq, sq=sq, rq=rq, aji=aggregated_jaccard_index(gt_lbl, pr_lbl),
        dice=d, iou=j, hd95=hausdorff95(pr_b, gt_b), bf1=boundary_f1(pr_b, gt_b),
        n_gt=int(len(np.unique(gt_lbl)) - (gt_lbl == 0).any()),
        n_pred=int(len(np.unique(pr_lbl)) - (pr_lbl == 0).any()),
    )
    row |= {f"ap_t{int(round(t * 100))}": v for t, v in ap.items()}
    row |= pixel_confusion_metrics(pr_b, gt_b)
    row |= instance_prf(gt_lbl, pr_lbl)
    row |= {f"err_{k}": v for k, v in error_taxonomy(gt_lbl, pr_lbl).items() if k in ERROR_TYPES}
    if prob is not None:
        row |= pixel_prob_metrics(prob, gt_b)
    return row
