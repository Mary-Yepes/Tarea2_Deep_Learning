import numpy as np
import pytest

from project.seg_metrics import (
    aggregated_jaccard_index,
    ap_per_threshold,
    boundary_f1,
    count_metrics,
    dice_iou,
    error_taxonomy,
    evaluate_image,
    hausdorff95,
    instance_iou_matrix,
    panoptic_quality,
    pixel_confusion_metrics,
    sartorius_score,
)


def two_squares():
    gt = np.zeros((64, 64), np.int32); gt[5:25, 5:25] = 1; gt[35:60, 35:60] = 2
    return gt


def test_perfect_and_empty_predictions():
    gt = two_squares()
    assert sartorius_score(gt, gt) == pytest.approx(1.0)
    assert sartorius_score(gt, np.zeros_like(gt)) == 0.0
    assert sartorius_score(np.zeros_like(gt), np.zeros_like(gt)) == 1.0
    pq, sq, rq = panoptic_quality(gt, gt)
    assert pq == pytest.approx(1.0)
    assert aggregated_jaccard_index(gt, gt) == pytest.approx(1.0)


def test_label_permutation_invariance():
    gt = two_squares(); pr = gt.copy(); pr[gt == 1] = 7; pr[gt == 2] = 3
    assert sartorius_score(gt, pr) == pytest.approx(1.0)


def test_dice_iou_relation():
    rng = np.random.default_rng(0)
    a = rng.random((64, 64)) > 0.5; b = rng.random((64, 64)) > 0.5
    d, j = dice_iou(a, b)
    assert d == pytest.approx(2 * j / (1 + j), rel=1e-6)


def test_known_partial_match():
    gt = np.zeros((20, 20), np.int32); gt[0:10, 0:10] = 1           # 100 px
    pr = np.zeros((20, 20), np.int32); pr[0:10, 0:8] = 1            # 80 px, IoU 0.8
    iou = instance_iou_matrix(gt, pr)
    assert iou[0, 0] == pytest.approx(0.8)
    ap = ap_per_threshold(gt, pr)
    assert ap[0.75] == 1.0 and ap[0.8] == 0.0                        # IoU > t estricto
    assert sartorius_score(gt, pr) == pytest.approx(6 / 10)          # umbrales 0.5..0.75


def test_hausdorff_shifted_squares():
    s1 = np.zeros((60, 60), bool); s1[10:40, 10:40] = True
    s2 = np.zeros((60, 60), bool); s2[10:40, 13:43] = True
    assert 2.0 <= hausdorff95(s1, s2) <= 4.0
    assert hausdorff95(s1, s1) == 0.0
    assert hausdorff95(s1, np.zeros_like(s1)) == float("inf")


def test_boundary_f1_tolerance():
    s1 = np.zeros((60, 60), bool); s1[10:40, 10:40] = True
    s2 = np.zeros((60, 60), bool); s2[10:40, 11:41] = True
    assert boundary_f1(s1, s1) == 1.0
    assert boundary_f1(s1, s2, tol=2) == 1.0
    assert boundary_f1(s1, s2, tol=0) < 1.0


def test_pixel_metrics_trivial_classifier():
    gt = np.zeros((100, 100), bool); gt[:5, :5] = True                # π = 0.0025
    m = pixel_confusion_metrics(np.zeros_like(gt), gt)
    assert m["accuracy"] > 0.99 and m["balanced_accuracy"] == pytest.approx(0.5)


def test_error_taxonomy_merge_split_missed_false():
    gt = np.zeros((40, 80), np.int32)
    gt[5:15, 5:15] = 1; gt[5:15, 15:25] = 2          # dos células en contacto -> fusión
    gt[25:35, 5:25] = 3                              # una célula -> división en dos
    gt[5:15, 40:50] = 4                              # omitida
    pr = np.zeros_like(gt)
    pr[5:15, 5:25] = 1                               # fusiona 1 y 2
    pr[25:35, 5:15] = 2; pr[25:35, 15:25] = 3        # divide 3
    pr[25:35, 60:70] = 4                             # falsa
    t = error_taxonomy(gt, pr)
    assert t["fusion"] == 2 and t["division"] == 1 and t["omitida"] == 1 and t["falsa"] == 1
    assert t["tp"] == 0


def test_error_taxonomy_displaced():
    gt = np.zeros((30, 30), np.int32); gt[5:15, 5:15] = 1
    pr = np.zeros_like(gt); pr[9:19, 9:19] = 1       # IoU = 36/164 ≈ 0.22
    assert error_taxonomy(gt, pr)["desplazada"] == 1


def test_aji_penalizes_unused_predictions():
    gt = two_squares(); pr = gt.copy(); pr[0:3, 60:63] = 9
    assert aggregated_jaccard_index(gt, pr) < 1.0


def test_count_metrics_and_evaluate_image():
    c = count_metrics(np.array([10, 20, 30]), np.array([12, 18, 30]))
    assert c["mae"] == pytest.approx(4 / 3) and c["bias"] == pytest.approx(0.0)
    row = evaluate_image(two_squares(), two_squares(), prob=(two_squares() > 0).astype(float))
    assert row["ap"] == pytest.approx(1.0) and row["auc_roc"] == pytest.approx(1.0) and row["n_gt"] == 2
