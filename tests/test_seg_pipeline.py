import numpy as np
import pytest

from project.classical import classical_prob_map, classical_segment
from project.data_seg import (
    check_integrity,
    density_contact,
    dhash,
    hamming,
    image_quality,
    instance_table,
    make_synthetic_image,
)
from project.losses import bce_loss, focal_loss, soft_dice_grad, soft_dice_loss, tversky_loss
from project.postprocess import tune_watershed, watershed_from_three_class
from project.preprocess_seg import (
    alignment_check_grid,
    augment_pair,
    distance_target,
    hv_target,
    instances_lost_by_resize,
    pad_to_multiple,
    perturb,
    stitch_tiles,
    three_class_target,
    tile_image,
    unpad,
)
from project.rle import rle_encode
from project.seg_metrics import sartorius_score


def test_classical_baseline_beats_zero_on_synthetic():
    sc = []
    for seed in range(4):             # promedio: una sola imagen sintética pequeña es demasiado ruidosa
        img, lbl = make_synthetic_image((128, 160), "astro", seed=seed)
        pred = classical_segment(img)
        assert pred.max() > 0
        sc.append(sartorius_score(lbl, pred))
    assert float(np.mean(sc)) > 0.1
    p = classical_prob_map(img); assert 0 <= p.min() and p.max() <= 1


def test_three_class_target_and_watershed_recovers_instances():
    _, lbl = make_synthetic_image((128, 160), "cort", seed=1)
    tgt = three_class_target(lbl)
    probs = np.eye(3)[tgt].astype(float)                       # «modelo perfecto»
    rec = watershed_from_three_class(probs, h=0.05, min_size=5)
    assert sartorius_score(lbl, rec) > 0.6
    best, s = tune_watershed([(probs, lbl)], watershed_from_three_class, {"h": [0.05, 0.2], "min_size": [5]},
                             sartorius_score)
    assert s >= 0.6 and "h" in best


def test_targets_shapes():
    _, lbl = make_synthetic_image((64, 64), "cort", seed=2)
    assert distance_target(lbl).max() <= 1.0
    hv = hv_target(lbl); assert hv.shape == (64, 64, 2) and np.abs(hv).max() <= 1.0


def test_augmentation_keeps_alignment():
    g, m = alignment_check_grid((96, 96))
    rng = np.random.default_rng(0)
    for level in ("geom", "geom+photo+elastic"):
        for _ in range(10):
            gi, mi = augment_pair(g, m, rng, "geom" if level == "geom" else "geom+photo+elastic")
            # misma geometría en imagen y máscara: alta correlación píxel a píxel
            assert np.corrcoef(gi.ravel(), (mi > 0).ravel())[0, 1] > (0.9 if level == "geom" else 0.6)
            assert gi.shape == mi.shape == (96, 96)


def test_pad_tile_stitch_roundtrip():
    x = np.random.default_rng(0).random((100, 130)).astype(np.float32)
    p, hw = pad_to_multiple(x, 32); assert p.shape == (128, 160) and np.allclose(unpad(p, hw), x)
    tiles, pos = tile_image(x, 64, 16)
    assert np.allclose(stitch_tiles(tiles, pos, x.shape), x, atol=1e-6)


def test_resize_loss_and_perturbations():
    _, lbl = make_synthetic_image((128, 160), "shsy5y", seed=3)
    r = instances_lost_by_resize(lbl, (32, 40), min_pixels=10)
    assert r["frac_perdidas"] > 0
    img = np.full((32, 32), 0.5, np.float32)
    for k in ("noise", "blur", "contrast", "brightness", "illumination", "jpeg"):
        assert perturb(img, k, 3).shape == img.shape


def test_eda_helpers_on_synthetic():
    img, lbl = make_synthetic_image((128, 160), "shsy5y", seed=4)
    q = image_quality(img); assert q["laplacian_var"] > 0 and q["snr"] > 0
    d = density_contact(lbl); assert 0 < d["cobertura"] < 1
    t = instance_table(lbl, "x"); assert len(t) == lbl.max() and (t["solidez"] <= 1).all()
    assert hamming(dhash(img), dhash(img)) == 0
    import pandas as pd
    ann = pd.DataFrame({"id": ["a"], "annotation": [rle_encode(lbl == 1)], "width": [160], "height": [128],
                        "cell_type": ["shsy5y"]})
    rep = check_integrity(ann); assert rep["rle_en_limites"] and rep["anotaciones_duplicadas"] == 0


def test_loss_references_and_gradient():
    rng = np.random.default_rng(0)
    p = rng.random(50); g = (rng.random(50) > 0.7).astype(float)
    num = np.array([(soft_dice_loss(p + 1e-6 * np.eye(50)[i], g) - soft_dice_loss(p - 1e-6 * np.eye(50)[i], g))
                    / 2e-6 for i in range(50)])
    assert np.allclose(num, soft_dice_grad(p, g), atol=1e-6)
    assert soft_dice_loss(g, g) == pytest.approx(0.0, abs=1e-9)
    assert tversky_loss(g, g) == pytest.approx(0.0, abs=1e-9)
    assert focal_loss(p, g) < bce_loss(p, g)


def test_hovernet_postprocess_with_perfect_maps_and_common_interface():
    from project.adapters_seg import ClassicalAdapter, HoverNetAdapter, evaluate_adapter

    img, lbl = make_synthetic_image((128, 160), "astro", seed=5)
    hn = HoverNetAdapter(lambda im: ((lbl > 0).astype(float), hv_target(lbl)))
    pred, prob = hn.predict(img)
    assert sartorius_score(lbl, pred) > 0.5 and prob.shape == lbl.shape
    rows = evaluate_adapter(ClassicalAdapter(), [img], [lbl], ["x"])
    assert rows[0]["modelo"] == "Baseline clásico" and "ap" in rows[0] and "auc_roc" in rows[0]
