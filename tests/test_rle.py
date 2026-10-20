import numpy as np
import pytest

from project.rle import instances_to_label_map, label_map_to_rles, masks_to_label_map, rle_area, rle_decode, rle_encode


def test_roundtrip_two_blocks():
    m = np.zeros((40, 50), np.uint8); m[5:15, 10:30] = 1; m[25:35, 5:20] = 1
    assert (rle_decode(rle_encode(m), m.shape) == m).all()


@pytest.mark.parametrize("order", ["C", "F"])
def test_roundtrip_random_both_orders(order):
    rng = np.random.default_rng(0)
    m = (rng.random((37, 23)) > 0.6).astype(np.uint8)
    assert (rle_decode(rle_encode(m, order), m.shape, order) == m).all()


def test_known_encoding_one_indexed_row_major():
    m = np.zeros((2, 3), np.uint8); m[0, 1:] = 1; m[1, 0] = 1  # aplanado C: 0 1 1 1 0 0
    assert rle_encode(m) == "2 3"
    assert rle_area("2 3") == 3


def test_edge_cases_empty_and_full():
    assert rle_encode(np.zeros((3, 3))) == ""
    assert rle_decode("", (3, 3)).sum() == 0
    assert rle_decode(None, (3, 3)).sum() == 0
    assert rle_decode(float("nan"), (3, 3)).sum() == 0
    assert rle_encode(np.ones((2, 2))) == "1 4"


def test_out_of_bounds_raises():
    with pytest.raises(ValueError):
        rle_decode("5 10", (3, 3))
    with pytest.raises(ValueError):
        rle_decode("1 2 3", (3, 3))


def test_label_map_overlap_first_wins():
    a = np.zeros((10, 10), np.uint8); a[2:6, 2:6] = 1
    b = np.zeros((10, 10), np.uint8); b[4:8, 4:8] = 1
    lbl, ov = instances_to_label_map([rle_encode(a), rle_encode(b)], (10, 10), return_overlap=True)
    assert ov == 4 and lbl[4, 4] == 1 and lbl[7, 7] == 2
    assert len(label_map_to_rles(lbl)) == 2


def test_masks_to_label_map_score_priority():
    a = np.zeros((6, 6), bool); a[0:4, 0:4] = True
    b = np.zeros((6, 6), bool); b[2:6, 2:6] = True
    lbl = masks_to_label_map([a, b], scores=[0.1, 0.9])
    assert lbl[3, 3] == 1  # gana b (mayor score) y recibe la etiqueta 1
