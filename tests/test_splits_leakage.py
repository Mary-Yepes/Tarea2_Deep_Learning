"""Ausencia de fuga entre particiones (segmentación por placa y NLP por clúster de duplicados)."""
import numpy as np
import pandas as pd
import pytest

from project.splits import assert_no_group_leakage, grouped_train_val_test_split, nested_cv, split_summary
from project.text_data import cross_split_duplicates, make_synthetic_text, near_duplicate_clusters


def plate_table(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t, n_plates in (("shsy5y", 14), ("astro", 12), ("cort", 20)):
        for p in range(n_plates):
            for i in range(int(rng.integers(5, 15))):
                rows.append(dict(id=f"{t}{p}_{i}", cell_type=t, sample_id=f"{t}_{p}"))
    return pd.DataFrame(rows)


def test_grouped_split_no_leakage_and_fractions():
    df = plate_table()
    s = grouped_train_val_test_split(df, "sample_id", "cell_type", 0.15, 0.15, seed=1)
    assert_no_group_leakage(s, df["sample_id"])
    summ = split_summary(df, s, "cell_type", "sample_id")
    assert set(summ.index) == {"train", "val", "test"}
    assert 0.08 < summ.loc["test", "frac"] < 0.25 and 0.08 < summ.loc["val", "frac"] < 0.25
    for c in ("p_shsy5y", "p_astro", "p_cort"):     # estratificación razonable
        assert summ[c].max() - summ[c].min() < 0.25


def test_leakage_detector_fires():
    s = pd.Series(["train", "test"]); g = pd.Series(["plateA", "plateA"])
    with pytest.raises(AssertionError):
        assert_no_group_leakage(s, g)


def test_near_duplicates_grouped_and_never_cross_split():
    df = make_synthetic_text(300, seed=0)
    dup = df.sample(30, random_state=0).assign(text=lambda d: d["text"] + " ok")   # casi-duplicados
    df = pd.concat([df, dup, df.iloc[:10]], ignore_index=True)                     # + exactos
    cl = near_duplicate_clusters(df["text"].tolist(), threshold=0.8)
    assert len(np.unique(cl)) <= len(df) - 35
    s = grouped_train_val_test_split(df.assign(g=cl), "g", "label", seed=0)
    assert cross_split_duplicates(cl, s.tolist()) == 0


def test_nested_cv_reports_optimism():
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier

    X, y = make_classification(300, 20, n_informative=3, random_state=0)
    g = np.arange(300) // 3
    grid = [dict(max_depth=d, min_samples_leaf=m) for d in (1, 2, 4, 8, None) for m in (1, 5, 20)]
    res = nested_cv(X, y, g, lambda p: DecisionTreeClassifier(random_state=0, **p), grid, accuracy_score,
                    k_outer=3, k_inner=3)
    assert len(res["folds"]) == 3 and np.isfinite(res["optimism"])
