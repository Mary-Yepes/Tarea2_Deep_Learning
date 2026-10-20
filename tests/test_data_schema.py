"""Pruebas de datos: esquema de anotaciones (pandera) e integridad.

Si existen los datos reales (``data/raw/sartorius/train.csv``) se validan; si no,
se valida un marco sintético con el mismo esquema (CI sin datos)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from project.data_seg import check_integrity, make_synthetic_dataset
from project.rle import rle_encode

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data/raw/sartorius/train.csv"


def annotations() -> pd.DataFrame:
    if REAL.exists():
        return pd.read_csv(REAL)
    imgs, lbls, meta = make_synthetic_dataset(2)
    rows = []
    for (_, r), lb in zip(meta.iterrows(), lbls):
        for k in range(1, lb.max() + 1):
            rows.append(dict(id=r["id_imagen"], annotation=rle_encode(lb == k), width=lb.shape[1],
                             height=lb.shape[0], cell_type=r["tipo_celular"], sample_id=r["sample_id"]))
    return pd.DataFrame(rows)


def test_integrity_checks():
    rep = check_integrity(annotations())
    assert rep["dims_validas"] and rep["rle_en_limites"] and rep["metadatos_consistentes_por_imagen"]
    assert rep["nulos"]["annotation"] == 0 and rep["areas_nulas"] == 0


def test_pandera_schema():
    pytest.importorskip("pandera")
    from project.data_seg import pandera_schema

    pandera_schema().validate(annotations())


def test_pandera_rejects_bad_rows():
    pytest.importorskip("pandera")
    from project.data_seg import pandera_schema

    bad = annotations().head(3).copy(); bad.loc[bad.index[0], "cell_type"] = "unknown"
    from pandera.errors import SchemaError

    with pytest.raises(SchemaError):
        pandera_schema().validate(bad)


def test_ids_unique_per_annotation():
    a = annotations()
    assert not a.duplicated(["id", "annotation"]).any()
    assert np.isfinite(a["width"]).all()
