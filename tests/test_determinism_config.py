import json

import numpy as np

from project.config import ExperimentConfig, environment_report, load_config, set_global_seed, sha256_file
from project.data_seg import make_synthetic_image
from project.inference import ModelRegistry
from project.text_data import make_synthetic_text


def test_seed_determinism_numpy_and_generators():
    set_global_seed(123); a = np.random.rand(5)
    set_global_seed(123); b = np.random.rand(5)
    assert np.array_equal(a, b)
    i1, l1 = make_synthetic_image(seed=7); i2, l2 = make_synthetic_image(seed=7)
    assert np.array_equal(i1, i2) and np.array_equal(l1, l2)
    assert make_synthetic_text(50, seed=3).equals(make_synthetic_text(50, seed=3))


def test_config_yaml_and_env(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("seed: 7\nseeds_eval: [1, 2]\nmodel: bilstm\nlr: 0.001\n")
    cfg = load_config(p, ExperimentConfig())
    assert cfg.seed == 7 and cfg.seeds_eval == (1, 2) and cfg.extra["model"] == "bilstm"
    rep = environment_report(("numpy",))
    assert rep["numpy"] == np.__version__
    f = tmp_path / "x.bin"; f.write_bytes(b"abc")
    assert sha256_file(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_registry_promote_and_rollback(tmp_path):
    r = ModelRegistry(tmp_path)
    r.register("text", "m-1.0.0"); r.promote("text")
    r.register("text", "m-1.1.0"); r.promote("text")
    assert r.data["text"]["production"] == "m-1.1.0"
    assert r.rollback("text") == "m-1.0.0"
    assert json.loads((tmp_path / "registry.json").read_text())["text"]["production"] == "m-1.0.0"
