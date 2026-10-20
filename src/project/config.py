"""Configuración, semillas, entorno y trazabilidad de artefactos."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass, field, replace
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuración global del protocolo (valores congelados en el pre-registro)."""

    seed: int = 42
    seeds_eval: tuple[int, ...] = (0, 1, 2, 3, 4)
    seeds_heavy: tuple[int, ...] = (0, 1, 2)
    alpha: float = 0.05
    n_boot: int = 10_000
    img_size: int = 512
    val_frac: float = 0.15
    test_frac: float = 0.15
    k_folds: int = 5
    data_dir: Path = Path("data")
    out_dir: Path = Path("reports")
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["data_dir"], d["out_dir"] = str(self.data_dir), str(self.out_dir)
        return d


CFG = ExperimentConfig()


def load_config(path: str | Path, base: ExperimentConfig = CFG) -> ExperimentConfig:
    """Carga un YAML y sobreescribe los campos conocidos de ``ExperimentConfig``.

    Las claves desconocidas se guardan en ``extra`` (configuración específica del
    experimento: modelo, hiperparámetros, rutas...).
    """
    import yaml

    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    known = {k: raw.pop(k) for k in list(raw) if k in ExperimentConfig.__dataclass_fields__}
    for k in ("seeds_eval", "seeds_heavy"):
        if k in known:
            known[k] = tuple(known[k])
    for k in ("data_dir", "out_dir"):
        if k in known:
            known[k] = Path(known[k])
    extra = {**base.extra, **raw}
    return replace(base, **known, extra=extra)


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Fija semillas en Python, NumPy, TensorFlow y PyTorch (si están instalados).

    Nota: ``PYTHONHASHSEED`` solo afecta a procesos *nuevos*; para que el hash de
    cadenas sea determinista debe exportarse antes de lanzar el intérprete (lo
    hace el ``Makefile``).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        if deterministic:
            tf.config.experimental.enable_op_determinism()
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


DEFAULT_PKGS = (
    "numpy", "pandas", "scipy", "scikit-learn", "scikit-image", "tensorflow", "torch",
    "xgboost", "optuna", "statsmodels", "transformers", "fastapi", "pandera",
)


def environment_report(pkgs: tuple[str, ...] = DEFAULT_PKGS) -> dict[str, Any]:
    """Versiones de software y hardware para el reporte de reproducibilidad."""
    rep: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    for p in pkgs:
        try:
            rep[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            rep[p] = None
    try:
        import tensorflow as tf

        rep["gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    except ImportError:
        rep["gpus"] = []
    return rep


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Hash SHA-256 de un archivo (para registrar versiones de datos y modelos)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_dir(path: str | Path, pattern: str = "**/*") -> str:
    """Hash estable de un directorio: hash de la lista ordenada (ruta relativa, hash)."""
    root = Path(path)
    h = hashlib.sha256()
    for f in sorted(p for p in root.glob(pattern) if p.is_file()):
        h.update(str(f.relative_to(root)).encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        import subprocess

        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def write_run_manifest(out: str | Path, cfg: ExperimentConfig, **extra: Any) -> dict[str, Any]:
    """Escribe un manifiesto JSON (commit, config, entorno, marca de tiempo)."""
    man = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git_commit(),
        "config": cfg.to_dict(),
        "environment": environment_report(),
        **extra,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(man, indent=2, default=str), encoding="utf-8")
    return man


def report_dirs(root: str | Path, real: bool) -> tuple[Path, Path]:
    """(tablas, figuras). En modo demo (datos sintéticos) se escribe en
    ``reports/demo/`` para que ninguna cifra de demostración se mezcle con los
    resultados reales en ``reports/tables`` y ``reports/figures``."""
    base = Path(root) / "reports" / ("" if real else "demo")
    tab, fig = base / "tables", base / "figures"
    tab.mkdir(parents=True, exist_ok=True); fig.mkdir(parents=True, exist_ok=True)
    return tab, fig


SMOKE = os.environ.get("SMOKE", "0") == "1"


def budget(domain: str | None = None, root: str | Path | None = None) -> dict[str, Any]:
    """Presupuesto del perfil activo (``BUDGET`` = rapido | completo) desde
    ``configs/budget.yaml``. En modo humo (``SMOKE=1``) reduce todo al mínimo."""
    import yaml

    root = Path(root) if root else Path(__file__).resolve().parents[2]
    prof = os.environ.get("BUDGET", "rapido")
    b = yaml.safe_load((root / "configs/budget.yaml").read_text(encoding="utf-8"))[prof]
    b["perfil"] = prof
    if SMOKE:
        s, n = b["seg"], b["nlp"]
        s.update(seeds=[0], loco_seeds=[0], fractions=[0.5, 1.0], hpo_trials=0)
        for k in ("unetpp", "hovernet"):
            s[k].update(max_epochs=1, patience=1, batch_size=2)
        s["maskrcnn"].update(epochs=1, patience=1, time_budget_h=0.05)
        s["cellpose"].update(n_epochs=2)
        s["ablation"].update(max_epochs=1, patience=1)
        s["ablation_levels"] = {"loss": ["bce_dice", "focal"]}
        n.update(seeds=[0], transformer_seeds=[0], k_folds=2, hpo_trials=1, keras_max_epochs=1)
        n["transformer"].update(epochs=1)
    return b if domain is None else {**b[domain], "perfil": prof}


def tables_dir(root: str | Path) -> Path:
    """``reports/tables`` (o ``reports/smoke/tables`` en modo humo)."""
    p = Path(root) / ("reports/smoke/tables" if SMOKE else "reports/tables")
    p.mkdir(parents=True, exist_ok=True)
    return p
