"""Seguimiento de experimentos: MLflow si está instalado (``MLFLOW_TRACKING_URI``,
por defecto ``file:./mlruns``); si no, un registro JSONL equivalente en
``reports/runs.jsonl``. Registra commit, configuración, semilla, métricas,
artefactos y tiempo; el registro de modelos con etapas y rollback está en
``project.inference.ModelRegistry`` (y opcionalmente en el Model Registry de MLflow)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import git_commit


def log_run(name: str, params: dict[str, Any], metrics: dict[str, float], artifacts: list[str] | None = None,
            tags: dict[str, str] | None = None, root: str | Path = ".") -> None:
    tags = {"git_commit": str(git_commit()), **(tags or {})}
    try:
        import mlflow

        with mlflow.start_run(run_name=name):
            mlflow.set_tags(tags)
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
            for a in artifacts or []:
                if Path(a).exists():
                    mlflow.log_artifact(a)
        return
    except ImportError:
        pass
    p = Path(root) / "reports/runs.jsonl"; p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(name=name, time=time.time(), params=params, metrics=metrics,
                                 artifacts=artifacts or [], tags=tags), default=str) + "\n")
