"""HPO con presupuesto equitativo (Optuna/TPE + poda) y espacios de búsqueda."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Espacio BiLSTM del enunciado: el producto cartesiano completo es inviable.
BILSTM_SPACE: dict[str, list[Any]] = dict(
    lstm_units=[32, 64, 128], num_lstm_layers=[1, 2], dropout_rate=[0.3, 0.5],
    learning_rate=[0.0005, 0.001, 0.003], batch_size=[32, 64], sequence_length=[128, 256, 512],
    max_epochs=[20, 50], embedding_dim=[100, 200],
)
GRID_SIZE = 1
for _v in BILSTM_SPACE.values():
    GRID_SIZE *= len(_v)  # = 576

UNETPP_SPACE: dict[str, Any] = dict(
    learning_rate=("log", 1e-4, 3e-3), batch_size=[4, 8], optimizer=["adam", "adamw"],
    scheduler=["cosine", "constant"], base_filters=[16, 32], loss=["bce_dice", "tversky", "focal"],
    h_maxima=("float", 0.02, 0.3), fg_thr=("float", 0.3, 0.7),
)
XGB_SPACE: dict[str, Any] = dict(
    max_depth=("int", 3, 10), learning_rate=("log", 0.01, 0.3), subsample=("float", 0.5, 1.0),
    colsample_bytree=("float", 0.2, 1.0), min_child_weight=("log", 1, 20), reg_lambda=("log", 1e-3, 10),
)


def suggest(trial: Any, space: dict[str, Any]) -> dict[str, Any]:
    """Traduce un espacio declarativo a llamadas ``trial.suggest_*``."""
    out = {}
    for k, v in space.items():
        if isinstance(v, list):
            out[k] = trial.suggest_categorical(k, v)
        elif v[0] == "log":
            out[k] = trial.suggest_float(k, v[1], v[2], log=True)
        elif v[0] == "float":
            out[k] = trial.suggest_float(k, v[1], v[2])
        elif v[0] == "int":
            out[k] = trial.suggest_int(k, v[1], v[2])
    return out


def suggest_bilstm_space(trial: Any) -> dict[str, Any]:
    return suggest(trial, BILSTM_SPACE)


def run_study(objective: Callable[[Any], float], n_trials: int = 30, seed: int = 0,
              direction: str = "maximize", timeout: float | None = None, storage: str | None = None,
              study_name: str | None = None) -> Any:
    """Mismo ``n_trials``, mismo ``timeout`` (horas-GPU) y misma semilla del
    sampler para todos los modelos comparables (presupuesto equitativo)."""
    import optuna

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner,
                                storage=storage, study_name=study_name, load_if_exists=bool(storage))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study


def importance_report(study: Any) -> dict[str, float]:
    """Importancia fANOVA de hiperparámetros (requiere >= ~20 trials completos)."""
    import optuna

    return dict(optuna.importance.get_param_importances(
        study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0)))


CNN1D_SPACE: dict[str, Any] = dict(
    learning_rate=[0.0005, 0.001, 0.003], batch_size=[32, 64], sequence_length=[128, 256, 512],
    embedding_dim=[100, 200], embedding=["skipgram", "fasttext", "random"],
)
# FastText se diseñó para millones de documentos (lr 0.1, 5 épocas); con ~1 700 documentos de entrenamiento hay
# pocas actualizaciones (una por documento y época), así que necesita lr alto y muchas épocas o queda sin entrenar
FASTTEXT_SPACE: dict[str, Any] = dict(lr=("log", 0.1, 1.0), epoch=[25, 50, 100], wordNgrams=[1, 2], dim=[50, 100])
TFIDF_LR_SPACE: dict[str, Any] = dict(C=("log", 0.05, 50.0), ngram=["1-1", "1-2"])
NLP_SPACES: dict[str, tuple[str, dict[str, Any]]] = {
    # nombre del modelo -> (clave de parámetros en el registro, espacio)
    "TF-IDF + LR": ("tfidf_lr", TFIDF_LR_SPACE),
    "TF-IDF + XGBoost": ("xgb", XGB_SPACE),
    "FastText": ("fasttext", FASTTEXT_SPACE),
    "CNN-1D": ("cnn1d", CNN1D_SPACE),
    "Word2Vec + BiLSTM": ("bilstm", BILSTM_SPACE),
}
