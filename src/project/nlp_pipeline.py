"""Interfaz común para los modelos de NLP: ``fit(train, y, val, y_val)`` y
``predict_proba(textos)``. Cada modelo declara su familia (para el
preprocesamiento por familia) y si se trata como ligero (validación anidada) o
pesado (HPO en un split + semillas)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .calibration import brier_score, expected_calibration_error
from .text_data import PREPROCESSORS


@dataclass
class TextModel:
    name: str
    family: str                                    # "classic" | "neural" | "transformer"
    builder: Callable[[dict[str, Any], int], Any]  # (params, seed) -> objeto con fit/predict_proba
    params: dict[str, Any] = field(default_factory=dict)
    light: bool = True
    preprocess: str = "standard"

    def fit(self, Xtr: Sequence[str], ytr: np.ndarray, Xva: Sequence[str], yva: np.ndarray, seed: int = 0) -> Any:
        pp = PREPROCESSORS[self.preprocess]
        self.m_ = self.builder(self.params, seed)
        tr, va = [pp(t) for t in Xtr], [pp(t) for t in Xva]
        if hasattr(self.m_, "fit_with_val"):
            self.m_.fit_with_val(tr, ytr, va, yva)
        else:
            self.m_.fit(tr, ytr)
        return self

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        pp = PREPROCESSORS[self.preprocess]
        return np.asarray(self.m_.predict_proba([pp(t) for t in X]))


# ------------------------------------------------------------- constructores
def _majority(p: dict[str, Any], seed: int) -> Any:
    from .models_text import majority_baseline
    return majority_baseline()


def _tfidf_lr(p: dict[str, Any], seed: int) -> Any:
    from .models_text import tfidf_logreg
    ng = p.get("ngram", (1, 2))
    ng = tuple(int(v) for v in ng.split("-")) if isinstance(ng, str) else tuple(ng)
    return tfidf_logreg(C=p.get("C", 4.0), ngram=ng)  # type: ignore[arg-type]


def _nbsvm(p: dict[str, Any], seed: int) -> Any:
    from .models_text import NBSVM
    return NBSVM(C=p.get("C", 4.0))


class _XGB:
    """TF-IDF + XGBoost con early stopping en validación (codificación de etiquetas propia)."""

    def __init__(self, p: dict[str, Any], seed: int):
        self.p, self.seed = p, seed

    def fit_with_val(self, tr: list[str], y: np.ndarray, va: list[str], yv: np.ndarray) -> _XGB:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from xgboost import XGBClassifier

        self.classes_ = np.unique(y); enc = {c: i for i, c in enumerate(self.classes_)}
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000, sublinear_tf=True)
        Xt = self.vec.fit_transform(tr); Xv = self.vec.transform(va)
        prm = dict(n_estimators=1000, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.5,
                   tree_method="hist", early_stopping_rounds=30, random_state=self.seed)
        try:                       # GPU si existe (≈10x más rápido con 24 clases); xgboost >= 2
            import torch

            if torch.cuda.is_available():
                prm["device"] = "cuda"
        except Exception:
            pass
        prm.update(self.p)
        self.m = XGBClassifier(**prm).fit(Xt, [enc[c] for c in y], eval_set=[(Xv, [enc.get(c, 0) for c in yv])],
                                          verbose=False)
        return self

    def predict_proba(self, X: list[str]) -> np.ndarray:
        return self.m.predict_proba(self.vec.transform(X))


class _KerasText:
    """BiLSTM o CNN-1D propios con Word2Vec/FastText entrenados solo con train."""

    def __init__(self, kind: str, p: dict[str, Any], seed: int):
        self.kind, self.p, self.seed = kind, p, seed

    def fit_with_val(self, tr: list[str], y: np.ndarray, va: list[str], yv: np.ndarray) -> _KerasText:
        from .config import set_global_seed
        from .models_text import (
            Vocabulary,
            build_bilstm,
            build_cnn1d,
            embedding_matrix,
            fit_keras_text,
            train_word_vectors,
        )

        set_global_seed(self.seed)
        p = self.p
        self.classes_ = np.unique(y); enc = {c: i for i, c in enumerate(self.classes_)}
        self.L = int(p.get("sequence_length", 100))
        self.vocab = Vocabulary(min_count=2).fit(tr)
        emb_kind, dim = p.get("embedding", "skipgram"), int(p.get("embedding_dim", 100))
        M = None
        if emb_kind != "random":
            try:
                wv = train_word_vectors(tr, emb_kind, dim, seed=self.seed)
                M, self.coverage = embedding_matrix(self.vocab, wv, dim, self.seed)
            except ImportError:      # gensim no disponible: se registra como desviación (embeddings aleatorios)
                import warnings

                warnings.warn("gensim no instalado: se usan embeddings aleatorios entrenables", stacklevel=2)
                self.embedding_fallback = True
        Xt, Xv = self.vocab.encode(tr, self.L), self.vocab.encode(va, self.L)
        yt, yvv = np.array([enc[c] for c in y]), np.array([enc.get(c, 0) for c in yv])
        if self.kind == "bilstm":
            self.model = build_bilstm(len(self.vocab), len(self.classes_), self.L, dim, int(p.get("lstm_units", 64)),
                                      int(p.get("num_lstm_layers", 1)), float(p.get("dropout_rate", 0.3)),
                                      p.get("pooling", "attention"), M, bool(p.get("trainable", True)),
                                      float(p.get("learning_rate", 1e-3)))
        else:
            self.model = build_cnn1d(len(self.vocab), len(self.classes_), self.L, dim, embedding_matrix=M,
                                     learning_rate=float(p.get("learning_rate", 1e-3)))
        self.hist = fit_keras_text(self.model, Xt, yt, Xv, yvv, int(p.get("batch_size", 32)),
                                   int(p.get("max_epochs", 20)), int(p.get("patience", 3)))
        return self

    def logits(self, X: list[str]) -> np.ndarray:
        return np.asarray(self.model.predict(self.vocab.encode(X, self.L), verbose=0))

    def predict_proba(self, X: list[str]) -> np.ndarray:
        from .calibration import softmax
        return softmax(self.logits(X))


class _FastText:
    def __init__(self, p: dict[str, Any], seed: int):
        self.p, self.seed = p, seed

    def fit(self, tr: list[str], y: np.ndarray) -> _FastText:
        from .transformer_text import fasttext_supervised
        self.m = fasttext_supervised(tr, y, seed=self.seed, **self.p)
        self.classes_ = self.m.classes_
        return self

    def predict_proba(self, X: list[str]) -> np.ndarray:
        return self.m.predict_proba(X)


class _Transformer:
    def __init__(self, p: dict[str, Any], seed: int):
        self.p, self.seed = p, seed

    def fit_with_val(self, tr: list[str], y: np.ndarray, va: list[str], yv: np.ndarray) -> _Transformer:
        from .transformer_text import TransformerConfig, fine_tune

        self.classes_ = np.unique(y); enc = {c: i for i, c in enumerate(self.classes_)}
        self.cfg = TransformerConfig(seed=self.seed, **self.p)
        self.res = fine_tune(tr, np.array([enc[c] for c in y]), va, np.array([enc.get(c, 0) for c in yv]),
                             len(self.classes_), self.cfg)
        return self

    def logits(self, X: list[str]) -> np.ndarray:
        import torch

        from .transformer_text import _encode
        m, tok = self.res["model"], self.res["tokenizer"]; dev = next(m.parameters()).device
        out = []
        m.eval()
        with torch.no_grad():
            for i in range(0, len(X), 64):
                enc = {k: v.to(dev) for k, v in _encode(tok, X[i:i + 64], self.cfg).items()}
                out.append(m(**enc).logits.float().cpu().numpy())
        return np.concatenate(out)

    def predict_proba(self, X: list[str]) -> np.ndarray:
        from .calibration import softmax
        return softmax(self.logits(X))


def registry(cfg_models: dict[str, Any] | None = None) -> dict[str, TextModel]:
    c = cfg_models or {}
    return {
        "Clase mayoritaria": TextModel("Clase mayoritaria", "classic", _majority),
        "TF-IDF + LR": TextModel("TF-IDF + LR", "classic", _tfidf_lr, c.get("tfidf_lr", {})),
        "NB-SVM": TextModel("NB-SVM", "classic", _nbsvm, c.get("nbsvm", {})),
        "TF-IDF + XGBoost": TextModel("TF-IDF + XGBoost", "classic", lambda p, s: _XGB(p, s), c.get("xgb", {})),
        "FastText": TextModel("FastText", "classic", lambda p, s: _FastText(p, s), c.get("fasttext", {}), preprocess="minimal"),
        "CNN-1D": TextModel("CNN-1D", "neural", lambda p, s: _KerasText("cnn1d", p, s), c.get("cnn1d", {})),
        "Word2Vec + BiLSTM": TextModel("Word2Vec + BiLSTM", "neural", lambda p, s: _KerasText("bilstm", p, s),
                                       c.get("bilstm", {})),
        "DistilBERT": TextModel("DistilBERT", "transformer", lambda p, s: _Transformer(p, s), c.get("transformer", {}),
                                light=False, preprocess="minimal"),
    }


# ------------------------------------------------------------------ métricas
def classification_metrics(y: np.ndarray, proba: np.ndarray, classes: Sequence[Any]) -> dict[str, float]:
    """Métricas del enunciado. Multiclase: ROC-AUC y PR-AUC uno-contra-resto (OvR, macro)."""
    cls = np.asarray(classes); yi = np.searchsorted(cls, y)
    pred = proba.argmax(1)
    out = dict(accuracy=accuracy_score(yi, pred), precision_macro=precision_score(yi, pred, average="macro", zero_division=0),
               recall_macro=recall_score(yi, pred, average="macro", zero_division=0),
               f1_macro=f1_score(yi, pred, average="macro"), f1_micro=f1_score(yi, pred, average="micro"),
               f1_weighted=f1_score(yi, pred, average="weighted"), mcc=matthews_corrcoef(yi, pred),
               balanced_accuracy=balanced_accuracy_score(yi, pred),
               log_loss=log_loss(yi, np.clip(proba, 1e-12, 1), labels=np.arange(len(cls))),
               brier=brier_score(proba, yi), ece=expected_calibration_error(proba, yi))
    try:
        if len(cls) == 2:
            out["roc_auc"] = roc_auc_score(yi, proba[:, 1]); out["pr_auc"] = average_precision_score(yi, proba[:, 1])
        else:
            oh = np.eye(len(cls))[yi]
            out["roc_auc"] = roc_auc_score(oh, proba, average="macro", multi_class="ovr")
            out["pr_auc"] = average_precision_score(oh, proba, average="macro")
    except ValueError:
        out["roc_auc"] = out["pr_auc"] = np.nan
    return {k: float(v) for k, v in out.items()}
