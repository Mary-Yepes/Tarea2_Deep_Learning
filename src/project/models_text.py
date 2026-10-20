"""Modelos de clasificación de texto.

* Baselines: clase mayoritaria, TF-IDF + Regresión Logística, NB-SVM (Wang &
  Manning, 2012) implementado aquí.
* TF-IDF + XGBoost (importación perezosa).
* Implementación propia en Keras: Word2Vec/FastText + **BiLSTM** (pooling último
  estado / max / atención, máscara de padding) y **CNN-1D** multi-escala (Kim, 2014).
* Vocabulario y vectorizador ajustados **solo con train**.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from .text_data import tokenize

PAD, UNK = "<PAD>", "<UNK>"


# --------------------------------------------------------------- baselines
def majority_baseline() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")


def tfidf_logreg(C: float = 4.0, ngram: tuple[int, int] = (1, 2), min_df: int = 2,
                 max_features: int | None = 200_000, class_weight: str | None = "balanced") -> Pipeline:
    return make_pipeline(
        TfidfVectorizer(ngram_range=ngram, min_df=min_df, max_features=max_features, sublinear_tf=True),
        LogisticRegression(C=C, max_iter=2000, class_weight=class_weight))


class NBSVM(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """NB-SVM (Wang & Manning, 2012) uno-contra-resto: escala las características
    binarias por la razón log-count de Naive Bayes y ajusta un clasificador lineal.
    Se usa regresión logística (variante NB-LR) para obtener probabilidades."""

    def __init__(self, C: float = 4.0, alpha: float = 1.0, ngram: tuple[int, int] = (1, 2), min_df: int = 2):
        self.C, self.alpha, self.ngram, self.min_df = C, alpha, ngram, min_df

    def fit(self, X: Sequence[str], y: Sequence[Any]) -> NBSVM:
        self.vec_ = TfidfVectorizer(ngram_range=self.ngram, min_df=self.min_df, binary=True,
                                    use_idf=False, norm=None)
        F = self.vec_.fit_transform(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.r_, self.clf_ = [], []
        for c in self.classes_:
            pos = F[y == c].sum(0) + self.alpha; neg = F[y != c].sum(0) + self.alpha
            r = np.asarray(np.log((pos / pos.sum()) / (neg / neg.sum()))).ravel()
            clf = LogisticRegression(C=self.C, max_iter=2000).fit(F.multiply(r).tocsr(), (y == c).astype(int))
            self.r_.append(r); self.clf_.append(clf)
        return self

    def decision_function(self, X: Sequence[str]) -> np.ndarray:
        F = self.vec_.transform(X)
        return np.column_stack([clf.decision_function(F.multiply(r).tocsr()) for r, clf in zip(self.r_, self.clf_)])

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        z = self.decision_function(X)
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict(self, X: Sequence[str]) -> np.ndarray:
        return self.classes_[self.decision_function(X).argmax(1)]


def tfidf_xgboost(params: dict[str, Any] | None = None, n_classes: int = 2, max_features: int = 50_000) -> Pipeline:
    """TF-IDF + XGBoost; early stopping se aplica pasando ``eval_set`` (validación)."""
    from xgboost import XGBClassifier

    p = dict(n_estimators=600, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.5,
             tree_method="hist", eval_metric="mlogloss" if n_classes > 2 else "logloss",
             early_stopping_rounds=30)
    p.update(params or {})
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features,
                                         sublinear_tf=True), XGBClassifier(**p))


# --------------------------------------------------------- vocabulario
class Vocabulary:
    """Vocabulario ajustado solo con train. Índice 0 = PAD, 1 = UNK."""

    def __init__(self, min_count: int = 2, max_size: int | None = 50_000):
        self.min_count, self.max_size = min_count, max_size
        self.itos: list[str] = [PAD, UNK]
        self.stoi: dict[str, int] = {PAD: 0, UNK: 1}

    def fit(self, texts: Sequence[str]) -> Vocabulary:
        c = Counter(w for t in texts for w in tokenize(t))
        words = [w for w, k in c.most_common(self.max_size) if k >= self.min_count]
        self.itos = [PAD, UNK] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        return self

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, texts: Sequence[str], max_len: int, truncation: str = "end") -> np.ndarray:
        """Índices con padding a la derecha. ``truncation``: "end" (conserva inicio),
        "start" (conserva final) o "head_tail"."""
        from .text_data import head_tail_truncate

        out = np.zeros((len(texts), max_len), np.int32)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            if len(toks) > max_len:
                toks = (toks[:max_len] if truncation == "end" else toks[-max_len:] if truncation == "start"
                        else head_tail_truncate(toks, max_len))
            ids = [self.stoi.get(w, 1) for w in toks]
            out[i, :len(ids)] = ids
        return out

    def unk_rate(self, texts: Sequence[str]) -> float:
        toks = [w for t in texts for w in tokenize(t)]
        return float(np.mean([w not in self.stoi for w in toks])) if toks else 0.0


def train_word_vectors(train_texts: Sequence[str], kind: str = "skipgram", dim: int = 100, window: int = 5,
                       min_count: int = 2, epochs: int = 10, seed: int = 0) -> Any:
    """Word2Vec (CBOW/Skip-gram) o FastText **solo con textos de train** (gensim)."""
    from gensim.models import FastText, Word2Vec

    sents = [tokenize(t) for t in train_texts]
    common = dict(vector_size=dim, window=window, min_count=min_count, epochs=epochs, seed=seed, workers=1)
    if kind == "fasttext":
        return FastText(sents, sg=1, min_n=3, max_n=6, **common)
    return Word2Vec(sents, sg=1 if kind == "skipgram" else 0, **common)


def embedding_matrix(vocab: Vocabulary, wv: Any, dim: int, seed: int = 0) -> tuple[np.ndarray, float]:
    """Matriz de embeddings alineada con el vocabulario y su cobertura.

    Word2Vec: palabras sin vector -> aleatorio N(0, 0.1) (equivalente a <UNK>).
    FastText: construye vectores OOV a partir de n-gramas de caracteres."""
    rng = np.random.default_rng(seed)
    M = rng.normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
    M[0] = 0
    hit = 0
    kv = getattr(wv, "wv", wv)
    is_ft = hasattr(kv, "buckets") or kv.__class__.__name__ == "FastTextKeyedVectors"
    for i, w in enumerate(vocab.itos[2:], start=2):
        if w in kv.key_to_index or is_ft:
            M[i] = kv[w]; hit += w in kv.key_to_index
    return M, hit / max(len(vocab) - 2, 1)


# ------------------------------------------------------------- Keras
def build_bilstm(vocab_size: int, n_classes: int, seq_len: int = 100, embedding_dim: int = 100,
                 lstm_units: int = 64, num_lstm_layers: int = 1, dropout_rate: float = 0.3,
                 pooling: str = "attention", embedding_matrix: np.ndarray | None = None,
                 trainable_embeddings: bool = True, learning_rate: float = 1e-3) -> Any:
    """BiLSTM propia. ``pooling`` en {"last", "max", "attention"}; la máscara de
    padding se aplica a las LSTM, al max-pooling y a la atención."""
    import tensorflow as tf
    from tensorflow import keras
    L = keras.layers

    inp = keras.Input((seq_len,), dtype="int32", name="tokens")
    emb = L.Embedding(vocab_size, embedding_dim, name="emb",
                      embeddings_initializer=(keras.initializers.Constant(embedding_matrix)
                                              if embedding_matrix is not None else "uniform"),
                      trainable=trainable_embeddings)(inp)
    # Máscara de padding explícita (índice 0 = PAD): se pasa a las LSTM (el
    # «último estado» es el del último token real) y se usa en max/atención.
    bool_mask = L.Lambda(lambda t: tf.not_equal(t, 0), name="pad_bool")(inp)
    mask = L.Lambda(lambda t: tf.cast(tf.not_equal(t, 0), tf.float32), name="pad_mask")(inp)
    x = L.SpatialDropout1D(dropout_rate)(emb)
    for k in range(num_lstm_layers):
        last = k == num_lstm_layers - 1
        x = L.Bidirectional(L.LSTM(lstm_units, return_sequences=not (last and pooling == "last"),
                                   dropout=dropout_rate), name=f"bilstm{k}")(x, mask=bool_mask)
    if pooling == "max":
        x = L.Lambda(lambda z: tf.reduce_max(z[0] + (z[1][..., None] - 1.0) * 1e9, axis=1),
                     name="masked_max")([x, mask])
    elif pooling == "attention":
        s = L.Dense(1, activation="tanh", name="att_score")(x)
        s = L.Lambda(lambda z: z[0][..., 0] + (z[1] - 1.0) * 1e9, name="att_mask")([s, mask])
        a = L.Softmax(name="att_weights")(s)
        x = L.Lambda(lambda z: tf.reduce_sum(z[0] * z[1][..., None], axis=1), name="att_pool")([x, a])
    x = L.Dropout(dropout_rate)(x)
    out = L.Dense(n_classes, name="logits")(x)
    model = keras.Model(inp, out, name=f"bilstm_{pooling}")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    return model


def build_cnn1d(vocab_size: int, n_classes: int, seq_len: int = 100, embedding_dim: int = 100,
                kernel_sizes: Sequence[int] = (2, 3, 4, 5), filters: int = 100, dropout_rate: float = 0.5,
                embedding_matrix: np.ndarray | None = None, learning_rate: float = 1e-3) -> Any:
    """CNN-1D de Kim (2014): filtros multi-escala en paralelo + global max pooling."""
    from tensorflow import keras
    L = keras.layers

    inp = keras.Input((seq_len,), dtype="int32", name="tokens")
    emb = L.Embedding(vocab_size, embedding_dim, name="emb",
                      embeddings_initializer=(keras.initializers.Constant(embedding_matrix)
                                              if embedding_matrix is not None else "uniform"))(inp)
    branches = [L.GlobalMaxPooling1D()(L.Conv1D(filters, k, activation="relu", padding="valid",
                                                name=f"conv_k{k}")(emb)) for k in kernel_sizes]
    x = L.Dropout(dropout_rate)(L.Concatenate()(branches))
    out = L.Dense(n_classes, name="logits")(x)
    model = keras.Model(inp, out, name="cnn1d_kim")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    return model


def fit_keras_text(model: Any, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray,
                   batch_size: int = 32, max_epochs: int = 50, patience: int = 3, verbose: int = 0) -> Any:
    """Entrena con early stopping sobre validación (paciencia declarada en el pre-registro)."""
    from tensorflow import keras

    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)]
    return model.fit(Xtr, ytr, validation_data=(Xva, yva), batch_size=batch_size, epochs=max_epochs,
                     callbacks=cb, verbose=verbose)


# --------------------------------------------------------------- complejidad
def bilstm_complexity(seq_len: int, emb: int, units: int, layers: int = 1) -> dict[str, float]:
    """MACs: por paso y dirección 4·u·(d + u); total Θ(T·u·(d+u)); memoria de
    activaciones para BPTT Θ(T·u). Secuencial en T (no paralelizable en el tiempo)."""
    macs, d = 0.0, emb
    for _ in range(layers):
        macs += 2 * seq_len * 4 * units * (d + units)
        d = 2 * units
    return {"macs": macs, "activations": 2 * layers * seq_len * units * 6}


def attention_complexity(seq_len: int, d_model: int) -> dict[str, float]:
    """Auto-atención de una cabeza: proyecciones Θ(T·d²) + QKᵀ y AV Θ(T²·d);
    memoria de la matriz de atención Θ(T²)."""
    return {"macs": 4 * seq_len * d_model ** 2 + 2 * seq_len ** 2 * d_model,
            "activations": seq_len ** 2 + 4 * seq_len * d_model}
