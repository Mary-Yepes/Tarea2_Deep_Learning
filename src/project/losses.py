"""Funciones de pérdida (implementación propia).

* Referencias en NumPy (con gradientes analíticos) usadas en los fundamentos
  teóricos y en las pruebas unitarias.
* Versiones Keras/TensorFlow (importación perezosa) para entrenar U-Net++.
"""

from __future__ import annotations

from typing import Any

import numpy as np

EPS = 1.0


# ------------------------------------------------------------------ NumPy
def soft_dice_loss(p: np.ndarray, g: np.ndarray, eps: float = EPS) -> float:
    """L = 1 − (2 Σ p g + ε) / (Σ p + Σ g + ε)."""
    p, g = np.asarray(p, float).ravel(), np.asarray(g, float).ravel()
    return float(1 - (2 * (p * g).sum() + eps) / (p.sum() + g.sum() + eps))


def soft_dice_grad(p: np.ndarray, g: np.ndarray, eps: float = EPS) -> np.ndarray:
    """∂L/∂p_i = −[2 g_i (S_p + S_g + ε) − (2 I + ε)] / (S_p + S_g + ε)²."""
    p, g = np.asarray(p, float), np.asarray(g, float)
    inter, s = (p * g).sum(), p.sum() + g.sum() + eps
    return -(2 * g * s - (2 * inter + eps)) / s ** 2


def bce_loss(p: np.ndarray, g: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(np.asarray(p, float), eps, 1 - eps); g = np.asarray(g, float)
    return float(-(g * np.log(p) + (1 - g) * np.log(1 - p)).mean())


def bce_grad(p: np.ndarray, g: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(np.asarray(p, float), eps, 1 - eps); g = np.asarray(g, float)
    return (-(g / p) + (1 - g) / (1 - p)) / p.size


def focal_loss(p: np.ndarray, g: np.ndarray, gamma: float = 2.0, alpha: float = 0.25, eps: float = 1e-7) -> float:
    """Focal (Lin et al., 2017): −α_t (1 − p_t)^γ log p_t."""
    p = np.clip(np.asarray(p, float), eps, 1 - eps); g = np.asarray(g, float)
    pt = np.where(g == 1, p, 1 - p); at = np.where(g == 1, alpha, 1 - alpha)
    return float((-at * (1 - pt) ** gamma * np.log(pt)).mean())


def tversky_loss(p: np.ndarray, g: np.ndarray, a: float = 0.3, b: float = 0.7, eps: float = EPS) -> float:
    """Tversky (Salehi et al., 2017): b > a penaliza más los FN (útil con objetos pequeños)."""
    p, g = np.asarray(p, float).ravel(), np.asarray(g, float).ravel()
    tp = (p * g).sum(); fp = (p * (1 - g)).sum(); fn = ((1 - p) * g).sum()
    return float(1 - (tp + eps) / (tp + a * fp + b * fn + eps))


# ------------------------------------------------------------------ Keras
def keras_losses() -> dict[str, Any]:
    """Diccionario de pérdidas Keras para salidas softmax (B, H, W, C) y
    objetivos one-hot. Se usa en la ablación de pérdidas (BCE/CE, Dice, CE+Dice,
    Tversky, focal)."""
    import tensorflow as tf

    def _flat(y_true: Any, y_pred: Any) -> tuple[Any, Any]:
        return tf.cast(y_true, y_pred.dtype), y_pred

    def ce(y_true: Any, y_pred: Any) -> Any:
        y_true, y_pred = _flat(y_true, y_pred)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        return -tf.reduce_mean(tf.reduce_sum(y_true * tf.math.log(y_pred), -1))

    def dice(y_true: Any, y_pred: Any, eps: float = EPS) -> Any:
        y_true, y_pred = _flat(y_true, y_pred)
        ax = (0, 1, 2)  # Dice por clase sobre el lote, luego media de clases de primer plano
        inter = tf.reduce_sum(y_true * y_pred, ax)
        den = tf.reduce_sum(y_true, ax) + tf.reduce_sum(y_pred, ax)
        d = (2 * inter + eps) / (den + eps)
        return 1 - tf.reduce_mean(d[1:])

    def tversky(y_true: Any, y_pred: Any, a: float = 0.3, b: float = 0.7, eps: float = EPS) -> Any:
        y_true, y_pred = _flat(y_true, y_pred)
        ax = (0, 1, 2)
        tp = tf.reduce_sum(y_true * y_pred, ax)
        fp = tf.reduce_sum((1 - y_true) * y_pred, ax)
        fn = tf.reduce_sum(y_true * (1 - y_pred), ax)
        t = (tp + eps) / (tp + a * fp + b * fn + eps)
        return 1 - tf.reduce_mean(t[1:])

    def focal(y_true: Any, y_pred: Any, gamma: float = 2.0) -> Any:
        y_true, y_pred = _flat(y_true, y_pred)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        return -tf.reduce_mean(tf.reduce_sum(y_true * (1 - y_pred) ** gamma * tf.math.log(y_pred), -1))

    def ce_dice(y_true: Any, y_pred: Any) -> Any:
        return ce(y_true, y_pred) + dice(y_true, y_pred)

    return {"bce": ce, "dice": dice, "bce_dice": ce_dice, "tversky": tversky, "focal": focal}
