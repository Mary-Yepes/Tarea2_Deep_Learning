"""U-Net++ implementada desde cero en Keras (Zhou et al., 2018) y su generador de datos.

* Nodos ``X[i][j]``: ``i`` = profundidad (0..L), ``j`` = columna de la ruta
  anidada. ``X[i][0]`` es el codificador; ``X[i][j] = H([X[i][0..j-1], U(X[i+1][j-1])])``.
* Supervisión profunda: una cabeza 1×1 + softmax en cada ``X[0][j]``, j = 1..L.
  En entrenamiento la pérdida es la media de las cabezas; en inferencia se usa
  la última (modo «preciso») o cualquiera anterior (modo «rápido», poda).
* Objetivo de 3 clases (fondo / interior / borde-contacto) + watershed.
* ``dropout`` > 0 activa *SpatialDropout* en los bloques para MC-Dropout.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .preprocess_seg import augment_pair, norm_percentile, three_class_target


def conv_block(x: Any, filters: int, name: str, dropout: float = 0.0) -> Any:
    from tensorflow import keras
    L = keras.layers
    for k in range(2):
        x = L.Conv2D(filters, 3, padding="same", use_bias=False, kernel_initializer="he_normal",
                     name=f"{name}_conv{k}")(x)
        x = L.BatchNormalization(name=f"{name}_bn{k}")(x)
        x = L.Activation("relu", name=f"{name}_relu{k}")(x)
    if dropout > 0:
        x = L.SpatialDropout2D(dropout, name=f"{name}_drop")(x)
    return x


RESNET50_TAPS = ("conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out")


def build_unetpp(input_shape: tuple[int | None, int | None, int] = (None, None, 1), n_classes: int = 3,
                 base_filters: int = 32, depth: int = 4, deep_supervision: bool = True,
                 dropout: float = 0.0, encoder: str = "scratch", freeze_encoder: bool = False) -> Any:
    """Construye U-Net++ (profundidad ``depth`` => entrada múltiplo de 2**depth).

    ``encoder``:
      * ``"scratch"``: codificador propio inicializado aleatoriamente (ablación «desde cero»).
      * ``"resnet50_imagenet"``: los nodos ``X[i][0]`` (i >= 1) se toman de ResNet50
        preentrenada en ImageNet (strides 2..32; fuerza ``depth = 5``). La imagen en
        gris se replica a 3 canales. Ablación «ImageNet».
      * Preentrenamiento en microscopía: entrenar primero sobre LIVECell y luego
        cargar esos pesos (``model.load_weights``) antes del ajuste fino.
    """
    from tensorflow import keras
    L = keras.layers
    inp = keras.Input(input_shape, name="image")
    if encoder == "resnet50_imagenet":
        depth = 5
        rgb = L.Concatenate(name="gray2rgb")([inp, inp, inp]) if input_shape[-1] == 1 else inp
        x255 = L.Rescaling(255.0, name="to255")(rgb)
        pre = keras.applications.resnet50.preprocess_input(x255)
        bb = keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=pre)
        bb.trainable = not freeze_encoder
        taps = [bb.get_layer(n).output for n in RESNET50_TAPS]
    elif encoder != "scratch":
        raise ValueError(encoder)
    f = [base_filters * 2 ** i for i in range(depth + 1)]
    X: list[list[Any]] = [[None] * (depth + 1) for _ in range(depth + 1)]
    for i in range(depth + 1):
        if encoder == "scratch":
            src = inp if i == 0 else L.MaxPooling2D(2, name=f"pool{i}")(X[i - 1][0])
            X[i][0] = conv_block(src, f[i], f"X{i}0", dropout if i >= depth - 1 else 0.0)
        else:
            X[i][0] = (conv_block(inp, f[0], "X00") if i == 0 else
                       L.Conv2D(f[i], 1, name=f"X{i}0_proj")(taps[i - 1]))
    for j in range(1, depth + 1):
        for i in range(depth + 1 - j):
            up = L.Conv2DTranspose(f[i], 2, strides=2, padding="same", name=f"up{i + 1}{j - 1}_to{i}{j}")(X[i + 1][j - 1])
            cat = L.Concatenate(name=f"cat{i}{j}")([*X[i][:j], up])
            X[i][j] = conv_block(cat, f[i], f"X{i}{j}", dropout if i >= depth - 1 else 0.0)
    heads = [L.Conv2D(n_classes, 1, activation="softmax", name=f"out{j}")(X[0][j])
             for j in range(1, depth + 1)]
    outputs = heads if deep_supervision else heads[-1]
    return keras.Model(inp, outputs, name="unetpp")


def compile_unetpp(model: Any, loss: str = "bce_dice", lr: float = 1e-3, epochs: int = 50,
                   steps_per_epoch: int = 100, schedule: str = "cosine", optimizer: str = "adam") -> Any:
    from tensorflow import keras

    from .losses import keras_losses

    if schedule == "cosine":
        warm = 2 * steps_per_epoch  # warmup lineal de 2 épocas y luego coseno
        lr_s: Any = keras.optimizers.schedules.CosineDecay(
            lr * 0.01, decay_steps=max(epochs * steps_per_epoch - warm, 1),
            warmup_target=lr, warmup_steps=warm)
    else:
        lr_s = lr
    opt = (keras.optimizers.AdamW(lr_s, weight_decay=1e-4) if optimizer == "adamw"
           else keras.optimizers.Adam(lr_s))
    loss_fn = keras_losses()[loss]
    n_out = len(model.outputs)
    model.compile(optimizer=opt, loss=[loss_fn] * n_out, loss_weights=[1.0 / n_out] * n_out)
    return model


def make_generator(images: Sequence[np.ndarray], labels: Sequence[np.ndarray], batch_size: int = 4,
                   crop: int = 256, augment: str = "geom+photo+elastic", n_outputs: int = 4,
                   normalize: Any = norm_percentile, seed: int = 0, shuffle: bool = True,
                   target: str = "three_class") -> Any:
    """Generador ``keras.utils.PyDataset`` con recorte aleatorio, augmentation y
    objetivo por modelo: ``"three_class"`` (U-Net++) o ``"hovernet"`` (NP one-hot +
    mapas HV con la máscara de primer plano como tercer canal). El aumento se
    aplica con la misma transformación a imagen y máscara (vecino más próximo en la
    máscara) y los objetivos se calculan **después** del aumento."""
    from tensorflow import keras

    from .preprocess_seg import hv_target

    Base = getattr(keras.utils, "PyDataset", None) or keras.utils.Sequence   # Keras 3 / Keras 2

    class SegSequence(Base):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            self.rng = np.random.default_rng(seed)
            self.idx = np.arange(len(images))
            self.on_epoch_end()

        def __len__(self) -> int:
            return int(np.ceil(len(images) / batch_size))

        def on_epoch_end(self) -> None:
            if shuffle:
                self.rng.shuffle(self.idx)

        def __getitem__(self, b: int) -> tuple[np.ndarray, Any]:
            xs, ys = [], []
            for i in self.idx[b * batch_size:(b + 1) * batch_size]:
                img, lbl = normalize(images[i]), labels[i]
                h, w = img.shape
                y0 = int(self.rng.integers(0, max(h - crop, 0) + 1)); x0 = int(self.rng.integers(0, max(w - crop, 0) + 1))
                img, lbl = img[y0:y0 + crop, x0:x0 + crop], lbl[y0:y0 + crop, x0:x0 + crop]
                img, lbl = augment_pair(img, lbl, self.rng, augment)
                pad_h, pad_w = crop - img.shape[0], crop - img.shape[1]
                if pad_h or pad_w:
                    img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")
                    lbl = np.pad(lbl, ((0, pad_h), (0, pad_w)))
                xs.append(img[..., None])
                if target == "hovernet":
                    fg = (lbl > 0).astype(np.float32)
                    ys.append((np.stack([1 - fg, fg], -1), np.dstack([hv_target(lbl), fg])))
                else:
                    ys.append(np.eye(3, dtype=np.float32)[three_class_target(lbl)])
            x = np.stack(xs).astype(np.float32)
            if target == "hovernet":
                return x, (np.stack([a for a, _ in ys]).astype(np.float32), np.stack([b for _, b in ys]).astype(np.float32))
            y = np.stack(ys)
            return x, (tuple([y] * n_outputs) if n_outputs > 1 else y)

    return SegSequence()


def predict_probs(model: Any, img: np.ndarray, normalize: Any = norm_percentile, head: int = -1,
                  mc_samples: int = 0) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Probabilidades (H, W, 3) en resolución nativa (padding a múltiplos de 32).

    Con ``mc_samples`` > 0 ejecuta MC-Dropout (``training=True``) y devuelve
    (media, varianza)."""
    from .preprocess_seg import pad_to_multiple, unpad

    x, hw = pad_to_multiple(normalize(img), 32)
    x = x[None, ..., None].astype(np.float32)

    def once(training: bool) -> np.ndarray:
        out = model(x, training=training)
        out = out[head] if isinstance(out, (list, tuple)) else out
        return unpad(np.asarray(out)[0], hw)

    if mc_samples <= 0:
        return once(False)
    s = np.stack([once(True) for _ in range(mc_samples)])
    return s.mean(0), s.var(0)


def unetpp_complexity(h: int, w: int, base_filters: int = 32, depth: int = 4, c_in: int = 1) -> dict[str, float]:
    """MACs y memoria de activaciones analíticas (convoluciones 3×3 dominantes).

    Cada nodo ``X[i][j]`` opera a resolución (h/2^i)·(w/2^i) con ``f_i`` filtros;
    su primera convolución recibe ``j·f_i + f_i`` canales (``j`` atajos + la
    entrada sobremuestreada; ``f_{i-1}`` para el codificador). Tiempo y memoria
    son Θ(H·W) para arquitectura fija: la suma geométrica sobre niveles converge.
    """
    f = [base_filters * 2 ** i for i in range(depth + 1)]
    macs = 0.0; act = 0.0
    for i in range(depth + 1):
        for j in range(depth + 1 - i):
            hw = (h / 2 ** i) * (w / 2 ** i)
            cin = (c_in if i == 0 else f[i - 1]) if j == 0 else (j + 1) * f[i]
            macs += hw * 9 * (cin * f[i] + f[i] * f[i])
            act += hw * 2 * f[i]
    return {"macs": macs, "activations": act}


# ===================================================================== Hover-Net
def build_hovernet(input_shape: tuple[int | None, int | None, int] = (None, None, 1), base_filters: int = 32,
                   depth: int = 4, dropout: float = 0.0) -> Any:
    """Hover-Net (Graham et al., 2019) **reimplementada** en Keras: codificador
    compartido y dos decodificadores con atajos, rama **NP** (primer plano, softmax
    de 2 clases) y rama **HV** (distancias horizontales/verticales normalizadas al
    centro de masa, ``tanh``). Sin rama de tipo nuclear (no hay etiquetas de tipo).

    Desviación declarada respecto al original: codificador tipo U-Net en lugar de
    ResNet-50 preactivada, para entrenar con el mismo generador y presupuesto que
    U-Net++; el post-procesamiento HV es el oficial (``adapters_seg.hovernet_postprocess``).
    """
    from tensorflow import keras
    L = keras.layers
    f = [base_filters * 2 ** i for i in range(depth + 1)]
    inp = keras.Input(input_shape, name="image")
    skips, x = [], inp
    for i in range(depth + 1):
        x = conv_block(x, f[i], f"enc{i}", dropout if i >= depth - 1 else 0.0)
        if i < depth:
            skips.append(x); x = L.MaxPooling2D(2, name=f"hpool{i}")(x)
    bottleneck = x

    def decoder(name: str) -> Any:
        y = bottleneck
        for i in range(depth - 1, -1, -1):
            y = L.Conv2DTranspose(f[i], 2, strides=2, padding="same", name=f"{name}_up{i}")(y)
            y = conv_block(L.Concatenate(name=f"{name}_cat{i}")([y, skips[i]]), f[i], f"{name}_dec{i}")
        return y

    np_out = L.Conv2D(2, 1, activation="softmax", name="np")(decoder("np"))
    hv_out = L.Conv2D(2, 1, activation="tanh", name="hv")(decoder("hv"))
    return keras.Model(inp, [np_out, hv_out], name="hovernet")


def hovernet_losses() -> tuple[Any, Any]:
    """NP: CE + Dice. HV: MSE + MSGE (error cuadrático del gradiente Sobel de H en x
    y de V en y, **solo dentro del primer plano**), como en el artículo original."""
    import tensorflow as tf

    from .losses import keras_losses

    ce_dice = keras_losses()["bce_dice"]

    def hv_loss(y_true: Any, y_pred: Any) -> Any:
        hv_t, fg = y_true[..., :2], y_true[..., 2:3]
        mse = tf.reduce_mean(tf.square(y_pred - hv_t))
        gt = tf.image.sobel_edges(hv_t); gp = tf.image.sobel_edges(y_pred)   # (B,H,W,2,[dy,dx])
        dh = gp[..., 0, 1] - gt[..., 0, 1]                                   # ∂H/∂x
        dv = gp[..., 1, 0] - gt[..., 1, 0]                                   # ∂V/∂y
        m = fg[..., 0]
        msge = (tf.reduce_sum(m * tf.square(dh)) + tf.reduce_sum(m * tf.square(dv))) / (tf.reduce_sum(m) + 1e-6)
        return mse + msge

    return ce_dice, hv_loss


def compile_hovernet(model: Any, lr: float = 1e-3, epochs: int = 50, steps_per_epoch: int = 100) -> Any:
    from tensorflow import keras

    np_loss, hv_loss = hovernet_losses()
    warm = 2 * steps_per_epoch
    sched = keras.optimizers.schedules.CosineDecay(lr * 0.01, max(epochs * steps_per_epoch - warm, 1),
                                                   warmup_target=lr, warmup_steps=warm)
    model.compile(optimizer=keras.optimizers.Adam(sched), loss=[np_loss, hv_loss], loss_weights=[1.0, 2.0])
    return model


def hovernet_infer_fn(model: Any, normalize: Any = norm_percentile) -> Any:
    """Devuelve ``f(img) -> (p_primer_plano (H, W), hv (H, W, 2))`` para ``HoverNetAdapter``."""
    from .preprocess_seg import pad_to_multiple, unpad

    def f(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, hw = pad_to_multiple(normalize(img), 32)
        npp, hv = model(x[None, ..., None].astype(np.float32), training=False)
        return unpad(np.asarray(npp)[0, ..., 1], hw), unpad(np.asarray(hv)[0], hw)

    return f


# ============================================================ persistencia robusta
def build_from_config(kind: str, cfg: dict[str, Any]) -> Any:
    """Reconstruye la arquitectura desde la configuración guardada en el JSON del modelo.
    Se guardan **solo los pesos** (``*.weights.h5``): evita problemas de serialización de
    pérdidas personalizadas entre versiones de Keras."""
    if kind == "hovernet":
        return build_hovernet((None, None, 1), base_filters=cfg.get("base_filters", 32), dropout=cfg.get("dropout", 0.0))
    return build_unetpp((None, None, 1), base_filters=cfg.get("base_filters", 32), depth=cfg.get("depth", 4),
                        deep_supervision=cfg.get("deep_supervision", True), dropout=cfg.get("dropout", 0.0),
                        encoder=cfg.get("encoder", "scratch"))


def load_seg_model(weights_path: str, meta: dict[str, Any]) -> Any:
    cfg = dict(meta["config"])
    model = build_from_config(meta.get("kind", "unetpp"), cfg)
    model.load_weights(weights_path)
    return model
