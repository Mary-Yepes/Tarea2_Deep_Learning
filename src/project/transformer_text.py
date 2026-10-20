"""Ajuste fino de DistilBERT/RoBERTa (PyTorch + Hugging Face ``transformers``).

Receta del enunciado: *warmup* lineal + decaimiento lineal, decaimiento de LR por
capa (LLRD), precisión mixta, >= 5 semillas, opcional LoRA (``peft``). El
preprocesamiento es **mínimo** (el tokenizer BPE/WordPiece maneja mayúsculas y
puntuación). Devuelve *logits* de validación y prueba para calibración (H5).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TransformerConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 256              # justificado por el percentil del EDA (p95 de tokens)
    truncation: str = "head"           # "head" | "head_tail"
    lr: float = 3e-5
    llrd: float = 0.9                  # factor multiplicativo por capa hacia abajo
    head_lr_mult: float = 20.0         # la cabeza (inicializada al azar) usa lr·20: con lr=3e-5 no aprende en pocas épocas
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    epochs: int = 4
    batch_size: int = 16
    patience: int = 2
    fp16: bool = True
    lora_r: int = 0                    # > 0 activa LoRA (peft)
    seed: int = 0


def _layer_groups(model: Any, cfg: TransformerConfig) -> list[dict[str, Any]]:
    """Grupos de parámetros con LLRD: clasificador con lr; capa L-1 con lr·llrd; ...;
    embeddings con lr·llrd^L. Sin weight decay en sesgos y LayerNorm."""
    import re

    n_layers: int = int(getattr(model.config, "num_hidden_layers", 0) or getattr(model.config, "n_layers", 6))
    groups: dict[tuple[float, float], list[Any]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        m = re.search(r"(?:layer|layers)\.(\d+)\.", name)
        if m:
            depth = n_layers - int(m.group(1))
        elif "embeddings" in name:
            depth = n_layers + 1
        else:
            depth = 0                                  # cabeza / pooler
        lr = cfg.lr * cfg.llrd ** depth * (cfg.head_lr_mult if depth == 0 else 1.0)
        wd = 0.0 if (name.endswith("bias") or "LayerNorm" in name or "layer_norm" in name) else cfg.weight_decay
        groups.setdefault((lr, wd), []).append(p)
    return [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()]


def _encode(tok: Any, texts: Sequence[str], cfg: TransformerConfig) -> dict[str, Any]:
    if cfg.truncation == "head_tail":
        ids = []
        for t in texts:
            x = tok(t, add_special_tokens=False)["input_ids"]
            m = cfg.max_length - 2
            if len(x) > m:
                x = x[: m // 4] + x[-(m - m // 4):]          # 1/4 cabeza + 3/4 cola (Sun et al., 2019)
            ids.append(tok.build_inputs_with_special_tokens(x))
        return tok.pad({"input_ids": ids}, padding=True, return_tensors="pt")
    return tok(list(texts), truncation=True, max_length=cfg.max_length, padding=True, return_tensors="pt")


def fine_tune(train_texts: Sequence[str], train_y: np.ndarray, val_texts: Sequence[str], val_y: np.ndarray,
              n_classes: int, cfg: TransformerConfig, test_texts: Sequence[str] | None = None) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    from .config import set_global_seed

    set_global_seed(cfg.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=n_classes)
    if cfg.lora_r > 0:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=2 * cfg.lora_r, lora_dropout=0.1,
                                                 task_type="SEQ_CLS"))
    model.to(dev)
    opt = torch.optim.AdamW(_layer_groups(model, cfg))
    steps = int(np.ceil(len(train_texts) / cfg.batch_size)) * cfg.epochs
    sch = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16 and dev == "cuda")
    g = torch.Generator().manual_seed(cfg.seed)
    ytr = torch.as_tensor(np.asarray(train_y), dtype=torch.long)

    def logits_for(texts: Sequence[str]) -> np.ndarray:
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                enc = {k: v.to(dev) for k, v in _encode(tok, texts[i:i + 64], cfg).items()}
                with torch.autocast(dev, enabled=cfg.fp16 and dev == "cuda"):
                    out.append(model(**enc).logits.float().cpu().numpy())
        return np.concatenate(out)

    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(train_texts), generator=g).numpy()
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            enc = {k: v.to(dev) for k, v in _encode(tok, [train_texts[j] for j in idx], cfg).items()}
            with torch.autocast(dev, enabled=cfg.fp16 and dev == "cuda"):
                loss = torch.nn.functional.cross_entropy(model(**enc).logits.float(), ytr[idx].to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
        zv = logits_for(list(val_texts))
        vl = float(torch.nn.functional.cross_entropy(torch.tensor(zv), torch.as_tensor(np.asarray(val_y))).item())
        hist.append(dict(epoch=ep, val_loss=vl))
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    res = {"val_logits": logits_for(list(val_texts)), "history": hist, "train_seconds": time.time() - t0,
           "n_params": int(sum(p.numel() for p in model.parameters())), "model": model, "tokenizer": tok}
    if test_texts is not None:
        res["test_logits"] = logits_for(list(test_texts))
    return res


# ----------------------------------------------------------------- FastText
def fasttext_supervised(train_texts: Sequence[str], train_y: Sequence[str], lr: float = 0.5, epoch: int = 50,
                        wordNgrams: int = 2, dim: int = 100, minn: int = 0, maxn: int = 0, seed: int = 0,
                        tmp_dir: str = "/tmp") -> Any:
    """FastText supervisado (Joulin et al., 2017) con n-gramas de palabra y de
    carácter. Devuelve un objeto con ``predict_proba(texts)`` y ``classes_``."""
    import os

    import fasttext

    path = os.path.join(tmp_dir, f"ft_train_{seed}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        for t, y in zip(train_texts, train_y):
            fh.write(f"__label__{str(y).replace(' ', '_')} {' '.join(str(t).split())}\n")
    m = fasttext.train_supervised(input=path, lr=lr, epoch=epoch, wordNgrams=wordNgrams, dim=dim, minn=minn,
                                  maxn=maxn, seed=seed, thread=1, verbose=0)
    classes = sorted({str(y).replace(" ", "_") for y in train_y})

    class _FT:
        classes_ = np.array(classes)

        def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
            labs, probs = m.predict([" ".join(str(t).split()) for t in texts], k=len(classes))
            out = np.zeros((len(texts), len(classes)))
            for i, (ls, ps) in enumerate(zip(labs, probs)):
                for lab, p in zip(ls, ps):
                    out[i, classes.index(lab.replace("__label__", ""))] = p
            return out / out.sum(1, keepdims=True)

        def predict(self, texts: Sequence[str]) -> np.ndarray:
            return self.classes_[self.predict_proba(texts).argmax(1)]

    return _FT()
