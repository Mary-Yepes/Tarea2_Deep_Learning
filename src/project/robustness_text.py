"""Pruebas tipo CheckList (Ribeiro et al., 2020) y ablación de atajos."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np

SYNONYMS = {"good": "great", "strong": "solid", "experience": "background", "team": "group",
            "skills": "abilities", "work": "job", "project": "initiative", "responsible": "accountable",
            "years": "yrs", "role": "position", "bueno": "buen", "equipo": "grupo", "trabajo": "empleo"}


def perturb_case(t: str) -> str:
    return t.upper()


def perturb_typos(t: str, rate: float = 0.05, seed: int = 0) -> str:
    """Intercambia caracteres adyacentes en ~``rate`` de las palabras."""
    rng = np.random.default_rng(seed)
    out = []
    for w in t.split():
        if len(w) > 3 and rng.random() < rate:
            i = int(rng.integers(0, len(w) - 1)); w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        out.append(w)
    return " ".join(out)


def perturb_synonyms(t: str) -> str:
    return re.sub(r"\b(" + "|".join(SYNONYMS) + r")\b", lambda m: SYNONYMS[m.group(0).lower()], t, flags=re.I)


def mask_tokens(t: str, tokens: set[str], mask: str = "") -> str:
    """Elimina/enmascara palabras o plantillas sospechosas de atajo."""
    tokens = {str(w) for w in tokens if str(w)}
    pat = r"\b(" + "|".join(map(re.escape, sorted(tokens, key=len, reverse=True))) + r")\b"
    return re.sub(r"\s+", " ", re.sub(pat, mask, t, flags=re.I)).strip() if tokens else t


INVARIANCE_TESTS: dict[str, Callable[[str], str]] = {
    "mayusculas": perturb_case, "typos_5pct": perturb_typos, "sinonimos": perturb_synonyms,
}


def invariance_report(predict: Callable[[Sequence[str]], np.ndarray], texts: Sequence[str],
                      tests: dict[str, Callable[[str], str]] = INVARIANCE_TESTS) -> dict[str, float]:
    """Tasa de fallo de invarianza: fracción de predicciones que cambian."""
    base = np.asarray(predict(list(texts)))
    return {k: float((np.asarray(predict([f(t) for t in texts])) != base).mean()) for k, f in tests.items()}
