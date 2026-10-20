"""Utilidades de texto: normalización por familia, deduplicación (exacta y
MinHash/LSH implementado desde cero), vocabulario (Zipf/Heaps, OOV), palabras
discriminativas (log-odds con prior de Dirichlet), diversidad léxica, PII,
ruido de etiquetas (*confident learning*) y truncamiento."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

STOP_EN = set("""a about above after again against all am an and any are as at be because been before being
below between both but by can did do does doing down during each few for from further had has have having he
her here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not
now of off on once only or other our ours ourselves out over own same she should so some such than that the
their theirs them themselves then there these they this those through to too under until up very was we were
what when where which while who whom why will with you your yours yourself yourselves""".split())
STOP_ES = set("""a al algo algunas algunos ante antes como con contra cual cuando de del desde donde durante e el
ella ellas ellos en entre era eras es esa esas ese eso esos esta estas este esto estos fue fueron ha han hasta
hay la las le les lo los mas me mi mis mucho muy nada ni no nos o otra otras otro otros para pero poco por porque
que quien se ser si sin sobre su sus también te tiene tu tus un una uno unos y ya yo""".split())
STOPWORDS = STOP_EN | STOP_ES
NEGATIONS = {"no", "not", "nor", "never", "ni", "nunca", "jamás", "n't"}

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
_NUM = re.compile(r"\d+([.,]\d+)?")
_TOKEN = re.compile(r"[A-Za-zÀ-ÿ0-9_]+(?:'[A-Za-z]+)?")


# ------------------------------------------------------------ normalización
def clean_minimal(t: str) -> str:
    """Solo limpieza de formato (para transformers): Unicode NFC, espacios, HTML básico."""
    t = unicodedata.normalize("NFC", str(t))
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def clean_standard(t: str, remove_stop: bool = True) -> str:
    """Pipeline de referencia: minúsculas, sin símbolos, sin stopwords (conservando negaciones)."""
    t = clean_minimal(t).lower()
    toks = _TOKEN.findall(t)
    if remove_stop:
        toks = [w for w in toks if w not in STOPWORDS or w in NEGATIONS]
    return " ".join(toks)


def clean_aggressive(t: str) -> str:
    """Agresivo: estándar + URLs/emails/números enmascarados + lematización ligera
    (sufijos ingleses/españoles frecuentes). Para lematización real use spaCy."""
    t = _URL.sub(" ", clean_minimal(t)); t = _EMAIL.sub(" ", t); t = _NUM.sub(" ", t)
    out = []
    for w in clean_standard(t).split():
        for suf in ("ingly", "edly", "ing", "ed", "ly", "es", "s", "mente", "ción", "ciones"):
            if len(w) > len(suf) + 3 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return " ".join(out)


PREPROCESSORS = {"minimal": clean_minimal, "standard": clean_standard, "aggressive": clean_aggressive}


def tokenize(t: str) -> list[str]:
    return _TOKEN.findall(str(t).lower())


# --------------------------------------------------------------- columnas
def detect_columns(df: pd.DataFrame, max_classes: int = 50) -> tuple[str, str]:
    """Heurística para (columna de texto, columna de etiqueta) cuando el esquema
    real difiere del supuesto. **Debe confirmarse manualmente** y registrarse."""
    obj = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]
    # se descartan columnas de HTML crudo (nombre «html» o texto que empieza por una etiqueta): sus primeros
    # tokens son plantilla y arruinan los modelos que truncan (CNN, BiLSTM, transformers)
    plain = [c for c in obj if "html" not in str(c).lower()
             and not df[c].astype(str).str.lstrip().str.startswith("<").mean() > 0.5]
    text_col = max(plain or obj, key=lambda c: df[c].astype(str).str.len().mean())
    cands = [c for c in df.columns if c != text_col and 1 < df[c].nunique() <= max_classes]
    pref = [c for c in cands if re.search(r"label|target|class|categor|y$", str(c), re.I)]
    if not cands:
        raise ValueError("No se encontró columna de etiqueta")
    return text_col, (pref or cands)[0]


# ---------------------------------------------------------- deduplicación
def exact_hash(t: str) -> str:
    return hashlib.md5(re.sub(r"\W+", " ", str(t).lower()).strip().encode()).hexdigest()


def shingles(t: str, k: int = 5) -> set[str]:
    s = re.sub(r"\s+", " ", str(t).lower()).strip()
    return {s[i:i + k] for i in range(max(len(s) - k + 1, 1))}


class MinHashLSH:
    """MinHash (hash universal (a·x + b) mod p, p = 2^31 − 1) + LSH por bandas.

    Probabilidad de colisión de un par con Jaccard s: 1 − (1 − s^r)^b; con
    b = 32, r = 4 (128 permutaciones) el umbral efectivo ≈ (1/b)^(1/r) ≈ 0.42; los
    candidatos se verifican después con Jaccard exacto >= ``threshold``.
    """

    PRIME = (1 << 31) - 1

    def __init__(self, num_perm: int = 128, bands: int = 32, seed: int = 0, k: int = 5):
        assert num_perm % bands == 0
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, self.PRIME, num_perm, dtype=np.uint64)
        self.b = rng.integers(0, self.PRIME, num_perm, dtype=np.uint64)
        self.num_perm, self.bands, self.rows, self.k = num_perm, bands, num_perm // bands, k

    def signature(self, text: str) -> np.ndarray:
        sh = shingles(text, self.k)
        h = np.array([int.from_bytes(hashlib.blake2b(x.encode(), digest_size=4).digest(), "little")
                      for x in sh], dtype=np.uint64) % np.uint64(self.PRIME)
        # a, h < 2^31 => a·h + b < 2^63: sin desbordamiento en uint64
        hv = (self.a[:, None] * h[None, :] + self.b[:, None]) % np.uint64(self.PRIME)
        return hv.min(1)

    def candidate_pairs(self, texts: Sequence[str]) -> set[tuple[int, int]]:
        sigs = [self.signature(t) for t in texts]
        pairs: set[tuple[int, int]] = set()
        for bnd in range(self.bands):
            buckets: dict[bytes, list[int]] = {}
            for i, s in enumerate(sigs):
                key = s[bnd * self.rows:(bnd + 1) * self.rows].tobytes()
                buckets.setdefault(key, []).append(i)
            for ids in buckets.values():
                for x in range(len(ids)):
                    for y in range(x + 1, len(ids)):
                        pairs.add((ids[x], ids[y]))
        return pairs


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(len(a | b), 1)


def near_duplicate_clusters(texts: Sequence[str], threshold: float = 0.8, **kw: Any) -> np.ndarray:
    """Id de clúster por texto (duplicados exactos y casi-duplicados comparten id).

    Se usa como **grupo** en la partición: los casi-duplicados nunca cruzan
    train/val/test."""
    n = len(texts)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    first: dict[str, int] = {}
    for i, t in enumerate(texts):
        h = exact_hash(t)
        if h in first:
            union(i, first[h])
        else:
            first[h] = i
    lsh = MinHashLSH(**kw)
    sh = [shingles(t, lsh.k) for t in texts]
    for i, j in lsh.candidate_pairs(texts):
        if jaccard(sh[i], sh[j]) >= threshold:
            union(i, j)
    return np.array([find(i) for i in range(n)])


def cross_split_duplicates(clusters: np.ndarray, split: Sequence[str]) -> int:
    """Número de clústeres de duplicados presentes en más de una partición (debe ser 0)."""
    df = pd.DataFrame({"c": clusters, "s": list(split)})
    return int((df.groupby("c")["s"].nunique() > 1).sum())


# ------------------------------------------------------------- vocabulario
def zipf_heaps(token_lists: Iterable[list[str]]) -> dict[str, Any]:
    """Pendiente de Zipf (log f vs log rango) y parámetros de Heaps V = K·N^β."""
    cnt: Counter[str] = Counter(); n_tok, vocab, ns, vs = 0, set(), [], []
    for toks in token_lists:
        cnt.update(toks)
        for w in toks:
            n_tok += 1; vocab.add(w)
            if n_tok & (n_tok - 1) == 0:  # potencias de 2
                ns.append(n_tok); vs.append(len(vocab))
    f = np.array(sorted(cnt.values(), reverse=True), float)
    r = np.arange(1, len(f) + 1)
    top = slice(0, max(10, len(f) // 2))
    zipf_s = -np.polyfit(np.log(r[top]), np.log(f[top]), 1)[0] if len(f) > 2 else np.nan
    beta, logk = np.polyfit(np.log(ns), np.log(vs), 1) if len(ns) > 2 else (np.nan, np.nan)
    return dict(zipf_exponent=float(zipf_s), heaps_beta=float(beta), heaps_K=float(np.exp(logk)),
                n_tokens=n_tok, vocab_size=len(vocab), freqs=f, heaps_n=np.array(ns), heaps_v=np.array(vs))


def oov_rate(train_tokens: Iterable[list[str]], test_tokens: Iterable[list[str]], min_count: int = 1) -> dict[str, float]:
    c: Counter[str] = Counter(w for t in train_tokens for w in t)
    vocab = {w for w, k in c.items() if k >= min_count}
    test = [w for t in test_tokens for w in t]
    types = set(test)
    return dict(oov_token_rate=float(np.mean([w not in vocab for w in test])) if test else 0.0,
                oov_type_rate=float(np.mean([w not in vocab for w in types])) if types else 0.0)


# ---------------------------------------------------- palabras discriminativas
def log_odds_dirichlet(tokens_a: Iterable[list[str]], tokens_b: Iterable[list[str]],
                       prior: Iterable[list[str]] | None = None, alpha0: float = 1000.0) -> pd.DataFrame:
    """Log-odds con prior informativo de Dirichlet (Monroe et al., 2008, ec. 22).

    Devuelve δ_w, su varianza aproximada y z = δ/σ (|z| > 1.96 ≈ IC 95 % excluye 0).
    """
    ca = Counter(w for t in tokens_a for w in t); cb = Counter(w for t in tokens_b for w in t)
    cp = Counter(w for t in prior for w in t) if prior is not None else ca + cb
    vocab = sorted(set(ca) | set(cb))
    tot_p = sum(cp.values())
    aw = np.array([alpha0 * cp[w] / tot_p for w in vocab]) + 1e-3
    ya = np.array([ca[w] for w in vocab], float); yb = np.array([cb[w] for w in vocab], float)
    na, nb, a0 = ya.sum(), yb.sum(), aw.sum()
    d = np.log((ya + aw) / (na + a0 - ya - aw)) - np.log((yb + aw) / (nb + a0 - yb - aw))
    var = 1 / (ya + aw) + 1 / (yb + aw)
    z = d / np.sqrt(var)
    return pd.DataFrame({"word": vocab, "delta": d, "se": np.sqrt(var), "z": z,
                         "ci_low": d - 1.96 * np.sqrt(var), "ci_high": d + 1.96 * np.sqrt(var),
                         "n_a": ya, "n_b": yb}).sort_values("z", ascending=False)


# ---------------------------------------------------------- diversidad léxica
def ttr(tokens: list[str]) -> float:
    return len(set(tokens)) / max(len(tokens), 1)


def mtld(tokens: list[str], threshold: float = 0.72) -> float:
    """MTLD (McCarthy & Jarvis, 2010), media de pasadas directa e inversa."""
    def one(toks: list[str]) -> float:
        factors, types, n = 0.0, set(), 0
        for w in toks:
            n += 1; types.add(w)
            if len(types) / n <= threshold:
                factors += 1; types, n = set(), 0
        if n > 0:
            factors += (1 - len(types) / n) / (1 - threshold) if len(types) / n < 1 else 0
        return len(toks) / factors if factors > 0 else float(len(toks))
    if not tokens:
        return 0.0
    return 0.5 * (one(tokens) + one(tokens[::-1]))


def flesch_reading_ease(text: str) -> float:
    """Flesch (inglés). Sílabas aproximadas por grupos vocálicos."""
    sents = max(len(re.findall(r"[.!?]+", text)), 1)
    words = tokenize(text)
    if not words:
        return np.nan
    syl = sum(max(len(re.findall(r"[aeiouy]+", w)), 1) for w in words)
    return 206.835 - 1.015 * len(words) / sents - 84.6 * syl / len(words)


# --------------------------------------------------------------------- PII
PII_PATTERNS = {"email": _EMAIL, "url": _URL, "telefono": _PHONE,
                "fecha": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
                "edad": re.compile(r"\b\d{2}\s*(?:years? old|años)\b", re.I)}
GENDER_TERMS = {"he", "she", "him", "her", "his", "hers", "man", "woman", "male", "female", "mr", "mrs", "ms",
                "él", "ella", "hombre", "mujer", "señor", "señora"}


def pii_scan(texts: Sequence[str]) -> pd.DataFrame:
    """Conteo de patrones de PII y de proxies de atributos protegidos por texto."""
    rows = []
    for t in texts:
        r = {k: len(p.findall(str(t))) for k, p in PII_PATTERNS.items()}
        toks = set(tokenize(t))
        r["terminos_genero"] = len(toks & GENDER_TERMS)
        r["nombres_propios_aprox"] = len(re.findall(r"(?<![.!?]\s)\b[A-Z][a-z]{2,}\b", str(t)))
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------- ruido de etiquetas
def confident_learning(oof_probs: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Confident learning (Northcutt et al., 2021), regla *prune by noise rate*.

    ``oof_probs``: probabilidades **fuera de muestra** (validación cruzada).
    Umbral por clase t_j = media de p_j sobre ejemplos etiquetados j; matriz de
    conjunta confiada C[ỹ, y*]; se marcan como sospechosos los ejemplos fuera de
    la diagonal con mayor margen. Devuelve índices sospechosos y la tasa estimada.
    """
    p = np.asarray(oof_probs, float); y = np.asarray(y)
    k = p.shape[1]
    t = np.array([p[y == j, j].mean() if (y == j).any() else 1.0 for j in range(k)])
    above = p >= t[None, :]
    ystar = np.where(above.any(1), np.argmax(np.where(above, p, -np.inf), 1), -1)
    C = np.zeros((k, k), int)
    for yi, ys in zip(y, ystar):
        if ys >= 0:
            C[yi, ys] += 1
    n_off = C.sum() - np.trace(C)
    cand = np.where((ystar >= 0) & (ystar != y))[0]
    margin = p[cand, ystar[cand]] - p[cand, y[cand]]
    suspects = cand[np.argsort(-margin)][:n_off]
    return {"suspects": np.sort(suspects), "est_noise_rate": float(n_off / max(C.sum(), 1)),
            "confident_joint": C, "thresholds": t}


# -------------------------------------------------------------- truncamiento
def length_percentiles(lengths: Sequence[int], q: Sequence[int] = (50, 90, 95, 99)) -> dict[str, float]:
    return {f"p{k}": float(np.percentile(lengths, k)) for k in q}


def truncation_rate_by_class(lengths: Sequence[int], y: Sequence[Any], max_len: int) -> pd.Series:
    return pd.DataFrame({"l": lengths, "y": y}).assign(t=lambda d: d["l"] > max_len).groupby("y")["t"].mean()


def head_tail_truncate(tokens: list[str], max_len: int, head: float = 0.5) -> list[str]:
    """Truncamiento cabeza+cola (Sun et al., 2019): conserva inicio y final."""
    if len(tokens) <= max_len:
        return tokens
    h = int(max_len * head)
    return tokens[:h] + tokens[len(tokens) - (max_len - h):]


# ---------------------------------------------------------------- sintético
def make_synthetic_text(n: int = 600, seed: int = 0, n_classes: int = 3, shortcut: float = 0.0,
                        vocab_shift: float = 0.0, length_scale: float = 1.0) -> pd.DataFrame:
    """Corpus sintético con palabras de clase, ruido común y un atajo opcional
    (token ``ref_<clase>`` con probabilidad ``shortcut``). ``vocab_shift``
    reemplaza palabras de clase por sinónimos no vistos (deriva de vocabulario).
    **Solo** para pruebas de humo y la simulación de deriva."""
    rng = np.random.default_rng(seed)
    topics = {
        0: "python sql pipeline data model deploy cloud spark etl dashboard".split(),
        1: "customer sales client negotiation target revenue crm pitch quota deal".split(),
        2: "design figma prototype user interface typography brand layout color sketch".split(),
        3: "nurse patient clinic care hospital shift triage ward dose chart".split(),
    }
    syn = {w: w + "x" for ws in topics.values() for w in ws}
    common = "experience team work project years skills strong role responsible good communication".split()
    rows = []
    for _ in range(n):
        c = int(rng.integers(n_classes))
        L = max(5, int(rng.gamma(4, 6) * length_scale))
        words = []
        for _ in range(L):
            if rng.random() < 0.35:
                w = str(rng.choice(topics[c]))
                if rng.random() < vocab_shift:
                    w = syn[w]
            elif rng.random() < 0.1:
                w = str(rng.choice(topics[int(rng.integers(n_classes))]))
            else:
                w = str(rng.choice(common))
            words.append(w)
        if rng.random() < shortcut:
            words.insert(0, f"ref_{c}")
        rows.append({"text": " ".join(words), "label": f"class_{c}"})
    return pd.DataFrame(rows)
