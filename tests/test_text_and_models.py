import numpy as np
import pytest

from project.fairness import counterfactual_flip_rate, fairness_report, gender_swap
from project.models_text import NBSVM, Vocabulary, bilstm_complexity, tfidf_logreg
from project.robustness_text import invariance_report, mask_tokens
from project.text_data import (
    MinHashLSH,
    clean_aggressive,
    clean_minimal,
    clean_standard,
    confident_learning,
    detect_columns,
    jaccard,
    log_odds_dirichlet,
    make_synthetic_text,
    mtld,
    oov_rate,
    pii_scan,
    shingles,
    tokenize,
    zipf_heaps,
)


def test_preprocessing_levels():
    t = "Hello <b>World</b>!  I'm NOT at http://x.com, call 555-123-4567 at 10 years old"
    assert "<b>" not in clean_minimal(t) and "Hello" in clean_minimal(t)
    s = clean_standard(t)
    assert s == s.lower() and "not" in s.split() and "at" not in s.split()
    assert "http" not in clean_aggressive(t) and "10" not in clean_aggressive(t)


def test_minhash_estimates_jaccard():
    a = "the quick brown fox jumps over the lazy dog " * 3
    b = a.replace("lazy", "sleepy")
    lsh = MinHashLSH(num_perm=128, bands=32)
    est = (lsh.signature(a) == lsh.signature(b)).mean()
    assert est == pytest.approx(jaccard(shingles(a), shingles(b)), abs=0.12)
    assert (0, 1) in lsh.candidate_pairs([a, b, "completely different text here"])


def test_vocabulary_fit_only_on_train_and_unk():
    v = Vocabulary(min_count=1).fit(["alpha beta", "beta gamma"])
    X = v.encode(["beta delta"], max_len=4)
    assert X.shape == (1, 4) and X[0, 1] == 1 and X[0, 2] == 0     # delta -> UNK, luego PAD
    assert v.unk_rate(["delta delta beta"]) == pytest.approx(2 / 3)
    X2 = v.encode(["alpha beta gamma alpha"], max_len=2, truncation="start")
    assert list(X2[0]) == [v.stoi["gamma"], v.stoi["alpha"]]


def test_baselines_learn_synthetic_signal():
    df = make_synthetic_text(600, seed=0)
    tr, te = df.iloc[:450], df.iloc[450:]
    for m in (tfidf_logreg(), NBSVM()):
        m.fit(tr["text"].tolist(), tr["label"].to_numpy())
        acc = (m.predict(te["text"].tolist()) == te["label"].to_numpy()).mean()
        assert acc > 0.8
        p = m.predict_proba(te["text"].tolist()); assert np.allclose(p.sum(1), 1)


def test_eda_helpers():
    df = make_synthetic_text(200, seed=1)
    toks = [tokenize(t) for t in df["text"]]
    z = zipf_heaps(toks)
    assert 0 < z["heaps_beta"] <= 1.0 and z["vocab_size"] > 10
    assert oov_rate(toks[:100], toks[100:])["oov_token_rate"] < 0.05
    lo = log_odds_dirichlet([t for t, y in zip(toks, df["label"]) if y == "class_0"],
                            [t for t, y in zip(toks, df["label"]) if y != "class_0"])
    assert lo.iloc[0]["word"] in "python sql pipeline data model deploy cloud spark etl dashboard".split()
    assert mtld("a b c d e f g h".split()) > mtld("a a a a a a a a".split())
    assert pii_scan(["mail me at a@b.com"])["email"].iloc[0] == 1
    assert detect_columns(df) == ("text", "label")


def test_confident_learning_finds_flipped_labels():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500); p = np.where(y[:, None] == np.arange(2), 0.9, 0.1)
    flipped = rng.choice(500, 25, replace=False); y_noisy = y.copy(); y_noisy[flipped] = 1 - y[flipped]
    res = confident_learning(p, y_noisy)
    assert set(res["suspects"]) == set(flipped) and res["est_noise_rate"] == pytest.approx(0.05)


def test_fairness_and_counterfactuals():
    rng = np.random.default_rng(0)
    g = np.array(["a"] * 500 + ["b"] * 500); yt = rng.integers(0, 2, 1000)
    yp = np.where(g == "a", rng.random(1000) < 0.6, rng.random(1000) < 0.3).astype(int)
    r = fairness_report(yt, yp, g, n_boot=200)
    assert r["metrics"]["dp_diff"][0] > 0.2 and not r["four_fifths_ok"]
    assert gender_swap("She is a strong woman.") == "He is a strong man."
    df = make_synthetic_text(300, seed=2)
    m = tfidf_logreg().fit(df["text"].tolist(), df["label"])
    cf = counterfactual_flip_rate(m.predict_proba, ["he leads the python team", "she leads the python team"],
                                  gender_swap)
    assert cf["n_changed"] == 2 and cf["mean_abs_delta"] < 0.1


def test_shortcut_ablation_and_invariance():
    df = make_synthetic_text(600, seed=3, shortcut=0.9)
    m = tfidf_logreg().fit(df["text"].tolist()[:450], df["label"][:450])
    te = df["text"].tolist()[450:]
    masked = [mask_tokens(t, {"ref_0", "ref_1", "ref_2"}) for t in te]
    assert all("ref_" not in t for t in masked)
    inv = invariance_report(m.predict, te[:50])
    assert set(inv) == {"mayusculas", "typos_5pct", "sinonimos"}
    assert bilstm_complexity(100, 100, 64)["macs"] == 2 * 100 * 4 * 64 * (100 + 64)
