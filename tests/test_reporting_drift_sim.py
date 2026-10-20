import numpy as np
import pandas as pd

from project.drift_sim import alarm, build_text_pools, calibrate_union, power_study, power_table, roc_auc_vs_null, sample_window
from project.efficiency import multicriteria, pareto_front, weight_sensitivity
from project.nlp_pipeline import classification_metrics, registry
from project.reporting import error_profile, ranking_agreement, results_table, simpson_check
from project.text_data import make_synthetic_text, tokenize


def per_image_frame(seed=0):
    rng = np.random.default_rng(seed); rows = []
    for i in range(30):
        base = rng.uniform(0.3, 0.7)
        for m, shift in (("A", 0.0), ("B", 0.05), ("C", -0.05)):
            rows.append(dict(modelo=m, id_imagen=f"i{i}", sample_id=f"p{i // 5}", tipo_celular=["x", "y"][i % 2],
                             ap=base + shift + rng.normal(0, 0.01), auc_roc=0.97 + rng.normal(0, 0.001),
                             err_fusion=int(rng.integers(0, 3)), err_omitida=1, n_gt=20))
    return pd.DataFrame(rows)


def test_results_table_and_contrasts():
    df = per_image_frame()
    t = results_table(df, ["ap"], primary="ap", n_boot=500)
    assert t.index[0] == "B" and "[" in t.loc["B", "ap"]
    assert simpson_check(df, "ap").shape[0] == 3
    ag = ranking_agreement(df, "ap", "ap", n_boot=50)
    assert ag["rho"] == 1.0
    assert (error_profile(df).values >= 0).all()


def test_text_pools_power_and_calibration():
    tr = make_synthetic_text(400, seed=0); st = make_synthetic_text(400, seed=1)
    m = registry()["TF-IDF + LR"].fit(tr.text.tolist(), tr.label.to_numpy(), tr.text.tolist(), tr.label.to_numpy())
    cls = np.unique(tr.label); vocab = {w for t in tr.text for w in tokenize(t)}
    pools = build_text_pools(st.text.tolist(), st.label, m.predict_proba, cls, vocab, severities=(0.0, 0.5), seed=0)
    ref = build_text_pools(tr.text.tolist(), tr.label, m.predict_proba, cls, vocab, severities=(0.0,))[("base", 0.0)]
    res = power_study(ref, pools, ["vocabulario"], ["len", "oov", "conf"], ["pred"], window=100, reps=30,
                      severities=(0.0, 0.5))
    pt = power_table(res).set_index(["tipo", "severidad"])
    assert pt.loc[("vocabulario", 0.5), "tasa_alarma"] > 0.9
    thr = calibrate_union(res[res.severidad == 0], 0.1)
    assert set(thr) == {"psi", "ks", "chi2"}
    assert roc_auc_vs_null(res).auc_score_comb.min() > 0.8
    w = sample_window(pools[("base", 0.0)], 100, np.random.default_rng(0), prior_shift=0.8, target_class=cls[0])
    assert (w.y == cls[0]).mean() > 0.7
    assert alarm({"psi_max": 1.0, "ks_p_min": 0.5}) and not alarm({"psi_max": 0.0, "ks_p_min": 0.5})


def test_classification_metrics_and_multicriteria():
    y = np.array(["a", "b", "a", "b"]); p = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.4, 0.6]])
    r = classification_metrics(y, p, ["a", "b"])
    assert r["accuracy"] == 1.0 and r["roc_auc"] == 1.0
    df = pd.DataFrame({"ap": [0.5, 0.6, 0.55], "lat": [10.0, 100.0, 12.0]}, index=["x", "y", "z"])
    assert set(pareto_front(df.reset_index(), "ap", "lat")["index"]) == {"x", "y", "z"}
    s = multicriteria(df, {"ap": 0.7, "lat": 0.3}, {"ap": True, "lat": False})
    assert s.idxmax() in df.index
    ws = weight_sensitivity(df, {"ap": 0.7, "lat": 0.3}, {"ap": True, "lat": False}, n=200)
    assert abs(ws.sum() - 1) < 1e-9
