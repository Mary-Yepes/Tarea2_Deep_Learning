"""Rellena automáticamente el Jupyter Book con los resultados reales.

Lee ``reports/tables`` y escribe fragmentos Markdown en ``book/_generado/`` que los
capítulos incluyen (``{include}``): tablas de resultados, contraste de H1–H6 con
su veredicto según los criterios PRE-REGISTRADOS, hallazgos para la discusión,
eficiencia, deriva y cifras de EDA para *datasheets* y *model cards*.

    python scripts/fill_report.py            # resultados reales
    python scripts/fill_report.py --src reports/smoke/tables   # probar con la corrida de humo

Si falta una tabla, el fragmento lo dice explícitamente («pendiente: …») en vez de
inventar cifras.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from project.nlp_pipeline import classification_metrics  # noqa: E402
from project.reporting import error_profile, ranking_agreement, results_table, simpson_check  # noqa: E402
from project.stats import bootstrap_ci, compare_models, corrected_resampled_ttest, fmt_ci, holm_adjust, paired_test  # noqa: E402

OUT = ROOT / "book/_generado"
PRE = yaml.safe_load((ROOT / "configs/prereg.yaml").read_text())
D_AP, D_F1, D_ECE = (PRE["efecto_minimo_relevante"][k] for k in ("imagenes_AP", "nlp_F1", "ECE"))
SEG_ORDER = ["Baseline clásico", "Mask R-CNN (+ Swin)", "Hover-Net", "U-Net++ + Watershed", "Cellpose 2.0 (ajustado)",
             "Cellpose 2.0 (generalista)", "SAM (cero-shot)", "SAM (prompts de caja)"]


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    d = df.reset_index().rename(columns={"index": "Modelo"}) if index else df
    cols = [str(c) for c in d.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        rows.append("| " + " | ".join(_fmt(v) for v in r.values) + " |")
    return "\n".join(rows) + "\n"


def _fmt(v: Any) -> str:
    if isinstance(v, (float, np.floating)):
        return "—" if np.isnan(v) else f"{v:.4g}"
    return str(v)


def write(name: str, text: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.md").write_text(text, encoding="utf-8")
    print("escrito", OUT / f"{name}.md")


def pend(what: str, cmd: str) -> str:
    return f"*Pendiente: {what} (`{cmd}`).*\n"


# ===================================================================== segmentación
def seg(src: Path) -> dict[str, str]:
    v: dict[str, str] = {}
    p = src / "seg_eval_test.csv"
    if not p.exists():
        for n in ("tabla_seg", "seg_estratificado", "seg_errores", "seg_contrastes"):
            write(n, pend("resultados de segmentación", "python scripts/seg_experiments.py main"))
        return {"H2": "pendiente", "H3": "pendiente"}
    df = pd.read_csv(p)
    cols = {"ap": "AP@[.5:.95]", "ap50": "AP50", "pq": "PQ", "dice": "Dice", "iou": "IoU", "precision": "Precision",
            "recall": "Recall", "auc_roc": "AUC", "hd95": "HD95", "balanced_accuracy": "Bal. Acc"}
    t = results_table(df, [c for c in cols if c in df], primary="ap").rename(columns=cols)
    t = t.reindex([m for m in SEG_ORDER if m in t.index] + [m for m in t.index if m not in SEG_ORDER])
    txt = ("Media [IC 95 %] por imagen del test sellado (bootstrap jerárquico placa → imagen; semillas promediadas "
           "por imagen). `rank_ap` = rango medio por imagen en la métrica primaria (menor = mejor).\n\n" + md_table(t))
    piv = df.groupby(["id_imagen", "modelo"]).ap.mean().unstack().dropna()
    if piv.shape[1] >= 3:
        cm = compare_models(piv, delta=D_AP)
        pw = cm["pairwise"][["A", "B", "mean_diff", "cliffs_delta", "cliffs_mag", "p_holm", "reject", "practically_relevant"]]
        txt += (f"\n**Friedman:** χ² = {cm['friedman_stat']:.2f}, p = {cm['friedman_p']:.2e} (n = {cm['n']} imágenes); "
                f"diferencia crítica de Nemenyi = {cm['cd_nemenyi']:.2f} rangos.\n\n**Wilcoxon por pares con Holm** "
                f"(δ = {D_AP}):\n\n" + md_table(pw.round(4), index=False))
        best = cm["mean_ranks"].index[0]
        v["mejor_seg"] = best
    write("tabla_seg", txt)

    # estratificado (H3) y por densidad
    df["dens"] = pd.qcut(df.groupby("id_imagen").n_gt.transform("first").rank(method="first"), 3, labels=["baja", "media", "alta"])
    st = df.groupby(["tipo_celular", "modelo"]).ap.mean().unstack(0).round(3)
    sd = df.groupby(["dens", "modelo"]).ap.mean().unstack(0).round(3)
    sc = simpson_check(df, "ap")
    same = sc.iloc[1:].orden.nunique() == 1 and sc.iloc[1:].orden.iloc[0] == sc.orden.iloc[0]
    v["H3"] = ("**Refutada**: el orden es idéntico en los tres tipos celulares." if same else
               "**Apoyada**: el orden de los modelos cambia al estratificar por tipo celular "
               f"(τ de Kendall con el global: {', '.join(f'{r.estrato} {r.tau:.2f}' for r in sc.iloc[1:].itertuples())}).")
    best_by_type = st.idxmax().to_dict()
    write("seg_estratificado", "AP medio por tipo celular:\n\n" + md_table(st) + "\nAP medio por tercil de densidad "
          "(número de células):\n\n" + md_table(sd) + "\nOrden por estrato:\n\n" + md_table(sc, index=False) +
          f"\n**H3.** {v['H3']}\n\n**Mejor modelo por tipo:** " + "; ".join(f"{k}: {m}" for k, m in best_by_type.items()) + ".\n")

    ep = error_profile(df).round(3)
    dom = ep.idxmax(axis=1)
    write("seg_errores", "Errores por objeto de referencia (taxonomía a priori):\n\n" + md_table(ep) +
          "\n**Error dominante:** " + "; ".join(f"{k[0]} / {k[1]}: {e.replace('_por_objeto', '')}" for k, e in dom.items()) + ".\n")

    # H2: píxel vs instancia en modelos de DL
    dl = df[~df.modelo.str.contains("clásico")]
    if "auc_roc" in dl and dl.auc_roc.notna().any() and dl.modelo.nunique() >= 3:
        ag = ranking_agreement(dl.dropna(subset=["auc_roc"]), "auc_roc", "ap")
        means = dl.groupby("modelo").auc_roc.mean()
        sat = bool((means > 0.95).all() and means.max() - means.min() < 0.02)
        refuted = ag["rho"] > 0.9 or not (means > 0.95).all()
        if np.isnan(ag["rho"]):
            v["H2"] = "no evaluable: las AUC medias no varían entre modelos"
        else:
            v["H2"] = (f"**{'Refutada' if refuted else 'Apoyada'}**: ρ de Spearman entre rankings por AUC de píxel y por AP = "
                       f"{ag['rho']:.2f} [{ag['ci_low']:.2f}, {ag['ci_high']:.2f}]; AUC medias {means.min():.3f}–{means.max():.3f} "
                       f"(saturación {'sí' if sat else 'no'}).")
    else:
        v["H2"] = "pendiente (se necesitan ≥ 3 modelos de DL con mapa de probabilidad)"
    write("seg_contrastes", f"**H2.** {v['H2']}\n")
    return v


def seg_extras(src: Path) -> dict[str, str]:
    v: dict[str, str] = {}
    fr = src / "seg_fractions.csv"
    if fr.exists():
        f = pd.read_csv(fr)
        piv = f.groupby(["fraccion", "id_imagen", "modelo"]).ap.mean().unstack()
        scratch = next((c for c in piv.columns if "U-Net" in c), None)
        pre = next((c for c in piv.columns if "Cellpose" in c), None)
        if scratch and pre and 0.25 in piv.index.get_level_values(0) and 1.0 in piv.index.get_level_values(0):
            d25 = (piv.loc[0.25, pre] - piv.loc[0.25, scratch]).dropna(); d100 = (piv.loc[1.0, pre] - piv.loc[1.0, scratch]).dropna()
            c25, c100 = bootstrap_ci(d25.to_numpy()), bootstrap_ci(d100.to_numpy())
            gap = bootstrap_ci((d25 - d100).dropna().to_numpy())
            ok = c25[1] > 0 and gap[1] > 0
            v["H1"] = (f"**{'Apoyada' if ok else 'Refutada'}**: Δ25 ({pre} − {scratch}) = {fmt_ci(*c25)}; Δ100 = {fmt_ci(*c100)}; "
                       f"Δ25 − Δ100 = {fmt_ci(*gap)}.")
            curve = f.groupby(["fraccion", "modelo"]).ap.agg(["mean", "std"]).round(4)
            write("seg_fracciones", "Curva de eficiencia de datos (AP en test):\n\n" + md_table(curve) + f"\n**H1.** {v['H1']}\n")
    if "H1" not in v:
        v["H1"] = "pendiente"; write("seg_fracciones", pend("curvas de datos", "seg_experiments.py fractions"))
    ab = src / "seg_ablations.csv"
    if ab.exists():
        a = pd.read_csv(ab); out = []
        for factor, g in a.groupby("factor"):
            pv = g.groupby(["id_imagen", "nivel"]).ap.mean().unstack().dropna()
            s = g.groupby("nivel").ap.agg(["mean", "std"]).round(4)
            out.append(f"**{factor}**\n\n" + md_table(s))
            if pv.shape[1] >= 3:
                pw = compare_models(pv, delta=D_AP)["pairwise"]
                sig = pw[pw.reject & pw.practically_relevant]
                out.append("Diferencias significativas (Holm) y ≥ δ: " +
                           ("; ".join(f"{r.A} vs {r.B} (Δ = {r.mean_diff:+.3f})" for r in sig.itertuples()) or "ninguna → factor irrelevante") + "\n")
            elif pv.shape[1] == 2:
                r = paired_test(pv.iloc[:, 0], pv.iloc[:, 1])
                out.append(f"{pv.columns[0]} − {pv.columns[1]}: Δ = {fmt_ci(r['mean_diff'], r['ci_low'], r['ci_high'])}, "
                           f"p = {r['p_wilcoxon']:.3g}\n")
        write("seg_ablaciones", "\n".join(out))
    else:
        write("seg_ablaciones", pend("ablaciones", "seg_experiments.py ablations"))
    lo, ev = src / "seg_loco.csv", src / "seg_eval_test.csv"
    if lo.exists() and ev.exists():
        l = pd.read_csv(lo); e = pd.read_csv(ev)
        base = e[e.modelo.str.contains("U-Net")].groupby("tipo_celular").ap.mean()
        o = l.groupby("excluido").ap.mean().to_frame("AP sin ver el tipo").join(base.rename("AP con todos los tipos"))
        o["caída relativa"] = 1 - o.iloc[:, 0] / o.iloc[:, 1]
        write("seg_loco", "Leave-one-cell-type-out (U-Net++):\n\n" + md_table(o.round(3)))
    else:
        write("seg_loco", pend("LOCO", "seg_experiments.py loco"))
    rb = src / "seg_robustness.csv"
    if rb.exists():
        from project.reporting import degradation_auc

        r = pd.read_csv(rb); r0 = r[r.tta == False]  # noqa: E712
        deg = degradation_auc(r0)
        txt = "Degradación del AP con severidad creciente (sin TTA):\n\n" + md_table(deg.round(3), index=False)
        if r.tta.nunique() > 1:
            t = r[r.severidad.isin([0, 3])].groupby(["perturbacion", "severidad", "tta"]).ap.mean().unstack().round(4)
            txt += "\nEfecto de TTA (AP medio):\n\n" + md_table(t)
        write("seg_robustez", txt)
    else:
        write("seg_robustez", pend("robustez", "seg_experiments.py robustness"))
    return v


# ===================================================================== NLP
def nlp(src: Path) -> dict[str, str]:
    v: dict[str, str] = {}
    tp, cvp = src / "nlp_test_preds.csv", src / "nlp_cv_scores.csv"
    if not tp.exists():
        write("tabla_nlp", pend("resultados de NLP", "python scripts/nlp_experiments.py all"))
        return {"H4": "pendiente", "H5": "pendiente"}
    pr = pd.read_csv(tp); test = pr[pr.split == "test"]
    pcols = [c for c in pr.columns if c.startswith("p_") and c != "p_T_max"]
    classes = np.array([c[2:] for c in pcols])
    rows = {}
    for m, g in test.groupby("modelo"):
        P = g.groupby("idx")[pcols].mean(); y = g.groupby("idx").y.first().astype(str).to_numpy()
        rng = np.random.default_rng(0); bs = []
        for _ in range(1000):
            i = rng.integers(0, len(y), len(y)); bs.append(classification_metrics(y[i], P.to_numpy()[i], classes))
        bs = pd.DataFrame(bs); pt = classification_metrics(y, P.to_numpy(), classes)
        rows[m] = {k: f"{pt[k]:.3f} [{bs[k].quantile(.025):.3f}, {bs[k].quantile(.975):.3f}]"
                   for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "mcc", "roc_auc", "pr_auc", "ece")}
        rows[m]["latencia (ms/texto)"] = f"{g.latencia_ms_por_texto.median():.3f}"
    t = pd.DataFrame(rows).T.sort_values("f1_macro", ascending=False)
    txt = "Test sellado, media [IC 95 %] (bootstrap sobre ejemplos; probabilidades promediadas entre semillas).\n\n" + md_table(t)
    if cvp.exists():
        cv = pd.read_csv(cvp); cv = cv[cv.tag == "cv"]
        piv = cv.pivot_table(index=["semilla", "fold"], columns="modelo", values="f1_macro")
        full = piv.dropna(axis=1, thresh=int(0.8 * len(piv)))
        n_tr, n_te = int(cv.n_train.mean()), int(cv.n_test.mean())
        pairs = []
        mods = list(full.columns)
        for i in range(len(mods)):
            for j in range(i + 1, len(mods)):
                d = (full[mods[i]] - full[mods[j]]).dropna()
                if len(d) > 2:
                    tt, p = corrected_resampled_ttest(d.to_numpy(), n_tr, n_te)
                    pairs.append(dict(A=mods[i], B=mods[j], dF1=d.mean(), n_rep=len(d), p_NB=p))
        pw = pd.DataFrame(pairs)
        if len(pw):
            pw["p_holm"] = holm_adjust(pw.p_NB.to_numpy())
            txt += "\nComparación por pares (t remuestreada corregida de Nadeau–Bengio sobre semilla×fold, Holm):\n\n" + md_table(pw.round(4), index=False)
        cvs = cv.groupby("modelo").f1_macro.agg(["mean", "std", "count"]).round(4).sort_values("mean", ascending=False)
        txt += "\nF1 macro en CV repetida (media, DE entre semillas×folds, n):\n\n" + md_table(cvs)
        tr = next((m for m in cv.modelo.unique() if "BERT" in m), None)
        if tr and "TF-IDF + XGBoost" in cv.modelo.unique():
            d = (piv[tr] - piv["TF-IDF + XGBoost"]).dropna().to_numpy()
            from scipy import stats as sst

            k = len(d); se = np.sqrt((1 / k + n_te / n_tr) * d.var(ddof=1))
            lo, hi = d.mean() - sst.t.ppf(0.95, k - 1) * se, d.mean() + sst.t.ppf(0.95, k - 1) * se
            row = pw[((pw.A == tr) & (pw.B == "TF-IDF + XGBoost")) | ((pw.B == tr) & (pw.A == "TF-IDF + XGBoost"))]
            ph = float(row.p_holm.iloc[0]) if len(row) else np.nan
            if ph < 0.05 and (lo > D_F1 or hi < -D_F1):
                verdict = "**Refutada** (diferencia significativa tras Holm y mayor que δ)"
            elif -D_F1 < lo and hi < D_F1:
                verdict = "**Apoyada** (equivalencia práctica: IC 90 % dentro de ±δ, TOST)"
            else:
                verdict = "**No concluyente** (ni diferencia relevante demostrada ni equivalencia)"
            v["H4"] = f"{verdict}: ΔF1 ({tr} − TF-IDF + XGBoost) = {d.mean():+.4f}, IC 90 % corregido [{lo:.4f}, {hi:.4f}], p_Holm = {ph:.4f}, {k} repeticiones."
    v.setdefault("H4", "pendiente (requiere transformer y TF-IDF + XGBoost en CV)")
    # H5
    tr = next((m for m in test.modelo.unique() if "BERT" in m), None)
    if tr and "TF-IDF + LR" in test.modelo.unique():
        from project.calibration import expected_calibration_error as ece

        pT = [c for c in pr.columns if c.startswith("pT_")]

        def ece_boot(g: pd.DataFrame, cols: list[str]) -> np.ndarray:
            P = g[cols].to_numpy(); y = np.searchsorted(classes, g.y.astype(str))
            r = np.random.default_rng(0)
            return np.array([ece(P[i], y[i]) for i in [r.integers(0, len(y), len(y)) for _ in range(500)]])
        gt = test[(test.modelo == tr) & (test.semilla == test[test.modelo == tr].semilla.min())]
        gl = test[(test.modelo == "TF-IDF + LR") & (test.semilla == 0)]
        e1 = ece_boot(gt, pcols); eT = ece_boot(gt, pT); el = ece_boot(gl, pcols)
        diff = np.percentile(e1 - el, [2.5, 97.5]); red = 1 - eT.mean() / max(e1.mean(), 1e-9)
        p1 = diff[0] > 0; p2 = red >= 0.5 and abs(eT.mean() - el.mean()) <= D_ECE
        v["H5"] = (f"Parte 1 {'apoyada' if p1 else 'refutada'}: ECE({tr}, T=1) − ECE(TF-IDF+LR) IC 95 % "
                   f"[{diff[0]:.4f}, {diff[1]:.4f}]. Parte 2 {'apoyada' if p2 else 'refutada'}: reducción de ECE con T* = {red:.0%}, "
                   f"ECE(T*) = {eT.mean():.4f} vs TF-IDF+LR {el.mean():.4f}. La exactitud no cambia (fundamento 5).")
    else:
        v["H5"] = "pendiente (requiere el transformer en test)"
    write("tabla_nlp", txt + f"\n**H4.** {v['H4']}\n\n**H5.** {v['H5']}\n")
    if (src / "nlp_eda_params.json").exists():
        e = json.loads((src / "nlp_eda_params.json").read_text())
        write("eda_nlp", f"Textos: {e['n_original']} originales → {e['n']} tras eliminar {e['duplicados_exactos']} duplicados exactos. "
              f"Clases: {e['clases']}. p95 de tokens = {e['p95_tokens']:.0f} → `max_length` = {e['max_length']}; "
              f"p95 de palabras = {e['p95_palabras']:.0f} → `sequence_length` = {e['sequence_length']}.\n")
    return v


# ===================================================================== deriva y EDA
def drift(src: Path) -> dict[str, str]:
    p = src / "drift_h6.csv"
    if not p.exists():
        write("deriva", pend("experimento de deriva con modelos reales", "notebook 07"))
        return {"H6": "pendiente"}
    h = pd.read_csv(p)
    d = h[h.umbrales == "por defecto"]
    st = d.H6.value_counts().to_dict()
    verdict = "Apoyada" if set(st) == {"apoyada"} else "Refutada" if "refutada" in st else "No concluyente"
    write("deriva", "Contraste de H6 (poder a severidad 2 y tasa de falsas alarmas, IC de Wilson):\n\n" + md_table(h, index=False))
    return {"H6": f"**{verdict}** con los umbrales pre-registrados (por tipo de deriva: {st}); ver la tabla para los umbrales calibrados."}


def eda_seg(src: Path) -> None:
    p = src / "seg_imagenes.csv"
    if not p.exists():
        write("eda_seg", pend("EDA de segmentación", "notebook 01")); return
    im = pd.read_csv(p); ins = pd.read_csv(src / "seg_instancias.csv") if (src / "seg_instancias.csv").exists() else None
    t = im.groupby("tipo_celular").agg(imagenes=("id_imagen", "size"), celulas=("n_instancias", "sum"),
                                       celulas_por_imagen=("n_instancias", "median"), cobertura=("cobertura", "median"),
                                       placas=("sample_id", "nunique"))
    txt = (f"Imágenes: {len(im)} ({im.alto.iloc[0]}×{im.ancho.iloc[0]} px); instancias: {int(im.n_instancias.sum())}; "
           f"grupos `sample_id`: {im.sample_id.nunique()}.\n\n" + md_table(t.round(3)))
    if ins is not None:
        txt += "\nÁrea de instancia (px), cuantiles por tipo:\n\n" + md_table(ins.groupby("tipo_celular").area.quantile([.05, .5, .95]).unstack().round(0))
    write("eda_seg", txt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "reports/tables"))
    a = ap.parse_args()
    src = Path(a.src)
    H: dict[str, str] = {}
    H.update(seg(src)); H.update(seg_extras(src)); H.update(nlp(src)); H.update(drift(src)); eda_seg(src)
    rows = "\n".join(f"| {k} | {H.get(k, 'pendiente')} |" for k in ("H1", "H2", "H3", "H4", "H5", "H6"))
    write("hipotesis", "| Hipótesis | Veredicto según el criterio pre-registrado |\n|---|---|\n" + rows + "\n")
    (OUT / "hipotesis.json").write_text(json.dumps(H, indent=2, ensure_ascii=False))
    pending = [k for k, v in H.items() if v.startswith("pendiente")]
    print("\nHipótesis pendientes:", pending or "ninguna")


if __name__ == "__main__":
    main()
