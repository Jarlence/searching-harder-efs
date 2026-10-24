"""Special studies: SMOTE-before-split (E5so, E5si), unit leakage on repeated measures (E6u, E6uf), long budgets (E7l),
subset stability across seeds (E4s, exploratory) and a descriptive table of honest performance (E1h)."""
import glob
import itertools
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Exact p-value calculation does not work if there are zeros")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load as LD  # noqa: E402
import statsx as SX  # noqa: E402

TAB = LD.TAB


def _cross(rows, key_cols, val_cols, n_boot=5000, seed=0):
    """rows: DataFrame with one value per (ds, run) per col in val_cols -> summary across data sets for each key."""
    out = []
    for key, g in rows.groupby(key_cols):
        key = key if isinstance(key, tuple) else (key,)
        for vc in val_cols:
            per = {}
            for ds, gg in g.groupby("ds"):
                v = gg[vc].dropna().to_numpy(dtype=float)
                if len(v):
                    per[ds] = [(v[:, None], 1.0)]
            if len(per) < 2:
                continue
            bt = SX.double_boot(per, n_boot, seed=seed)
            if bt is None:
                continue
            v = np.array(list(bt["ds_values"].values()))
            out.append(dict(zip(key_cols, key), outcome=vc, n_ds=len(v), mean=bt["point"], lo=bt["lo"], hi=bt["hi"], n_pos=int((v > 0).sum()),
                            delta=SX.sign_delta(v), p=SX.wilcoxon(v)))
    return pd.DataFrame(out)


def smote_outer(t):
    S = t["smote_outer"]
    if len(S) == 0:
        return None
    a = S[S.protocol == "before_split"].set_index(["ds", "learner", "rep"])
    b = S[S.protocol == "honest"].set_index(["ds", "learner", "rep"])
    idx = a.index.intersection(b.index)
    d = (a.loc[idx, list(LD.METRICS)] - b.loc[idx, list(LD.METRICS)]).reset_index()
    d.to_csv(os.path.join(TAB, "E5so_paired_diffs.csv"), index=False, encoding="utf-8-sig")
    res = _cross(d.assign(all="all"), ["learner"], ["ba", "auc", "mcc", "f1", "acc"], seed=31)
    res["p_holm"] = np.nan
    for o, g in res.groupby("outcome"):
        res.loc[g.index, "p_holm"] = SX.holm(g.p.to_numpy())
    res.to_csv(os.path.join(TAB, "E5so_summary.csv"), index=False, encoding="utf-8-sig")
    return res


def smote_decomp(t):
    """X9s (docs/DEVIATIONS.md #9).  The over-statement of SMOTE-before-split mixes two things: (i) the TEST SET changes (it now contains synthetic minority points
    that interpolate training points, and the class prior is 50 %), which moves prevalence-dependent metrics (F1, MCC) mechanically; (ii) LEAKAGE proper
    (synthetic copies of test-fold real points in the training part).  With the same seeds the flawed pipeline is re-run and scored on all rows and on the real rows
    only:  total = all rows - honest;  composition = all rows - real rows;  leakage = real rows - honest."""
    S = t["smote_outer"]
    recs = []
    for f in glob.glob(os.path.join(LD.EXTRA, "X9s", "*.json")):
        r = json.load(open(f))
        if r.get("status") == "ok":
            recs.append(r)
    if not recs or len(S) == 0:
        return None
    hon = S[S.protocol == "honest"].set_index(["ds", "learner", "rep"])
    bef = S[S.protocol == "before_split"].set_index(["ds", "learner", "rep"])
    rows = []; worst = 0.0
    for r in recs:
        sp = r["spec"]; key = (sp["ds"], sp["learner"], sp["rep"])
        if key not in hon.index or key not in bef.index:
            continue
        row = dict(ds=sp["ds"], learner=sp["learner"], rep=sp["rep"])
        for m in ("ba", "auc", "mcc", "f1", "acc"):
            a, re_, h = r["result"]["all"][m], r["result"]["real"][m], float(hon.loc[key, m])
            worst = max(worst, abs(a - float(bef.loc[key, m])))                         # the replay must reproduce E5so exactly
            row["total_" + m] = a - h; row["composition_" + m] = a - re_; row["leakage_" + m] = re_ - h
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TAB, "X9s_paired_diffs.csv"), index=False, encoding="utf-8-sig")
    cols = [c for c in d.columns if c.split("_")[0] in ("total", "composition", "leakage")]
    res = _cross(d, ["learner"], cols, seed=33)
    res["component"] = [o.split("_")[0] for o in res.outcome]; res["metric"] = [o.split("_")[1] for o in res.outcome]
    res.to_csv(os.path.join(TAB, "X9s_decomposition.csv"), index=False, encoding="utf-8-sig")
    json.dump(dict(n_jobs=len(d), max_abs_diff_vs_E5so=worst), open(os.path.join(TAB, "X9s_check.json"), "w"))
    return res


def smote_inner(t):
    S = t["smote_inner"]
    if len(S) == 0:
        return None
    S = S.copy(); S["gap_B"] = (1 - S.inner_err) - S.test_ba
    keys = ["ds", "learner", "rep", "fold", "cp"]
    c = S[S["mode"] == "clean"].set_index(keys); l = S[S["mode"] == "leaky"].set_index(keys)
    idx = c.index.intersection(l.index)
    d = pd.DataFrame({"d_test_ba": l.loc[idx, "test_ba"] - c.loc[idx, "test_ba"], "d_test_auc": l.loc[idx, "test_auc"] - c.loc[idx, "test_auc"],
                      "d_gapB": l.loc[idx, "gap_B"] - c.loc[idx, "gap_B"], "d_k": l.loc[idx, "k"] - c.loc[idx, "k"],
                      "gapB_clean": c.loc[idx, "gap_B"], "gapB_leaky": l.loc[idx, "gap_B"]}).reset_index()
    d.to_csv(os.path.join(TAB, "E5si_paired_diffs.csv"), index=False, encoding="utf-8-sig")
    res = _cross(d, ["learner", "cp"], ["d_test_ba", "d_test_auc", "d_gapB", "gapB_clean", "gapB_leaky"], seed=32)
    res.to_csv(os.path.join(TAB, "E5si_summary.csv"), index=False, encoding="utf-8-sig")
    return res


def unit(t):
    H = t["honest"]; U = H[(H.suite == "E6u")].copy()
    F = t["full"]; Uf = F[F.suite == "E6uf"].copy()
    if len(U) == 0:
        return None
    U["gap_B"] = (1 - U.inner_err) - U.test_ba
    g = U.groupby(["algo", "learner", "cp", "outer_group", "inner_group"]).agg(mean_ba=("test_ba", "mean"), mean_auc=("test_auc", "mean"),
                                                                              mean_fit=("inner_err", lambda x: 1 - x.mean()), gap_B=("gap_B", "mean"),
                                                                              n=("test_ba", "size")).reset_index()
    g.to_csv(os.path.join(TAB, "E6u_means.csv"), index=False, encoding="utf-8-sig")
    # paired by (algo, learner, cp, inner_group, rep): record-wise outer - subject-wise outer
    rows = []
    rep_mean = U.groupby(["algo", "learner", "cp", "outer_group", "inner_group", "rep"]).agg(ba=("test_ba", "mean"), auc=("test_auc", "mean"),
                                                                                              gapB=("gap_B", "mean")).reset_index()
    for (algo, lr, cp, ig), gg in rep_mean.groupby(["algo", "learner", "cp", "inner_group"]):
        a = gg[~gg.outer_group].set_index("rep"); b = gg[gg.outer_group].set_index("rep")
        idx = a.index.intersection(b.index)
        if len(idx) < 3:
            continue
        for o in ("ba", "auc", "gapB"):
            d = (a.loc[idx, o] - b.loc[idx, o]).to_numpy()
            # descriptive: the repeats are repetitions of cross-validation on ONE data set (same subjects), not independent evidence, so no p-value
            rows.append(dict(algo=algo, learner=lr, cp=cp, inner_group=bool(ig), outcome=o, n_rep=len(idx), mean_record_minus_subject=float(d.mean()),
                             sd=float(d.std(ddof=1)), n_pos=int((d > 0).sum())))
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "E6u_record_vs_subject.csv"), index=False, encoding="utf-8-sig")
    # does a group-aware inner CV help when the outer test is honest (subject-wise)?
    rows2 = []
    for (algo, lr, cp), gg in rep_mean[rep_mean.outer_group].groupby(["algo", "learner", "cp"]):
        a = gg[gg.inner_group].set_index("rep"); b = gg[~gg.inner_group].set_index("rep"); idx = a.index.intersection(b.index)
        if len(idx) >= 3:
            d = (a.loc[idx, "ba"] - b.loc[idx, "ba"]).to_numpy()
            rows2.append(dict(algo=algo, learner=lr, cp=cp, n_rep=len(idx), mean_groupinner_minus_recordinner=float(d.mean()), n_pos=int((d > 0).sum()),
                              sd=float(d.std(ddof=1))))
    pd.DataFrame(rows2).to_csv(os.path.join(TAB, "E6u_inner_group_benefit.csv"), index=False, encoding="utf-8-sig")
    out = dict(record_vs_subject=rows, inner_group_benefit=rows2)
    if len(Uf):
        Uf["fit_ba"] = 1 - Uf.reported_err
        f = Uf.groupby(["algo", "learner", "cp", "group"]).agg(leaky_cv_ba=("leaky_cv_ba", "mean"), fit_ba=("fit_ba", "mean"), k=("k", "mean"),
                                                               n=("seed", "size")).reset_index()
        f.to_csv(os.path.join(TAB, "E6uf_means.csv"), index=False, encoding="utf-8-sig")
        out["E6uf"] = f.to_dict(orient="records")
    return out


def long_budget(t):
    H = t["honest"]; L = H[H.suite == "E7l"].copy()
    if len(L) == 0:
        return None
    L["gap_B"] = (1 - L.inner_err) - L.test_ba
    g = L.groupby(["ds", "learner", "algo", "cp"]).agg(fit_ba=("inner_err", lambda x: 1 - x.mean()), test_ba=("test_ba", "mean"), gap_B=("gap_B", "mean"),
                                                       k=("k", "mean"), n=("test_ba", "size")).reset_index()
    g.to_csv(os.path.join(TAB, "E7l_means.csv"), index=False, encoding="utf-8-sig")
    cps = sorted(L.cp.unique())
    rows = []
    for (ds, lr, algo), gg in L.groupby(["ds", "learner", "algo"]):
        lo, hi = cps[0], int(gg.cp.max())
        a = gg[gg.cp == hi].set_index(["rep", "fold"]); b = gg[gg.cp == lo].set_index(["rep", "fold"]); idx = a.index.intersection(b.index)
        if len(idx) < 3 or hi == lo:
            continue
        cm = gg.groupby("cp").gap_B.mean()
        rows.append(dict(ds=ds, learner=lr, algo=algo, cp_lo=lo, cp_hi=hi, n=len(idx), d_gapB=float((a.loc[idx, "gap_B"] - b.loc[idx, "gap_B"]).mean()),
                         d_test_ba=float((a.loc[idx, "test_ba"] - b.loc[idx, "test_ba"]).mean()), d_fit=float(((1 - a.loc[idx, "inner_err"]) - (1 - b.loc[idx, "inner_err"])).mean()),
                         rho_gapB=SX.spearman_cell(cm.index.to_numpy(), cm.to_numpy())))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(TAB, "E7l_hi_vs_lo.csv"), index=False, encoding="utf-8-sig")
    # one value per DATA SET (mean over learners and, for "all", algorithms); only four data sets, so the summary is descriptive: mean and range over
    # data sets, no interval or test (docs/DEVIATIONS.md #8)
    pooled = []
    for tag, g in [("all", r)] + [(a, r[r.algo == a]) for a in sorted(r.algo.unique())]:
        dsv = g.groupby("ds")[["d_fit", "d_test_ba", "d_gapB"]].mean()
        row = dict(algo=tag, n_cells=len(g), n_ds=len(dsv))
        for col in ("d_fit", "d_test_ba", "d_gapB"):
            v = dsv[col].to_numpy(dtype=float)
            row[col] = float(v.mean()); row[col + "_min"] = float(v.min()); row[col + "_max"] = float(v.max())
        pooled.append(row)
    pd.DataFrame(pooled).to_csv(os.path.join(TAB, "E7l_pooled.csv"), index=False, encoding="utf-8-sig")
    byb = L.assign(fit=1 - L.inner_err).groupby(["learner", "algo", "cp"]).agg(fit=("fit", "mean"), test_ba=("test_ba", "mean"), gap_B=("gap_B", "mean"),
                                                                                  k=("k", "mean"), n=("test_ba", "size")).reset_index()
    byb.to_csv(os.path.join(TAB, "E7l_by_budget.csv"), index=False, encoding="utf-8-sig")
    return r


def stability(t):
    """Exploratory: mean pairwise Jaccard of the selected subsets of independent seeds on the same fold, and the spread of their test BA."""
    H = t["honest"]; E = H[H.suite == "E4s"]
    if len(E) == 0:
        return None
    rows = []
    for (ds, algo, lr, cp, fold), g in E.groupby(["ds", "algo", "learner", "cp", "fold"]):
        sets = [set(map(int, s)) for s in g.sel]
        if len(sets) < 3:
            continue
        js = [len(a & b) / max(len(a | b), 1) for a, b in itertools.combinations(sets, 2)]
        rows.append(dict(ds=ds, algo=algo, learner=lr, cp=cp, fold=fold, n_seeds=len(sets), jaccard=float(np.mean(js)), mean_k=float(g.k.mean()),
                         ba_mean=float(g.test_ba.mean()), ba_sd=float(g.test_ba.std(ddof=1))))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(TAB, "E4_stability.csv"), index=False, encoding="utf-8-sig")
    return r


def honest_perf(t):
    H = t["honest"]; E = H[H.suite == "E1h"]
    g = E.groupby(["ds", "learner", "algo", "cp"]).agg(test_ba=("test_ba", "mean"), test_auc=("test_auc", "mean"), test_mcc=("test_mcc", "mean"),
                                                       k=("k", "mean"), fit_ba=("inner_err", lambda x: 1 - x.mean()), n=("test_ba", "size")).reset_index()
    g.to_csv(os.path.join(TAB, "E1h_honest_means.csv"), index=False, encoding="utf-8-sig")
    return g


def run():
    t = LD.load_all(use_cache=False)
    out = {}
    for name, fn in (("smote_outer", smote_outer), ("smote_decomp", smote_decomp), ("smote_inner", smote_inner), ("unit", unit), ("long", long_budget), ("stability", stability),
                     ("honest_perf", honest_perf)):
        r = fn(t)
        out[name] = r
        print("==", name, "done" if r is not None else "(no data)")
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
    out = run()
    for k in ("smote_outer", "smote_inner"):
        if out[k] is not None:
            print(out[k].round(4).to_string())
    if out["unit"] is not None:
        print(pd.DataFrame(out["unit"]["record_vs_subject"]).round(4).head(24).to_string())
    if out["long"] is not None:
        print(out["long"].round(4).head(20).to_string())
    if out["stability"] is not None:
        print(out["stability"].groupby(["algo", "learner", "cp"]).jaccard.mean().round(3).to_string())
