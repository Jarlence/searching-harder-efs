"""Q5 (EFS vs budget-matched random search and simple baselines under HONEST evaluation) and Q6 (fitness metric)."""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", message="Exact p-value calculation does not work if there are zeros")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load as LD  # noqa: E402
import statsx as SX  # noqa: E402

TAB = LD.TAB
EFS = ["ga", "bpso", "bde", "nsga2", "rs"]
BASE = ["anova", "rf", "all", "l1"]
METHODS = EFS + BASE


def method_series(H, B, ds, lr, cp, value="test_ba"):
    """{method: Series indexed by (rep, fold)} of honest outer-test values for one data set and learner."""
    out = {}
    h = H[(H.ds == ds) & (H.learner == lr) & (H.cp == cp)]
    for a in EFS:
        g = h[h.algo == a]
        if len(g):
            out[a] = g.set_index(["rep", "fold"])[value]
    for m in ("anova", "rf", "all"):
        g = B[(B.ds == ds) & (B.learner == lr) & (B.algo == m)]
        if len(g):
            out[m] = g.set_index(["rep", "fold"])[value]
    g = B[(B.ds == ds) & (B.algo == "l1")]
    if len(g):
        out["l1"] = g.set_index(["rep", "fold"])[value]
    return out


def pairs():
    P = [(a, "rs") for a in EFS if a != "rs"]
    P += [(a, b) for a in EFS for b in BASE]
    return P


PRIMARY = int(os.environ.get("EFS_CPREF", 4000))             # B* of docs/ANALYSIS_PLAN.md (only changed when debugging on the fixture)


def advantage_inflation(H, core, cp_refs, n_boot):
    """Exploratory (not pre-registered): when a paper compares algorithms through the REPORTED number (the search fitness) instead of the honest
    outer-test BA, how much does the claimed advantage over budget-matched random search differ from the true one?
    advantage_reported = fit(a) - fit(rs),  advantage_true = test_BA(a) - test_BA(rs)   (paired per outer run, same learner / budget)."""
    rows, rk_rows = [], []
    for cp in cp_refs:
        for lr in ("ridge", "knn"):
            for a in EFS:
                if a == "rs":
                    continue
                per = {"rep": {}, "true": {}, "infl": {}}
                for ds in core:
                    h = H[(H.ds == ds) & (H.learner == lr) & (H.cp == cp) & H.algo.isin([a, "rs"])]
                    if len(h) == 0:
                        continue
                    h = h.assign(fit=1.0 - h.inner_err)
                    fa = h[h.algo == a].set_index(["rep", "fold"]); fr = h[h.algo == "rs"].set_index(["rep", "fold"])
                    idx = fa.index.intersection(fr.index)
                    if len(idx) < 5:
                        continue
                    d_rep = (fa.loc[idx, "fit"] - fr.loc[idx, "fit"]).to_numpy(dtype=float)
                    d_true = (fa.loc[idx, "test_ba"] - fr.loc[idx, "test_ba"]).to_numpy(dtype=float)
                    per["rep"][ds] = [(d_rep[:, None], 1.0)]; per["true"][ds] = [(d_true[:, None], 1.0)]; per["infl"][ds] = [((d_rep - d_true)[:, None], 1.0)]
                if len(per["rep"]) < 3:
                    continue
                row = dict(cp=cp, learner=lr, algo=a)
                for key in ("rep", "true", "infl"):
                    bt = SX.double_boot(per[key], n_boot, seed=20261013)
                    if bt is None:
                        break
                    row[key] = bt["point"]; row[key + "_lo"] = bt["lo"]; row[key + "_hi"] = bt["hi"]
                    row["n_ds"] = len(bt["ds_values"])
                else:
                    rows.append(row)
        # ranking by reported fitness vs honest BA (mean rank over data sets, 1 = best), the 5 search algorithms
        for lr in ("ridge", "knn"):
            by = {"fitness": {}, "honest": {}}
            for ds in core:
                h = H[(H.ds == ds) & (H.learner == lr) & (H.cp == cp)]
                if h.algo.nunique() < 5:
                    continue
                h = h.assign(fit=1.0 - h.inner_err)
                by["fitness"][ds] = h.groupby("algo").fit.mean(); by["honest"][ds] = h.groupby("algo").test_ba.mean()
            if len(by["honest"]) >= 5:
                R = {k: pd.DataFrame(v).T.rank(axis=1, ascending=False).mean(axis=0) for k, v in by.items()}
                tau = stats.kendalltau(R["fitness"].reindex(EFS), R["honest"].reindex(EFS)).correlation
                for algo in EFS:
                    rk_rows.append(dict(cp=cp, learner=lr, algo=algo, mean_rank_by_fitness=float(R["fitness"][algo]), mean_rank_by_honest=float(R["honest"][algo]),
                                        kendall_tau_between_rankings=float(tau), n_ds=len(by["honest"])))
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "Q5b_advantage_inflation.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(rk_rows).to_csv(os.path.join(TAB, "Q5b_rank_by_criterion.csv"), index=False, encoding="utf-8-sig")
    return rows, rk_rows


def run(cp_refs=None, n_boot=10000):
    cp_refs = cp_refs or tuple(dict.fromkeys((PRIMARY, 1000, 16000)))
    t = LD.load_all(use_cache=False)
    H = t["honest"]; B = t["base"]
    H = H[H.suite == "E1h"]
    core = [d for d in LD.CORE if d in set(H.ds)]
    out = {}
    allrows, ds_rows, mean_rows = [], [], []
    for cp in cp_refs:
        for lr in ("ridge", "knn"):
            diffs = {}                                             # (a,b) -> {ds: Series of paired differences}
            for ds in core:
                ms = method_series(H, B, ds, lr, cp)
                if "rs" not in ms:
                    continue
                for m, s in ms.items():
                    mean_rows.append(dict(cp=cp, learner=lr, ds=ds, method=m, mean_ba=float(s.mean()), n_runs=len(s)))
                # within-data-set rows are DESCRIPTIVE only: the outer runs of one data set share samples and training data, so no p-value is computed
                # from them (docs/DEVIATIONS.md #8); inference uses one value per data set (below)
                for a, b in pairs():
                    if a in ms and b in ms:
                        d = (ms[a] - ms[b]).dropna()
                        if len(d) >= 5:
                            diffs.setdefault((a, b), {})[ds] = d
                            ds_rows.append(dict(cp=cp, learner=lr, ds=ds, a=a, b=b, n=len(d), mean_diff=float(d.mean()), sd=float(d.std(ddof=1)),
                                                delta=SX.sign_delta(d.to_numpy())))
            # across data sets
            fam = []
            for (a, b), dd in diffs.items():
                per = {ds: [(d.to_numpy(dtype=float)[:, None], 1.0)] for ds, d in dd.items()}
                bt = SX.double_boot(per, n_boot if cp == PRIMARY else 2000, seed=20261011)
                if bt is None:
                    continue
                v = np.array(list(bt["ds_values"].values()))
                fam.append(dict(cp=cp, learner=lr, a=a, b=b, n_ds=len(v), mean_diff=bt["point"], lo=bt["lo"], hi=bt["hi"], n_pos=int((v > 0).sum()),
                                n_neg=int((v < 0).sum()), delta=SX.sign_delta(v), p=SX.wilcoxon(v)))
            if fam:
                adj = SX.holm([f["p"] for f in fam])
                for f, q in zip(fam, adj):
                    f["p_holm"] = q                                    # conservative: all 24 pairwise comparisons of this learner / budget in one family
                # pre-registered family (docs/ANALYSIS_PLAN.md section 2, Q5): "each method vs rs" = 4 EFS algorithms - rs and rs - 4 baselines (8 tests)
                pre = [i for i, f in enumerate(fam) if f["b"] == "rs" or (f["a"] == "rs" and f["b"] in BASE)]
                adj8 = SX.holm([fam[i]["p"] for i in pre])
                for f in fam:
                    f["p_holm_prereg"] = np.nan
                for i, q in zip(pre, adj8):
                    fam[i]["p_holm_prereg"] = q
                allrows += fam
            # Friedman over methods with a complete data-set x method table
            mr = pd.DataFrame([r for r in mean_rows if r["cp"] == cp and r["learner"] == lr])
            if len(mr):
                piv = mr.pivot_table(index="ds", columns="method", values="mean_ba").dropna(axis=0)
                piv = piv[[m for m in METHODS if m in piv.columns]]
                if piv.shape[0] >= 5 and piv.shape[1] >= 3:
                    fr = stats.friedmanchisquare(*[piv[c].to_numpy() for c in piv.columns])
                    ranks = piv.rank(axis=1, ascending=False).mean(axis=0)
                    out["friedman_cp%d_%s" % (cp, lr)] = dict(n_ds=int(piv.shape[0]), chi2=float(fr.statistic), p=float(fr.pvalue),
                                                              mean_rank={k: float(v) for k, v in ranks.items()})
    pd.DataFrame(allrows).to_csv(os.path.join(TAB, "Q5_across_datasets.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(ds_rows).to_csv(os.path.join(TAB, "Q5_within_dataset.csv"), index=False, encoding="utf-8-sig")
    wd = pd.DataFrame(ds_rows)
    if len(wd):
        cnt = wd.groupby(["cp", "learner", "a", "b"]).apply(lambda g: pd.Series(dict(
            n_ds=len(g), a_higher_mean=int((g.mean_diff > 0).sum()), a_lower_mean=int((g.mean_diff < 0).sum())))).reset_index()
        cnt.to_csv(os.path.join(TAB, "Q5_within_counts.csv"), index=False, encoding="utf-8-sig")
        wd = wd.assign(family=np.where(wd.ds.isin(LD.HD), "HD", "IMB"))          # exploratory: 9 high-dimensional small-sample vs 4 imbalanced low-dimensional
        fm = wd.groupby(["cp", "learner", "family", "a", "b"]).apply(lambda g: pd.Series(dict(
            n_ds=len(g), mean_diff=float(g.mean_diff.mean()), n_pos=int((g.mean_diff > 0).sum())))).reset_index()
        fm.to_csv(os.path.join(TAB, "Q5_by_family.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(mean_rows).to_csv(os.path.join(TAB, "Q5_mean_ba.csv"), index=False, encoding="utf-8-sig")
    out["Q5_across"] = allrows
    out["Q5b"], out["Q5b_rank"] = advantage_inflation(H, core, cp_refs, n_boot if n_boot < 3000 else 3000)
    # ---------------------------------------------------------------- Q6 fitness metric (exploratory)
    E1 = t["honest"][t["honest"].suite == "E1h"]; E3 = t["honest"][t["honest"].suite == "E3m"]
    q6 = []
    if len(E3):
        for cp in tuple(dict.fromkeys((PRIMARY, 1000))):
            for lr in ("ridge", "knn"):
                for algo in ("ga", "nsga2"):
                    for ds in core:
                        e3 = E3[(E3.ds == ds) & (E3.learner == lr) & (E3.algo == algo) & (E3.cp == cp)]
                        if len(e3) == 0:
                            continue
                        reps = sorted(e3.rep.unique())
                        e1 = E1[(E1.ds == ds) & (E1.learner == lr) & (E1.algo == algo) & (E1.cp == cp) & (E1.rep.isin(reps))]
                        base = e1.set_index(["rep", "fold"])
                        for metric in ("acc", "gm", "auc"):
                            m3 = e3[e3.metric == metric].set_index(["rep", "fold"])
                            idx = base.index.intersection(m3.index)
                            if len(idx) < 3:
                                continue
                            row = dict(cp=cp, learner=lr, algo=algo, ds=ds, metric=metric, n=len(idx))
                            for o in ("ba", "auc", "gm", "mcc", "f1", "acc", "tpr", "tnr"):
                                row["d_" + o] = float((m3.loc[idx, "test_" + o] - base.loc[idx, "test_" + o]).mean())
                            row["d_k"] = float((m3.loc[idx, "k"] - base.loc[idx, "k"]).mean())
                            q6.append(row)
    q6 = pd.DataFrame(q6)
    q6.to_csv(os.path.join(TAB, "Q6_metric_by_dataset.csv"), index=False, encoding="utf-8-sig")
    q6s = []
    if len(q6):
        for (cp, lr, algo, metric), g in q6.groupby(["cp", "learner", "algo", "metric"]):
            for o in ("ba", "auc", "mcc", "tpr", "tnr", "k"):
                v = g["d_" + o].to_numpy()
                q6s.append(dict(cp=cp, learner=lr, algo=algo, fitness_metric=metric, outcome=o, n_ds=len(v), mean_diff_vs_BA_fitness=float(v.mean()),
                                n_pos=int((v > 0).sum()), p=SX.wilcoxon(v)))
        q6s = pd.DataFrame(q6s)
        q6s["p_holm"] = np.nan
        for (cp, o), g in q6s.groupby(["cp", "outcome"]):
            q6s.loc[g.index, "p_holm"] = SX.holm(g.p.to_numpy())
        q6s.to_csv(os.path.join(TAB, "Q6_metric_summary.csv"), index=False, encoding="utf-8-sig")
        out["Q6"] = q6s.to_dict(orient="records")
    json.dump(out, open(os.path.join(TAB, "compare_summary.json"), "w"), ensure_ascii=False, indent=1, default=float)
    return out


if __name__ == "__main__":
    out = run(n_boot=int(sys.argv[1]) if len(sys.argv) > 1 else 10000)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    a = pd.DataFrame(out.get("Q5_across", []))
    if len(a):
        print(a[(a.cp == PRIMARY) & (a.b == "rs")].round(4).to_string())
        print(a[(a.cp == PRIMARY) & (a.b != "rs")].round(4).head(30).to_string())
    for k, v in out.items():
        if k.startswith("friedman"):
            print(k, {x: (round(y, 3) if not isinstance(y, dict) else {m: round(z, 2) for m, z in y.items()}) for x, y in v.items()})
