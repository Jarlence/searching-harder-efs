"""Analysis of the extra studies X1s (scaler refit), X2k (inner folds), X3u (synthetic units), X4r (random-subset landscape)."""
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load as LD  # noqa: E402
import statsx as SX  # noqa: E402

TAB = LD.TAB
BASE = os.environ.get("EFS_JOBS_EXTRA", os.path.join(LD.ROOT, "results", "jobs_extra"))
METRICS = LD.METRICS


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def ci_t(x):
    """Student-t interval of the mean of one value per synthetic data set (the unit of inference); same rule as for the real data sets."""
    return SX.t_interval(np.asarray(x, float))


def load_extra():
    out = {}
    shas = set()
    for suite in ("X1s", "X2k", "X3u", "X4r", "X5s"):
        rows = []
        for f in glob.glob(os.path.join(BASE, suite, "*.json")):
            rec = json.load(open(f))
            if rec.get("status") != "ok":
                continue
            shas.add(rec["code_sha"])
            s = rec["spec"]
            base = {k: v for k, v in s.items() if k not in ("cps", "kind")}
            for key, r in rec["result"].items():
                row = dict(base)
                if suite == "X4r":
                    row.update(k=int(key), **r)
                else:
                    row.update(cp=int(key), k=r["k"], inner_err=r["inner_err"], **{"test_" + m: r["test_" + m] for m in METRICS})
                    if "n_inf_selected" in r:
                        row["n_inf_selected"] = r["n_inf_selected"]
                rows.append(row)
        out[suite] = pd.DataFrame(rows)
    out["_shas"] = shas
    return out


def scaler(E, H):
    X = E["X1s"]
    if len(X) == 0:
        return None
    m = H[H.suite == "E1h"]
    keys = ["ds", "algo", "learner", "rep", "fold", "cp"]
    a = X.set_index(keys); b = m.set_index(keys)
    idx = a.index.intersection(b.index)
    a = a.loc[idx]; b = b.loc[idx]
    d = pd.DataFrame({"d_ba": a.test_ba - b.test_ba, "d_auc": a.test_auc - b.test_auc, "d_fit": (1 - a.inner_err) - (1 - b.inner_err),
                      "d_gapB": ((1 - a.inner_err) - a.test_ba) - ((1 - b.inner_err) - b.test_ba), "d_k": a.k - b.k}).reset_index()
    d.to_csv(os.path.join(TAB, "X1s_paired.csv"), index=False, encoding="utf-8-sig")
    rows = []
    for key, g in d.groupby(["ds", "learner", "algo", "cp"]):                      # descriptive means per cell (no interval: the runs of a cell share samples)
        r = dict(zip(["ds", "learner", "algo", "cp"], key), n=len(g))
        for o in ("d_ba", "d_gapB", "d_fit"):
            r[o] = float(g[o].mean())
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TAB, "X1s_summary.csv"), index=False, encoding="utf-8-sig")
    # one value per DATA SET (three data sets: descriptive only, no interval and no equivalence claim; docs/DEVIATIONS.md #8)
    dsv = d.groupby("ds")[["d_ba", "d_gapB", "d_fit", "d_k"]].mean()
    tot = {o: dict(mean=float(dsv[o].mean()), min=float(dsv[o].min()), max=float(dsv[o].max()), per_ds={k: float(v) for k, v in dsv[o].items()}) for o in dsv.columns}
    tot["n_pairs"] = int(len(d)); tot["n_ds"] = int(len(dsv))
    return dict(table=t, pooled=tot)


def innerk(E, H):
    X = E["X2k"]
    if len(X) == 0:
        return None
    m = H[(H.suite == "E1h") & (H.ds.isin(X.ds.unique())) & (H.algo.isin(X.algo.unique())) & (H.cp.isin(X.cp.unique()))].copy()
    m["inner_k"] = 5
    A = pd.concat([X, m[X.columns.intersection(m.columns)]], ignore_index=True)
    A["gap_B"] = (1 - A.inner_err) - A.test_ba
    A["fit"] = 1 - A.inner_err
    keys = ["ds", "learner", "algo", "cp", "inner_k"]
    t = A.groupby(keys).agg(test_ba=("test_ba", "mean"), fit=("fit", "mean"), gap_B=("gap_B", "mean"), k=("k", "mean"), n=("test_ba", "size")).reset_index()
    t.to_csv(os.path.join(TAB, "X2k_means.csv"), index=False, encoding="utf-8-sig")
    # paired differences vs 5 inner folds on the same outer run
    rows = []
    for (ds, lr, algo, cp), g in A.groupby(["ds", "learner", "algo", "cp"]):
        ref = g[g.inner_k == 5].set_index(["rep", "fold"])
        for ik in (3, 10):
            o = g[g.inner_k == ik].set_index(["rep", "fold"]); idx = o.index.intersection(ref.index)
            if len(idx) < 5:
                continue
            for name, col in (("test_ba", "test_ba"), ("gap_B", "gap_B"), ("fit", "fit")):
                d = (o.loc[idx, col] - ref.loc[idx, col]).to_numpy()
                lo, hi = boot_ci(d)
                rows.append(dict(ds=ds, learner=lr, algo=algo, cp=cp, inner_k=ik, outcome=name, n=len(idx), mean_diff_vs_k5=float(d.mean()), lo=lo, hi=hi))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(TAB, "X2k_paired.csv"), index=False, encoding="utf-8-sig")
    pooled = r.groupby(["inner_k", "outcome", "cp"]).mean_diff_vs_k5.mean().reset_index()
    return dict(table=t, paired=r, pooled=pooled)


def synth(E):
    X = E["X3u"]
    if len(X) == 0:
        return None
    X = X.copy(); X["gap_B"] = (1 - X.inner_err) - X.test_ba; X["fit"] = 1 - X.inner_err
    X["precision"] = X.n_inf_selected / X.k.clip(lower=1); X["recall"] = X.n_inf_selected / 10.0
    keys = ["U", "sig_f", "algo", "cp", "outer_group", "inner_group"]
    t = X.groupby(keys).agg(test_ba=("test_ba", "mean"), fit=("fit", "mean"), gap_B=("gap_B", "mean"), precision=("precision", "mean"),
                            recall=("recall", "mean"), k=("k", "mean"), n=("test_ba", "size")).reset_index()
    t.to_csv(os.path.join(TAB, "X3u_means.csv"), index=False, encoding="utf-8-sig")
    rows = []
    for (U, sf, algo, cp), g in X.groupby(["U", "sig_f", "algo", "cp"]):
        def arm(og, ig):
            return g[(g.outer_group == og) & (g.inner_group == ig)].set_index(["inst", "fold"])
        ff, ft, tf, tt = arm(False, False), arm(False, True), arm(True, False), arm(True, True)
        for name, a, b in (("record_minus_subject_eval(ff-tt)", ff, tt), ("record_minus_subject_eval_same_inner(ff-tf)", ff, tf),
                           ("unit_aware_search_benefit(tt-tf)", tt, tf)):
            idx = a.index.intersection(b.index)
            if len(idx) < 5:
                continue
            dd = (a.loc[idx, "test_ba"] - b.loc[idx, "test_ba"]).groupby(level="inst").mean().to_numpy()      # one value per synthetic instance (= data set)
            lo, hi = ci_t(dd)
            rows.append(dict(U=U, sig_f=sf, algo=algo, cp=cp, contrast=name, n=len(dd), mean_diff=float(dd.mean()), lo=lo, hi=hi))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(TAB, "X3u_contrasts.csv"), index=False, encoding="utf-8-sig")
    # pooled over U, algo, cp (pairs = (U, algo, cp, inst, fold)); the sig_f = 0 row is the pre-written control (|difference| < 0.02)
    pooled = []
    for sf, g in X.groupby("sig_f"):
        for name, (oa, ia), (ob, ib) in (("record_minus_subject_eval(ff-tt)", (False, False), (True, True)),
                                         ("unit_aware_search_benefit(tt-tf)", (True, True), (True, False))):
            keys = ["U", "algo", "cp", "inst", "fold"]
            a = g[(g.outer_group == oa) & (g.inner_group == ia)].set_index(keys).test_ba
            b = g[(g.outer_group == ob) & (g.inner_group == ib)].set_index(keys).test_ba
            idx = a.index.intersection(b.index)
            dd = (a.loc[idx] - b.loc[idx]).groupby(level=["U", "inst"]).mean().to_numpy()                  # one value per synthetic instance (= data set)
            lo, hi = ci_t(dd)
            pooled.append(dict(sig_f=sf, contrast=name, n=len(dd), n_pairs=int(len(idx)), mean_diff=float(dd.mean()), lo=lo, hi=hi,
                               control_ok=bool(abs(dd.mean()) < 0.02) if sf == 0 else None))
    pd.DataFrame(pooled).to_csv(os.path.join(TAB, "X3u_pooled_by_sigma.csv"), index=False, encoding="utf-8-sig")
    return dict(table=t, contrasts=r, pooled=pd.DataFrame(pooled))


def land(E):
    X = E["X4r"]
    if len(X) == 0:
        return None
    X = X.copy(); X["perm"] = X["perm"].fillna(-1)
    rows = []
    for (ds, lr, k, perm), g in X.groupby(["ds", "learner", "k", "perm"]):
        for B in (250, 1000, 4000, 16000):
            z = float(stats.norm.ppf((B - 0.375) / (B + 0.25)))                       # Blom approximation of E[max of B N(0,1)]
            obs = (g["max_inn_%d" % B] - g["tst_at_max_%d" % B]).mean()
            pred = ((g.inn_mean - g.tst_mean) + z * (g.inn_sd - g.pearson * g.tst_sd)).mean()
            rows.append(dict(ds=ds, learner=lr, k=k, perm=int(perm), B=B, z_B=z, n_splits=len(g), inn_mean=g.inn_mean.mean(), inn_sd=g.inn_sd.mean(),
                             tst_mean=g.tst_mean.mean(), pearson=g.pearson.mean(), max_inn=g["max_inn_%d" % B].mean(),
                             tst_at_max=g["tst_at_max_%d" % B].mean(), gap_obs=float(obs), gap_pred_normal=float(pred)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TAB, "X4r_landscape.csv"), index=False, encoding="utf-8-sig")
    r2 = {}
    for perm, g in t.groupby("perm"):
        o, p = g.gap_obs.to_numpy(), g.gap_pred_normal.to_numpy()
        ss = ((o - o.mean()) ** 2).sum()
        r2[int(perm)] = dict(n_cells=len(g), r2_pred_vs_obs=float(1 - ((o - p) ** 2).sum() / ss) if ss > 0 else float("nan"),
                             pearson=float(np.corrcoef(o, p)[0, 1]), mean_abs_err=float(np.abs(o - p).mean()))
    return dict(table=t, r2=r2)


def rs100(E, H):
    """X5s: Q5 re-done with a size-matched random-search null (subset size 1..100).  Paired with E1h on the same outer runs."""
    X = E.get("X5s")
    if X is None or len(X) == 0:
        return None
    m = H[(H.suite == "E1h") & H.cp.isin([1000, 4000])]
    keys = ["ds", "learner", "rep", "fold", "cp"]
    r100 = X.set_index(keys)
    rows = []
    for algo in LD.ALGOS:
        for cp in (4000, 1000):
            for lr in ("ridge", "knn"):
                per = {}
                for ds in sorted(set(X.ds)):
                    a = m[(m.algo == algo) & (m.ds == ds) & (m.learner == lr) & (m.cp == cp)].set_index(keys)
                    b = r100[(r100.index.get_level_values("ds") == ds) & (r100.index.get_level_values("learner") == lr) & (r100.index.get_level_values("cp") == cp)]
                    idx = a.index.intersection(b.index)
                    if len(idx) >= 5:
                        per[ds] = [((a.loc[idx, "test_ba"] - b.loc[idx, "test_ba"]).to_numpy(dtype=float)[:, None], 1.0)]
                if len(per) < 3:
                    continue
                bt = SX.double_boot(per, 5000, seed=20261016)
                if bt is None:
                    continue
                v = np.array(list(bt["ds_values"].values()))
                rows.append(dict(algo=algo, cp=cp, learner=lr, n_ds=len(v), mean_diff_vs_rs100=bt["point"], lo=bt["lo"], hi=bt["hi"], n_pos=int((v > 0).sum()),
                                 p=SX.wilcoxon(v)))
    t = pd.DataFrame(rows)
    if len(t):
        t["p_holm"] = np.nan
        for (cp, lr), g in t.groupby(["cp", "learner"]):
            t.loc[g.index, "p_holm"] = SX.holm(g.p.to_numpy())
    t.to_csv(os.path.join(TAB, "X5s_vs_rs100.csv"), index=False, encoding="utf-8-sig")
    k = X.groupby(["learner", "cp"]).k.mean().reset_index()
    k.to_csv(os.path.join(TAB, "X5s_k.csv"), index=False, encoding="utf-8-sig")
    return dict(table=t, k=k)


def run():
    E = load_extra()
    t = LD.load_all(use_cache=False)
    H = t["honest"]
    out = dict(shas=sorted(E["_shas"]))
    for name, fn, args in (("scaler", scaler, (E, H)), ("innerk", innerk, (E, H)), ("synth", synth, (E,)), ("land", land, (E,)), ("rs100", rs100, (E, H))):
        out[name] = fn(*args)
        print("==", name, "done" if out[name] is not None else "(no data)")
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
    o = run()
    if o["scaler"]:
        print(json.dumps(o["scaler"]["pooled"], indent=1))
    if o["innerk"]:
        print(o["innerk"]["pooled"].round(4).to_string())
    if o["synth"]:
        print(o["synth"]["contrasts"].round(3).head(30).to_string())
    if o["land"]:
        print(o["land"]["r2"]); print(o["land"]["table"][o["land"]["table"].k == 20].round(3).head(24).to_string())
