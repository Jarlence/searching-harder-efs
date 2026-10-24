"""Analyses added after the external review of the first draft (docs/DEVIATIONS.md #7-#9, docs/REVISION_1.md).  All are exploratory or corrections;
none was pre-specified.  The unit of inference is the DATA SET: every comparison is first averaged within a data set (over outer runs), the
data-set values are summarised by their mean, a cluster-bootstrap interval over data sets and a Wilcoxon signed-rank test over data sets.

  X8p  front statistics of the Pareto archive (size, ties), D by definition                       -> X8p_front_stats.csv
  X6d  GA / NSGA-II compared at equal DISTINCT evaluated masks instead of equal wrapper requests  -> X6d_distinct_matched.csv
  X5m  random search with the subset size of each algorithm's incumbent ("same-size RS")           -> X5m_same_size_rs.csv, X5m_size_effect_on_rs.csv
  X7d  requests vs distinct masks (cache hits) and run time per algorithm                          -> X7d_distinct_diag.csv, E1h_seconds_by_algo.csv"""
import glob
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
NB = 10000


def load_suite(name):
    out = []
    for f in glob.glob(os.path.join(LD.EXTRA, name, "*.json")):
        rec = json.load(open(f))
        if rec.get("status") == "ok":
            out.append(rec)
    return out


def _summ(per, seed, n_boot=NB):
    """per: {ds: array of paired differences over outer runs} -> dict with dataset-level summary."""
    per = {k: [(np.asarray(v, float)[:, None], 1.0)] for k, v in per.items() if len(v)}
    if len(per) < 3:
        return None
    bt = SX.double_boot(per, n_boot, seed=seed)
    if bt is None:
        return None
    v = np.array(list(bt["ds_values"].values()))
    return dict(n_ds=len(v), mean_diff=bt["point"], lo=bt["lo"], hi=bt["hi"], lo_db=bt["lo_db"], hi_db=bt["hi_db"], n_pos=int((v > 0).sum()),
                p=SX.wilcoxon(v), ds_min=float(v.min()), ds_max=float(v.max()))


def _holm_within(df, by):
    df = df.copy(); df["p_holm"] = np.nan
    for _, g in df.groupby(by):
        df.loc[g.index, "p_holm"] = SX.holm(g.p.to_numpy())
    return df


def pareto_check(H):
    """The replay X8p must reproduce every incumbent number stored by E1h bit by bit (same seeds, same trajectory)."""
    E = H[(H.suite == "E1h") & (H.algo == "nsga2")].set_index(["ds", "learner", "rep", "fold", "cp"])
    fields = ("k", "inner_err", "test_ba", "test_auc", "test_mcc", "test_f1", "test_acc", "test_gm", "test_tpr", "test_tnr")
    n_num = n_diff = n_rec = 0
    for rec in load_suite("X8p"):
        s = rec["spec"]
        for cp, r in rec["result"].items():
            o = E.loc[(s["ds"], s["learner"], s["rep"], s["fold"], int(cp))]
            n_rec += 1
            for k in fields:
                n_num += 1
                n_diff += int(float(o[k]) != float(r[k]))
    out = dict(n_records=n_rec, n_numbers=n_num, n_different=n_diff, n_expected_records=int(len(E)))
    json.dump(out, open(os.path.join(TAB, "X8p_check.json"), "w"))
    return out


def front_stats(P):
    if len(P) == 0:
        return None
    g = P.groupby("cp").agg(n=("d_merged", "size"), mean_front=("n_front", "mean"), mean_tied=("n_tied", "mean"), mean_calls=("calls", "mean"),
                            mean_distinct=("distinct", "mean"), mean_D_merged=("d_merged", "mean"), mean_D_ties=("d_ties", "mean")).reset_index()
    g["share_distinct"] = g.mean_distinct / g.mean_calls
    g.to_csv(os.path.join(TAB, "X8p_front_stats.csv"), index=False, encoding="utf-8-sig")
    return g


def distinct_matched(H):
    """X6d vs E1h.  (i) the same algorithm at the same wrapper REQUESTS (E1h) -> what equalising distinct masks changes for the algorithm itself;
    (ii) the algorithm at N DISTINCT masks vs budget-matched random search at N requests (E1h; RS evaluates ~N distinct masks)."""
    X = load_suite("X6d")
    if not X:
        return None
    rows_x = []
    for rec in X:
        s = rec["spec"]
        for cp, r in rec["result"].items():
            rows_x.append(dict(ds=s["ds"], learner=s["learner"], algo=s["algo"], rep=s["rep"], fold=s["fold"], target=int(cp), test_ba=r["test_ba"], k=r["k"],
                               calls=r["calls"], distinct=r["distinct"], reached=r["reached"]))
    X = pd.DataFrame(rows_x)
    X.to_csv(os.path.join(TAB, "X6d_runs.csv"), index=False, encoding="utf-8-sig")
    E = H[H.suite == "E1h"]
    key = ["ds", "learner", "rep", "fold"]
    out = []
    for lr in ("ridge", "knn"):
        for algo in ("ga", "nsga2"):
            for tgt in (1000, 4000):
                xa = X[(X.learner == lr) & (X.algo == algo) & (X.target == tgt)].set_index(key)
                ea = E[(E.learner == lr) & (E.algo == algo) & (E.cp == tgt)].set_index(key)
                er = E[(E.learner == lr) & (E.algo == "rs") & (E.cp == tgt)].set_index(key)
                for comp, ref in (("same_algo_same_requests", ea), ("vs_random_search", er)):
                    per = {}
                    for ds in sorted(set(xa.index.get_level_values("ds"))):
                        a = xa[xa.index.get_level_values("ds") == ds]; b = ref[ref.index.get_level_values("ds") == ds]
                        idx = a.index.intersection(b.index)
                        if len(idx) >= 5:
                            per[ds] = (a.loc[idx, "test_ba"] - b.loc[idx, "test_ba"]).to_numpy(float)
                    sm = _summ(per, 20261021)
                    if sm is None:
                        continue
                    out.append(dict(learner=lr, algo=algo, target=tgt, comparison=comp, mean_calls=float(xa.calls.mean()), mean_distinct=float(xa.distinct.mean()),
                                    unreached=int((~xa.reached.astype(bool)).sum()), **sm))
    t = pd.DataFrame(out)
    if len(t):
        t = _holm_within(t, ["learner", "target", "comparison"])
        # the same gain over RS when both are given the same number of REQUESTS (E1h) - for the side-by-side table
        base = []
        for lr in ("ridge", "knn"):
            for algo in ("ga", "nsga2"):
                for tgt in (1000, 4000):
                    ea = E[(E.learner == lr) & (E.algo == algo) & (E.cp == tgt)].set_index(key)
                    er = E[(E.learner == lr) & (E.algo == "rs") & (E.cp == tgt)].set_index(key)
                    per = {}
                    for ds in sorted(set(ea.index.get_level_values("ds"))):
                        a = ea[ea.index.get_level_values("ds") == ds]; b = er[er.index.get_level_values("ds") == ds]
                        idx = a.index.intersection(b.index)
                        if len(idx) >= 5:
                            per[ds] = (a.loc[idx, "test_ba"] - b.loc[idx, "test_ba"]).to_numpy(float)
                    sm = _summ(per, 20261022)
                    if sm:
                        base.append(dict(learner=lr, algo=algo, target=tgt, **{"req_" + k: v for k, v in sm.items()}))
        t = t.merge(pd.DataFrame(base), on=["learner", "algo", "target"], how="left")
    t.to_csv(os.path.join(TAB, "X6d_distinct_matched.csv"), index=False, encoding="utf-8-sig")
    return t


def same_size_rs(H):
    X = load_suite("X5m")
    if not X:
        return None
    rows = []
    for rec in X:
        s = rec["spec"]
        for cp, r in rec["result"].items():
            rows.append(dict(ds=s["ds"], learner=s["learner"], ref=s["algo"], rep=s["rep"], fold=s["fold"], budget=int(s["budget"]), k=s["k"], rsk_ba=r["test_ba"],
                             rsk_fit=1 - r["inner_err"]))
    X = pd.DataFrame(rows)
    X.to_csv(os.path.join(TAB, "X5m_runs.csv"), index=False, encoding="utf-8-sig")
    E = H[H.suite == "E1h"]
    key = ["ds", "learner", "rep", "fold"]
    out, out2 = [], []
    for lr in ("ridge", "knn"):
        for b in (1000, 4000):
            er = E[(E.learner == lr) & (E.algo == "rs") & (E.cp == b)].set_index(key)
            for ref in ("ga", "bpso", "bde", "nsga2"):
                xa = X[(X.learner == lr) & (X.ref == ref) & (X.budget == b)].set_index(key)
                ea = E[(E.learner == lr) & (E.algo == ref) & (E.cp == b)].set_index(key)
                per, per2, per3 = {}, {}, {}
                for ds in sorted(set(xa.index.get_level_values("ds"))):
                    a = xa[xa.index.get_level_values("ds") == ds]; e = ea[ea.index.get_level_values("ds") == ds]; r_ = er[er.index.get_level_values("ds") == ds]
                    idx = a.index.intersection(e.index)
                    if len(idx) >= 5:
                        per[ds] = (e.loc[idx, "test_ba"] - a.loc[idx, "rsk_ba"]).to_numpy(float)           # algorithm - same-size RS
                    idx2 = a.index.intersection(r_.index)
                    if len(idx2) >= 5:
                        per2[ds] = (a.loc[idx2, "rsk_ba"] - r_.loc[idx2, "test_ba"]).to_numpy(float)       # same-size RS - ordinary RS (size effect)
                    idx3 = e.index.intersection(r_.index)
                    if len(idx3) >= 5:
                        per3[ds] = (e.loc[idx3, "test_ba"] - r_.loc[idx3, "test_ba"]).to_numpy(float)      # algorithm - ordinary RS (E1h; for the side-by-side table)
                s1 = _summ(per, 20261023); s2 = _summ(per2, 20261024); s3 = _summ(per3, 20261025)
                # Match the data-set-equal weighting of the manuscript's feature counts.
                mean_k = float(xa.groupby(level="ds").k.mean().mean())
                if s1:
                    out.append(dict(learner=lr, budget=b, ref=ref, mean_k=mean_k, **s1,
                                    **({"vs_rs_" + k: v for k, v in s3.items()} if s3 else {})))
                if s2:
                    out2.append(dict(learner=lr, budget=b, ref=ref, mean_k=mean_k, **s2))
    t = pd.DataFrame(out); t2 = pd.DataFrame(out2)
    if len(t):
        t = _holm_within(t, ["learner", "budget"])
    t.to_csv(os.path.join(TAB, "X5m_same_size_rs.csv"), index=False, encoding="utf-8-sig")
    t2.to_csv(os.path.join(TAB, "X5m_size_effect_on_rs.csv"), index=False, encoding="utf-8-sig")
    return t, t2


def diag(H, secs):
    X = load_suite("X7d")
    if not X:
        return None
    rows = []
    for rec in X:
        s = rec["spec"]
        for cp, r in rec["result"].items():
            rows.append(dict(ds=s["ds"], learner=s["learner"], algo=s["algo"], fold=s["fold"], cp=int(cp), calls=r["calls"], distinct=r["distinct"], k=r["k"],
                             seconds=rec["seconds"]))
    D = pd.DataFrame(rows)
    D["hit_rate"] = 1.0 - D.distinct / D.calls
    g = D.groupby(["algo", "cp"]).agg(n=("calls", "size"), mean_calls=("calls", "mean"), mean_distinct=("distinct", "mean"), mean_hit_rate=("hit_rate", "mean")).reset_index()
    # run time of a whole search up to the largest budget of the cell (seconds of the job), averaged over cells
    sec = pd.DataFrame([dict(algo=r["spec"]["algo"], ds=r["spec"]["ds"], learner=r["spec"]["learner"], seconds=r["seconds"]) for r in X])
    g2 = sec.groupby("algo").seconds.agg(["mean", "median", "sum"]).reset_index().rename(columns={"mean": "mean_seconds", "median": "median_seconds", "sum": "total_seconds"})
    g.to_csv(os.path.join(TAB, "X7d_distinct_diag.csv"), index=False, encoding="utf-8-sig")
    g2.to_csv(os.path.join(TAB, "X7d_seconds_by_algo.csv"), index=False, encoding="utf-8-sig")
    # run time in the main experiment (E1h jobs include the test-part evaluations of all checkpoints)
    e = secs[secs.suite == "E1h"].groupby("algo").seconds.agg(["count", "mean", "median", "sum"]).reset_index()
    e.to_csv(os.path.join(TAB, "E1h_seconds_by_algo.csv"), index=False, encoding="utf-8-sig")
    return g, g2


def run():
    t = LD.load_all(use_cache=False)
    H = t["honest"]
    P = LD.load_pareto()
    out = {"check": pareto_check(H), "front": front_stats(P), "distinct": distinct_matched(H), "same_size": same_size_rs(H), "diag": diag(H, t["secs"])}
    for k, v in out.items():
        print("==", k, "done" if v is not None else "(no data)")
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    o = run()
    for k, v in o.items():
        if v is None:
            continue
        print("-----", k)
        if isinstance(v, dict):
            print(v)
        elif isinstance(v, tuple):
            for x in v:
                print(x.round(4).to_string())
        else:
            print(v.round(4).to_string())
