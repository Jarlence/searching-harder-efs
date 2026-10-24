"""Builds a TINY copy of every suite (same spec keys, short budgets, few repeats) in analysis/_fixture/jobs so that the
analysis scripts can be debugged end-to-end before the real runs finish.  Fixture numbers are NEVER reported anywhere.
Uses the frozen job code read-only (does not touch results/jobs)."""
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import suites as ST  # noqa: E402

FIX = os.path.join(ROOT, "analysis", "_fixture", "jobs")
CP = [250, 1000]
DS3 = ["colon", "cns", "kc3"]
DS2 = ["colon", "cns"]


def specs():
    J = []
    for ds in DS3:
        for lr in ("ridge", "knn"):
            for algo in ST.ALGOS:
                for rep in range(2):
                    for fold in range(5):
                        J.append(dict(suite="E1h", kind="honest", ds=ds, algo=algo, learner=lr, rep=rep, fold=fold, cps=CP, metric="ba"))
                for seed in range(3):
                    J.append(dict(suite="E1f", kind="full", ds=ds, algo=algo, learner=lr, seed=seed, cps=CP))
    for ds in DS2:
        for lr in ("ridge", "knn"):
            for algo in ("ga", "nsga2", "rs"):
                for pid in range(3):
                    J.append(dict(suite="Ep_h", kind="honest", ds=ds, algo=algo, learner=lr, rep=pid, fold=0, perm=pid, cps=CP, metric="ba"))
                    J.append(dict(suite="Ep_f", kind="full", ds=ds, algo=algo, learner=lr, seed=pid, perm=pid, cps=CP))
            for method in ("anova", "rf", "all"):
                for rep in range(2):
                    for fold in range(5):
                        J.append(dict(suite="E2b", kind="baseline", ds=ds, method=method, learner=lr, rep=rep, fold=fold))
            for algo in ("ga", "nsga2"):
                for metric in ("acc", "gm", "auc"):
                    for rep in range(2):
                        for fold in range(5):
                            J.append(dict(suite="E3m", kind="honest", ds=ds, algo=algo, learner=lr, rep=rep, fold=fold, cps=CP, metric=metric))
                for fold in range(3):
                    for seed in range(6):
                        J.append(dict(suite="E4s", kind="honest", ds=ds, algo=algo, learner=lr, rep=0, fold=fold, seed=seed, cps=CP, metric="ba"))
            for proto in ("honest", "before_split"):
                for rep in range(3):
                    J.append(dict(suite="E5so", kind="smote_outer", ds=ds, learner=lr, k_feat=20, protocol=proto, rep=rep))
            for mode in ("clean", "leaky"):
                for rep in range(2):
                    for fold in range(5):
                        J.append(dict(suite="E5si", kind="smote_inner", ds=ds, algo="ga", learner=lr, mode=mode, rep=rep, fold=fold, cps=CP))
            for algo in ("ga", "nsga2", "rs"):
                for fold in range(5):
                    J.append(dict(suite="E7l", kind="honest", ds=ds, algo=algo, learner=lr, rep=0, fold=fold, cps=CP, metric="ba"))
        for method in ("l1",):
            for rep in range(2):
                for fold in range(5):
                    J.append(dict(suite="E2b", kind="baseline", ds=ds, method=method, learner="l1", rep=rep, fold=fold))
    for og in (False, True):
        for ig in (False, True):
            for algo in ("ga", "nsga2", "rs"):
                for rep in range(2):
                    for fold in range(5):
                        J.append(dict(suite="E6u", kind="honest", ds="parkinson", algo=algo, learner="ridge", rep=rep, fold=fold, cps=CP,
                                      metric="ba", outer_group=og, inner_group=ig))
    for algo in ("ga", "nsga2", "rs"):
        for grp in (False, True):
            for seed in range(3):
                J.append(dict(suite="E6uf", kind="full", ds="parkinson", algo=algo, learner="ridge", seed=seed, cps=CP, group=grp))
    return J


def k_baseline_fixed(s):
    """Candidate fix for experiments/suites.py:k_baseline (learner 'l1' is not a key of L.SPECS) - tested here before it is applied."""
    ds, lr, rep, fold, method = s["ds"], s["learner"], s["rep"], s["fold"], s["method"]
    X, y, g = ST.data(ds)
    tr, te = ST.outer_split(y, ds, rep, fold)
    Xtr, Xte = ST.P.standardize(X[tr], X[te])
    ikey = ("binner", ds, lr, rep, fold)
    if method in ("anova", "rf"):
        k, m = ST.F.topk_pipeline(Xtr, y[tr], Xte, y[te], ST.L.SPECS[lr], method, ikey)
    elif method == "all":
        k, m = ST.F.all_features_pipeline(Xtr, y[tr], Xte, y[te], ST.L.SPECS[lr])
    elif method == "l1":
        k, m = ST.F.l1_logistic_pipeline(Xtr, y[tr], Xte, y[te], ikey)
    else:
        raise ValueError(method)
    return {"k": int(k), **{"test_" + a: float(b) for a, b in m.items()}}


def run_one(s):
    import traceback
    path = os.path.join(FIX, s["suite"], ST.job_id(s) + ".json")
    if os.path.exists(path):
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.time()
    try:
        res = k_baseline_fixed(s) if s["kind"] == "baseline" else ST.KINDS[s["kind"]](s)
    except Exception:  # noqa: BLE001
        ed = os.path.join(os.path.dirname(FIX), "errors"); os.makedirs(ed, exist_ok=True)
        open(os.path.join(ed, ST.job_id(s) + ".txt"), "w").write(traceback.format_exc())
        return -1
    rec = {"spec": s, "status": "ok", "seconds": round(time.time() - t0, 2), "code_sha": ST.CODE_SHA, "result": res}
    json.dump(rec, open(path, "w"))
    return 1


if __name__ == "__main__":
    from joblib import Parallel, delayed
    J = specs()
    print("fixture jobs:", len(J), flush=True)
    n = Parallel(n_jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 4, backend="loky", verbose=0)(delayed(run_one)(s) for s in J)
    print("done", sum(1 for x in n if x == 1), "errors", sum(1 for x in n if x == -1), flush=True)
