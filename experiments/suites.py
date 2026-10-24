"""Job definitions for every pre-registered suite (docs/PROTOCOL.md).  One job = one JSON file under results/jobs/<suite>/."""
import hashlib
import json
import os
import sys
import time
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np  # noqa: E402

from efsbench import datasets as D, filters as F, learners as L, metrics as M, protocols as P, splits as S  # noqa: E402

JOBS = os.path.join(ROOT, "results", "jobs")
HD = ["colon", "cns", "dlbcl", "leukemia", "prostate", "gli", "smk", "ovarian", "breast"]
IMB = ["oil_spill", "ozone", "speech", "kc3"]
ALGOS = ["ga", "bpso", "bde", "nsga2", "rs"]
STD_CP = (250, 1000, 4000, 16000)
SHORT_CP = (250, 1000, 4000)
NO_KNN = ("ozone", "speech")                      # n > 2000: 5-NN wrapper is too slow
SLOW_KNN = ("oil_spill", "kc3", "parkinson")      # n in 350..800: 5-NN only up to budget 4000
COST = {"speech": 40, "ozone": 25, "oil_spill": 30, "parkinson": 30, "kc3": 12, "ovarian": 10, "smk": 9}   # rough sort key


def _code_sha():
    h = hashlib.sha256()
    for d in (os.path.join(ROOT, "src", "efsbench"), os.path.dirname(os.path.abspath(__file__))):
        for f in sorted(os.listdir(d)):
            if f.endswith(".py") and f not in ("run_suite.py", "analyze_tmp.py"):
                h.update(open(os.path.join(d, f), "rb").read())
    return h.hexdigest()[:16]


CODE_SHA = _code_sha()
_DATA = {}


def data(ds):
    if ds not in _DATA:
        _DATA[ds] = D.load(ds)
    return _DATA[ds]


def learners_for(ds):
    return ["ridge"] if ds in NO_KNN else ["ridge", "knn"]


# docs/DEVIATIONS.md #1: fewer repeats for the four big / slow data sets (the nine core high-dimensional sets keep the full design)
SCALE = {("speech", "ridge"): 0.4, ("ozone", "ridge"): 0.6, ("oil_spill", "ridge"): 0.6, ("oil_spill", "knn"): 0.3,
         ("kc3", "knn"): 0.5}


def nrep(ds, learner, n):
    return max(2, int(round(n * SCALE.get((ds, learner), 1.0))))


def cps_for(ds, learner):
    if ds in NO_KNN or (learner == "knn" and ds in SLOW_KNN):     # docs/DEVIATIONS.md #2: n > 2000 -> budgets up to 4000 only
        return SHORT_CP
    return STD_CP


def permuted(y, ds, pid):
    yp = y.copy()
    S.rng_for("perm", ds, pid).shuffle(yp)
    return yp


def outer_split(y, ds, rep, fold, tag="outer", groups=None):
    return S.make_folds(y, 5, S.rng_for(tag, ds, rep), groups)[fold]


# ------------------------------------------------------------------------------------------ job kinds
def k_honest(s):
    ds, algo, lr, rep, fold = s["ds"], s["algo"], s["learner"], s["rep"], s["fold"]
    X, y, g = data(ds)
    pid = s.get("perm")
    if pid is not None:
        y = permuted(y, ds, pid)
    use_g = g if s.get("outer_group", False) else None
    in_g = s.get("inner_group", False)
    tag = "outer" if pid is None else "outer_perm%s" % pid
    tr, te = outer_split(y, ds, rep, fold, tag, use_g)
    return P.honest_core(X[tr], y[tr], X[te], y[te], algo, lr,
                         ("inner", ds, lr, rep, fold, pid, s.get("outer_group", False), in_g),
                         ("algo", ds, algo, lr, rep, fold, pid, s.get("seed", 0), s.get("outer_group", False), in_g),
                         tuple(s["cps"]), s.get("metric", "ba"), 5, (g[tr] if in_g else None))


def k_full(s):
    ds, algo, lr, seed = s["ds"], s["algo"], s["learner"], s["seed"]
    X, y, g = data(ds)
    pid = s.get("perm")
    if pid is not None:
        y = permuted(y, ds, pid)
    return P.fulldata_core(X, y, algo, lr, ("innerfull", ds, lr, seed, pid), ("algofull", ds, algo, lr, seed, pid),
                           tuple(s["cps"]), groups=(g if s.get("group") else None),
                           reeval_group=(g if s.get("group") else None))


def k_baseline(s):
    ds, lr, rep, fold, method = s["ds"], s["learner"], s["rep"], s["fold"], s["method"]
    X, y, g = data(ds)
    tr, te = outer_split(y, ds, rep, fold)
    Xtr, Xte = P.standardize(X[tr], X[te])
    ikey = ("binner", ds, lr, rep, fold)
    if method in ("anova", "rf"):
        k, m = F.topk_pipeline(Xtr, y[tr], Xte, y[te], L.SPECS[lr], method, ikey)
    elif method == "all":
        k, m = F.all_features_pipeline(Xtr, y[tr], Xte, y[te], L.SPECS[lr])
    elif method == "l1":
        k, m = F.l1_logistic_pipeline(Xtr, y[tr], Xte, y[te], ikey)
    else:
        raise ValueError(method)
    return {"k": int(k), **{"test_" + a: float(b) for a, b in m.items()}}


def k_smote_outer(s):
    X, y, _ = data(s["ds"])
    m = P.smote_outer_core(X, y, s["learner"], s["k_feat"], s["protocol"], s["ds"], s["rep"])
    return {a: float(b) for a, b in m.items()}


def k_smote_inner(s):
    ds, algo, lr, rep, fold = s["ds"], s["algo"], s["learner"], s["rep"], s["fold"]
    X, y, _ = data(ds)
    tr, te = outer_split(y, ds, rep, fold)
    return P.smote_inner_core(X[tr], y[tr], X[te], y[te], algo, lr, s["mode"], ("sinner", ds, lr, rep, fold),
                              ("salgo", ds, algo, lr, rep, fold), tuple(s["cps"]))


KINDS = {"honest": k_honest, "full": k_full, "baseline": k_baseline, "smote_outer": k_smote_outer,
         "smote_inner": k_smote_inner}


def job_id(s):
    parts = [s["suite"], s["ds"], s.get("algo", s.get("method", "-")), s.get("learner", "-")]
    for k in ("rep", "fold", "seed", "perm", "metric", "mode", "protocol", "k_feat", "outer_group", "inner_group", "group"):
        if k in s and s[k] is not None:
            parts.append("%s%s" % (k[:3], s[k]))
    return "__".join(str(p) for p in parts)


def job_file(s):
    return os.path.join(JOBS, s["suite"], job_id(s) + ".json")


def run_job(s):
    path = job_file(s)
    if os.path.exists(path):
        return "skip"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.time()
    try:
        res = KINDS[s["kind"]](s); status = "ok"
    except Exception:  # noqa: BLE001
        res = {"error": traceback.format_exc()[-1800:]}; status = "error"
    rec = {"spec": s, "status": status, "seconds": round(time.time() - t0, 2), "code_sha": CODE_SHA, "result": res}
    if status == "ok":
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f)
        os.replace(tmp, path)
    else:
        ed = os.path.join(JOBS, s["suite"], "errors"); os.makedirs(ed, exist_ok=True)
        with open(os.path.join(ed, job_id(s) + ".json"), "w") as f:
            json.dump(rec, f)
    return status


# ------------------------------------------------------------------------------------------ enumerators
def _order(jobs):
    """repeat-major, expensive datasets first inside a repeat -> early partial results are balanced, workers stay busy."""
    return sorted(jobs, key=lambda s: (s.get("rep", s.get("seed", 0)) if s.get("rep", None) is not None else s.get("seed", 0),
                                       -COST.get(s["ds"], 5), s["ds"], s.get("algo", ""), s.get("learner", "")))


def E1_honest(n_rep=10):
    jobs = []
    for ds in HD + IMB:
        for lr in learners_for(ds):
            for algo in ALGOS:
                for rep in range(nrep(ds, lr, n_rep)):
                    for fold in range(5):
                        jobs.append(dict(suite="E1h", kind="honest", ds=ds, algo=algo, learner=lr, rep=rep, fold=fold,
                                         cps=list(cps_for(ds, lr)), metric="ba"))
    return _order(jobs)


def E1_full(n_seeds=30):
    jobs = []
    for ds in HD + IMB:
        for lr in learners_for(ds):
            for algo in ALGOS:
                for seed in range(nrep(ds, lr, n_seeds)):
                    jobs.append(dict(suite="E1f", kind="full", ds=ds, algo=algo, learner=lr, seed=seed,
                                     cps=list(cps_for(ds, lr))))
    return _order(jobs)


def E_perm(n_perm=5):
    jobs = []
    for ds in HD + IMB:
        for lr in learners_for(ds):
            for algo in ("ga", "nsga2", "rs"):
                for pid in range(n_perm):
                    jobs.append(dict(suite="Ep_h", kind="honest", ds=ds, algo=algo, learner=lr, rep=pid, fold=0, perm=pid,
                                     cps=list(SHORT_CP), metric="ba"))
                    jobs.append(dict(suite="Ep_f", kind="full", ds=ds, algo=algo, learner=lr, seed=pid, perm=pid,
                                     cps=list(SHORT_CP)))
    return _order(jobs)


def E_baselines(n_rep=10):
    jobs = []
    for ds in HD + IMB:
        for lr in learners_for(ds):
            for method in ("anova", "rf", "all"):
                for rep in range(n_rep):
                    for fold in range(5):
                        jobs.append(dict(suite="E2b", kind="baseline", ds=ds, method=method, learner=lr, rep=rep, fold=fold))
        for rep in range(n_rep):
            for fold in range(5):
                jobs.append(dict(suite="E2b", kind="baseline", ds=ds, method="l1", learner="l1", rep=rep, fold=fold))
    return _order(jobs)


def E_metric(n_rep=4):
    jobs = []
    for ds in HD + IMB:
        for lr in learners_for(ds):
            for algo in ("ga", "nsga2"):
                for metric in ("acc", "gm", "auc"):
                    for rep in range(nrep(ds, lr, n_rep)):
                        for fold in range(5):
                            jobs.append(dict(suite="E3m", kind="honest", ds=ds, algo=algo, learner=lr, rep=rep, fold=fold,
                                             cps=list(cps_for(ds, lr)), metric=metric))
    return _order(jobs)


def E_seeds(n_seeds=20):
    jobs = []
    for ds in HD + IMB:
        for lr in learners_for(ds):
            for algo in ("ga", "nsga2"):
                for fold in range(3):
                    for seed in range(nrep(ds, lr, n_seeds)):
                        jobs.append(dict(suite="E4s", kind="honest", ds=ds, algo=algo, learner=lr, rep=0, fold=fold,
                                         seed=seed, cps=[1000, 4000], metric="ba"))
    return _order(jobs)


def E_smote_outer(n_rep=10):
    jobs = []
    for ds in HD + IMB:
        for lr in ("ridge", "knn"):
            for proto in ("honest", "before_split"):
                for rep in range(nrep(ds, lr, n_rep)):
                    jobs.append(dict(suite="E5so", kind="smote_outer", ds=ds, learner=lr, k_feat=20, protocol=proto, rep=rep))
    return _order(jobs)


def E_smote_inner(n_rep=2):
    jobs = []
    for ds in HD:
        for lr in ("ridge", "knn"):
            for mode in ("clean", "leaky"):
                for rep in range(n_rep):
                    for fold in range(5):
                        jobs.append(dict(suite="E5si", kind="smote_inner", ds=ds, algo="ga", learner=lr, mode=mode, rep=rep,
                                         fold=fold, cps=[250, 1000, 4000]))
    return _order(jobs)


def E_unit(n_rep=10):
    jobs = []
    for og in (False, True):
        for ig in (False, True):
            for algo in ("ga", "nsga2", "rs"):
                for rep in range(n_rep):
                    for fold in range(5):
                        jobs.append(dict(suite="E6u", kind="honest", ds="parkinson", algo=algo, learner="ridge", rep=rep,
                                         fold=fold, cps=list(STD_CP), metric="ba", outer_group=og, inner_group=ig))
            for rep in range(4):
                for fold in range(5):
                    jobs.append(dict(suite="E6u", kind="honest", ds="parkinson", algo="ga", learner="knn", rep=rep,
                                     fold=fold, cps=list(SHORT_CP), metric="ba", outer_group=og, inner_group=ig))
    for algo in ("ga", "nsga2", "rs"):                        # flaw A on the unit data: record-wise vs group-wise re-estimation
        for grp in (False, True):
            for seed in range(30):
                jobs.append(dict(suite="E6uf", kind="full", ds="parkinson", algo=algo, learner="ridge", seed=seed,
                                 cps=list(STD_CP), group=grp))
    return _order(jobs)


def E_long(n_rep=2):
    jobs = []
    for ds in ("colon", "dlbcl", "prostate", "ovarian"):
        for lr in ("ridge", "knn"):
            cps = [4000, 16000, 64000, 256000] if lr == "ridge" else [4000, 16000, 64000]
            for algo in ("ga", "nsga2", "rs"):
                for rep in range(n_rep):
                    for fold in range(5):
                        jobs.append(dict(suite="E7l", kind="honest", ds=ds, algo=algo, learner=lr, rep=rep, fold=fold,
                                         cps=cps, metric="ba"))
    return _order(jobs)


SUITES = {"E1h": E1_honest, "E1f": E1_full, "Eperm": E_perm, "E2b": E_baselines, "E3m": E_metric, "E4s": E_seeds,
          "E5so": E_smote_outer, "E5si": E_smote_inner, "E6u": E_unit, "E7l": E_long}
