"""Fourth batch of extra studies (written after the external review of the first draft; docs/DEVIATIONS.md #9).  Exploratory, not pre-specified.

X9s  SMOTE decomposition : the 'before_split' pipeline of E5so (SMOTE on all rows, then K-fold over real + synthetic rows, ANOVA top-20, learner),
     replayed with the same seeds.  The metrics are computed (a) on ALL rows of the augmented data set (what E5so reports; must reproduce E5so
     bit by bit), (b) on the REAL rows only (original class balance, no synthetic test points) and (c) on the synthetic rows only.
     (a) - (b) is the effect of changing the TEST SET (synthetic minority points that interpolate training points, and a 50 % class
     prior); (b) - honest is the effect of LEAKAGE alone (synthetic copies of test-fold real points in the training part)."""
import json
import os
import sys
import time
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "experiments")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from efsbench import learners as L, metrics as M, protocols as P, smote as SM, splits as S  # noqa: E402
from efsbench.filters import anova_f  # noqa: E402
import suites as ST  # noqa: E402
import ex_suites3 as E3  # noqa: E402

JOBS = E3.JOBS
CODE_SHA = E3.CODE_SHA


def k_smote_dec(s):
    ds, lr, rep, k_feat = s["ds"], s["learner"], s["rep"], s["k_feat"]
    X, y, _ = ST.data(ds)
    spec = L.SPECS[lr]; thr = L.threshold(spec)
    rng = S.rng_for("smote_outer", ds, rep, "before_split")                    # exactly the streams of protocols.smote_outer_core
    Xs, = P.standardize(X)
    Xa, ya = SM.smote(Xs, y, rng, ratio=1.0)
    n_real = len(y)
    folds = S.stratified_kfold(ya, 5, S.rng_for("smote_outer_folds", ds, rep))
    pred = np.empty(len(ya), np.int64); sco = np.empty(len(ya))
    for tr, te in folds:
        top = np.argsort(-anova_f(Xa[tr], ya[tr]), kind="stable")[:k_feat]
        sc = L.fit_predict(Xa[tr][:, top], ya[tr], Xa[te][:, top], spec)
        sco[te] = sc; pred[te] = (sc > thr)
    real = np.arange(len(ya)) < n_real
    out = {"all": {a: float(b) for a, b in M.all_metrics(ya, pred, sco).items()},
           "real": {a: float(b) for a, b in M.all_metrics(ya[real], pred[real], sco[real]).items()},
           "synthetic_tpr": float((pred[~real] == 1).mean()) if (~real).any() else float("nan"),
           "n_real": int(real.sum()), "n_synthetic": int((~real).sum()), "n_real_minority": int((y == 1).sum())}
    return out


KINDS = {"smote_dec": k_smote_dec}


def job_id(s):
    return "__".join([s["suite"], s["ds"], s["learner"], "rep%d" % s["rep"]])


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
        with open(path + ".tmp", "w") as f:
            json.dump(rec, f)
        os.replace(path + ".tmp", path)
    else:
        ed = os.path.join(JOBS, s["suite"], "errors"); os.makedirs(ed, exist_ok=True)
        with open(os.path.join(ed, job_id(s) + ".json"), "w") as f:
            json.dump(rec, f)
    return status


def X9_smote(n_rep=10):
    J = []
    for ds in ST.HD + ST.IMB:
        for lr in ("ridge", "knn"):
            for rep in range(ST.nrep(ds, lr, n_rep)):                           # the same jobs as the 'before_split' half of E5so
                J.append(dict(suite="X9s", kind="smote_dec", ds=ds, learner=lr, rep=rep, k_feat=20))
    return ST._order(J)


SUITES = {"X9s": X9_smote}
