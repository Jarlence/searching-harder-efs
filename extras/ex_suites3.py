"""Third batch of extra studies - written AFTER an external review of the first full draft of the paper, when all main results were known
(docs/DEVIATIONS.md #7-#9, docs/REVISION_1.md).  They are corrections and robustness checks, NOT pre-specified tests.  Results go to
results/jobs_extra/<suite>/.

X8p  pareto replay : the NSGA-II honest runs of E1h, replayed with the same seeds.  The over-statement D of "pick the Pareto point on the test set"
                     is now defined by ONE candidate set: the Pareto ARCHIVE (all non-dominated candidates evaluated up to the checkpoint), which
                     contains the incumbent.  (E1h took the incumbent from all evaluated candidates but the front from the previous generation.)
X6d  distinct      : GA and NSGA-II run (same seeds, hence the same trajectory as E1h) until 1,000 / 4,000 DISTINCT masks were evaluated;
                     E1h counts wrapper requests, cache hits included.
X5m  same-size RS  : random search whose subsets all have the size of the incumbent that algorithm `algo` returned in the same outer run
                     (E1h), 1,000 / 4,000 requests.  Isolates "which features" from "how many features".
X7d  diagnostics   : requests vs distinct masks (cache-hit rate) of all five searches on a panel (repeat 0, folds 0-4); no test data touched."""
import hashlib
import json
import os
import sys
import time
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np  # noqa: E402

from efsbench import algos as A, protocols as P, tracking as T  # noqa: E402
import suites as ST  # noqa: E402

JOBS = os.environ.get("EFS_JOBS_EXTRA", os.path.join(ROOT, "results", "jobs_extra"))
ALGOS5 = ["ga", "bpso", "bde", "nsga2", "rs"]
CALL_CAP = 150000                                    # safety stop of a distinct-mode run (never reached except on tiny feature spaces)


def _code_sha():
    h = hashlib.sha256()
    h.update(open(os.path.abspath(__file__), "rb").read())
    d = os.path.join(ROOT, "src", "efsbench")
    for f in sorted(os.listdir(d)):
        if f.endswith(".py"):
            h.update(open(os.path.join(d, f), "rb").read())
    return h.hexdigest()[:16]


CODE_SHA = _code_sha()
_CUR = {"k": None}


def _rs_fixed_k(run, rng, init_p):
    """random search with subsets of exactly _CUR['k'] features (runtime registration only; the efsbench sources are untouched)"""
    p = run.p
    k = int(min(_CUR["k"], p))
    while not run.done():
        m = np.zeros(p, bool); m[rng.choice(p, k, replace=False)] = True
        run.eval(m)


A.ALGOS["rsk"] = _rs_fixed_k


def _keys(ds, lr, rep, fold, algo):
    return ("inner", ds, lr, rep, fold, None, False, False), ("algo", ds, algo, lr, rep, fold, None, 0, False, False)


# ------------------------------------------------------------------------------------------------ job kinds
def k_pareto_replay(s):
    ds, lr, rep, fold = s["ds"], s["learner"], s["rep"], s["fold"]
    X, y, _ = ST.data(ds)
    tr, te = ST.outer_split(y, ds, rep, fold)
    ik, ak = _keys(ds, lr, rep, fold, "nsga2")
    return T.honest_core_tracked(X[tr], y[tr], X[te], y[te], "nsga2", lr, ik, ak, tuple(s["cps"]), mode="calls", archive=True)


def k_distinct(s):
    ds, lr, rep, fold, algo = s["ds"], s["learner"], s["rep"], s["fold"], s["algo"]
    X, y, _ = ST.data(ds)
    tr, te = ST.outer_split(y, ds, rep, fold)
    ik, ak = _keys(ds, lr, rep, fold, algo)
    return T.honest_core_tracked(X[tr], y[tr], X[te], y[te], algo, lr, ik, ak, tuple(s["dcps"]), mode="distinct", archive=False,
                                 call_cap=CALL_CAP)


def k_rsk(s):
    ds, lr, rep, fold, ref, b = s["ds"], s["learner"], s["rep"], s["fold"], s["algo"], int(s["budget"])
    X, y, _ = ST.data(ds)
    tr, te = ST.outer_split(y, ds, rep, fold)
    _CUR["k"] = int(s["k"])
    ik = ("inner", ds, lr, rep, fold, None, False, False)                       # the SAME inner folds as E1h: a paired comparison
    ak = ("algo", ds, "rsk", lr, rep, fold, ref, b)
    return P.honest_core(X[tr], y[tr], X[te], y[te], "rsk", lr, ik, ak, (b,), "ba", 5, None)


def k_diag(s):
    ds, lr, rep, fold, algo = s["ds"], s["learner"], s["rep"], s["fold"], s["algo"]
    X, y, _ = ST.data(ds)
    tr, te = ST.outer_split(y, ds, rep, fold)
    ik, ak = _keys(ds, lr, rep, fold, algo)
    return T.honest_core_tracked(X[tr], y[tr], X[te], y[te], algo, lr, ik, ak, tuple(s["cps"]), mode="calls", archive=False,
                                 test_metrics=False)


KINDS = {"pareto_replay": k_pareto_replay, "distinct": k_distinct, "rsk": k_rsk, "diag": k_diag}


def job_id(s):
    parts = [s["suite"], s["ds"], s.get("algo", "-"), s.get("learner", "-"), "rep%s" % s["rep"], "fol%s" % s["fold"]]
    if "budget" in s:
        parts.append("b%d" % s["budget"])
    return "__".join(str(x) for x in parts)


def job_file(s):
    return os.path.join(JOBS, s["suite"], job_id(s) + ".json")


def run_job(s):
    """Entry point used by run_extra3.py: defined HERE so that every worker process imports this module (which registers rsk) first."""
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


# ------------------------------------------------------------------------------------------------ enumerators
def _e1h():
    return ST.E1_honest()


def X8_pareto():
    return [dict(suite="X8p", kind="pareto_replay", ds=s["ds"], algo="nsga2", learner=s["learner"], rep=s["rep"], fold=s["fold"],
                 cps=list(s["cps"])) for s in _e1h() if s["algo"] == "nsga2"]


def X6_distinct():
    return ST._order([dict(suite="X6d", kind="distinct", ds=s["ds"], algo=s["algo"], learner=s["learner"], rep=s["rep"], fold=s["fold"],
                           dcps=[1000, 4000]) for s in _e1h() if s["algo"] in ("ga", "nsga2")])


def X5_rsk():
    J = []
    for s in _e1h():
        if s["algo"] not in ("ga", "bpso", "bde", "nsga2"):
            continue
        rec = json.load(open(ST.job_file(s)))
        for b in (1000, 4000):
            J.append(dict(suite="X5m", kind="rsk", ds=s["ds"], algo=s["algo"], learner=s["learner"], rep=s["rep"], fold=s["fold"],
                          budget=b, k=int(rec["result"][str(b)]["k"])))
    return ST._order(J)


def X7_diag():
    J = []
    for ds in ST.HD + ST.IMB:
        for lr in ST.learners_for(ds):
            for algo in ALGOS5:
                for fold in range(5):
                    J.append(dict(suite="X7d", kind="diag", ds=ds, algo=algo, learner=lr, rep=0, fold=fold,
                                  cps=list(ST.cps_for(ds, lr))))
    return ST._order(J)


SUITES = {"X8p": X8_pareto, "X6d": X6_distinct, "X5m": X5_rsk, "X7d": X7_diag}
