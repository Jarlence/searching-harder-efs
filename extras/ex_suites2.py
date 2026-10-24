"""Second batch of extra studies (written after the first batch had been analysed, BEFORE any X5 result existed).

X5s  rs100 : subset-size confound of the random-search null.  The pre-registered null (RS) draws subset sizes uniformly from 1..40, BDE ends up with ~70-100
             features (it hits the KMAX = 100 cap) while GA / BPSO sit at ~20 and NSGA-II at ~7-9.  If larger subsets are simply better for ridge / 5-NN, BDE's
             honest advantage over RS could come from subset size instead of search quality.  RS100 draws sizes uniformly from 1..100 (matched to BDE's range).
             Same outer splits / inner folds as E1h; budgets 1000 and 4000; outer repeats 0-4 (25 outer runs per cell); all 13 core data sets.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ex_suites as ES  # noqa: E402
from efsbench import algos as A, protocols as P  # noqa: E402

CORE = ["colon", "cns", "dlbcl", "leukemia", "prostate", "gli", "smk", "ovarian", "breast", "oil_spill", "ozone", "speech", "kc3"]
NO_KNN = ("ozone", "speech")


def _rs_kmax(kmax):
    def f(run, rng, init_p):
        A.random_search(run, rng, init_p, kmax=kmax)
    return f


A.ALGOS["rs100"] = _rs_kmax(100)            # runtime registration only; the efsbench sources are untouched


def k_rs100(s):
    ds, lr, rep, fold = s["ds"], s["learner"], s["rep"], s["fold"]
    X, y, _ = ES.data(ds)
    tr, te = ES.outer_split(y, ds, rep, fold)
    return P.honest_core(X[tr], y[tr], X[te], y[te], "rs100", lr, ("inner", ds, lr, rep, fold, None, False, False),
                         ("algo", ds, "rs100", lr, rep, fold, None, 0, False, False), tuple(s["cps"]), "ba", 5)


ES.KINDS["rs100"] = k_rs100


def X5_rs100(n_rep=5):
    J = []
    for ds in CORE:
        for lr in (["ridge"] if ds in NO_KNN else ["ridge", "knn"]):
            for rep in range(n_rep):
                for fold in range(5):
                    J.append(dict(suite="X5s", kind="rs100", ds=ds, algo="rs100", learner=lr, rep=rep, fold=fold, cps=[1000, 4000]))
    return J


SUITES = {"X5s": X5_rs100}


def run_job(s):
    """Entry point used by run_extra.py: defined HERE so that every worker process imports this module (which registers rs100) before running a job."""
    return ES.run_job(s)
