"""Tiny fixture for the extra studies (debugging of analysis/extras_analysis.py only; never reported)."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ["EFS_JOBS_EXTRA"] = os.path.join(ROOT, "analysis", "_fixture", "extra")
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(ROOT, "extras"))
import ex_suites as ES  # noqa: E402

CP = [250, 1000]


def specs():
    J = []
    for ds in ("colon", "cns"):
        for lr in ("ridge", "knn"):
            for algo in ("ga", "rs"):
                for rep in range(2):
                    for fold in range(5):
                        J.append(dict(suite="X1s", kind="scaler", ds=ds, algo=algo, learner=lr, rep=rep, fold=fold, cps=CP))
                        for ik in (3, 10):
                            J.append(dict(suite="X2k", kind="innerk", ds=ds, algo=algo, learner=lr, rep=rep, fold=fold, inner_k=ik, cps=CP))
    for U in (50, 100):
        for sf in (0.0, 1.0):
            for og in (False, True):
                for ig in (False, True):
                    for algo in ("ga", "rs"):
                        for inst in range(2):
                            for fold in range(5):
                                J.append(dict(suite="X3u", kind="synth", algo=algo, learner="ridge", U=U, sig_f=sf, inst=inst, fold=fold,
                                              outer_group=og, inner_group=ig, cps=[250, 1000]))
    for lr in ("ridge", "knn"):
        for pid in (None, 0):
            for rep in range(2):
                for fold in range(5):
                    J.append(dict(suite="X4r", kind="land", ds="colon", learner=lr, rep=rep, fold=fold, perm=pid))
    return J


if __name__ == "__main__":
    from joblib import Parallel, delayed
    ES.LAND_NSUB = 400
    J = specs()
    print("jobs", len(J), flush=True)

    def go(s):
        ES.LAND_NSUB = 400
        return ES.run_job(s)
    r = Parallel(n_jobs=3, backend="loky")(delayed(go)(s) for s in J)
    print({k: r.count(k) for k in set(r)})
