"""Extra studies (outside the hashed main code; results go to results/jobs_extra/<suite>/).

X1s  scaler  : honest EFS run where the inner CV re-fits the standardisation on every inner-training part
               (docs/PROTOCOL.md section 6 / 7 'E-scaler').  Paired with E1h (same seeds, same outer splits).
X2k  innerk  : inner fold count 3 and 10 instead of 5 (E1h uses 5).
X3u  synth   : synthetic unit-structured data (U units x 3 recordings, unit fingerprint strength sig_f) - dose-response of the
               record-wise vs subject-wise evaluation inflation (docs/PROTOCOL.md section 7 'E-unit-synth').
X4r  land    : random-subset landscape - the selection effect without any search: inner-CV BA and held-out BA of 16 384 random subsets,
               max-of-B statistics (true labels and permuted labels).
"""
import hashlib
import json
import os
import sys
import time
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np  # noqa: E402

from efsbench import algos as A, datasets as D, learners as L, metrics as M, protocols as P, splits as S, wrapper as W  # noqa: E402

JOBS = os.environ.get("EFS_JOBS_EXTRA", os.path.join(ROOT, "results", "jobs_extra"))
DS3 = ["colon", "prostate", "kc3"]                     # chosen before any result: small HD, larger HD, low-dimensional imbalanced
CPS = [1000, 4000]


def _code_sha():
    h = hashlib.sha256()
    h.update(open(os.path.abspath(__file__), "rb").read())
    d = os.path.join(ROOT, "src", "efsbench")
    for f in sorted(os.listdir(d)):
        if f.endswith(".py"):
            h.update(open(os.path.join(d, f), "rb").read())
    return h.hexdigest()[:16]


CODE_SHA = _code_sha()
_DATA = {}


def data(ds):
    if ds not in _DATA:
        _DATA[ds] = D.load(ds)
    return _DATA[ds]


def outer_split(y, ds, rep, fold, tag="outer", groups=None):
    return S.make_folds(y, 5, S.rng_for(tag, ds, rep), groups)[fold]


def permuted(y, ds, pid):
    yp = y.copy()
    S.rng_for("perm", ds, pid).shuffle(yp)
    return yp


# ------------------------------------------------------------------------------------------------ X1s
class FoldScaledWrapper(W.Wrapper):
    """Inner folds with the standardisation re-fitted on each inner-training part (statistics never see the inner validation part)."""

    def __init__(self, Xraw, y, folds, spec, metric="ba"):
        self.y, self.folds, self.spec, self.metric = y, folds, spec, metric
        self.aux = L.prepare_cv(y, folds, spec)
        self.thr = L.threshold(spec)
        self.calls = 0
        self.cache = {}
        self.p = Xraw.shape[1]
        self.parts = []
        for tr, te in folds:
            a, b = P.standardize(Xraw[tr], Xraw[te])
            self.parts.append((np.asfortranarray(a), np.asfortranarray(b)))

    def evaluate(self, mask):
        self.calls += 1
        idx = np.flatnonzero(mask)
        key = idx.astype(np.int32).tobytes()
        v = self.cache.get(key)
        if v is None:
            s = np.empty(len(self.y))
            for (tr, te), (A_, B_) in zip(self.folds, self.parts):
                s[te] = L.fit_predict(A_[:, idx], self.y[tr], B_[:, idx], self.spec)
            v = self._err_from_scores(s)
            self.cache[key] = v
        return v


def k_scaler(s):
    ds, algo, lr, rep, fold = s["ds"], s["algo"], s["learner"], s["rep"], s["fold"]
    X, y, _ = data(ds)
    tr, te = outer_split(y, ds, rep, fold)
    spec = L.SPECS[lr]; thr = L.threshold(spec)
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    Xtr_s, Xte_s = P.standardize(Xtr, Xte)
    inner = S.make_folds(ytr, 5, S.rng_for("inner", ds, lr, rep, fold, None, False, False))      # identical keys to E1h
    w = FoldScaledWrapper(Xtr, ytr, inner, spec, "ba")
    run = A.Run(w, s["cps"])
    A.run_algo(algo, run, S.rng_for("algo", ds, algo, lr, rep, fold, None, 0, False, False))
    out = {}
    for cp in s["cps"]:
        idx, err = P._sel(algo, run.snap[cp])
        sc = L.fit_predict(Xtr_s[:, idx], ytr, Xte_s[:, idx], spec)
        m = M.all_metrics(yte, (sc > thr).astype(np.int64), sc)
        out[int(cp)] = dict(k=int(len(idx)), inner_err=float(err), sel=[int(i) for i in idx[:400]], **{"test_" + a: float(b) for a, b in m.items()})
    return out


# ------------------------------------------------------------------------------------------------ X2k
def k_innerk(s):
    ds, algo, lr, rep, fold, ik = s["ds"], s["algo"], s["learner"], s["rep"], s["fold"], s["inner_k"]
    X, y, _ = data(ds)
    tr, te = outer_split(y, ds, rep, fold)
    return P.honest_core(X[tr], y[tr], X[te], y[te], algo, lr, ("inner", ds, lr, rep, fold, None, False, False, "k%d" % ik),
                         ("algo", ds, algo, lr, rep, fold, None, 0, False, False), tuple(s["cps"]), "ba", ik)


# ------------------------------------------------------------------------------------------------ X3u
SYN_P, SYN_Q, SYN_DELTA, SYN_R, SYN_PREV = 200, 10, 0.5, 3, 0.35


def synth_units(U, sig_f, inst, p=SYN_P, q=SYN_Q, delta=SYN_DELTA, R=SYN_R, prev=SYN_PREV):
    """U units x R recordings; unit-level label (minority = 1); q informative features (class means +-delta/2); every unit has a
    random fingerprint offset ~ N(0, sig_f^2) on all features; recording noise N(0,1).  Columns are shuffled (instance-specific)."""
    rng = S.rng_for("synth", U, R, p, q, delta, sig_f, inst)
    lab = np.zeros(U, dtype=np.int64); lab[rng.permutation(U)[: int(round(prev * U))]] = 1
    b = rng.normal(0.0, sig_f, size=(U, p)) if sig_f > 0 else np.zeros((U, p))
    X = rng.normal(0.0, 1.0, size=(U * R, p)) + np.repeat(b, R, axis=0)
    sign = np.where(np.repeat(lab, R) == 1, 1.0, -1.0)
    X[:, :q] += sign[:, None] * (delta / 2.0)
    perm = S.rng_for("synth_perm", p, inst).permutation(p)
    X = X[:, perm]
    informative = np.flatnonzero(perm < q)
    return X.astype(np.float32), np.repeat(lab, R), np.repeat(np.arange(U), R), informative


def k_synth(s):
    U, sf, inst, fold, og, ig, algo = s["U"], s["sig_f"], s["inst"], s["fold"], s["outer_group"], s["inner_group"], s["algo"]
    X, y, g, inf = synth_units(U, sf, inst)
    tr, te = S.make_folds(y, 5, S.rng_for("synouter", U, sf, inst, og), g if og else None)[fold]
    res = P.honest_core(X[tr], y[tr], X[te], y[te], algo, "ridge", ("syninner", U, sf, inst, fold, og, ig),
                        ("synalgo", algo, U, sf, inst, fold, og, ig), tuple(s["cps"]), "ba", 5, g[tr] if ig else None)
    infset = set(int(i) for i in inf)
    for cp, r in res.items():
        sel = r["sel"]
        r["n_inf_selected"] = int(sum(1 for i in sel if i in infset))
    return res


# ------------------------------------------------------------------------------------------------ X4r
LAND_NSUB = 16384
LAND_BS = (250, 1000, 4000, 16000)


def _ks(p):
    return [5, 20, 50] if p >= 100 else [3, 10, 20]


def _boot_max(inn, tst, B, rng, R=400):
    n = len(inn)
    mi = np.empty(R); mt = np.empty(R)
    for r in range(R):
        ii = rng.integers(0, n, B)
        v = inn[ii]; mx = v.max()
        mi[r] = mx; mt[r] = tst[ii][v == mx].mean()
    return float(mi.mean()), float(mt.mean())


def k_land(s):
    ds, lr, rep, fold, pid = s["ds"], s["learner"], s["rep"], s["fold"], s.get("perm")
    X, y, _ = data(ds)
    if pid is not None:
        y = permuted(y, ds, pid)
    tr, te = outer_split(y, ds, rep, fold, "outer" if pid is None else "outer_perm%s" % pid)
    spec = L.SPECS[lr]; thr = L.threshold(spec)
    Xtr_s, Xte_s = P.standardize(X[tr], X[te]); Xf = np.asfortranarray(Xtr_s)
    ytr, yte = y[tr], y[te]
    inner = S.make_folds(ytr, 5, S.rng_for("landinner", ds, lr, rep, fold, pid))
    aux = L.prepare_cv(ytr, inner, spec)
    rng = S.rng_for("landsub", ds, lr, rep, fold, pid)
    p = X.shape[1]
    out = {}
    for k in _ks(p):
        inn = np.empty(LAND_NSUB); tst = np.empty(LAND_NSUB)
        for i in range(LAND_NSUB):
            idx = np.sort(rng.choice(p, k, replace=False))
            sc = L.oof_scores(Xf[:, idx], ytr, inner, spec, aux)
            inn[i] = M.ba(ytr, (sc > thr).astype(np.int64))
            st = L.fit_predict(Xtr_s[:, idx], ytr, Xte_s[:, idx], spec)
            tst[i] = M.ba(yte, (st > thr).astype(np.int64))
        from scipy import stats
        rec = dict(inn_mean=float(inn.mean()), inn_sd=float(inn.std()), tst_mean=float(tst.mean()), tst_sd=float(tst.std()),
                   pearson=float(np.corrcoef(inn, tst)[0, 1]) if inn.std() > 0 and tst.std() > 0 else 0.0,
                   spearman=float(stats.spearmanr(inn, tst).correlation) if inn.std() > 0 and tst.std() > 0 else 0.0)
        br = S.rng_for("landboot", ds, lr, rep, fold, pid, k)
        for B in LAND_BS:
            mi, mt = _boot_max(inn, tst, B, br)
            rec["max_inn_%d" % B] = mi; rec["tst_at_max_%d" % B] = mt
        out[int(k)] = rec
    return out


KINDS = {"scaler": k_scaler, "innerk": k_innerk, "synth": k_synth, "land": k_land}


def job_id(s):
    parts = [s["suite"], s["ds"] if "ds" in s else "synth", s.get("algo", "-"), s.get("learner", "-")]
    for k in ("rep", "fold", "inner_k", "U", "sig_f", "inst", "outer_group", "inner_group", "perm"):
        if k in s and s[k] is not None:
            parts.append("%s%s" % (k[:3], s[k]))
    return "__".join(str(x) for x in parts)


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


# ------------------------------------------------------------------------------------------------ enumerators
def X1_scaler(n_rep=5):
    J = []
    for ds in DS3:
        for lr in ("ridge", "knn"):
            for algo in ("ga", "rs"):
                for rep in range(n_rep):
                    for fold in range(5):
                        J.append(dict(suite="X1s", kind="scaler", ds=ds, algo=algo, learner=lr, rep=rep, fold=fold, cps=CPS))
    return J


def X2_innerk(n_rep=5):
    J = []
    for ds in DS3:
        for lr in ("ridge", "knn"):
            for algo in ("ga", "rs"):
                for ik in (3, 10):
                    for rep in range(n_rep):
                        for fold in range(5):
                            J.append(dict(suite="X2k", kind="innerk", ds=ds, algo=algo, learner=lr, rep=rep, fold=fold, inner_k=ik, cps=CPS))
    return J


def X3_synth(n_inst=5):
    J = []
    for U in (50, 100, 200):
        for sf in (0.0, 0.5, 1.0, 2.0):
            for og in (False, True):
                for ig in (False, True):
                    for algo in ("ga", "rs"):
                        for inst in range(n_inst):
                            for fold in range(5):
                                J.append(dict(suite="X3u", kind="synth", algo=algo, learner="ridge", U=U, sig_f=sf, inst=inst, fold=fold,
                                              outer_group=og, inner_group=ig, cps=[250, 1000, 4000]))
    return J


def X4_land(n_rep=3):
    J = []
    for ds in DS3:
        for lr in ("ridge", "knn"):
            for pid in (None, 0):
                for rep in range(n_rep):
                    for fold in range(5):
                        J.append(dict(suite="X4r", kind="land", ds=ds, learner=lr, rep=rep, fold=fold, perm=pid))
    return J


SUITES = {"X1s": X1_scaler, "X2k": X2_innerk, "X3u": X3_synth, "X4r": X4_land}
