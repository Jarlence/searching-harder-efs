"""Evaluation protocols.

HONEST   honest_core(): everything that is data-driven (scaling, folds, search) sees the outer-training part only.
FLAWED   fulldata_core(): feature search on ALL rows, then 'evaluate' on the same rows (selection-before-split leakage);
         the same module reports the optimistic numbers produced by common reporting habits
         (fitness-as-report, best-of-front-by-test, best-of-seeds, SMOTE-before-split, record-wise CV on repeated measures).
Every function takes plain arrays so that tests can swap the held-out part for garbage and check that selection is unchanged."""
import numpy as np

from . import algos as A
from . import learners as L
from . import metrics as M
from . import smote as SM
from . import splits as S
from .wrapper import Wrapper

CHECKPOINTS = (250, 1000, 4000, 16000)


def standardize(Xtr, *others):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd = np.where(sd > 0, sd, 1.0)
    return tuple((X - mu) / sd for X in (Xtr,) + others)


def _sel(algo, snap):
    return (snap["lex_idx"], snap["lex_err"]) if algo == "nsga2" else (snap["wsum_idx"], snap["wsum_err"])


# ---------------------------------------------------------------------------------------------------- HONEST
def honest_core(Xtr, ytr, Xte, yte, algo, learner, inner_key, algo_key, checkpoints=CHECKPOINTS, metric="ba",
                inner_k=5, groups_tr=None, with_front=True):
    spec = L.SPECS[learner]; thr = L.threshold(spec)
    Xtr_s, Xte_s = standardize(Xtr, Xte)
    inner = S.make_folds(ytr, inner_k, S.rng_for(*inner_key), groups_tr)
    w = Wrapper(np.asfortranarray(Xtr_s), ytr, inner, spec, metric)
    run = A.Run(w, checkpoints)
    A.run_algo(algo, run, S.rng_for(*algo_key))
    out = {}
    for cp in checkpoints:
        sn = run.snap[cp]
        idx, err = _sel(algo, sn)
        sc = L.fit_predict(Xtr_s[:, idx], ytr, Xte_s[:, idx], spec)
        m = M.all_metrics(yte, (sc > thr).astype(np.int64), sc)
        rec = dict(k=int(len(idx)), inner_err=float(err), sel=[int(i) for i in idx[:400]],
                   **{"test_" + k: float(v) for k, v in m.items()})
        if algo == "nsga2" and with_front and sn["front"] is not None:
            fidx, ferr = sn["front"]
            rec["front_k"] = [int(len(ii)) for ii in fidx]
            rec["front_inner_err"] = [float(e) for e in ferr]
            fb = []
            for ii in fidx:
                s2 = L.fit_predict(Xtr_s[:, ii], ytr, Xte_s[:, ii], spec)
                fb.append(float(M.ba(yte, (s2 > thr).astype(np.int64))))
            rec["front_test_ba"] = fb
        out[int(cp)] = rec
    return out


# ---------------------------------------------------------------------------------------------------- FLAW A
def fulldata_core(X, y, algo, learner, inner_key, algo_key, checkpoints=CHECKPOINTS, metric="ba", groups=None,
                  reeval_repeats=3, reeval_group=None):
    """Search on all rows (scaler, folds and fitness all see every row) and report (i) the fitness of the incumbent and
    (ii) a fresh repeated 5-fold CV of the selected subset on the same rows - the two numbers papers usually report."""
    spec = L.SPECS[learner]; thr = L.threshold(spec)
    Xs, = standardize(X)
    inner = S.make_folds(y, 5, S.rng_for(*inner_key), groups)
    w = Wrapper(np.asfortranarray(Xs), y, inner, spec, metric)
    run = A.Run(w, checkpoints)
    A.run_algo(algo, run, S.rng_for(*algo_key))
    rg = groups if reeval_group is None else reeval_group
    out = {}
    for cp in checkpoints:
        idx, err = _sel(algo, run.snap[cp])
        Xsel = Xs[:, idx]
        bas, aucs = [], []
        for r in range(reeval_repeats):
            folds = S.make_folds(y, 5, S.rng_for("reeval", *inner_key, r), rg)
            s = L.oof_scores(Xsel, y, folds, spec, L.prepare_cv(y, folds, spec))
            bas.append(M.ba(y, (s > thr).astype(np.int64))); aucs.append(M.auc(y, s))
        out[int(cp)] = dict(k=int(len(idx)), reported_err=float(err), leaky_cv_ba=float(np.mean(bas)),
                            leaky_cv_auc=float(np.mean(aucs)))
    return out


# ---------------------------------------------------------------------------------------------------- SMOTE inside the wrapper
class SmoteCleanWrapper(Wrapper):
    """Inner fold = real training part + synthetic minority rows built from THAT training part only; scored on real rows."""

    def __init__(self, Xs, y, folds, spec, metric, rng):
        super().__init__(Xs, y, folds, spec, metric)
        self.plans = []
        for tr, _ in folds:
            self.plans.append(SM.smote_plan(Xs[tr], y[tr], rng))
        self.synth = [SM.smote_apply(Xs[tr], pl) for (tr, _), pl in zip(folds, self.plans)]

    def evaluate(self, mask):
        self.calls += 1
        idx = np.flatnonzero(mask)
        key = idx.astype(np.int32).tobytes()
        v = self.cache.get(key)
        if v is None:
            Xsel = self.Xs[:, idx]
            s = np.empty(len(self.y))
            for (tr, te), syn in zip(self.folds, self.synth):
                Xt = np.vstack([Xsel[tr], syn[:, idx]]); yt = np.concatenate([self.y[tr], np.ones(len(syn), np.int64)])
                s[te] = L.fit_predict(Xt, yt, Xsel[te], self.spec)
            v = self._err_from_scores(s)
            self.cache[key] = v
        return v


def smote_inner_core(Xtr, ytr, Xte, yte, algo, learner, mode, inner_key, algo_key, checkpoints=(250, 1000, 4000)):
    """EFS on a SMOTE-balanced training set.  mode='clean': SMOTE inside every inner fold.
    mode='leaky': SMOTE once on the whole outer-training set, then cross-validate over real+synthetic rows (common practice).
    Final model: trained on SMOTE(outer-train); scored on the REAL outer-test rows in both modes."""
    spec = L.SPECS[learner]; thr = L.threshold(spec)
    Xtr_s, Xte_s = standardize(Xtr, Xte)
    rng = S.rng_for("smote", *inner_key)
    if mode == "clean":
        inner = S.make_folds(ytr, 5, S.rng_for(*inner_key))
        w = SmoteCleanWrapper(np.asfortranarray(Xtr_s), ytr, inner, spec, "ba", rng)
    else:
        Xa, ya = SM.smote(Xtr_s, ytr, rng)
        inner = S.make_folds(ya, 5, S.rng_for(*inner_key))
        w = Wrapper(np.asfortranarray(Xa), ya, inner, spec, "ba")
    run = A.Run(w, checkpoints)
    A.run_algo(algo, run, S.rng_for(*algo_key))
    Xf, yf = SM.smote(Xtr_s, ytr, S.rng_for("smote_final", *inner_key))
    out = {}
    for cp in checkpoints:
        idx, err = _sel(algo, run.snap[cp])
        sc = L.fit_predict(Xf[:, idx], yf, Xte_s[:, idx], spec)
        m = M.all_metrics(yte, (sc > thr).astype(np.int64), sc)
        out[int(cp)] = dict(k=int(len(idx)), inner_err=float(err), **{"test_" + k: float(v) for k, v in m.items()})
    return out


# ---------------------------------------------------------------------------------------------------- SMOTE at the outer level
def smote_outer_core(X, y, learner, k_feat, protocol, dataset, rep, ratio=1.0, folds_k=5):
    """Fixed pipeline: scale -> [SMOTE] -> ANOVA top-k -> learner.  protocol='honest': SMOTE only on each training part and
    scoring on real held-out rows;  protocol='before_split': SMOTE on all rows first, then K-fold over real+synthetic rows."""
    spec = L.SPECS[learner]; thr = L.threshold(spec)
    from .filters import anova_f
    rng = S.rng_for("smote_outer", dataset, rep, protocol)
    if protocol == "before_split":
        Xs, = standardize(X)
        Xa, ya = SM.smote(Xs, y, rng, ratio=ratio)
        folds = S.stratified_kfold(ya, folds_k, S.rng_for("smote_outer_folds", dataset, rep))
        pred = np.empty(len(ya), np.int64); sco = np.empty(len(ya))
        for tr, te in folds:
            top = np.argsort(-anova_f(Xa[tr], ya[tr]), kind="stable")[:k_feat]
            sc = L.fit_predict(Xa[tr][:, top], ya[tr], Xa[te][:, top], spec)
            sco[te] = sc; pred[te] = (sc > thr)
        return M.all_metrics(ya, pred, sco)
    folds = S.stratified_kfold(y, folds_k, S.rng_for("smote_outer_folds", dataset, rep))
    pred = np.empty(len(y), np.int64); sco = np.empty(len(y))
    for tr, te in folds:
        Xtr, Xte = standardize(X[tr], X[te])
        Xa, ya = SM.smote(Xtr, y[tr], rng, ratio=ratio)
        top = np.argsort(-anova_f(Xa, ya), kind="stable")[:k_feat]
        sc = L.fit_predict(Xa[:, top], ya, Xte[:, top], spec)
        sco[te] = sc; pred[te] = (sc > thr)
    return M.all_metrics(y, pred, sco)
