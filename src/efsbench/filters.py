"""Non-evolutionary baselines: ANOVA-F filter with k chosen by inner CV, random-forest importance, L1-logistic."""
import numpy as np

from . import learners as L
from . import metrics as M
from . import splits as S

K_GRID = (5, 10, 20, 50, 100)


def anova_f(X, y):
    n1 = y.sum(); n0 = len(y) - n1
    m1 = X[y == 1].mean(0); m0 = X[y == 0].mean(0); m = X.mean(0)
    ssb = n1 * (m1 - m) ** 2 + n0 * (m0 - m) ** 2
    ssw = ((X[y == 1] - m1) ** 2).sum(0) + ((X[y == 0] - m0) ** 2).sum(0)
    return ssb / np.maximum(ssw / max(len(y) - 2, 1), 1e-12)


def rf_importance(X, y, seed):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample", n_jobs=1, random_state=seed % (2 ** 31))
    rf.fit(X, y)
    return rf.feature_importances_


def _score_fn(kind, seed):
    if kind == "anova":
        return lambda X, y: anova_f(X, y)
    if kind == "rf":
        return lambda X, y: rf_importance(X, y, seed)
    raise ValueError(kind)


def choose_k_by_cv(Xtr, ytr, spec, kind, inner_key, inner_k=5, groups_tr=None):
    """Pick k from K_GRID by inner-CV balanced accuracy; the ranking is recomputed inside every inner training part."""
    folds = S.make_folds(ytr, inner_k, S.rng_for(*inner_key), groups_tr)
    scorer = _score_fn(kind, int(S.rng_for("rf", *inner_key).integers(0, 2 ** 31 - 1)))
    p = Xtr.shape[1]
    ks = [k for k in K_GRID if k <= p] or [p]
    pred = {k: np.empty(len(ytr), dtype=np.int64) for k in ks}
    for tr, te in folds:
        order = np.argsort(-scorer(Xtr[tr], ytr[tr]), kind="stable")
        for k in ks:
            sc = L.fit_predict(Xtr[tr][:, order[:k]], ytr[tr], Xtr[te][:, order[:k]], spec)
            pred[k][te] = (sc > L.threshold(spec)).astype(np.int64)
    bas = {k: M.ba(ytr, pred[k]) for k in ks}
    return max(ks, key=lambda k: (bas[k], -k))


def topk_pipeline(Xtr, ytr, Xte, yte, spec, kind, inner_key, groups_tr=None):
    k = choose_k_by_cv(Xtr, ytr, spec, kind, inner_key, groups_tr=groups_tr)
    scorer = _score_fn(kind, int(S.rng_for("rf", *inner_key).integers(0, 2 ** 31 - 1)))
    order = np.argsort(-scorer(Xtr, ytr), kind="stable")[:k]
    sc = L.fit_predict(Xtr[:, order], ytr, Xte[:, order], spec)
    return k, M.all_metrics(yte, (sc > L.threshold(spec)).astype(np.int64), sc)


def all_features_pipeline(Xtr, ytr, Xte, yte, spec):
    sc = L.fit_predict(Xtr, ytr, Xte, spec)
    return Xtr.shape[1], M.all_metrics(yte, (sc > L.threshold(spec)).astype(np.int64), sc)


def l1_logistic_pipeline(Xtr, ytr, Xte, yte, inner_key, groups_tr=None, Cs=(0.01, 0.1, 1.0)):
    from sklearn.linear_model import LogisticRegression
    folds = S.make_folds(ytr, 5, S.rng_for(*inner_key), groups_tr)
    seed = int(S.rng_for("l1_solver", *inner_key).integers(0, 2 ** 31 - 1))        # liblinear shuffles coordinates: seed it (docs/DEVIATIONS.md #5)
    best = None
    for C in Cs:
        pred = np.empty(len(ytr), dtype=np.int64)
        for tr, te in folds:
            clf = LogisticRegression(penalty="l1", solver="liblinear", C=C, class_weight="balanced", max_iter=200, random_state=seed)
            clf.fit(Xtr[tr], ytr[tr]); pred[te] = clf.predict(Xtr[te])
        b = M.ba(ytr, pred)
        if best is None or b > best[0]:
            best = (b, C)
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=best[1], class_weight="balanced", max_iter=200, random_state=seed)
    clf.fit(Xtr, ytr)
    sc = clf.decision_function(Xte)
    return int((clf.coef_ != 0).sum()), M.all_metrics(yte, (sc > 0).astype(np.int64), sc)
