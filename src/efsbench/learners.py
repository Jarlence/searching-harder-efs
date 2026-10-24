"""Fast wrapper learners on standardised, column-subsetted matrices.  Hyper-parameters are fixed a priori (no tuning,
hence no extra selection step): 5-NN (Euclidean) and a class-balanced weighted ridge classifier (lambda = 1)."""
import numpy as np

SPECS = {"knn": ("knn", 5), "ridge": ("ridge", 1.0)}


def threshold(spec):
    return 0.5 if spec[0] == "knn" else 0.0


def balanced_weights(y):
    n1 = int(y.sum()); n0 = len(y) - n1
    return np.where(y == 1, len(y) / (2.0 * max(n1, 1)), len(y) / (2.0 * max(n0, 1))).astype(np.float64)


def prepare_cv(y, folds, spec):
    if spec[0] != "ridge":
        return [None] * len(folds)
    return [balanced_weights(y[tr]) for tr, _ in folds]


def _sqdist(A, B):
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
    np.maximum(d, 0.0, out=d)
    return d


def knn_oof(Xs, y, folds, k):
    sq = np.einsum("ij,ij->i", Xs, Xs)
    D = sq[:, None] + sq[None, :] - 2.0 * (Xs @ Xs.T)
    s = np.empty(len(y))
    for tr, te in folds:
        sub = D[np.ix_(te, tr)]
        nn = np.argpartition(sub, k - 1, axis=1)[:, :k]
        s[te] = y[tr][nn].mean(1)
    return s


def _ridge_fit(Xtr, ttr, w, lam):
    sw = w.sum()
    mu = (w @ Xtr) / sw
    mt = float((w @ ttr) / sw)
    Xc = Xtr - mu
    tc = ttr - mt
    m = Xc.shape[1]
    if m <= len(ttr):
        Xw = Xc * w[:, None]
        A = Xc.T @ Xw
        A.flat[:: m + 1] += lam
        beta = np.linalg.solve(A, Xw.T @ tc)
    else:                                              # dual form when more features than samples
        r = np.sqrt(w)
        Z = Xc * r[:, None]
        K = Z @ Z.T
        K.flat[:: len(ttr) + 1] += lam
        beta = Z.T @ np.linalg.solve(K, r * tc)
    return mu, mt, beta


def ridge_oof(Xs, y, folds, aux, lam):
    t = 2.0 * y - 1.0
    s = np.empty(len(y))
    for (tr, te), w in zip(folds, aux):
        mu, mt, beta = _ridge_fit(Xs[tr], t[tr], w, lam)
        s[te] = (Xs[te] - mu) @ beta + mt
    return s


def oof_scores(Xs, y, folds, spec, aux):
    return knn_oof(Xs, y, folds, spec[1]) if spec[0] == "knn" else ridge_oof(Xs, y, folds, aux, spec[1])


def fit_predict(Xtr, ytr, Xte, spec):
    """Train on (Xtr, ytr) and return real-valued scores for Xte (higher = more minority-like)."""
    if spec[0] == "knn":
        D = _sqdist(Xte, Xtr)
        nn = np.argpartition(D, spec[1] - 1, axis=1)[:, : spec[1]]
        return ytr[nn].mean(1)
    mu, mt, beta = _ridge_fit(Xtr, 2.0 * ytr - 1.0, balanced_weights(ytr), spec[1])
    return (Xte - mu) @ beta + mt
