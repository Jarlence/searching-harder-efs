"""Minimal SMOTE (Chawla et al. 2002): synthetic minority points on segments to one of the k nearest minority neighbours."""
import numpy as np


def smote_plan(X, y, rng, k=5, ratio=1.0):
    """Return (i, j, lam) arrays: synthetic = X[i] + lam * (X[j] - X[i]); i, j are minority row indices.
    ratio=1.0 -> enough synthetic points to equalise the class sizes."""
    pos = np.flatnonzero(y == 1)
    n_new = int(round((np.sum(y == 0) - len(pos)) * ratio))
    if len(pos) < 2 or n_new <= 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0)
    Xp = X[pos]
    sq = (Xp * Xp).sum(1)
    D = sq[:, None] + sq[None, :] - 2.0 * (Xp @ Xp.T)
    np.fill_diagonal(D, np.inf)
    kk = min(k, len(pos) - 1)
    nn = np.argpartition(D, kk - 1, axis=1)[:, :kk]
    i_loc = rng.integers(0, len(pos), n_new)
    j_loc = nn[i_loc, rng.integers(0, kk, n_new)]
    return pos[i_loc], pos[j_loc], rng.random(n_new)


def smote_apply(X, plan):
    i, j, lam = plan
    if len(i) == 0:
        return np.empty((0, X.shape[1]), X.dtype)
    return X[i] + lam[:, None] * (X[j] - X[i])


def smote(X, y, rng, k=5, ratio=1.0):
    plan = smote_plan(X, y, rng, k, ratio)
    Xn = smote_apply(X, plan)
    return np.vstack([X, Xn]), np.concatenate([y, np.ones(len(Xn), dtype=y.dtype)])
