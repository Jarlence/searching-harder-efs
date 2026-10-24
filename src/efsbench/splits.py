"""Deterministic stratified / stratified-group K-fold splitters and seeding helpers."""
import hashlib

import numpy as np


def rng_for(*keys):
    """numpy Generator seeded from an arbitrary tuple of keys (stable across runs / machines)."""
    h = hashlib.sha256("|".join(str(k) for k in keys).encode()).hexdigest()
    return np.random.default_rng(int(h[:16], 16))


def stratified_kfold(y, k, rng):
    """Round-robin assignment inside each class => fold sizes and per-class counts differ by <= 1."""
    y = np.asarray(y)
    folds = [[] for _ in range(k)]
    ptr = int(rng.integers(k))
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        for j in idx:
            folds[ptr % k].append(int(j)); ptr += 1
    out = []
    allidx = np.arange(len(y))
    for f in folds:
        te = np.array(sorted(f), dtype=np.int64)
        tr = np.setdiff1d(allidx, te)
        out.append((tr, te))
    return out


def stratified_group_kfold(y, groups, k, rng):
    """Whole groups go to one fold.  Greedy balancing of the minority-class count, then of the fold size.
    Assumes (as in subject / machine data) that a group has one dominant label; the label of a group = its majority label."""
    y = np.asarray(y); groups = np.asarray(groups)
    ug = np.unique(groups)
    glab = {g: int(np.round(y[groups == g].mean())) for g in ug}
    gsize = {g: int((groups == g).sum()) for g in ug}
    order = list(ug)
    rng.shuffle(order)
    order.sort(key=lambda g: -gsize[g])             # big groups first (stable after the shuffle)
    cls_cnt = np.zeros((k, 2)); tot = np.zeros(k)
    assign = {}
    for g in order:
        lab = glab[g]
        cand = np.flatnonzero(cls_cnt[:, lab] == cls_cnt[:, lab].min())
        cand = cand[np.argsort(tot[cand], kind="stable")]
        best = cand[tot[cand] == tot[cand].min()]
        f = int(best[rng.integers(len(best))])
        assign[g] = f; cls_cnt[f, lab] += gsize[g]; tot[f] += gsize[g]
    fold_of = np.array([assign[g] for g in groups])
    allidx = np.arange(len(y))
    return [(allidx[fold_of != f], allidx[fold_of == f]) for f in range(k)]


def make_folds(y, k, rng, groups=None):
    return stratified_group_kfold(y, groups, k, rng) if groups is not None else stratified_kfold(y, k, rng)


def outer_splits(y, k, repeats, dataset, groups=None, tag="outer"):
    """List of dict(rep, fold, tr, te) for repeated K-fold; deterministic in (dataset, rep)."""
    out = []
    for r in range(repeats):
        folds = make_folds(y, k, rng_for(tag, dataset, r), groups)
        for f, (tr, te) in enumerate(folds):
            out.append(dict(rep=r, fold=f, tr=tr, te=te))
    return out
