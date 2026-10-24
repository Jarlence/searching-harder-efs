"""Wrapper fitness with exact evaluation counting and a per-run cache."""
import numpy as np

from . import learners as L
from . import metrics as M


class Wrapper:
    """error(mask) = 1 - metric of the pooled out-of-fold predictions on the given (fixed) inner folds.
    `calls` counts every request (cache hits included) so that budgets are comparable across algorithms."""

    def __init__(self, Xs, y, folds, spec, metric="ba"):
        self.Xs, self.y, self.folds, self.spec, self.metric = Xs, y, folds, spec, metric
        self.aux = L.prepare_cv(y, folds, spec)
        self.thr = L.threshold(spec)
        self.calls = 0
        self.cache = {}
        self.p = Xs.shape[1]

    def _err_from_scores(self, s):
        if self.metric == "auc":
            return 1.0 - M.auc(self.y, s)
        pred = (s > self.thr).astype(np.int64)
        if self.metric == "ba":
            return 1.0 - M.ba(self.y, pred)
        if self.metric == "acc":
            return 1.0 - M.acc(self.y, pred)
        if self.metric == "gm":
            return 1.0 - M.gmean(self.y, pred)
        raise ValueError(self.metric)

    def evaluate(self, mask):
        self.calls += 1
        idx = np.flatnonzero(mask)
        key = idx.astype(np.int32).tobytes()
        v = self.cache.get(key)
        if v is None:
            s = L.oof_scores(self.Xs[:, idx], self.y, self.folds, self.spec, self.aux)
            v = self._err_from_scores(s)
            self.cache[key] = v
        return v

    def oof_metrics(self, mask):
        """full metric dict of the pooled out-of-fold predictions (does not count as a call)."""
        idx = np.flatnonzero(mask)
        s = L.oof_scores(self.Xs[:, idx], self.y, self.folds, self.spec, self.aux)
        return M.all_metrics(self.y, (s > self.thr).astype(np.int64), s)
