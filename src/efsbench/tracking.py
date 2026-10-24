"""Passive bookkeeping around a search run (added after the main experiments; docs/DEVIATIONS.md #7).

`TrackRun` is a drop-in replacement for `algos.Run`.  It draws no random numbers and does not change the search trajectory
(tests/test_tracking.py checks bit-equality with `algos.Run`).  In addition it
  * counts checkpoints either in wrapper REQUESTS (mode="calls", identical to `Run`) or in DISTINCT evaluated masks (mode="distinct");
  * keeps every distinct evaluated candidate and, at every checkpoint, returns the Pareto ARCHIVE: the candidates that are non-dominated
    in (inner error, number of features) among ALL candidates evaluated up to that moment.  Candidates with identical objective vectors are
    merged; the representative is the first one evaluated (the same tie rule as the incumbent of `Run`);
  * records the number of requests and of distinct masks at every checkpoint.

`honest_core_tracked` is `protocols.honest_core` on top of `TrackRun`; for NSGA-II it evaluates every archive member on the outer test part, so that
the over-statement D of 'choose the Pareto point on the test set' is defined by ONE set of candidates: the archive that contains the incumbent."""
import numpy as np

from . import algos as A
from . import learners as L
from . import metrics as M
from . import splits as S
from .protocols import standardize
from .wrapper import Wrapper


def pareto_archive(cands):
    """cands: list of (err, k, idx) of DISTINCT masks in evaluation order.
    Returns the positions (into `cands`) of the non-dominated candidates (minimise err and k), one per objective vector (the first evaluated),
    ordered by increasing k.  A candidate is dominated if another has err <= and k <=, with at least one strict."""
    n = len(cands)
    if n == 0:
        return []
    err = np.fromiter((c[0] for c in cands), float, n)
    k = np.fromiter((c[1] for c in cands), np.int64, n)
    order = np.lexsort((np.arange(n), err, k))               # k ascending, then error, then evaluation order
    keep, best, last_k = [], np.inf, -1
    for i in order:
        if k[i] == last_k:                                   # only the first member of a k-group (lowest error, earliest) can qualify
            continue
        last_k = k[i]
        if err[i] < best:
            keep.append(int(i)); best = err[i]
    return keep


class TrackRun(A.Run):
    def __init__(self, wrapper, checkpoints, mode="calls", archive=False, call_cap=None):
        super().__init__(wrapper, (10 ** 12,))                # the base class never takes a snapshot itself
        if mode not in ("calls", "distinct"):
            raise ValueError(mode)
        self.tcp = sorted(int(c) for c in checkpoints)
        self.mode, self.archive = mode, bool(archive)
        self.call_cap = int(call_cap) if call_cap else None
        self._tn = 0
        self._cand = []
        self.tsnap = {}

    def _count(self):
        return self.w.calls if self.mode == "calls" else len(self.w.cache)

    def done(self):
        return self._count() >= self.tcp[-1] or (self.call_cap is not None and self.w.calls >= self.call_cap)

    def snapshot(self):
        sn = dict(wsum_idx=self.best_idx, wsum_err=self.best_err, lex_idx=self.lex_idx, lex_err=self.lex_err,
                  front=self._front, calls=int(self.w.calls), distinct=int(len(self.w.cache)))
        if self.archive:
            keep = pareto_archive(self._cand)
            sn["arch"] = [self._cand[i] for i in keep]
            groups = {}                                       # every distinct mask that shares an objective vector with a front member
            for c in self._cand:
                groups.setdefault((c[0], c[1]), []).append(c[2])
            sn["arch_ties"] = [groups[(self._cand[i][0], self._cand[i][1])] for i in keep]
        return sn

    def eval(self, mask):
        n0 = len(self.w.cache)
        err, k, f = super().eval(mask)                        # repairs the mask in place, evaluates, updates the incumbents
        if self.archive and len(self.w.cache) > n0:           # a new distinct mask
            self._cand.append((err, k, np.flatnonzero(mask).astype(np.int32)))
        while self._tn < len(self.tcp) and self._count() >= self.tcp[self._tn]:
            self.tsnap[self.tcp[self._tn]] = self.snapshot()
            self._tn += 1
        return err, k, f


def _sel(algo, sn):
    return (sn["lex_idx"], sn["lex_err"]) if algo == "nsga2" else (sn["wsum_idx"], sn["wsum_err"])


def honest_core_tracked(Xtr, ytr, Xte, yte, algo, learner, inner_key, algo_key, checkpoints, mode="calls", archive=False,
                        call_cap=None, metric="ba", inner_k=5, test_metrics=True):
    """Same data flow as protocols.honest_core (everything data-driven sees the outer-training part only), on top of TrackRun.
    Returns {checkpoint: record}; each record has k, inner_err, calls, distinct, reached, test_* and, with archive=True, the archive
    (arch_k, arch_err, arch_test_ba, ordered by increasing k).  With test_metrics=False nothing is evaluated on the test part.
    If `call_cap` stops the search before a checkpoint is reached, that checkpoint holds the final state with reached=False."""
    spec = L.SPECS[learner]; thr = L.threshold(spec)
    Xtr_s, Xte_s = standardize(Xtr, Xte)
    inner = S.make_folds(ytr, inner_k, S.rng_for(*inner_key), None)
    w = Wrapper(np.asfortranarray(Xtr_s), ytr, inner, spec, metric)
    run = TrackRun(w, checkpoints, mode, archive, call_cap)
    A.run_algo(algo, run, S.rng_for(*algo_key))
    out = {}
    for cp in run.tcp:
        reached = cp in run.tsnap
        sn = run.tsnap[cp] if reached else run.snapshot()
        idx, err = _sel(algo, sn)
        rec = dict(k=int(len(idx)), inner_err=float(err), calls=sn["calls"], distinct=sn["distinct"], reached=bool(reached))
        if test_metrics:
            sc = L.fit_predict(Xtr_s[:, idx], ytr, Xte_s[:, idx], spec)
            m = M.all_metrics(yte, (sc > thr).astype(np.int64), sc)
            rec.update({"test_" + a: float(b) for a, b in m.items()})
            if archive:
                ak, ae, ab, tn, tm = [], [], [], [], []
                for (e, kk, ii), ties in zip(sn["arch"], sn["arch_ties"]):
                    s2 = L.fit_predict(Xtr_s[:, ii], ytr, Xte_s[:, ii], spec)
                    b0 = float(M.ba(yte, (s2 > thr).astype(np.int64)))
                    ak.append(int(kk)); ae.append(float(e)); ab.append(b0)
                    best = b0                                                  # the best test BA among ALL masks tied at this objective vector
                    for jj in ties:
                        if jj is ii:
                            continue
                        s3 = L.fit_predict(Xtr_s[:, jj], ytr, Xte_s[:, jj], spec)
                        best = max(best, float(M.ba(yte, (s3 > thr).astype(np.int64))))
                    tn.append(int(len(ties))); tm.append(best)
                rec.update(arch_k=ak, arch_err=ae, arch_test_ba=ab, arch_tie_n=tn, arch_tie_max_ba=tm)
        elif archive:
            rec.update(arch_k=[int(c[1]) for c in sn["arch"]], arch_err=[float(c[0]) for c in sn["arch"]])
        out[int(cp)] = rec
    return out
