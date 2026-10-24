"""Binary evolutionary / swarm feature-selection searches with exact evaluation-budget accounting.

All algorithms (GA, sparsity-anchored BPSO, binary DE, NSGA-II, budget-matched random search) start from the same
sparse initial distribution (P[bit=1] = min(0.5, 20/p)) and evaluate through Run.eval(), which keeps the incumbents and
takes a snapshot exactly when the wrapper has been called `checkpoint` times.  Nothing here touches test data."""
import numpy as np

ALPHA = 0.99          # single-objective fitness f = ALPHA*err + (1-ALPHA)*k/p   (usual weighted-sum form)
K0 = 20               # expected initial subset size
KMAX = 100            # hard cardinality cap for EVERY algorithm (random repair of the excess bits, written back in place)


def default_init_p(p):
    return min(0.5, K0 / float(p))


class Run:
    def __init__(self, wrapper, checkpoints, alpha=ALPHA):
        self.w, self.p, self.alpha = wrapper, wrapper.p, alpha
        self.cp = sorted(int(c) for c in checkpoints)
        self.next = 0
        self.best_f = np.inf; self.best_idx = None; self.best_err = None
        self.lex = (np.inf, 10 ** 9); self.lex_idx = None; self.lex_err = None
        self.snap = {}
        self._front = None
        self._rep_rng = np.random.default_rng(20261008)      # repair randomness (deterministic given the call sequence)

    def done(self):
        return self.w.calls >= self.cp[-1]

    def set_front(self, idx_list, errs):
        self._front = (idx_list, np.asarray(errs, dtype=np.float64))

    def eval(self, mask):
        k = int(mask.sum())
        if k > KMAX:                                   # repair in place so the population keeps the repaired individual
            on = np.flatnonzero(mask)
            mask[self._rep_rng.choice(on, k - KMAX, replace=False)] = False
            k = KMAX
        err = self.w.evaluate(mask)
        f = self.alpha * err + (1.0 - self.alpha) * k / self.p
        if f < self.best_f:
            self.best_f, self.best_err, self.best_idx = f, err, np.flatnonzero(mask).astype(np.int32)
        if (err, k) < self.lex:
            self.lex, self.lex_err, self.lex_idx = (err, k), err, np.flatnonzero(mask).astype(np.int32)
        while self.next < len(self.cp) and self.w.calls >= self.cp[self.next]:
            self.snap[self.cp[self.next]] = dict(
                wsum_idx=self.best_idx, wsum_err=self.best_err,
                lex_idx=self.lex_idx, lex_err=self.lex_err, front=self._front)
            self.next += 1
        return err, k, f


def _init_mask(p, init_p, rng):
    m = rng.random(p) < init_p
    if not m.any():
        m[rng.integers(p)] = True
    return m


def _flip(c, rng):
    p = len(c)
    nf = int(rng.poisson(1.0))
    if nf:
        pos = rng.integers(0, p, nf)
        c[pos] = ~c[pos]
    if not c.any():
        c[rng.integers(p)] = True
    return c


# ------------------------------------------------------------------ GA
def ga(run, rng, init_p, N=50, pc=0.9, ts=3):
    p = run.p
    P = np.zeros((N, p), bool); F = np.empty(N); K = np.empty(N, np.int64)
    for i in range(N):
        P[i] = _init_mask(p, init_p, rng)
    for i in range(N):
        if run.done():
            return
        _, K[i], F[i] = run.eval(P[i])
    while not run.done():
        kids = np.zeros((N, p), bool); kF = np.empty(N); kK = np.empty(N, np.int64); n_ok = 0
        for i in range(N):
            a = rng.integers(0, N, ts); a = a[np.argmin(F[a])]
            b = rng.integers(0, N, ts); b = b[np.argmin(F[b])]
            c = np.where(rng.random(p) < 0.5, P[a], P[b]) if rng.random() < pc else P[a].copy()
            c = _flip(c, rng)
            if run.done():
                return
            _, kK[i], kF[i] = run.eval(c); kids[i] = c; n_ok += 1
        allP = np.vstack([P, kids]); allF = np.concatenate([F, kF]); allK = np.concatenate([K, kK])
        o = np.lexsort((allK, allF))[:N]
        P, F, K = allP[o], allF[o], allK[o]


# ------------------------------------------------------------------ BPSO (sparsity anchored)
def bpso(run, rng, init_p, N=30, w=0.7, c1=1.5, c2=1.5, vmax=6.0):
    """Velocity is the deviation from the logit of the initial density, so that without attraction the swarm keeps the
    sparse prior instead of drifting to 50 % density (the classic failure of plain BPSO when p >> n)."""
    p = run.p
    v0 = float(np.log(init_p / (1.0 - init_p))) if init_p < 0.5 else 0.0
    X = np.zeros((N, p), bool); V = np.zeros((N, p), np.float32)
    pb = np.zeros((N, p), bool); pF = np.full(N, np.inf)
    gb = None; gF = np.inf
    for i in range(N):
        X[i] = _init_mask(p, init_p, rng)
    for i in range(N):
        if run.done():
            return
        _, _, f = run.eval(X[i]); pb[i] = X[i]; pF[i] = f
        if f < gF:
            gF, gb = f, X[i].copy()
    while not run.done():
        r1 = rng.random((N, p), dtype=np.float32); r2 = rng.random((N, p), dtype=np.float32)
        V = w * V + c1 * r1 * (pb.astype(np.float32) - X.astype(np.float32)) \
            + c2 * r2 * (gb.astype(np.float32) - X.astype(np.float32))
        np.clip(V, -vmax, vmax, out=V)
        X = rng.random((N, p), dtype=np.float32) < (1.0 / (1.0 + np.exp(-(V + v0))))
        for i in range(N):
            if not X[i].any():
                X[i, rng.integers(p)] = True
            if run.done():
                return
            _, _, f = run.eval(X[i])
            if f < pF[i]:
                pF[i] = f; pb[i] = X[i]
                if f < gF:
                    gF, gb = f, X[i].copy()


# ------------------------------------------------------------------ binary DE (DE/rand/1/bin with XOR difference vectors)
def bde(run, rng, init_p, N=50, Fm=0.5, CR=0.5):
    """Mutant = x_r1 with the bits flipped (each with prob. F) where x_r2 and x_r3 differ; binomial crossover with the target.
    The number of changed bits scales with the subset size, not with p, so sparsity is preserved.  (A first version used
    continuous vectors thresholded at 0.5; additive mutation made the subsets dense and ~10x slower - see docs/DEVIATIONS.md.)"""
    p = run.p
    P = np.zeros((N, p), bool); F = np.empty(N)
    for i in range(N):
        P[i] = _init_mask(p, init_p, rng)
    for i in range(N):
        if run.done():
            return
        _, _, F[i] = run.eval(P[i])
    while not run.done():
        for i in range(N):
            r = rng.choice(N - 1, 3, replace=False); r = r + (r >= i)
            diff = np.flatnonzero(P[r[1]] ^ P[r[2]])
            v = P[r[0]].copy()
            if len(diff):
                flip = diff[rng.random(len(diff)) < Fm]
                v[flip] = ~v[flip]
            cross = rng.random(p) < CR; cross[rng.integers(p)] = True
            trial = np.where(cross, v, P[i])
            if not trial.any():
                trial[rng.integers(p)] = True
            if run.done():
                return
            _, _, f = run.eval(trial)
            if f <= F[i]:
                P[i] = trial; F[i] = f


# ------------------------------------------------------------------ NSGA-II  (err, k)
def fast_nds(Fm):
    n = len(Fm)
    le = (Fm[:, None, :] <= Fm[None, :, :]).all(2)
    lt = (Fm[:, None, :] < Fm[None, :, :]).any(2)
    dom = le & lt
    cnt = dom.sum(0).astype(np.int64)
    fronts = []; rem = np.ones(n, bool)
    while rem.any():
        cur = np.flatnonzero(rem & (cnt == 0))
        fronts.append(cur); rem[cur] = False
        cnt -= dom[cur].sum(0)
    return fronts


def crowding(Fm, front):
    d = np.zeros(len(front)); Ff = Fm[front]
    for m in range(Fm.shape[1]):
        o = np.argsort(Ff[:, m], kind="stable"); d[o[0]] = d[o[-1]] = np.inf
        r = Ff[o[-1], m] - Ff[o[0], m]
        if r > 0 and len(front) > 2:
            d[o[1:-1]] += (Ff[o[2:], m] - Ff[o[:-2], m]) / r
    return d


def _publish_front(run, P, Fm):
    f0 = fast_nds(Fm)[0]
    run.set_front([np.flatnonzero(P[i]).astype(np.int32) for i in f0], Fm[f0, 0])


def nsga2(run, rng, init_p, N=50, pc=0.9):
    p = run.p
    P = np.zeros((N, p), bool); Fm = np.empty((N, 2))
    for i in range(N):
        P[i] = _init_mask(p, init_p, rng)
    for i in range(N):
        if run.done():
            return
        err, k, _ = run.eval(P[i]); Fm[i] = (err, k)
    _publish_front(run, P, Fm)
    while not run.done():
        rank = np.zeros(N, np.int64); cd = np.zeros(N)
        for r, fr in enumerate(fast_nds(Fm)):
            rank[fr] = r; cd[fr] = crowding(Fm, fr)
        def pick():
            a, b = rng.integers(0, N, 2)
            return a if (rank[a], -cd[a]) <= (rank[b], -cd[b]) else b
        kids = np.zeros((N, p), bool); kF = np.empty((N, 2))
        for i in range(N):
            a, b = pick(), pick()
            c = np.where(rng.random(p) < 0.5, P[a], P[b]) if rng.random() < pc else P[a].copy()
            c = _flip(c, rng)
            if run.done():
                return
            err, k, _ = run.eval(c); kids[i] = c; kF[i] = (err, k)
        allP = np.vstack([P, kids]); allF = np.vstack([Fm, kF]); keep = []
        for fr in fast_nds(allF):
            if len(keep) + len(fr) <= N:
                keep += list(fr)
            else:
                cdv = crowding(allF, fr)
                keep += list(fr[np.argsort(-cdv, kind="stable")[: N - len(keep)]])
                break
        P, Fm = allP[keep], allF[keep]
        _publish_front(run, P, Fm)


# ------------------------------------------------------------------ budget-matched random search (null model)
def random_search(run, rng, init_p, kmax=2 * K0):
    p = run.p
    kmax = min(kmax, p)
    while not run.done():
        k = int(rng.integers(1, kmax + 1))
        m = np.zeros(p, bool); m[rng.choice(p, k, replace=False)] = True
        run.eval(m)


ALGOS = {"ga": ga, "bpso": bpso, "bde": bde, "nsga2": nsga2, "rs": random_search}


def run_algo(name, run, rng, init_p=None):
    ALGOS[name](run, rng, default_init_p(run.p) if init_p is None else init_p)
    return run
