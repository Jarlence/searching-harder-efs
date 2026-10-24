"""Small statistics helpers (scipy / numpy only; statsmodels is not installed in the frozen env)."""
import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Exact p-value calculation does not work if there are zeros")


def holm(p):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(p, float); n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (n - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def sign_delta(d):
    """Cliff's delta of paired differences against zero: (n_pos - n_neg) / n."""
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    return float(((d > 0).sum() - (d < 0).sum()) / len(d)) if len(d) else float("nan")


def wilcoxon(d, alternative="two-sided"):
    """Wilcoxon signed-rank of differences d against 0; p = 1 if all differences are zero."""
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    if len(d) == 0 or np.all(d == 0):
        return 1.0
    try:
        return float(stats.wilcoxon(d, alternative=alternative, zero_method="wilcox", mode="auto").pvalue)
    except ValueError:
        return 1.0


def _cell_stat(M, idx):
    """M: [runs x cells]; idx: [B x runs] resample indices -> [B x cells] nan-means."""
    R = M[idx]                                        # B x runs x cells
    with np.errstate(all="ignore"):
        return np.nanmean(R, axis=1)


def ds_value(srcs):
    """Point estimate for one data set: sum_s sign_s * nanmean_rows(M_s) -> per cell -> equal-weight mean over cells."""
    v = 0.0
    for M, sign in srcs:
        with np.errstate(all="ignore"):
            v = v + sign * np.nanmean(M, axis=0)
    v = np.asarray(v)
    return float(np.nanmean(v)) if np.any(~np.isnan(v)) else float("nan")


def ds_boot(srcs, n_boot, rng, chunk=500):
    """Run-level bootstrap of one data set's value: returns array [n_boot]. Sources are resampled independently."""
    out = np.empty(n_boot)
    for a in range(0, n_boot, chunk):
        b = min(chunk, n_boot - a)
        v = 0.0
        for M, sign in srcs:
            idx = rng.integers(0, M.shape[0], size=(b, M.shape[0]))
            v = v + sign * _cell_stat(M, idx)
        with np.errstate(all="ignore"):
            out[a:a + b] = np.nanmean(v, axis=1)
    return out


def cluster_boot(values, n_boot=10000, seed=0):
    """Percentile bootstrap of the mean over CLUSTERS: the units are the data-set-level values (one number per data set), resampled with replacement.
    Outer runs of one data set are not independent and are therefore not resampled.  With 9-13 units the percentile interval is somewhat too
    narrow (docs/DEVIATIONS.md #8), so the primary interval is the t interval; this one is a sensitivity check and the interval of bounded shares."""
    v = np.asarray(values, float); v = v[~np.isnan(v)]
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(int(seed) + 7919)
    m = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def t_interval(values, level=0.95):
    """Student-t interval of the mean of the data-set-level values (sensitivity check of the bootstrap interval; n is small)."""
    v = np.asarray(values, float); v = v[~np.isnan(v)]
    n = len(v)
    if n < 2:
        return (float("nan"), float("nan"))
    h = stats.t.ppf(0.5 + level / 2.0, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return (float(v.mean() - h), float(v.mean() + h))


def double_boot(per_ds, n_boot=10000, seed=0):
    """per_ds: {name: [(matrix, sign), ...]}.  Returns dict(point, lo, hi, lo_cb, hi_cb, lo_db, hi_db, ds_values, boot).
    lo / hi: Student-t interval of the mean of the data-set-level values (PRIMARY; with 9-13 data sets the percentile bootstrap is too narrow).
    lo_cb / hi_cb: percentile cluster bootstrap over data sets.  lo_db / hi_db: the two-level bootstrap of the first analysis (data sets, then outer
    runs within each data set).  lo_t / hi_t duplicate lo / hi."""
    rng = np.random.default_rng(seed)
    names = [n for n in per_ds if per_ds[n]]
    vals = {n: ds_value(per_ds[n]) for n in names}
    names = [n for n in names if not np.isnan(vals[n])]
    if not names:
        return None
    V = np.stack([ds_boot(per_ds[n], n_boot, rng) for n in names])           # nd x n_boot
    nd = len(names)
    pick = rng.integers(0, nd, size=(n_boot, nd))
    col = rng.integers(0, n_boot, size=(n_boot, nd))
    boot = V[pick, col].mean(axis=1)
    point = float(np.mean([vals[n] for n in names]))
    dv = [vals[n] for n in names]
    lo_cb, hi_cb = cluster_boot(dv, n_boot, seed)
    lo_t, hi_t = t_interval(dv)
    return dict(point=point, lo=lo_t, hi=hi_t, lo_t=lo_t, hi_t=hi_t, lo_cb=lo_cb, hi_cb=hi_cb,
                lo_db=float(np.nanpercentile(boot, 2.5)), hi_db=float(np.nanpercentile(boot, 97.5)),
                ds_values={n: vals[n] for n in names}, boot=boot)


def double_boot_best(per_ds, n_boot=10000, seed=0, chunk=250):
    """Double bootstrap for the 'report the best of the algorithms' statistic.
    per_ds: {name: array [runs, algos, cells]} (complete: no NaN).  data-set value = mean over cells of
    ( max over algos of the run-mean  -  mean over algos of the run-mean ).  Runs are resampled jointly (rows), data sets afterwards."""
    rng = np.random.default_rng(seed)
    names = list(per_ds)

    def stat(m):                                        # m: [..., algos, cells]
        return (m.max(axis=-2) - m.mean(axis=-2)).mean(axis=-1)

    vals = {n: float(stat(per_ds[n].mean(axis=0))) for n in names}
    V = []
    for n in names:
        A = per_ds[n]; out = np.empty(n_boot)
        for a in range(0, n_boot, chunk):
            b = min(chunk, n_boot - a)
            idx = rng.integers(0, A.shape[0], size=(b, A.shape[0]))
            out[a:a + b] = stat(A[idx].mean(axis=1))
        V.append(out)
    V = np.stack(V); nd = len(names)
    pick = rng.integers(0, nd, size=(n_boot, nd)); col = rng.integers(0, n_boot, size=(n_boot, nd))
    boot = V[pick, col].mean(axis=1)
    dv = list(vals.values())
    lo_cb, hi_cb = cluster_boot(dv, n_boot, seed)
    lo_t, hi_t = t_interval(dv)
    return dict(point=float(np.mean(dv)), lo=lo_t, hi=hi_t, lo_t=lo_t, hi_t=hi_t, lo_cb=lo_cb, hi_cb=hi_cb,
                lo_db=float(np.percentile(boot, 2.5)), hi_db=float(np.percentile(boot, 97.5)), ds_values=vals, boot=boot)


def spearman_cell(budgets, values):
    b = np.log2(np.asarray(budgets, float)); v = np.asarray(values, float)
    ok = ~np.isnan(v)
    if ok.sum() < 3 or np.ptp(v[ok]) == 0:
        return float("nan")
    return float(stats.spearmanr(b[ok], v[ok]).correlation)


def exact_expected_max(x, s):
    """E[max of s draws without replacement from the sample x] (exact order-statistic formula)."""
    x = np.sort(np.asarray(x, float)); n = len(x)
    if s > n:
        return float("nan")
    from math import comb
    w = np.array([comb(i, s - 1) for i in range(n)], float)       # i = 0-based rank -> C(i, s-1)
    return float((x * w).sum() / comb(n, s))
