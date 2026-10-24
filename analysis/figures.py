"""Figures (PNG, 200 dpi).  English labels on purpose (fonts); the SUMMARY text is Chinese."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load as LD  # noqa: E402

PAPER = os.environ.get("EFS_PAPER") == "1"          # EFS_PAPER=1 python analysis/figures.py  -> vector PDFs sized for the manuscript in paper/figures
FIG = os.path.join(LD.ROOT, "paper", "figures") if PAPER else LD.FIG
TAB = LD.TAB
os.makedirs(FIG, exist_ok=True)
def FS(default, paper):
    """figure size: the paper-mode size is chosen so that the PDF is placed at (about) its native size in the manuscript"""
    return paper if PAPER else default


if PAPER:
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
                         "lines.linewidth": 1.4, "lines.markersize": 4})
def _save(fig, name):
    """PNG for quick viewing, vector PDF for the paper."""
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    fig.savefig(os.path.join(FIG, name + ".pdf"))


COL = {"fit": "#d62728", "leaky": "#ff7f0e", "pareto": "#9467bd", "honest": "#1f77b4", "perm": "#7f7f7f"}


def _boot_ci(vals, n=2000, seed=1):
    """95% Student-t interval of the mean of the data-set-level values (name kept from the first version, which used a percentile bootstrap)."""
    from scipy import stats as _st
    v = np.asarray(vals, float); v = v[~np.isnan(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    h = _st.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return (v.mean() - h, v.mean() + h)


def fig_budget(t):
    H = t["honest"]; F = t["full"]
    Hm = H[H.suite == "E1h"]; Fm = F[F.suite == "E1f"]
    if len(Hm) == 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=FS((15, 4.4), (6.4, 2.45)), sharey=True)
    full_cps = {250, 1000, 4000, 16000}
    have = Hm.groupby(["ds", "learner"]).cp.apply(lambda s: set(s) >= full_cps)         # data sets that have every budget (for that learner)
    ok_lr = {lr: {ds for (ds, l), v in have.items() if l == lr and v} for lr in ("ridge", "knn")}
    for ax, (title, sel) in zip(axes, (("both learners", None), ("ridge", "ridge"), ("5-NN", "knn"))):
        keep = (ok_lr["ridge"] & ok_lr["knn"]) if sel is None else ok_lr[sel]       # fixed data-set set across budgets (no composition drift)
        h = Hm[Hm.ds.isin(keep)] if sel is None else Hm[(Hm.learner == sel) & Hm.ds.isin(keep)]
        f = Fm[Fm.ds.isin(keep)] if sel is None else Fm[(Fm.learner == sel) & Fm.ds.isin(keep)]
        series = {}
        h = h.assign(fit=1 - h.inner_err)
        for name, df, col in (("fitness as report (B)", h, "fit"), ("honest nested test BA", h, "test_ba")):
            series[name] = df.groupby(["ds", "cp"])[col].mean().groupby("cp")
        if len(f):
            series["select on all rows, re-CV (A)"] = f.groupby(["ds", "cp"]).leaky_cv_ba.mean().groupby("cp")
        hn = h[h.algo == "nsga2"]
        if len(hn):
            # D from the Pareto-archive replay X8p (docs/DEVIATIONS.md #7): best test BA among the non-dominated candidates evaluated so far
            P_ = LD.load_pareto()
            if len(P_):
                hn = hn.merge(P_[["ds", "learner", "rep", "fold", "cp", "d_merged"]], on=["ds", "learner", "rep", "fold", "cp"], how="left")
                hn = hn.assign(par=hn.test_ba + hn.d_merged)
            else:                                                       # fixture / debugging only
                hn = hn.assign(par=[np.max(a) for a in hn.front_test_ba])
            series["best Pareto point chosen on test (D)"] = hn.groupby(["ds", "cp"]).par.mean().groupby("cp")
        colors = {"fitness as report (B)": COL["fit"], "honest nested test BA": COL["honest"], "select on all rows, re-CV (A)": COL["leaky"],
                  "best Pareto point chosen on test (D)": COL["pareto"]}
        for name, grp in series.items():
            xs, ys, lo, hi = [], [], [], []
            for cp, v in grp:
                xs.append(cp); ys.append(v.mean()); l, h2 = _boot_ci(v.to_numpy()); lo.append(l); hi.append(h2)
            ax.plot(xs, ys, "o-", color=colors[name], label=name, lw=2)
            ax.fill_between(xs, lo, hi, color=colors[name], alpha=0.15)
        ax.set_xscale("log", base=2); ax.set_xticks([250, 1000, 4000, 16000]); ax.set_xticklabels(["250", "1k", "4k", "16k"])
        ax.set_title("%s (%d data sets)" % (title, len(keep))); ax.set_xlabel("search budget (wrapper requests)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("balanced accuracy (mean over data sets)" if not PAPER else "balanced accuracy")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=2, frameon=False, fontsize=(7 if PAPER else 9))
    fig.tight_layout(rect=[0, 0.15, 1, 1]); _save(fig, "fig1_budget"); plt.close(fig)


def fig_by_dataset():
    p = os.path.join(TAB, "H1_by_dataset.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p); d = d[d.breakdown == "all"]
    cards = LD.cards()
    order = sorted([x for x in d.ds.unique()], key=lambda x: np.log(cards[x]["p"] / cards[x]["n"]))
    fig, ax = plt.subplots(figsize=FS((10, 4.8), (6.4, 3.1)))
    colors = {"B": COL["fit"], "A": COL["leaky"], "D": COL["pareto"], "E5": "#2ca02c"}
    names = {"B": "B fitness-as-report", "A": "A select-then-split", "D": "D pick Pareto by test", "E5": "E best of 5 seeds"}
    for j, fl in enumerate(["B", "A", "D", "E5"]):
        g = d[d.flaw == fl].set_index("ds")
        xs = [order.index(x) + (j - 1.5) * 0.18 for x in g.index]
        ax.scatter(xs, g.gap.to_numpy(), color=colors[fl], label=names[fl], s=38, zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(["%s\n(p/n=%.1f)" % (x, cards[x]["p"] / cards[x]["n"]) for x in order], fontsize=7, rotation=0)
    ax.set_ylabel("over-statement of balanced accuracy"); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    fig.tight_layout(); _save(fig, "fig2_gap_by_dataset"); plt.close(fig)


def fig_pn():
    p = os.path.join(TAB, "H1_by_dataset.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p); d = d[(d.breakdown == "all") & (d.flaw.isin(["B", "A"]))]
    cards = LD.cards()
    fig, axes = plt.subplots(1, 2, figsize=FS((10, 4), (6.0, 2.5)))
    for ax, fl in zip(axes, ["B", "A"]):
        g = d[d.flaw == fl]
        x = np.array([np.log(cards[s]["p"] / cards[s]["n"]) for s in g.ds]); y = g.gap.to_numpy()
        ax.scatter(x, y, s=40, color=COL["fit"] if fl == "B" else COL["leaky"])
        for xi, yi, s in zip(x, y, g.ds):
            ax.annotate(s, (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
        if len(x) >= 4:
            from scipy import stats
            ts = stats.theilslopes(y, x); xx = np.linspace(x.min(), x.max(), 20); ax.plot(xx, ts[1] + ts[0] * xx, "k--", lw=1)
        ax.set_xlabel("log(p / n)"); ax.set_ylabel("gap (BA points)"); ax.set_title("flaw %s" % fl); ax.grid(alpha=0.3)
    fig.tight_layout(); _save(fig, "fig3_gap_vs_pn"); plt.close(fig)


def fig_perm(t):
    H = t["honest"]; F = t["full"]
    Hp = H[H.suite == "Ep_h"]; Fp = F[F.suite == "Ep_f"]
    if len(Hp) == 0:
        return
    vals = {"honest\ntest BA": (Hp.assign(v=Hp.test_ba).groupby("ds").v.mean()),
            "fitness\nas report (B)": (Hp.assign(v=1 - Hp.inner_err).groupby("ds").v.mean())}
    if len(Fp):
        vals["select on all\n+ re-CV (A)"] = Fp.groupby("ds").leaky_cv_ba.mean()
        vals["select on all\n+ fitness"] = (1 - Fp.reported_err).groupby(Fp.ds).mean()
    fig, ax = plt.subplots(figsize=FS((7.5, 4.2), (4.4, 2.7)))
    for i, (k, v) in enumerate(vals.items()):
        ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(v)), v.to_numpy(), s=30, alpha=0.8)
        ax.hlines(v.mean(), i - 0.3, i + 0.3, color="k")
    ax.axhline(0.5, color="r", ls="--", lw=1); ax.set_xticks(range(len(vals))); ax.set_xticklabels(list(vals), fontsize=8)
    ax.set_ylabel("reported BA on permuted labels (truth = 0.5)"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); _save(fig, "fig4_permutation"); plt.close(fig)


def fig_efs_vs_rs():
    p = os.path.join(TAB, "Q5_across_datasets.csv")
    if not os.path.exists(p):
        return
    a = pd.read_csv(p)
    if len(a) == 0:
        return
    a = a[a.cp == 4000]
    fig, axes = plt.subplots(1, 2, figsize=FS((12, 5), (6.4, 6.0)), sharex=True)
    for ax, lr in zip(axes, ("ridge", "knn")):
        g = a[a.learner == lr]
        labels, y = [], 0
        for _, r in g.iterrows():
            ax.errorbar(r["mean_diff"], y, xerr=[[r["mean_diff"] - r["lo"]], [r["hi"] - r["mean_diff"]]], fmt="o", color="C0" if r["p_holm"] >= 0.05 else "C3", capsize=2)
            labels.append("%s - %s" % (r["a"], r["b"])); y += 1
        ax.axvline(0, color="k", lw=0.8); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=(7.5 if PAPER else 7))
        ax.set_title("ridge" if lr == "ridge" else "5-NN", fontsize=9)
        ax.set_xlabel("honest BA difference at budget 4000")
        ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); _save(fig, "fig5_efs_vs_baselines"); plt.close(fig)


def fig_long():
    p = os.path.join(TAB, "E7l_means.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    if len(d) == 0:
        return
    dss = sorted(d.ds.unique())
    fig, axes = plt.subplots(2, len(dss), figsize=FS((3.6 * len(dss), 6), (6.4, 3.5)), sharey="row", squeeze=False)
    for j, ds in enumerate(dss):
        for i, lr in enumerate(("ridge", "knn")):
            ax = axes[i, j]
            g = d[(d.ds == ds) & (d.learner == lr)]
            for algo, gg in g.groupby("algo"):
                gg = gg.sort_values("cp")
                ax.plot(gg.cp, gg.fit_ba, "--", label=algo + " fitness"); ax.plot(gg.cp, gg.test_ba, "-o", label=algo + " test")
            ax.set_xscale("log", base=4); ax.set_title("%s / %s" % (ds, lr), fontsize=9); ax.grid(alpha=0.3)
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=(6.5 if PAPER else 8))
    fig.tight_layout(rect=[0, 0.1, 1, 1]); _save(fig, "fig6_long_budget"); plt.close(fig)


def fig_unit():
    p = os.path.join(TAB, "E6u_means.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    d = d[(d.learner == "ridge")]
    if len(d) == 0:
        return
    fig, ax = plt.subplots(figsize=FS((7.5, 4.2), (4.2, 2.6)))
    lab = {(False, False): "record-wise outer / record-wise inner", (False, True): "record-wise outer / subject-wise inner",
           (True, False): "subject-wise outer / record-wise inner", (True, True): "subject-wise outer / subject-wise inner"}
    for (og, ig), g in d.groupby(["outer_group", "inner_group"]):
        gg = g.groupby("cp").mean_ba.mean()
        ax.plot(gg.index, gg.values, "o-", label=lab[(og, ig)])
    ax.set_xscale("log", base=2); ax.set_xlabel("search budget"); ax.set_ylabel("mean test BA (Parkinson, ridge)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); _save(fig, "fig7_unit_leakage"); plt.close(fig)


def fig_advantage():
    p = os.path.join(TAB, "Q5b_advantage_inflation.csv")
    if not os.path.exists(p):
        return
    a = pd.read_csv(p)
    a = a[a.cp == 4000]
    if len(a) == 0:
        return
    order = [(lr, algo) for lr in ("ridge", "knn") for algo in ("ga", "bpso", "bde", "nsga2")]
    a = a.set_index(["learner", "algo"]).loc[order].reset_index()
    assert len(a) == 8, "the advantage plot must show all eight comparisons"
    fig, ax = plt.subplots(figsize=FS((9, 5.5), (6.0, 3.6)))
    names = {"ga": "GA", "bpso": "BPSO", "bde": "BDE", "nsga2": "NSGA-II"}
    labels = []
    for y, r in a.iterrows():
        point, lo, hi = 100.0 * r["infl"], 100.0 * r["infl_lo"], 100.0 * r["infl_hi"]
        color = COL["honest"] if r.learner == "ridge" else COL["pareto"]
        ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o", color=color, capsize=3, zorder=3)
        labels.append("%s / %s" % ("ridge" if r.learner == "ridge" else "5-NN", names[r.algo]))
        ax.text(1.02, y, "%.2f [%.2f, %.2f]" % (point, lo, hi), transform=ax.get_yaxis_transform(), va="center", fontsize=7.5 if PAPER else 10)
    ax.axvline(0, color="k", lw=0.9, ls="--")
    ax.axhline(3.5, color="0.8", lw=0.7)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8 if PAPER else 10)
    ax.set_ylim(len(labels) - 0.5, -0.8); ax.set_xlim(-1.6, 10.5)
    ax.set_xticks([-1, 0, 2, 4, 6, 8, 10]); ax.grid(alpha=0.25, axis="x")
    ax.set_xlabel("Paired inflation of advantage over RS (BA points)")
    ax.text(1.02, -0.62, "Mean [95% CI]", transform=ax.get_yaxis_transform(), va="center", fontsize=8 if PAPER else 10)
    fig.subplots_adjust(left=0.22, right=0.72, bottom=0.17, top=0.97)
    _save(fig, "fig8_advantage_inflation"); plt.close(fig)


def fig_synth():
    p = os.path.join(TAB, "X3u_means.csv")
    if not os.path.exists(p):
        return
    m = pd.read_csv(p)
    m = m[m.cp == m.cp.max()]
    if len(m) == 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=FS((13, 3.9), (6.4, 2.3)), sharey=True)
    lab = {(False, False): "record-wise outer / inner", (False, True): "record-wise outer, subject-wise inner",
           (True, False): "subject-wise outer, record-wise inner", (True, True): "subject-wise outer / inner"}
    for ax, U in zip(axes, sorted(m.U.unique())):
        for (og, ig), g in m[(m.U == U) & (m.algo == "ga")].groupby(["outer_group", "inner_group"]):
            gg = g.groupby("sig_f").test_ba.mean()
            ax.plot(gg.index, gg.values, "o-", label=lab[(og, ig)])
        ax.axhline(0.5, color="gray", ls=":"); ax.set_title("U = %d subjects" % U); ax.set_xlabel(r"fingerprint strength $\sigma_f$"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean test BA (GA)")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=2, frameon=False, fontsize=(6.5 if PAPER else 8))
    fig.tight_layout(rect=[0, 0.16, 1, 1]); _save(fig, "fig9_synthetic_units"); plt.close(fig)


def fig_land():
    p = os.path.join(TAB, "X4r_landscape.csv")
    if not os.path.exists(p):
        return
    t = pd.read_csv(p)
    if len(t) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=FS((10, 4), (6.4, 2.6)))
    for ax, perm in zip(axes, (-1, 0)):
        g = t[t.perm == perm]
        for (ds, lr, k), gg in g.groupby(["ds", "learner", "k"]):
            if k != 20:
                continue
            ax.plot(gg.B, gg.gap_obs, "o-", label="%s/%s obs" % (ds, lr), lw=1.5)
            ax.plot(gg.B, gg.gap_pred_normal, "--", color=ax.lines[-1].get_color(), lw=1)
        ax.set_xscale("log", base=4); ax.set_title("%s labels" % ("permuted" if perm == 0 else "true"), fontsize=9)
        ax.set_xlabel("number of random subsets $B$"); ax.set_ylabel("inner BA $-$ held-out BA of the best"); ax.grid(alpha=0.3)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=(6.5 if PAPER else 8))
    fig.tight_layout(rect=[0, 0.14, 1, 1]); _save(fig, "fig10_random_subset_landscape"); plt.close(fig)


def run():
    t = LD.load_all(use_cache=False)
    for fn, args in ((fig_budget, (t,)), (fig_by_dataset, ()), (fig_pn, ()), (fig_perm, (t,)), (fig_efs_vs_rs, ()), (fig_long, ()), (fig_unit, ()),
                     (fig_advantage, ()), (fig_synth, ()), (fig_land, ())):
        try:
            fn(*args); print("ok", fn.__name__)
        except Exception as e:  # noqa: BLE001
            print("FAILED", fn.__name__, type(e).__name__, e)


if __name__ == "__main__":
    run()
