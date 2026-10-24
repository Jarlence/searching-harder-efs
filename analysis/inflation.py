"""Q1-Q4 of docs/PROTOCOL.md: how much do flawed protocols over-state performance, does it grow with search budget and p/n,
and what do protocols report on permuted labels.  Writes results/tables/*.csv and results/tables/inflation_summary.json."""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load as LD  # noqa: E402
import statsx as SX  # noqa: E402

TAB = LD.TAB
CK = ["algo", "learner", "cp"]
NB_MAIN, NB_SUB = 10000, 2000
MIN_CP = int(os.environ.get("EFS_MIN_CP", 3))               # min. number of budget points for a per-cell Spearman (2 only when debugging on the fixture)


def cells_of(df, filt=None):
    c = sorted({tuple(r) for r in df[CK].drop_duplicates().itertuples(index=False, name=None)})
    return [x for x in c if (filt is None or filt(x))]


def mat(df, value, rowkeys, cells):
    if len(df) == 0:
        return np.full((1, len(cells)), np.nan)
    pv = df.pivot_table(index=list(rowkeys), columns=CK, values=value, aggfunc="mean")
    pv = pv.reindex(columns=pd.MultiIndex.from_tuples(cells, names=CK))
    return pv.to_numpy(dtype=float)


def prep(t):
    """D (pick the Pareto point on the test set).  E1h stored the incumbent of ALL evaluated candidates but the population front of the previous
    generation, so D could be negative (110 of 4,190 records).  D is now computed from the Pareto ARCHIVE of the replay X8p (docs/DEVIATIONS.md #7):
    gap_D = merged archive (one representative per objective vector; primary), gap_D_ties = all masks tied at a non-dominated vector may be picked,
    gap_D_old = the superseded E1h definition (kept for the record only)."""
    H = t["honest"]; F = t["full"]
    Hm = H[(H.suite == "E1h")].copy()
    Hm["gap_B"] = (1.0 - Hm.inner_err) - Hm.test_ba
    nsg = Hm.algo == "nsga2"
    Hm["gap_D"] = np.nan; Hm["gap_D_ties"] = np.nan; Hm["gap_D_old"] = np.nan
    Hm.loc[nsg, "gap_D_old"] = [np.max(a) - b for a, b in zip(Hm.loc[nsg, "front_test_ba"], Hm.loc[nsg, "test_ba"])]
    P = LD.load_pareto()
    if len(P):
        key = ["ds", "learner", "rep", "fold", "cp"]
        m = Hm.loc[nsg, key + ["test_ba", "inner_err", "k"]].merge(P, on=key, how="left", suffixes=("", "_x8"))
        if m.d_merged.isna().any():
            raise RuntimeError("X8p replay is incomplete: %d NSGA-II records without an archive" % int(m.d_merged.isna().sum()))
        if not (np.array_equal(m.test_ba.to_numpy(), m.test_ba_x8.to_numpy()) and np.array_equal(m.inner_err.to_numpy(), m.inner_err_x8.to_numpy())):
            raise RuntimeError("X8p incumbents differ from E1h: the replay is not bit-identical")
        Hm.loc[nsg, "gap_D"] = m.d_merged.to_numpy(); Hm.loc[nsg, "gap_D_ties"] = m.d_ties.to_numpy()
    elif os.environ.get("EFS_JOBS"):                                   # debugging on a fixture only
        Hm.loc[nsg, "gap_D"] = Hm.loc[nsg, "gap_D_old"]; Hm.loc[nsg, "gap_D_ties"] = Hm.loc[nsg, "gap_D_old"]
    else:
        raise RuntimeError("results/jobs_extra/X8p is missing: run  python extras/run_extra3.py --suites X8p")
    Fm = F[F.suite == "E1f"].copy()
    Fm["fit_ba"] = 1.0 - Fm.reported_err
    return Hm, Fm


def sources_for(flaw, Hm, Fm, ds, filt):
    h = Hm[Hm.ds == ds]; f = Fm[Fm.ds == ds]
    if flaw == "B":
        cells = cells_of(h, filt); return [(mat(h, "gap_B", ("rep", "fold"), cells), 1.0)]
    if flaw in ("D", "Dties", "Dold"):
        col = {"D": "gap_D", "Dties": "gap_D_ties", "Dold": "gap_D_old"}[flaw]
        hn = h[h.algo == "nsga2"]; cells = cells_of(hn, filt)
        return [(mat(hn, col, ("rep", "fold"), cells), 1.0)] if cells else []
    if flaw in ("A", "Afit"):
        cells = [c for c in cells_of(h, filt) if len(f[(f.algo == c[0]) & (f.learner == c[1]) & (f.cp == c[2])])]
        if not cells:
            return []
        col = "leaky_cv_ba" if flaw == "A" else "fit_ba"
        return [(mat(f, col, ("seed",), cells), 1.0), (mat(h, "test_ba", ("rep", "fold"), cells), -1.0)]
    raise ValueError(flaw)


def summarize(name, flaw, per_ds_vals, boot, extra=None):
    v = np.array([per_ds_vals[k] for k in per_ds_vals])
    p = SX.wilcoxon(v)
    row = dict(breakdown=name, flaw=flaw, n_ds=len(v), mean=boot["point"], lo=boot["lo"], hi=boot["hi"], n_pos=int((v > 0).sum()),
               delta=SX.sign_delta(v), p_wilcoxon=p, lo_cb=boot.get("lo_cb"), hi_cb=boot.get("hi_cb"), lo_db=boot.get("lo_db"), hi_db=boot.get("hi_db"), lo_t=boot.get("lo_t"), hi_t=boot.get("hi_t"))
    if extra:
        row.update(extra)
    return row


def honest_budget_effect(Hm, core, n_boot):
    """Exploratory (not pre-registered): does a larger search budget change the HONEST outer-test BA, the fitness and the subset size?
    Paired per outer run (rep, fold) and cell (algo, learner); data-set level mean; double bootstrap over data sets."""
    rows = []
    for (lo, hi) in ((250, 1000), (1000, 4000), (250, 4000), (4000, 16000)):
        for tag, filt in (("all", None), ("ridge", lambda c: c[1] == "ridge"), ("knn", lambda c: c[1] == "knn")) + tuple(
                ("algo=" + a, (lambda c, a=a: c[0] == a)) for a in LD.ALGOS):
            for outcome in ("test_ba", "fit", "k", "gap_B"):
                per = {}
                for ds in core:
                    h = Hm[(Hm.ds == ds) & Hm.cp.isin([lo, hi])]
                    if len(h) == 0:
                        continue
                    h = h.assign(fit=1.0 - h.inner_err)
                    cells = sorted({(a, l) for a, l in h[["algo", "learner"]].drop_duplicates().itertuples(index=False, name=None)
                                    if (filt is None or filt((a, l, 0)))})
                    cols = []
                    for a, l in cells:
                        g = h[(h.algo == a) & (h.learner == l)]
                        pv = g.pivot_table(index=["rep", "fold"], columns="cp", values=outcome, aggfunc="mean")
                        if lo in pv.columns and hi in pv.columns:
                            cols.append((pv[hi] - pv[lo]).rename((a, l)))
                    if cols:
                        M = pd.concat(cols, axis=1)
                        per[ds] = [(M.to_numpy(dtype=float), 1.0)]
                if len(per) < 3:
                    continue
                bt = SX.double_boot(per, n_boot, seed=20261012)
                if bt is None:
                    continue
                v = np.array(list(bt["ds_values"].values()))
                rows.append(dict(cp_lo=lo, cp_hi=hi, breakdown=tag, outcome=outcome, n_ds=len(v), mean_diff=bt["point"], lo=bt["lo"], hi=bt["hi"],
                                 n_pos=int((v > 0).sum()), p_wilcoxon=SX.wilcoxon(v)))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(TAB, "Q2b_honest_vs_budget.csv"), index=False, encoding="utf-8-sig")
    return r


def gap_E_table(t):
    """E4s: exact E[max of S seeds] - mean, per (ds, algo, learner, cp, fold)."""
    H = t["honest"]; E = H[H.suite == "E4s"]
    rows = []
    for (ds, algo, lr, cp, fold), g in E.groupby(["ds", "algo", "learner", "cp", "fold"]):
        x = g.test_ba.to_numpy()
        for s in (3, 5, 10, 20):
            if len(x) >= s:
                rows.append(dict(ds=ds, algo=algo, learner=lr, cp=cp, fold=fold, S=s, n_seeds=len(x), gap=SX.exact_expected_max(x, s) - x.mean(),
                                 sd=x.std(ddof=1)))
    return pd.DataFrame(rows)


def run(n_main=NB_MAIN, n_sub=NB_SUB):
    t = LD.load_all(use_cache=False)
    Hm, Fm = prep(t)
    core = [d for d in LD.CORE if d in set(Hm.ds)]
    out = {"n_honest_rows": int(len(Hm)), "n_full_rows": int(len(Fm)), "datasets": core}
    # ---------------------------------------------------------------- H1 main (all cells), then breakdowns
    rows = []; ds_tab = []
    flaws = ["B", "A", "D", "Afit", "Dties", "Dold"]            # Dties / Dold: sensitivity of D to the tie rule / the superseded definition
    breakdowns = [("all", None)]
    breakdowns += [("cp=%d" % cp, (lambda c, cp=cp: c[2] == cp)) for cp in (250, 1000, 4000, 16000)]
    breakdowns += [("learner=%s" % lr, (lambda c, lr=lr: c[1] == lr)) for lr in ("ridge", "knn")]
    breakdowns += [("algo=%s" % a, (lambda c, a=a: c[0] == a)) for a in LD.ALGOS]
    for bname, filt in breakdowns:
        nb = n_main if bname == "all" else n_sub
        for flaw in flaws:
            per = {ds: sources_for(flaw, Hm, Fm, ds, filt) for ds in core}
            per = {k: v for k, v in per.items() if v}
            if not per:
                continue
            bt = SX.double_boot(per, nb, seed=20261008)
            if bt is None:
                continue
            rows.append(summarize(bname, flaw, bt["ds_values"], bt))
            for ds, v in bt["ds_values"].items():
                ds_tab.append(dict(breakdown=bname, flaw=flaw, ds=ds, gap=v))
    h1 = pd.DataFrame(rows)
    # gap_E from E4s (cells = (algo, learner, cp); rows = folds)
    GE = gap_E_table(t)
    if len(GE):
        for S in (3, 5, 10, 20):
            g = GE[GE.S == S]
            per = {}
            for ds in core:
                gg = g[g.ds == ds]
                if len(gg) == 0:
                    continue
                cells = cells_of(gg)
                per[ds] = [(mat(gg, "gap", ("fold",), cells), 1.0)]
            bt = SX.double_boot(per, n_main, seed=20261009 + S) if per else None
            if bt is not None:
                rows_e = summarize("all", "E%d" % S, bt["ds_values"], bt)
                h1 = pd.concat([h1, pd.DataFrame([rows_e])], ignore_index=True)
                for ds, v in bt["ds_values"].items():
                    ds_tab.append(dict(breakdown="all", flaw="E%d" % S, ds=ds, gap=v))
    # exploratory flaw G: after seeing the outer-test results, report the best of the five search algorithms (per outer run, learner, budget)
    perG = {}
    for ds in core:
        h = Hm[Hm.ds == ds]
        pv = h.pivot_table(index=["rep", "fold", "learner", "cp"], columns="algo", values="test_ba").dropna()
        if len(pv) == 0:
            continue
        d = (pv.max(axis=1) - pv.mean(axis=1)).rename("g").reset_index()
        perG[ds] = [(d.pivot_table(index=["rep", "fold"], columns=["learner", "cp"], values="g").to_numpy(dtype=float), 1.0)]
    btG = SX.double_boot(perG, n_main, seed=20261014) if perG else None
    if btG is not None:
        h1 = pd.concat([h1, pd.DataFrame([summarize("all", "G", btG["ds_values"], btG)])], ignore_index=True)
        for ds, v in btG["ds_values"].items():
            ds_tab.append(dict(breakdown="all", flaw="G", ds=ds, gap=v))
    # exploratory flaw G2: report, for every data set, the algorithm with the best MEAN test BA (the usual "best method per data set" table)
    perG2 = {}
    for ds in core:
        h = Hm[Hm.ds == ds]
        arrs = []
        for (lr, cp), g in h.groupby(["learner", "cp"]):
            pv = g.pivot_table(index=["rep", "fold"], columns="algo", values="test_ba").dropna()
            if pv.shape[1] == 5 and len(pv) >= 5:
                arrs.append((pv.index, pv[LD.ALGOS].to_numpy()))
        if not arrs:
            continue
        common = sorted(set.intersection(*[set(i) for i, _ in arrs]))
        if len(common) < 5:
            continue
        perG2[ds] = np.stack([a[[list(i).index(c) for c in common]] for i, a in arrs], axis=-1)           # [runs, algos, cells]
    btG2 = SX.double_boot_best(perG2, n_main, seed=20261015) if perG2 else None
    if btG2 is not None:
        h1 = pd.concat([h1, pd.DataFrame([summarize("all", "G2", btG2["ds_values"], btG2)])], ignore_index=True)
        for ds, v in btG2["ds_values"].items():
            ds_tab.append(dict(breakdown="all", flaw="G2", ds=ds, gap=v))
    # Holm over the pre-registered family {B, A, D, E5} for the headline breakdown
    fam = h1[(h1.breakdown == "all") & (h1.flaw.isin(["B", "A", "D", "E5"]))].copy()
    if len(fam):
        fam["p_holm"] = SX.holm(fam.p_wilcoxon.to_numpy())
        h1 = h1.merge(fam[["breakdown", "flaw", "p_holm"]], on=["breakdown", "flaw"], how="left")
    h1.to_csv(os.path.join(TAB, "H1_inflation.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(ds_tab).to_csv(os.path.join(TAB, "H1_by_dataset.csv"), index=False, encoding="utf-8-sig")
    GE.to_csv(os.path.join(TAB, "E4_gapE_fold_level.csv"), index=False, encoding="utf-8-sig")
    out["H1"] = h1.to_dict(orient="records")
    # ---------------------------------------------------------------- H2: Spearman(gap, log2 budget) per (ds, algo, learner)
    sp = []
    for flaw in ("B", "A", "D", "Afit", "Dties", "Dold"):
        for ds in core:
            h = Hm[Hm.ds == ds]
            for (algo, lr), g in h.groupby(["algo", "learner"]):
                if flaw in ("D", "Dties", "Dold") and algo != "nsga2":
                    continue
                if flaw in ("B", "D", "Dties", "Dold"):
                    col = {"B": "gap_B", "D": "gap_D", "Dties": "gap_D_ties", "Dold": "gap_D_old"}[flaw]
                    cm = g.groupby("cp")[col].mean()
                else:
                    f = Fm[(Fm.ds == ds) & (Fm.algo == algo) & (Fm.learner == lr)]
                    if len(f) == 0:
                        continue
                    a = f.groupby("cp")["leaky_cv_ba" if flaw == "A" else "fit_ba"].mean(); b = g.groupby("cp").test_ba.mean()
                    cm = (a - b).dropna()
                if len(cm) < MIN_CP:
                    continue
                sp.append(dict(flaw=flaw, ds=ds, algo=algo, learner=lr, n_cp=len(cm), rho=SX.spearman_cell(cm.index.to_numpy(), cm.to_numpy()),
                               gap_first=float(cm.iloc[0]), gap_last=float(cm.iloc[-1])))
    spd = pd.DataFrame(sp, columns=["flaw", "ds", "algo", "learner", "n_cp", "rho", "gap_first", "gap_last"])
    spd.to_csv(os.path.join(TAB, "H2_spearman_cells.csv"), index=False, encoding="utf-8-sig")
    h2 = []
    for flaw, g in spd.groupby("flaw"):
        r = g.rho.dropna(); n = len(r); k = int((r > 0).sum()); k1 = int((r >= 0.999).sum()); kneg = int((r < 0).sum())
        lo, hi = SX.clopper_pearson(k, n)
        # cluster version (docs/DEVIATIONS.md #8): the cells of one data set are not independent, so the share is formed within each data set first and the
        # data sets are the units: mean share over data sets with a bootstrap interval over data sets, and the number of data sets with a share >= 75 %
        per_ds = g.dropna(subset=["rho"]).groupby("ds").rho.apply(lambda x: float((x > 0).mean()))
        c_lo, c_hi = SX.cluster_boot(per_ds.to_numpy(), 10000, 20261017)
        h2.append(dict(flaw=flaw, n_cells=n, n_rho_pos=k, share_pos=k / n if n else np.nan, cp_lo=lo, cp_hi=hi, n_rho_eq1=k1, n_rho_neg=kneg,
                       mean_rho=float(r.mean()), median_rho=float(r.median()),
                       mean_gap_first=float(g.gap_first.mean()), mean_gap_last=float(g.gap_last.mean()),
                       n_ds=int(len(per_ds)), share_ds_mean=float(per_ds.mean()), share_ds_lo=c_lo, share_ds_hi=c_hi, n_ds_ge75=int((per_ds >= 0.75).sum()),
                       n_ds_all_pos=int((per_ds >= 0.999).sum())))
    pd.DataFrame(h2).to_csv(os.path.join(TAB, "H2_summary.csv"), index=False, encoding="utf-8-sig")
    out["H2"] = h2
    out["Q2b"] = honest_budget_effect(Hm, core, n_sub).to_dict(orient="records")
    # ---------------------------------------------------------------- H3: dataset-level gap vs log(p/n)
    cards = LD.cards(); h3 = []
    dsv = pd.DataFrame(ds_tab)
    xfun = {"log_p_over_n": lambda c: np.log(c["p"] / c["n"]),                       # pre-registered (H3)
            "log_n": lambda c: np.log(c["n"]),                                            # exploratory
            "log_n_minority": lambda c: np.log(c["n_minority"])}                          # exploratory
    for flaw in ("B", "A"):
        for lr in ("all", "ridge", "knn"):
            bn = "all" if lr == "all" else "learner=%s" % lr
            g = dsv[(dsv.breakdown == bn) & (dsv.flaw == flaw) & (dsv.ds.isin(LD.CORE))]
            if len(g) < 5:
                continue
            for xn, xf in xfun.items():
                x = np.array([xf(cards[d]) for d in g.ds]); y = g.gap.to_numpy()
                ts = stats.theilslopes(y, x); kt = stats.kendalltau(x, y)
                h3.append(dict(flaw=flaw, learner=lr, x=xn, n_ds=len(g), slope=float(ts[0]), intercept=float(ts[1]), slope_lo=float(ts[2]), slope_hi=float(ts[3]),
                               kendall_tau=float(kt.correlation), p_tau=float(kt.pvalue)))
    pd.DataFrame(h3).to_csv(os.path.join(TAB, "H3_pn.csv"), index=False, encoding="utf-8-sig")
    out["H3"] = h3
    # ---------------------------------------------------------------- H4: permuted labels
    Hp = t["honest"][t["honest"].suite == "Ep_h"].copy(); Fp = t["full"][t["full"].suite == "Ep_f"].copy()
    h4 = []
    if len(Hp):
        Hp["a_honest"] = Hp.test_ba - 0.5; Hp["a_fit"] = (1 - Hp.inner_err) - 0.5
        Fp["a_leaky"] = Fp.leaky_cv_ba - 0.5; Fp["a_fitfull"] = (1 - Fp.reported_err) - 0.5
        for tag, df, col in (("honest_test_BA", Hp, "a_honest"), ("fitness_as_report(B)", Hp, "a_fit"), ("select_on_all+reCV(A)", Fp, "a_leaky"),
                             ("select_on_all+fitness(Afit)", Fp, "a_fitfull")):
            if len(df) == 0:
                continue
            per = {}
            for ds, g in df.groupby("ds"):
                cells = cells_of(g)
                rk = ("rep", "fold") if tag.startswith(("honest", "fitness")) else ("seed",)
                per[ds] = [(mat(g, col, rk, cells), 1.0)]
            bt = SX.double_boot(per, n_main, seed=20261010)
            if bt is None:
                continue
            v = np.array(list(bt["ds_values"].values()))
            p_two = SX.wilcoxon(v); p_gt = SX.wilcoxon(v, "greater")
            h4.append(dict(report=tag, n_ds=len(v), mean_minus_half=bt["point"], lo=bt["lo"], hi=bt["hi"], n_pos=int((v > 0).sum()),
                           p_two_sided=p_two, p_greater=p_gt, max_abs_ds=float(np.abs(v).max())))
        pd.DataFrame(h4).to_csv(os.path.join(TAB, "H4_permutation.csv"), index=False, encoding="utf-8-sig")
    out["H4"] = h4
    json.dump(out, open(os.path.join(TAB, "inflation_summary.json"), "w"), ensure_ascii=False, indent=1, default=float)
    return out, h1, spd


if __name__ == "__main__":
    nm = int(sys.argv[1]) if len(sys.argv) > 1 else NB_MAIN
    ns = int(sys.argv[2]) if len(sys.argv) > 2 else NB_SUB
    out, h1, spd = run(nm, ns)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(h1[h1.breakdown == "all"].round(4).to_string())
    print(pd.DataFrame(out["H2"]).round(3).to_string())
    print(pd.DataFrame(out["H3"]).round(3).to_string())
    print(pd.DataFrame(out["H4"]).round(4).to_string())
