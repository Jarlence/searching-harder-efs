"""Assemble results/SUMMARY.md from the tables written by inflation.py, compare.py, special.py, extras_analysis.py.
All verdicts follow the rules written down in docs/PROTOCOL.md and docs/ANALYSIS_PLAN.md; nothing is hand-edited."""
import collections
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load as LD  # noqa: E402

TAB = LD.TAB
OUT = os.path.join(LD.OUT, "SUMMARY.md")


def rd(name):
    p = os.path.join(TAB, name)
    if os.path.exists(p):
        try:
            return pd.read_csv(p)
        except Exception:  # noqa: BLE001
            return None
    return None


def f3(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else ("%." + str(nd) + "f") % x


def ci(m, lo, hi, nd=3):
    return "%s [%s, %s]" % (f3(m, nd), f3(lo, nd), f3(hi, nd))


def pv(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "-"
    return "<0.001" if p < 0.001 else "%.3f" % p


def md(rows, headers):
    s = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        s.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(s)


FLAW_NAME = {"B": "B 适应度即报告", "A": "A 先选后分（全数据搜索 + 重估）", "Afit": "A' 先选后分 + 直接报适应度", "D": "D 按测试集挑 Pareto 解（NSGA-II；Pareto 档案，每个目标向量取首个评估的代表）", "Dties": "D' 同上，但同一目标向量下的所有掩码都可被挑（并列全部保留）", "Dold": "D\" 旧定义（E1h：当前最优解 + 上一代的种群前沿；已废弃，仅留档）",
             "E3": "E 3 个种子取最好", "E5": "E 5 个种子取最好", "E10": "E 10 个种子取最好", "E20": "E 20 个种子取最好",
             "G": "G 每个外层运行事后在 5 个算法里挑测试最好的（探索性，极端情形）",
             "G2": "G2 每个数据集事后挑平均测试 BA 最好的算法（探索性，\"各数据集最佳方法\"表）"}


def section_integrity():
    L = ["## 0. 完整性与溯源", ""]
    from collections import Counter
    shas = collections.defaultdict(Counter); total = Counter(); err = Counter()
    for d in sorted(glob.glob(os.path.join(LD.JOBS, "*"))):
        if not os.path.isdir(d):
            continue
        suite = os.path.basename(d)
        for f in glob.glob(os.path.join(d, "*.json")):
            try:
                rec = json.load(open(f))
                shas[suite][rec.get("code_sha")] += 1; total[suite] += 1
            except Exception:  # noqa: BLE001
                pass
        err[suite] = len(glob.glob(os.path.join(d, "errors", "*.json")))
    rows = []
    for s in sorted(total):
        rows.append((s, total[s], err[s], ", ".join("%s×%d" % (k, v) for k, v in shas[s].items())))
    L.append(md(rows, ["套件", "完成作业", "失败（见 errors/）", "code_sha × 作业数"]))
    n_all = sum(total.values()); n_err = sum(err.values())
    L += ["", "- 主实验作业数 %d，失败 %d（%.2f%%；事先设定的阈值 1%%）。" % (n_all, n_err, 100.0 * n_err / max(n_all + n_err, 1))]
    L.append("- 代码冻结哈希：`docs/FREEZE_dev2.json`（主版本，E1h 的前 1 229 个作业）→ `FREEZE_dev3.json`（仅 `experiments/suites.py` 两处改动，见 `docs/DEVIATIONS.md` #3；其余作业）→ `FREEZE_dev4.json`（仅 `src/efsbench/filters.py` 给 L1 基线的 liblinear 求解器固定随机种子，见 #5；`E2b` 的 650 个 `l1` 作业用 dev4 重跑，旧结果保留在 `results/superseded/`）。")
    xdir = os.environ.get("EFS_JOBS_EXTRA", os.path.join(LD.ROOT, "results", "jobs_extra"))
    if os.path.isdir(xdir):
        xs = collections.defaultdict(Counter); xt = Counter(); xe = Counter()
        for d in sorted(glob.glob(os.path.join(xdir, "*"))):
            if os.path.isdir(d):
                suite = os.path.basename(d)
                for f in glob.glob(os.path.join(d, "*.json")):
                    rec = json.load(open(f)); xs[suite][rec.get("code_sha")] += 1; xt[suite] += 1
                xe[suite] = len(glob.glob(os.path.join(d, "errors", "*.json")))
        if xt:
            L += ["", "**额外研究作业**（`results/jobs_extra/`，不属于冻结的主代码；`code_sha` 覆盖各套件驱动指定的实现与依赖；X6r 的指纹含 ex_suites5.py、run_extra5.py 及核心实现）", "",
                  md([(s, xt[s], xe[s], ", ".join("%s×%d" % (k, v) for k, v in xs[s].items())) for s in sorted(xt)], ["套件", "完成作业", "失败", "code_sha × 作业数"])]
    cards = LD.cards()
    rows = [(k, v["n"], v["p"], "%.1f%%" % (100 * v["minority_fraction"]), v["n_groups"] or "-", v["source"], v["processed_sha256"][:12]) for k, v in cards.items()]
    L += ["", "**数据集**（全部公开，md5 已校验；少数类 = 样本较少的类 = 正类）", "", md(rows, ["名称", "n", "p", "少数类占比", "受试者数", "来源", "sha256 前 12 位"])]
    return L


def section_audit():
    p = os.path.join(LD.ROOT, "audit", "audit_summary.md")
    if not os.path.exists(p):
        return []
    L = ["## 1. 文献审计（Europe PMC 开放获取全文）", "", "来源：`audit/audit_summary.md`（AI 按 `audit/CODEBOOK.md` 编码；独立规则编码器与重编码一致性见 `audit/RELIABILITY.md`）。", ""]
    L += [("#" + x if x.startswith("## ") else x) for x in open(p, encoding="utf-8").read().splitlines()[2:]]
    rel = os.path.join(LD.ROOT, "audit", "RELIABILITY.md")
    if os.path.exists(rel):
        L += ["", "**编码可靠性**", ""] + open(rel, encoding="utf-8").read().splitlines()
    return L


def verdicts(h1, h2, h3, h4):
    rows = []
    if h1 is not None:
        for fl in ("B", "A", "D", "E5"):
            g = h1[(h1.breakdown == "all") & (h1.flaw == fl)]
            if len(g):
                r = g.iloc[0]
                ok = (r.lo > 0) and (r.get("p_holm", 1.0) < 0.05)
                rows.append(("H1 · " + FLAW_NAME[fl], "平均高估 > 0（CI 下限 > 0 且 Holm 校正后 p < 0.05）",
                             "%s；%d/%d 个数据集为正；p_Holm=%s" % (ci(r["mean"], r.lo, r.hi), r.n_pos, r.n_ds, pv(r.get("p_holm"))), "**支持**" if ok else "**不支持**"))
    if h2 is not None:
        for _, r in h2.iterrows():
            rows.append(("H2 · " + FLAW_NAME.get(r.flaw, r.flaw), "≥ 75% 的格子 Spearman ρ > 0",
                         "%d/%d = %s（Clopper–Pearson [%s, %s]，把格子当独立）；按数据集聚类：数据集平均占比 %s [%s, %s]，%d/%d 个数据集的占比 ≥ 75%%；ρ=1 的格子 %d" % (
                             r.n_rho_pos, r.n_cells, f3(r.share_pos, 2), f3(r.cp_lo, 2), f3(r.cp_hi, 2), f3(r.share_ds_mean, 2), f3(r.share_ds_lo, 2), f3(r.share_ds_hi, 2),
                             r.n_ds_ge75, r.n_ds, r.n_rho_eq1),
                         "**支持**" if r.share_pos >= 0.75 else "**不支持**"))
    if h3 is not None and len(h3):
        for _, r in h3[(h3.learner == "all") & (h3.x == "log_p_over_n")].iterrows():
            ok = (r.slope_lo > 0) and (r.kendall_tau > 0)
            rows.append(("H3（探索性）· 缺陷 " + r.flaw, "gap 随 log(p/n) 增大（Theil–Sen 斜率区间不含 0）",
                         "斜率 %s；Kendall τ=%s（p=%s）" % (ci(r.slope, r.slope_lo, r.slope_hi), f3(r.kendall_tau, 2), pv(r.p_tau)), "支持" if ok else "不支持"))
    if h4 is not None and len(h4):
        hon = h4[h4.report == "honest_test_BA"]
        if len(hon):
            r = hon.iloc[0]
            ok = abs(r.mean_minus_half) < 0.03
            rows.append(("H4 · 诚实协议（置换标签）", "|BA − 0.5| < 0.03", "BA − 0.5 = %s；各数据集最大 |偏差| %s" % (ci(r.mean_minus_half, r.lo, r.hi, 4), f3(r.max_abs_ds, 3)),
                         "**达标**" if ok else "**未达标**"))
        for _, r in h4[h4.report != "honest_test_BA"].iterrows():
            ok = (r.lo > 0) and (r.p_greater < 0.05)
            rows.append(("H4 · " + r.report, "> 0.5", "BA − 0.5 = %s；p(单侧)=%s" % (ci(r.mean_minus_half, r.lo, r.hi, 3), pv(r.p_greater)), "**支持**" if ok else "不支持"))
    return rows


def section_core():
    h1 = rd("H1_inflation.csv"); h2 = rd("H2_summary.csv"); h3 = rd("H3_pn.csv"); h4 = rd("H4_permutation.csv")
    L = ["## 2. 事先设定假设的判定（内部协议；不是外部预注册）", "", "判定规则写在 `docs/PROTOCOL.md` §4（主实验运行前冻结并哈希）与 `docs/ANALYSIS_PLAN.md`（主实验运行期间、查看任何性能结果之前写成，未被哈希），脚本机械执行；事后的改动见 `docs/DEVIATIONS.md`。", ""]
    hp = rd("E1h_honest_means.csv")
    if hp is not None and len(hp):
        hp = hp.assign(gap=hp.fit_ba - hp.test_ba)
        rows = []
        for lr in ("ridge", "knn"):
            for algo in LD.ALGOS:
                for cp in (250, 1000, 4000, 16000):
                    g = hp[(hp.learner == lr) & (hp.algo == algo) & (hp.cp == cp)]
                    if len(g):
                        rows.append((lr, algo, cp, len(g), f3(g.fit_ba.mean()), f3(g.test_ba.mean()), f3(g.gap.mean()), f3(g.k.mean(), 1)))
        L += ["**诚实评估总表**（E1h；各数据集先对外层运行取平均，再对数据集平均；注意预算 16 000 缺 ozone、speech 以及 5-NN 的 oil_spill、kc3，故数据集数不同）", "",
              md(rows, ["学习器", "算法", "预算", "数据集数", "报告的适应度 BA", "诚实测试 BA", "高估（B）", "平均子集大小 k"]), ""]
    rows = verdicts(h1, h2, h3, h4)
    if rows:
        L += ["**判定表**（H1–H4；Q5 / Q6 见第 3、4 节）", "", md(rows, ["假设", "判据", "结果", "判定"])]
    else:
        L.append("（尚无结果）")
    if h1 is not None:
        L += ["", "### 2.1 Q1：各类做法的平均高估（平衡准确率点数；所有算法 × 学习器 × 预算格子等权；先数据集内平均再跨数据集）", ""]
        rows = []
        for fl in ("B", "A", "Afit", "D", "Dties", "Dold", "E3", "E5", "E10", "E20", "G", "G2"):
            g = h1[(h1.breakdown == "all") & (h1.flaw == fl)]
            if len(g):
                r = g.iloc[0]
                rows.append((FLAW_NAME[fl], r.n_ds, ci(r["mean"], r.lo, r.hi), "%d/%d" % (r.n_pos, r.n_ds), f3(r.delta, 2), pv(r.p_wilcoxon), pv(r.get("p_holm"))))
        L.append(md(rows, ["缺陷", "数据集数", "平均高估 [95% CI：数据集层面的 t 区间]", "正向数据集", "Cliff's δ", "p（Wilcoxon）", "p（Holm）"]))
        rows = []
        for fl in ("B", "A", "D", "E5"):
            g = h1[(h1.breakdown == "all") & (h1.flaw == fl)]
            if len(g):
                r = g.iloc[0]
                rows.append((fl, ci(r["mean"], r.lo, r.hi), ci(r["mean"], r.lo_cb, r.hi_cb), ci(r["mean"], r.lo_db, r.hi_db)))
        L += ["", "**区间的敏感性**：t 区间（主；13 个数据集层面的值）vs 对数据集层面的值做百分位聚类自助 vs 首次分析的两层自助（数据集内再对外层运行重抽样）。"
              "数据集只有 9–13 个时，百分位自助区间偏窄，所以主区间用 t 区间。", "", md(rows, ["缺陷", "t 区间（主）", "聚类自助", "两层自助（旧）"])]
        L += ["", "**按预算、学习器、算法拆分**（自助 2 000 次）", ""]
        rows = []
        for bn in ["cp=250", "cp=1000", "cp=4000", "cp=16000", "learner=ridge", "learner=knn"] + ["algo=%s" % a for a in LD.ALGOS]:
            cells = []
            for fl in ("B", "A", "D"):
                g = h1[(h1.breakdown == bn) & (h1.flaw == fl)]
                cells.append(ci(g.iloc[0]["mean"], g.iloc[0].lo, g.iloc[0].hi) if len(g) else "-")
            rows.append((bn, *cells))
        L.append(md(rows, ["拆分", "B 适应度即报告", "A 先选后分", "D 按测试挑 Pareto"]))
        d = rd("H1_by_dataset.csv")
        if d is not None:
            cards = LD.cards()
            piv = d[d.breakdown == "all"].pivot_table(index="ds", columns="flaw", values="gap")
            order = sorted(piv.index, key=lambda x: np.log(cards[x]["p"] / cards[x]["n"]))
            rows = [(x, cards[x]["n"], cards[x]["p"], f3(cards[x]["p"] / cards[x]["n"], 2), *[f3(piv.loc[x].get(c)) for c in ("B", "A", "D", "E5")]) for x in order]
            L += ["", "**各数据集**（按 p/n 升序）", "", md(rows, ["数据集", "n", "p", "p/n", "B", "A", "D", "E(5 种子)"])]
    h2c = rd("H2_spearman_cells.csv")
    if h2 is not None:
        L += ["", "### 2.2 Q2：高估是否随搜索预算增大（每个 数据集×算法×学习器 格子：gap 与 log2(预算) 的 Spearman ρ）", ""]
        rows = [(FLAW_NAME.get(r.flaw, r.flaw), r.n_cells, "%d (%s)" % (r.n_rho_pos, f3(r.share_pos, 2)), "[%s, %s]" % (f3(r.cp_lo, 2), f3(r.cp_hi, 2)),
                 r.n_rho_eq1, r.n_rho_neg, f3(r.mean_rho, 2), f3(r.mean_gap_first), f3(r.mean_gap_last)) for _, r in h2.iterrows()]
        L.append(md(rows, ["缺陷", "格子数", "ρ>0（比例）", "Clopper–Pearson", "ρ=1", "ρ<0", "平均 ρ", "最小预算处平均 gap", "最大预算处平均 gap"]))
    q2b = rd("Q2b_honest_vs_budget.csv")
    if q2b is not None and len(q2b):
        L += ["", "**（探索性，非事先设定）搜索预算增大时，诚实 BA、适应度、高估和子集大小的变化**（逐外层运行配对；先格子等权再跨数据集；自助 2 000 次）", ""]
        nm = {"test_ba": "诚实测试 BA", "fit": "适应度 BA（1 − 内层误差）", "gap_B": "高估 B", "k": "子集大小 k"}
        rows = [("%d → %d" % (r.cp_lo, r.cp_hi), nm[r.outcome], r.n_ds, ci(r.mean_diff, r.lo, r.hi, 3 if r.outcome != "k" else 1), "%d/%d" % (r.n_pos, r.n_ds), pv(r.p_wilcoxon))
                for _, r in q2b[q2b.breakdown == "all"].iterrows()]
        L.append(md(rows, ["预算", "指标", "数据集", "平均变化 [95% CI]", "增大的数据集", "p（Wilcoxon，未校正）"]))
    if h3 is not None and len(h3):
        L += ["", "### 2.3 Q3（探索性）：gap 与 log(p/n)", ""]
        xn = {"log_p_over_n": "log(p/n)（事先设定）", "log_n": "log(n)（探索）", "log_n_minority": "log(少数类样本数)（探索）"}
        rows = [(r.flaw, r.learner, xn[r.x], r.n_ds, ci(r.slope, r.slope_lo, r.slope_hi), f3(r.kendall_tau, 2), pv(r.p_tau)) for _, r in h3.iterrows()]
        L.append(md(rows, ["缺陷", "学习器", "自变量", "数据集", "Theil–Sen 斜率 [95% CI]", "Kendall τ", "p"]))
    if h4 is not None and len(h4):
        L += ["", "### 2.4 Q4：标签完全置换后各协议报告的 BA − 0.5（真值为 0）", ""]
        rows = [(r.report, r.n_ds, ci(r.mean_minus_half, r.lo, r.hi, 4), "%d/%d" % (r.n_pos, r.n_ds), pv(r.p_two_sided), pv(r.p_greater)) for _, r in h4.iterrows()]
        L.append(md(rows, ["报告方式", "数据集数", "BA − 0.5 [95% CI]", "为正的数据集", "p（双侧）", "p（> 0 单侧）"]))
    return L


def section_q5():
    a = rd("Q5_across_datasets.csv")
    L = ["## 3. Q5：诚实评估下，EFS 是否优于预算匹配随机搜索与简单基线（双侧；主参考预算 4 000）", ""]
    if a is None or len(a) == 0:
        return L + ["（尚无结果）"]
    cmp_ = json.load(open(os.path.join(TAB, "compare_summary.json"))) if os.path.exists(os.path.join(TAB, "compare_summary.json")) else {}
    prim = int(a.cp.max()) if 4000 not in set(a.cp) else 4000
    for lr in ("ridge", "knn"):
        g = a[(a.cp == prim) & (a.learner == lr)]
        if len(g) == 0:
            continue
        L += ["", "### 学习器 = %s" % ("岭分类" if lr == "ridge" else "5-NN"), "", "EFS 算法 − 预算匹配随机搜索（跨数据集平均的 BA 差；区间为数据集层面的 t 区间；Holm 校正）", ""]
        rows = [(r.a, r.n_ds, ci(r.mean_diff, r.lo, r.hi), "%d/%d" % (r.n_pos, r.n_ds), pv(r.p), pv(r.get("p_holm_prereg")), pv(r.p_holm)) for _, r in g[g.b == "rs"].iterrows()]
        L.append(md(rows, ["算法 − rs", "数据集", "BA 差 [95% CI]", "正 / 总", "p（Wilcoxon）", "p（Holm，事先设定的族：8 个\"方法 vs rs\"）", "p（Holm，保守：全部 24 个两两比较）"]))
        wc = rd("Q5_within_counts.csv")
        cmap = {}
        if wc is not None and len(wc):
            for _, r in wc[(wc.cp == prim) & (wc.learner == lr)].iterrows():
                cmap[(r.a, r.b)] = "%d / %d（共 %d）" % (r.a_higher_mean, r.a_lower_mean, r.n_ds)
        rows = [(r.a, r.b, ci(r.mean_diff, r.lo, r.hi), "%d/%d" % (r.n_pos, r.n_ds), pv(r.p_holm), cmap.get((r.a, r.b), "-")) for _, r in g[g.b != "rs"].iterrows()]
        L += ["", "各搜索算法（含 rs）− 简单基线（anova / rf = 同一学习器上的 top-k 过滤，k 由内层 CV 选；all = 不做特征选择；l1 = L1 逻辑回归，自带分类器，与学习器无关）", "",
              md(rows, ["算法", "基线", "BA 差 [95% CI]", "正 / 总", "p（Holm）", "数据集内均值更高 / 更低（仅描述，无检验）"])]
        fr = cmp_.get("friedman_cp%d_%s" % (prim, lr))
        if fr:
            L += ["", "Friedman 检验（%d 个数据集 × 方法）：χ²=%.2f，p=%s；平均名次（越小越好）：%s" % (fr["n_ds"], fr["chi2"], pv(fr["p"]), "，".join("%s %.2f" % (k, v) for k, v in sorted(fr["mean_rank"].items(), key=lambda kv: kv[1])))]
    fmy = rd("Q5_by_family.csv")
    if fmy is not None and len(fmy):
        L += ["", "### 按数据集类型拆分（探索性）：高维小样本 HD（9 个，p ≫ n）vs 重度不平衡低维 IMB（4 个，n ≫ p），预算 %d" % prim, "",
              "每格 = 数据集平均的 BA 差（正 / 总）；行 = 搜索算法，列 = 被比较的方法。", ""]
        for lr in ("ridge", "knn"):
            g = fmy[(fmy.cp == prim) & (fmy.learner == lr)]
            if len(g) == 0:
                continue
            rows = []
            for a in ("ga", "bpso", "bde", "nsga2", "rs"):
                cells = []
                for fam in ("HD", "IMB"):
                    for b in ("rs", "anova", "rf", "all", "l1"):
                        r = g[(g.family == fam) & (g.a == a) & (g.b == b)]
                        cells.append("%s (%d/%d)" % (f3(r.iloc[0].mean_diff), r.iloc[0].n_pos, r.iloc[0].n_ds) if len(r) else "-")
                rows.append((a, *cells))
            L += ["学习器 = %s" % lr, "", md(rows, ["算法"] + ["HD − " + b for b in ("rs", "anova", "rf", "all", "l1")] + ["IMB − " + b for b in ("rs", "anova", "rf", "all", "l1")]), ""]
    q5b = rd("Q5b_advantage_inflation.csv")
    if q5b is not None and len(q5b):
        L += ["", "### 通过\"报告的适应度\"比较算法会得到什么结论？（探索性，非事先设定；预算 %d）" % prim, "",
              "把\"算法 − 预算匹配随机搜索\"的优势分别按**报告的适应度**与**诚实外层测试 BA**来算（逐外层运行配对；先数据集内平均再跨数据集）。", ""]
        g = q5b[q5b.cp == prim]
        rows = [(r.learner, r.algo, r.n_ds, ci(r.rep, r.rep_lo, r.rep_hi), ci(r.true, r.true_lo, r.true_hi), ci(r.infl, r.infl_lo, r.infl_hi),
                 f3(r.rep / r.true, 1) if abs(r.true) > 1e-9 else "-") for _, r in g.iterrows()]
        L.append(md(rows, ["学习器", "算法", "数据集", "报告的优势（适应度）", "真实优势（诚实 BA）", "优势被夸大的量", "倍数"]))
        rk = rd("Q5b_rank_by_criterion.csv")
        if rk is not None and len(rk):
            r0 = rk[rk.cp == prim]
            rows = [(r.learner, r.algo, f3(r.mean_rank_by_fitness, 2), f3(r.mean_rank_by_honest, 2), f3(r.kendall_tau_between_rankings, 2)) for _, r in r0.iterrows()]
            L += ["", "五个搜索算法的平均名次（1 = 最好）：按适应度排名 vs 按诚实 BA 排名", "", md(rows, ["学习器", "算法", "按适应度的平均名次", "按诚实 BA 的平均名次", "两种排名的 Kendall τ"])]
    m = rd("Q5_mean_ba.csv")
    if m is not None:
        piv = m[(m.cp == prim)].pivot_table(index=["learner", "ds"], columns="method", values="mean_ba")
        cols = [c for c in ("ga", "bpso", "bde", "nsga2", "rs", "anova", "rf", "all", "l1") if c in piv.columns]
        rows = [(lr, ds, *[f3(piv.loc[(lr, ds)][c]) for c in cols]) for lr, ds in piv.index]
        L += ["", "**各数据集平均诚实 BA（预算 %d）**" % prim, "", md(rows, ["学习器", "数据集"] + cols)]
    q6 = rd("Q6_metric_summary.csv")
    if q6 is not None and len(q6):
        L += ["", "## 4. Q6（探索性）：适应度指标（相对 BA 适应度的诚实性能差）", ""]
        g = q6[(q6.outcome.isin(["ba", "auc", "tpr", "tnr"])) & (q6.cp == q6.cp.max())]
        rows = [(r.learner, r.algo, r.fitness_metric, r.outcome, r.n_ds, f3(r.mean_diff_vs_BA_fitness, 4), "%d/%d" % (r.n_pos, r.n_ds), pv(r.p_holm)) for _, r in g.iterrows()]
        L.append(md(rows, ["学习器", "算法", "适应度指标", "评价指标", "数据集", "平均差（该指标 − BA 适应度）", "正 / 总", "p（Holm）"]))
    return L


def section_special():
    L = ["## 5. 专项研究", ""]
    s = rd("E5so_summary.csv")
    if s is not None and len(s):
        L += ["### 5.1 SMOTE 先于划分（缺陷 C，外层固定管线：标准化 → [SMOTE] → ANOVA top-20 → 学习器）", "",
              "指标差 = 先过采样后划分 − 只在训练部分过采样（数据集层面平均，真实样本评估 vs 含合成样本评估）", ""]
        rows = [(r.learner, r.outcome, r.n_ds, ci(r["mean"], r.lo, r.hi), "%d/%d" % (r.n_pos, r.n_ds), pv(r.get("p_holm", r.p))) for _, r in s.iterrows()]
        L.append(md(rows, ["学习器", "指标", "数据集", "平均差 [95% CI]", "正 / 总", "p（Holm）"]))
    s = rd("X9s_decomposition.csv")
    if s is not None and len(s):
        L += ["", "### 5.1b SMOTE 先于划分的分解（X9s；探索性，事后加做）", "",
              "同样的种子重放\"先过采样后划分\"的流水线，并把同一套预测分别算在（a）含合成点的全部行（E5so 原流程报告的就是这个）和（b）只含真实行上：**总流程差 = 全部行 − 诚实；测试样本组成评分差 = 全部行 − 真实行；真实样本上的剩余流程差异 = 真实行 − 诚实**。"
              "这是描述性代数拆分，不是因果效应识别。错误流程与诚实流程的真实样本分折、标准化统计和合成训练集不同；剩余项不能解释成纯泄漏。组成评分差在同一预测下改变计分样本及类别比例，F1、MCC 对此尤其敏感。", ""]
        s = s.assign(m=s["metric"].map({"ba": 1, "auc": 2, "mcc": 3, "f1": 4, "acc": 5}), c=s["component"].map({"total": 1, "composition": 2, "leakage": 3}))
        labels = {"total": "总流程差", "composition": "测试样本组成评分差", "leakage": "真实样本上的剩余流程差异"}  # legacy CSV component key retained for numerical reproducibility
        rows = [(r.learner, r.metric, labels[r.component], r.n_ds, ci(r["mean"], r.lo, r.hi), "%d/%d" % (r.n_pos, r.n_ds)) for _, r in s.sort_values(["learner", "m", "c"]).iterrows()]
        L.append(md(rows, ["学习器", "指标", "分量", "数据集", "平均差 [95% CI]", "正 / 总"]))
    s = rd("E5si_summary.csv")
    if s is not None and len(s):
        L += ["", "### 5.2 EFS 内层 SMOTE（泄漏版 − 干净版；配对）", ""]
        rows = [(r.learner, r.cp, r.outcome, r.n_ds, ci(r["mean"], r.lo, r.hi), "%d/%d" % (r.n_pos, r.n_ds)) for _, r in s.iterrows() if r.outcome in ("d_test_ba", "d_gapB", "gapB_clean", "gapB_leaky")]
        L.append(md(rows, ["学习器", "预算", "指标", "数据集", "平均 [95% CI]", "正 / 总"]))
    u = rd("E6u_record_vs_subject.csv")
    if u is not None and len(u):
        L += ["", "### 5.3 重复测量的单元泄漏（帕金森，252 人 × 3 条录音；按记录 − 按受试者划分外层折，按重复次数配对）", ""]
        rows = [(r.algo, r.learner, r.cp, "按受试者" if r.inner_group else "按记录", r.outcome, r.n_rep, "%s ± %s" % (f3(r.mean_record_minus_subject), f3(r.sd)), "%d/%d" % (r.n_pos, r.n_rep))
                for _, r in u[u.outcome == "ba"].iterrows()]
        L.append(md(rows, ["算法", "学习器", "预算", "内层折", "指标", "重复数", "平均差 ± SD", "正 / 总"]))
        L.append("\n重复 = 同一批受试者上的交叉验证重复，不是独立证据，故不给 p 值（仅描述）。")
    u = rd("E6u_inner_group_benefit.csv")
    if u is not None and len(u):
        L += ["", "外层按受试者评估时，内层折改为按受试者（相对按记录）的诚实 BA 差", ""]
        rows = [(r.algo, r.learner, r.cp, r.n_rep, f3(r.mean_groupinner_minus_recordinner, 4), "%d/%d" % (r.n_pos, r.n_rep)) for _, r in u.iterrows()]
        L.append(md(rows, ["算法", "学习器", "预算", "重复数", "BA 差", "正 / 总"]))
    u = rd("E6uf_means.csv")
    if u is not None and len(u):
        rows = [(r.algo, r.learner, r.cp, "按受试者" if r.group else "按记录", f3(r.leaky_cv_ba), f3(r.fit_ba), f3(r.k, 1), r.n) for _, r in u.iterrows()]
        L += ["", "缺陷 A 在单元数据上（全数据搜索后重估：按记录 vs 按受试者折）", "", md(rows, ["算法", "学习器", "预算", "折划分", "重估 BA", "适应度 BA", "k", "种子数"])]
    t = rd("E7l_hi_vs_lo.csv")
    if t is not None and len(t):
        pl = rd("E7l_pooled.csv"); bb = rd("E7l_by_budget.csv")
        L += ["", "### 5.4 长预算（E7l：4 个数据集 colon / dlbcl / prostate / ovarian × {岭 256 000，5-NN 64 000} × {GA, NSGA-II, RS}；每个格子 10 个外层运行）", ""]
        if pl is not None and len(pl):
            rows = [(r.algo, r.n_ds, "%s（范围 %s ~ %s）" % (f3(r.d_fit), f3(r.d_fit_min), f3(r.d_fit_max)), "%s（范围 %s ~ %s）" % (f3(r.d_test_ba), f3(r.d_test_ba_min), f3(r.d_test_ba_max)),
                     "%s（范围 %s ~ %s）" % (f3(r.d_gapB), f3(r.d_gapB_min), f3(r.d_gapB_max))) for _, r in pl.iterrows()]
            L += ["从 4 000 次评估增加到该学习器的最大预算（岭 256 000 / 5-NN 64 000）后的变化（每个数据集一个值 = 学习器平均；只有 4 个数据集，仅描述：均值与数据集间范围，不给区间或检验）：", "",
                  md(rows, ["算法", "数据集数", "Δ适应度 BA（均值，范围）", "Δ诚实 BA（均值，范围）", "Δ高估（均值，范围）"]), ""]
        if bb is not None and len(bb):
            rows = [(r.learner, r.algo, r.cp, f3(r.fit), f3(r.test_ba), f3(r.gap_B), f3(r.k, 1)) for _, r in bb.sort_values(["learner", "algo", "cp"]).iterrows()]
            L += ["按预算的平均（4 个数据集平均）：", "", md(rows, ["学习器", "算法", "预算", "适应度 BA", "诚实 BA", "高估", "k"]), "", "分格子明细：", ""]
        rows = [(r.ds, r.learner, r.algo, "%d→%d" % (r.cp_lo, r.cp_hi), r.n, f3(r.d_fit), f3(r.d_test_ba), f3(r.d_gapB), f3(r.rho_gapB, 2)) for _, r in t.iterrows()]
        L.append(md(rows, ["数据集", "学习器", "算法", "预算", "运行数", "Δ适应度 BA", "Δ诚实 BA", "Δ高估", "ρ(gap, 预算)"]))
    return L


def section_extras():
    L = ["## 6. 稳健性与机制（额外研究；不属于主冻结代码，结果在 `results/jobs_extra/`）", ""]
    p = os.path.join(TAB, "X1s_summary.csv")
    d = rd("X1s_paired.csv")
    if d is not None and len(d):
        from extras_analysis import boot_ci  # noqa: WPS433
        L += ["### 6.1 标准化是否逐内层折重拟合（E-scaler；与 E1h 逐运行配对）", ""]
        dsv = d.groupby("ds")[["d_ba", "d_gapB", "d_fit"]].mean()
        rows = [(ds_, f3(r_.d_ba, 4), f3(r_.d_gapB, 4), f3(r_.d_fit, 4)) for ds_, r_ in dsv.iterrows()]
        rows.append(("**%d 个数据集的平均**" % len(dsv), f3(dsv.d_ba.mean(), 4), f3(dsv.d_gapB.mean(), 4), f3(dsv.d_fit.mean(), 4)))
        L.append(md(rows, ["数据集", "Δ诚实 BA（逐折重拟合 − 共享标准化）", "Δ高估 B", "Δ适应度 BA"]))
        lo, hi = boot_ci(d["d_ba"])
        L.append("\n只有 %d 个数据集（%d 个配对运行共享样本和训练数据），所以**只描述，不给区间，也不做\"可忽略\"的等价性判断**。"
                 "首次分析把这 %d 个配对运行当作独立样本，给出 Δ诚实 BA 的区间 [%s, %s]（旧做法，已废弃），其上界略超出事先写定的 ±0.01，当时就没有判为可忽略。" % (
                     len(dsv), len(d), len(d), f3(lo, 4), f3(hi, 4)))
    t = rd("X2k_means.csv")
    if t is not None and len(t):
        pooled = rd("X2k_paired.csv")
        L += ["", "### 6.2 内层折数（3 / 5 / 10）", ""]
        if pooled is not None and len(pooled):
            g = pooled.groupby(["inner_k", "outcome", "cp"]).mean_diff_vs_k5.mean().reset_index()
            rows = [(r.inner_k, r.outcome, r.cp, f3(r.mean_diff_vs_k5, 4)) for _, r in g.iterrows()]
            L.append(md(rows, ["内层折数", "指标", "预算", "相对 5 折的平均差（各格子平均）"]))
    t = rd("X3u_contrasts.csv")
    pl = rd("X3u_pooled_by_sigma.csv")
    if pl is not None and len(pl):
        L += ["", "### 6.3 合成单元数据：按记录 vs 按受试者评估的剂量–反应（ridge；U = 受试者数，sig_f = 个体指纹强度）", "",
              "合并 U、算法、预算后的汇总（事先写定的对照：sig_f = 0 时按记录评估 − 全按受试者评估应 |差| < 0.02）", ""]
        rows = [(r.sig_f, "按记录评估 − 全按受试者" if r.contrast.startswith("record") else "外层按受试者时：按受试者内层折 − 按记录内层折", r.n, ci(r.mean_diff, r.lo, r.hi),
                 ("对照通过" if r.control_ok else "对照未通过") if (r.sig_f == 0 and isinstance(r.control_ok, (bool, np.bool_)) and r.contrast.startswith("record")) else "") for _, r in pl.iterrows()]
        L.append(md(rows, ["sig_f", "对比", "实例数（聚类：每个实例是一个独立的合成数据集）", "平均差（BA）[95% 数据集层面 t 区间]", "备注"]))
    if t is not None and len(t):
        L += ["", "分 U、算法的明细（最大预算）", ""]
        g = t[(t.cp == t.cp.max()) & (t.contrast == "record_minus_subject_eval(ff-tt)")]
        rows = [(r.U, r.sig_f, r.algo, r.n, ci(r.mean_diff, r.lo, r.hi)) for _, r in g.iterrows()]
        L.append(md(rows, ["U", "sig_f", "算法", "实例数", "按记录评估 − 全按受试者评估（BA）"]))
        g = t[(t.cp == t.cp.max()) & (t.contrast == "unit_aware_search_benefit(tt-tf)")]
        rows = [(r.U, r.sig_f, r.algo, r.n, ci(r.mean_diff, r.lo, r.hi)) for _, r in g.iterrows()]
        L += ["", "外层按受试者评估时，内层按受试者折相对按记录折的收益（BA）", "", md(rows, ["U", "sig_f", "算法", "实例数", "收益 [95% CI]"])]
    t = rd("X5s_vs_rs100.csv")
    if t is not None and len(t):
        kk = rd("X5s_k.csv")
        L += ["", "### 6.4 扩大尺寸范围的随机基线（X5s，RS100）：Q5 的\"算法 − RS\"在 RS100（子集大小均匀取 1..100）下重算", "",
              "RS100 只是把随机子集的大小范围从 1..40 扩大到 1..100，**并没有匹配各算法自己的大小分布**（GA 约 23、NSGA-II 约 9、DE 约 74），所以它只部分控制了子集大小；真正的同尺寸对照见 6.6（X5m）。"
              "RS100 的平均子集大小：%s。" % (
                  "；".join("%s/%d: %.1f" % (r.learner, r.cp, r.k) for _, r in kk.iterrows()) if kk is not None else "-"), ""]
        rows = [(r.learner, r.cp, r.algo, r.n_ds, ci(r.mean_diff_vs_rs100, r.lo, r.hi), "%d/%d" % (r.n_pos, r.n_ds), pv(r.p_holm)) for _, r in t.sort_values(["learner", "cp", "algo"]).iterrows()]
        L.append(md(rows, ["学习器", "预算", "算法（含 rs）", "数据集", "诚实 BA 差：算法 − RS100 [95% CI]", "正 / 总", "p（Holm）"]))
    t = rd("X4r_landscape.csv")
    if t is not None and len(t):
        L += ["", "### 6.5 无搜索的选择效应：16 384 个随机子集的\"取最大\"统计（诚实标签 vs 置换标签）", ""]
        g = t[(t.B == 16000)]
        rows = [(r.ds, r.learner, r.k, "置换" if r.perm == 0 else "真实", f3(r.inn_mean), f3(r.inn_sd), f3(r.pearson, 2), f3(r.max_inn), f3(r.tst_at_max), f3(r.gap_obs), f3(r.gap_pred_normal)) for _, r in g.iterrows()]
        L.append(md(rows, ["数据集", "学习器", "k", "标签", "内层 BA 均值", "内层 BA SD", "内层-测试相关", "最大内层 BA（B=16000）", "该解的测试 BA", "观测 gap", "正态极值预测 gap"]))
    L += section_revision()
    return L


def section_revision():
    """6.6-6.9: the studies added after the external review (docs/DEVIATIONS.md #7-#9)."""
    L = []
    t = rd("X5m_same_size_rs.csv")
    if t is not None and len(t):
        t2 = rd("X5m_size_effect_on_rs.csv")
        L += ["", "### 6.6 同尺寸随机搜索（X5m；探索性，事后加做）", "",
              "对每个算法 a 的每个外层运行，随机搜索只抽**恰好与 a 在该运行中返回的子集同样大小**的随机子集（同样的内层折、相同的请求预算）。"
              "它把\"选哪些特征\"和\"选多少个特征\"分开：算法 − 同尺寸 RS 的差只能来自特征的选择。注意同尺寸 RS 免费拿到了算法通过内层 CV 找到的大小，对算法不利。", ""]
        rows = [(r.learner, r.budget, r.ref, f3(r.mean_k, 1), r.n_ds, ci(r.mean_diff, r.lo, r.hi), "%d/%d" % (r.n_pos, r.n_ds), pv(r.p), pv(r.p_holm),
                 ci(r.vs_rs_mean_diff, r.vs_rs_lo, r.vs_rs_hi)) for _, r in t.sort_values(["learner", "budget", "ref"]).iterrows()]
        L.append(md(rows, ["学习器", "预算", "算法 a", "平均子集大小", "数据集", "诚实 BA：a − 同尺寸 RS [95% CI]", "正 / 总", "p", "p（Holm，每个学习器×预算 4 个）", "对照：a − 普通 RS（预算匹配，尺寸 1..40）"]))
        if t2 is not None and len(t2):
            rows = [(r.learner, r.budget, r.ref, f3(r.mean_k, 1), r.n_ds, ci(r.mean_diff, r.lo, r.hi), "%d/%d" % (r.n_pos, r.n_ds), pv(r.p)) for _, r in t2.sort_values(["learner", "budget", "ref"]).iterrows()]
            L += ["", "同尺寸 RS − 普通 RS（只因为抽的子集大小不同而带来的诚实 BA 差）", "", md(rows, ["学习器", "预算", "尺寸取自算法", "平均子集大小", "数据集", "同尺寸 RS − 普通 RS [95% CI]", "正 / 总", "p"])]
    t = rd("X6r_distinct_matched.csv")
    if t is not None and len(t):
        L += ["", "### 6.7 双方按相同不同候选数匹配预算（X6d + X6r；探索性，事后加做）", "",
              "E1h 的预算按\"包装器请求次数\"计，缓存命中也算；GA 与 NSGA-II 会重复请求已评估过的掩码，因此同样 4 000 次请求只评估了约 2 800~3 100 个不同掩码，而随机搜索约 4 000 个。"
              "X6d 用原种子让 GA / NSGA-II 到达 1 000 / 4 000 个不同掩码；修订 2 的 X6r 用原随机搜索种子把 RS 也补到相同不同掩码数。表中不再把 RS 的请求数近似当作不同候选数，旧近似结果仍保存在 X6d_distinct_matched.csv。", ""]
        rows = [(r.learner, r.algo, r.target, "同一算法，E1h 同请求数" if r.comparison.startswith("same") else "对随机搜索（双方相同不同候选数）", f3(r.mean_calls, 0), f3(r.mean_distinct, 0),
                 r.n_ds, ci(r.mean_diff, r.lo, r.hi), "%d/%d" % (r.n_pos, r.n_ds), pv(r.p), pv(r.p_holm),
                 ci(r.req_mean_diff, r.req_lo, r.req_hi) if r.comparison.startswith("vs") else "-") for _, r in t.sort_values(["learner", "target", "algo", "comparison"]).iterrows()]
        L.append(md(rows, ["学习器", "算法", "目标（不同掩码数）", "比较", "平均请求数", "平均不同掩码数", "数据集", "诚实 BA 差 [95% CI]", "正 / 总", "p", "p（Holm）", "对照：同请求数下的差 [95% CI]"]))
    t = rd("X7d_distinct_diag.csv")
    if t is not None and len(t):
        sec = rd("X7d_seconds_by_algo.csv"); e1 = rd("E1h_seconds_by_algo.csv")
        L += ["", "### 6.8 请求数 vs 不同掩码数、缓存命中率与耗时（X7d；各数据集 × 学习器的 5 个外层折）", ""]
        rows = [(r.algo, r.cp, r.n, f3(r.mean_calls, 0), f3(r.mean_distinct, 0), f3(100 * r.mean_hit_rate, 1) + "%") for _, r in t.sort_values(["algo", "cp"]).iterrows()]
        L.append(md(rows, ["算法", "预算（请求数）", "运行数", "平均请求数", "平均不同掩码数", "缓存命中率"]))
        if sec is not None and e1 is not None:
            rows = [(r.algo, f3(r.mean_seconds, 1), f3(r.median_seconds, 1)) for _, r in sec.iterrows()]
            L += ["", "一次搜索（到该格子的最大预算）的耗时（秒；X7d 不含测试部分评估）", "", md(rows, ["算法", "平均", "中位数"])]
    t = rd("X8p_front_stats.csv")
    if t is not None and len(t):
        L += ["", "### 6.9 D 的重放（X8p）：Pareto 档案", "",
              "E1h 的 NSGA-II 记录里，当前最优解取自全部已评估候选，前沿却取自上一代的种群，4 190 条记录中有 110 条 D 为负（最低 -22.5 个点）。X8p 用相同的种子重放全部 1 070 个 NSGA-II 作业，"
              "在每个检查点把 D 定义在同一个候选集合上：截至该点的全部已评估候选的非支配集（Pareto 档案；目标向量相同的掩码合并为首个评估的代表）；当前最优解必在其中，故 D ≥ 0。"
              "重放逐位复现了 E1h 的全部当前最优解（41 900 个数字，0 个不同）。", ""]
        rows = [(r.cp, r.n, f3(r.mean_front, 1), f3(r.mean_tied, 1), f3(r.mean_calls, 0), f3(r.mean_distinct, 0), f3(100 * r.mean_D_merged, 2), f3(100 * r.mean_D_ties, 2)) for _, r in t.iterrows()]
        L.append(md(rows, ["预算", "记录数", "档案平均大小", "含并列的平均掩码数", "平均请求数", "平均不同掩码数", "D（合并，点）", "D'（并列全保留，点）"]))
    return L


def section_tldr():
    """One-screen summary composed from the tables (so that no number is transcribed by hand)."""
    L = ["## 要点（由表格自动组成）", ""]
    h1 = rd("H1_inflation.csv"); h2 = rd("H2_summary.csv"); h4 = rd("H4_permutation.csv"); q2b = rd("Q2b_honest_vs_budget.csv")
    a5 = rd("Q5_across_datasets.csv"); q5b = rd("Q5b_advantage_inflation.csv"); rk = rd("Q5b_rank_by_criterion.csv"); x3 = rd("X3u_pooled_by_sigma.csv")
    so = rd("E5so_summary.csv"); u = rd("E6u_record_vs_subject.csv")

    def row(fl, bn="all"):
        if h1 is None:
            return None
        g = h1[(h1.breakdown == bn) & (h1.flaw == fl)]
        return g.iloc[0] if len(g) else None
    for fl, txt in (("B", "把搜索适应度当性能报告（B）"), ("A", "先在全部样本上选特征、再重估（A）"), ("D", "按测试集挑 Pareto 解（D）"), ("E5", "5 个种子取最好（E）"),
                    ("G2", "每个数据集事后挑平均测试最好的算法（G2，探索性）")):
        r = row(fl)
        if r is not None:
            L.append("- **%s**：平均高估 BA %s 个点（95%% CI %s 到 %s；%d/%d 个数据集为正）。" % (txt, f3(100 * r["mean"], 1), f3(100 * r.lo, 1), f3(100 * r.hi, 1), r.n_pos, r.n_ds))
    if h4 is not None and len(h4):
        d = {r.report: r for _, r in h4.iterrows()}
        if "honest_test_BA" in d:
            L.append("- **标签完全置换**时，诚实嵌套协议的 BA − 0.5 = %s 个点（%s）；适应度即报告 %s 个点，先选后分 + 重估 %s 个点。" % (
                f3(100 * d["honest_test_BA"].mean_minus_half, 2), "事先设定的门槛 |差| < 3 个点：" + ("达标" if abs(d["honest_test_BA"].mean_minus_half) < 0.03 else "未达标"),
                f3(100 * d["fitness_as_report(B)"].mean_minus_half, 1) if "fitness_as_report(B)" in d else "-",
                f3(100 * d["select_on_all+reCV(A)"].mean_minus_half, 1) if "select_on_all+reCV(A)" in d else "-"))
    if h2 is not None and len(h2):
        g = h2[h2.flaw == "B"]
        if len(g):
            r = g.iloc[0]
            L.append("- **搜索预算越大，报告的适应度越乐观**：%d/%d 个 (数据集, 算法, 学习器) 格子里 高估与 log2(预算) 正相关（事先设定的阈值 75%%）。" % (r.n_rho_pos, r.n_cells))
    if h2 is not None and len(h2):
        g = h2[h2.flaw == "B"]
        if len(g):
            r = g.iloc[0]
            L.append("- 按数据集聚类后，B 的\"随预算增大\"：数据集平均占比 %s [%s, %s]，%d/%d 个数据集的占比 ≥ 75%%。" % (f3(r.share_ds_mean, 2), f3(r.share_ds_lo, 2), f3(r.share_ds_hi, 2), r.n_ds_ge75, r.n_ds))
    if q2b is not None and len(q2b):
        g = q2b[(q2b.breakdown == "all") & (q2b.cp_lo == 250) & (q2b.cp_hi == 4000)].set_index("outcome")
        g2 = q2b[(q2b.breakdown == "all") & (q2b.cp_lo == 4000) & (q2b.cp_hi == 16000)].set_index("outcome")
        if len(g) and "test_ba" in g.index:
            s = "- **诚实 BA 增益很小**（在这个设定里；大部分增益在 250 → 1000）：预算 250 → 4000 时适应度 +%s 个点、诚实 BA 只 +%s 个点" % (f3(100 * g.loc["fit", "mean_diff"], 1), f3(100 * g.loc["test_ba", "mean_diff"], 1))
            if len(g2) and "test_ba" in g2.index:
                s += "；4000 → 16000 时适应度 +%s、诚实 BA +%s（95%% CI %s 到 %s）。" % (f3(100 * g2.loc["fit", "mean_diff"], 1), f3(100 * g2.loc["test_ba", "mean_diff"], 1),
                                                                         f3(100 * g2.loc["test_ba", "lo"], 1), f3(100 * g2.loc["test_ba", "hi"], 1))
            L.append(s)
    if a5 is not None and len(a5):
        pr = 4000 if 4000 in set(a5.cp) else int(a5.cp.max())
        g = a5[(a5.cp == pr) & (a5.b == "rs")]
        if len(g):
            parts = ["%s/%s %s 个点（事先设定的族 Holm p=%s）" % (r.learner, r.a, f3(100 * r.mean_diff, 1), pv(r.get("p_holm_prereg", r.p_holm))) for _, r in g.iterrows()]
            L.append("- **诚实评估下 EFS 相对预算匹配随机搜索的真实优势**（预算 %d）：" % pr + "；".join(parts) + "。")
    if q5b is not None and len(q5b):
        pr = 4000 if 4000 in set(q5b.cp) else int(q5b.cp.max())
        g = q5b[q5b.cp == pr]
        if len(g):
            L.append("- **用\"报告的适应度\"比较算法会夸大它们相对随机搜索的优势**（预算 %d）：报告的优势 %s 个点 vs 真实优势 %s 个点（各 algo × learner 的范围）。" % (
                pr, "%s ~ %s" % (f3(100 * g["rep"].min(), 1), f3(100 * g["rep"].max(), 1)), "%s ~ %s" % (f3(100 * g["true"].min(), 1), f3(100 * g["true"].max(), 1))))
    if rk is not None and len(rk):
        pr = 4000 if 4000 in set(rk.cp) else int(rk.cp.max())
        t = rk[rk.cp == pr].groupby("learner").kendall_tau_between_rankings.first()
        L.append("- 五个搜索算法按\"适应度\"排名与按\"诚实 BA\"排名的 Kendall τ（预算 %d）：%s。" % (pr, "，".join("%s %.2f" % (k, v) for k, v in t.items())))
    if so is not None and len(so):
        g = so[so.outcome == "ba"]
        if len(g):
            L.append("- **SMOTE 先于划分**（外层固定管线）使 BA 虚高 %s 个点（%s）。" % ("%s ~ %s" % (f3(100 * g["mean"].min(), 1), f3(100 * g["mean"].max(), 1)),
                                                               "，".join("%s %s" % (r.learner, pv(r.get("p_holm", r.p))) for _, r in g.iterrows())))
    dd = rd("X7d_distinct_diag.csv")
    if dd is not None and len(dd):
        g = dd[dd.cp == 4000].set_index("algo")
        L.append("- **同样 4 000 次请求，不同掩码数并不相同**：%s。" % "，".join("%s %s" % (a, f3(g.loc[a, "mean_distinct"], 0)) for a in ("ga", "nsga2", "bpso", "bde", "rs") if a in g.index))
    m5 = rd("X5m_same_size_rs.csv")
    if m5 is not None and len(m5):
        g = m5[m5.budget == 4000]
        L.append("- **与同尺寸随机搜索相比**（预算 4 000，诚实 BA）：%s。" % "；".join("%s/%s %s 个点（Holm p=%s）" % (r.learner, r.ref, f3(100 * r.mean_diff, 1), pv(r.p_holm)) for _, r in g.iterrows()))
    dm = rd("X6r_distinct_matched.csv")
    if dm is not None and len(dm):
        g = dm[(dm.target == 4000) & dm.comparison.str.startswith("vs")]
        L.append("- **双方均到达 4 000 个不同掩码后，相对随机搜索的优势**：%s。" % "；".join("%s/%s %s 个点（同请求数下 %s）" % (r.learner, r.algo, f3(100 * r.mean_diff, 1), f3(100 * r.req_mean_diff, 1)) for _, r in g.iterrows()))
    sd = rd("X9s_decomposition.csv")
    if sd is not None and len(sd):
        parts = []

        def dv(m, comp):                                         # "ridge&knn" in a fixed order
            return "&".join(f3(100 * sd[(sd.metric == m) & (sd.component == comp) & (sd.learner == lr)].iloc[0]["mean"], 1) for lr in ("ridge", "knn"))
        for m in ("ba", "f1", "mcc"):
            if len(sd[sd.metric == m]):
                parts.append("%s：总流程差 %s / 测试样本组成评分差 %s / 真实样本剩余流程差 %s" % (m.upper(), dv(m, "total"), dv(m, "composition"), dv(m, "leakage")))
        L.append("- **SMOTE 先于划分的描述性分解**（岭&5-NN，点）：" + "；".join(parts) + "；剩余项不是严格隔离的纯泄漏效应。")
    if x3 is not None and len(x3):
        g = x3[x3.contrast.str.startswith("record")]
        L.append("- **合成重复测量数据**：按记录评估比全按受试者评估多报 BA（sig_f = %s）：%s 个点。" % ("/".join(str(x) for x in g.sig_f), "/".join(f3(100 * x, 1) for x in g.mean_diff)))
    if u is not None and len(u):
        g = u[(u.outcome == "ba") & (~u.inner_group)]
        if len(g):
            L.append("- **帕金森（252 人 × 3 条录音）真实数据**：外层按记录划分比按受试者划分多报 BA %s ~ %s 个点（各 算法 × 预算）。" % (f3(100 * g.mean_record_minus_subject.min(), 1), f3(100 * g.mean_record_minus_subject.max(), 1)))
    return L + [""]


def main():
    L = ["# 实验结果汇总（自动生成，%s UTC）" % time.strftime("%Y-%m-%d %H:%M", time.gmtime()), "",
         "本文件由 `analysis/report.py` 从 `results/tables/*.csv` 机械生成；事先冻结的内部协议见 `docs/PROTOCOL.md`，分析细则见 `docs/ANALYSIS_PLAN.md`，偏离见 `docs/DEVIATIONS.md`。", ""]
    for fn in (section_tldr, section_integrity, section_audit, section_core, section_q5, section_special, section_extras):
        try:
            L += fn() + [""]
        except Exception as e:  # noqa: BLE001
            L += ["（%s 生成失败：%s: %s）" % (fn.__name__, type(e).__name__, e), ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", OUT, "(%d lines)" % len(L))


if __name__ == "__main__":
    main()
