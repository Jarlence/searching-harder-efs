"""Pre-specified R22 summaries. Refuses partial experiment coverage.

Only writes results/r22_strict_scaling; original results and manuscript stay intact.
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "extras"))
import ex_suites6 as E
import numpy as np
from scipy import stats

OUT = ROOT / "results/r22_strict_scaling"
FAMILY = 9


def score_record(record):
    F = 1.0 - record["inner_err"]
    T = record["test_ba"]
    return dict(F=F, T=T, B=F-T, k=record["k"])


def load_shared(jobs):
    records = {}
    for s in jobs:
        record = json.loads(E.comparator_path(s).read_text(encoding="utf-8"))
        for cp in E.CPS:
            key = (s["ds"], s["learner"], s["algo"], s["rep"], s["fold"], cp)
            if key in records:
                raise RuntimeError("Repeated main comparator key")
            records[key] = score_record(record["result"][str(cp)])
    return records


def load_strict(jobs):
    records, provenance, replay_count = {}, {}, 0
    labels, expected_splits, execution = {}, {}, []
    for s in jobs:
        p = E.job_path(s)
        if not p.exists():
            raise RuntimeError("Incomplete suite: " + str(p))
        r = json.loads(p.read_text(encoding="utf-8"))
        E.validate_record(r, s)
        execution.append(dict(dataset=s["ds"], learner=s["learner"], algorithm=s["algo"],
                              repeat=s["rep"], fold=s["fold"], seconds=r["seconds"]))
        provenance[str(p.relative_to(ROOT)).replace("\\", "/")] = E.sha_file(p)
        replay_count += bool(r["x1_replay"]["available"])
        outer_train, outer_test = r["splits"]["outer_train"], r["splits"]["outer_test"]
        if set(outer_train) & set(outer_test):
            raise RuntimeError("Outer split overlap")
        validation_indices = []
        for inner in r["splits"]["inner_relative_to_outer_train"]:
            if set(inner["train"]) & set(inner["validation"]):
                raise RuntimeError("Inner split overlap")
            if sorted(inner["train"] + inner["validation"]) != list(range(len(outer_train))):
                raise RuntimeError("Incomplete inner partition")
            validation_indices += inner["validation"]
        if sorted(validation_indices) != list(range(len(outer_train))):
            raise RuntimeError("Incomplete inner OOF partition")
        # Independently regenerate fold indices from the same source keys.
        split_key = (s["ds"], s["learner"], s["rep"], s["fold"])
        if s["ds"] not in labels:
            with np.load(ROOT / "data/processed" / (s["ds"] + ".npz")) as data:
                labels[s["ds"]] = data["y"].astype(np.int64)
        if split_key not in expected_splits:
            y = labels[s["ds"]]
            tr, te = E.ST.outer_split(y, s["ds"], s["rep"], s["fold"])
            inner = E.S.make_folds(y[tr], 5, E.S.rng_for("inner", s["ds"], s["learner"], s["rep"], s["fold"], None, False, False))
            expected_splits[split_key] = dict(outer_train=tr.tolist(), outer_test=te.tolist(),
                inner_relative_to_outer_train=[{"train": a.tolist(), "validation": b.tolist()} for a,b in inner])
        if r["splits"] != expected_splits[split_key]:
            raise RuntimeError("Main/strict outer or inner indices differ")
        for cp in E.CPS:
            key = (s["ds"], s["learner"], s["algo"], s["rep"], s["fold"], cp)
            if key in records:
                raise RuntimeError("Repeated strict key")
            records[key] = score_record(r["result"][str(cp)])
            records[key].update(calls=r["result"][str(cp)]["calls"], distinct=r["result"][str(cp)]["distinct"])
    return records, provenance, replay_count, execution


def t_summary(dataset_values, focal=False):
    values = np.asarray(list(dataset_values.values()), dtype=float) * 100.0
    if not np.isfinite(values).all() or len(values) < 2:
        raise ValueError("Invalid data-set-level values")
    n = len(values)
    mean, se = float(values.mean()), float(values.std(ddof=1)/math.sqrt(n))
    half = float(stats.t.ppf(.975, n-1))*se
    row = dict(n_datasets=n, mean_pp=mean, ci95_lo_pp=mean-half, ci95_hi_pp=mean+half,
               sd_pp=float(values.std(ddof=1)), median_pp=float(np.median(values)), min_pp=float(values.min()),
               max_pp=float(values.max()), n_positive=int((values > 0).sum()), focal=focal)
    if focal:
        simultaneous = float(stats.t.ppf(1-.05/(2*FAMILY), n-1))*se
        row.update(simultaneous_family_n=FAMILY, simultaneous_confidence=1-.05/FAMILY,
                   simultaneous_lo_pp=mean-simultaneous, simultaneous_hi_pp=mean+simultaneous)
    return row


def equal_dataset_means(cells):
    datasets = defaultdict(list)
    for (ds, learner, algo), values in cells.items():
        datasets[ds].append(float(np.mean(values)))
    return {ds: float(np.mean(values)) for ds, values in sorted(datasets.items())}


def budget_values(records, lo, hi, field, learner="all"):
    cells = defaultdict(list)
    for key, low in records.items():
        ds, lr, algo, rep, fold, cp = key
        if cp != lo or (learner != "all" and lr != learner):
            continue
        high = records[(ds, lr, algo, rep, fold, hi)]
        cells[(ds, lr, algo)].append(high[field]-low[field])
    return equal_dataset_means(cells)


def level_values(records, cp, field, learner="all", algorithm="all"):
    cells = defaultdict(list)
    for key, value in records.items():
        ds, lr, algo, rep, fold, budget = key
        if budget == cp and learner in ("all", lr) and algorithm in ("all", algo):
            cells[(ds, lr, algo)].append(value[field])
    return equal_dataset_means(cells)


def advantage_values(records, learner, algo, cp, field):
    datasets = defaultdict(list)
    for key, value in records.items():
        ds, lr, a, rep, fold, c = key
        if (lr,a,c) != (learner,algo,cp):
            continue
        ref = records[(ds,lr,"rs",rep,fold,c)]
        datasets[ds].append(value[field]-ref[field])
    return {ds: float(np.mean(values)) for ds,values in sorted(datasets.items())}


def make_summaries(protocols):
    summaries, data_rows = [], []
    def add(meta, values, focal=False):
        summaries.append(dict(meta, **t_summary(values, focal)))
        data_rows.extend(dict(meta, dataset=ds, value_pp=100*value) for ds,value in values.items())
    for protocol, records in protocols.items():
        for cp in E.CPS:
            for learner in ("all", "ridge", "knn"):
                for field in ("F", "T", "B"):
                    add(dict(protocol=protocol, estimand="score_level", learner=learner, algorithm="all",
                             low_budget=cp, high_budget=cp, field=field), level_values(records, cp, field, learner))
        for lo,hi in ((250,1000),(1000,4000),(250,4000)):
            for learner in ("all","ridge","knn"):
                for field in ("F","T","B"):
                    focal = (protocol,lo,hi,learner,field)==("inner_fold",250,4000,"all","B")
                    add(dict(protocol=protocol, estimand="budget_increment", learner=learner, algorithm="all",
                             low_budget=lo, high_budget=hi, field=field), budget_values(records,lo,hi,field,learner), focal)
        for learner in ("ridge","knn"):
            for algo in ("ga","bpso","bde","nsga2"):
                for cp in E.CPS:
                    for field in ("F","T","B"):
                        focal=(protocol=="inner_fold" and cp==4000 and field=="B")
                        add(dict(protocol=protocol, estimand="algorithm_minus_rs", learner=learner, algorithm=algo,
                                 low_budget=cp, high_budget=cp, field=field), advantage_values(records,learner,algo,cp,field),focal)
    return summaries,data_rows


def search_properties(strict):
    rows = []
    for learner in ("ridge", "knn"):
        for algo in ("ga", "bpso", "bde", "nsga2", "rs"):
            for cp in E.CPS:
                masks = level_values(strict, cp, "distinct", learner, algo)
                features = level_values(strict, cp, "k", learner, algo)
                rows.append(dict(learner=learner, algorithm=algo, budget=cp, n_datasets=len(masks),
                                 mean_distinct_masks=float(np.mean(list(masks.values()))),
                                 mean_selected_features=float(np.mean(list(features.values())))))
    return rows


def plot_results(summary, dataset_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    focal = [r for r in summary if r["focal"]]
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 9), constrained_layout=True)
    for ax, row in zip(axes.flat, focal):
        matching = [r for r in dataset_rows if all(r[k] == row[k] for k in
                    ("protocol", "estimand", "learner", "algorithm", "low_budget", "high_budget", "field"))]
        values = np.array([r["value_pp"] for r in matching])
        # Fixed, deterministic offsets; the plot does not consume search RNGs.
        offsets = np.linspace(-.1, .1, len(values))
        ax.scatter(values, .3 + offsets, s=22, color="#637b89", alpha=.85, label="Dataset mean")
        ax.plot([row["simultaneous_lo_pp"], row["simultaneous_hi_pp"]], [0, 0], lw=1.3, color="#172d42")
        ax.plot([row["ci95_lo_pp"], row["ci95_hi_pp"]], [0, 0], lw=4, color="#177e89")
        ax.scatter([row["mean_pp"]], [0], color="#172d42", s=40, marker="D", zorder=3)
        ax.axvline(0, color=".55", lw=.8, ls="--")
        title = "Budget-gap increase: 250 to 4000" if row["estimand"] == "budget_increment" else row["learner"] + " / " + row["algorithm"] + " vs RS"
        ax.set_title(title, fontsize=10)
        ax.set_yticks([]); ax.set_ylim(-.18, .55)
        ax.set_xlabel("BA percentage points")
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.text(.02, .95, "N = %d datasets" % row["n_datasets"], transform=ax.transAxes, va="top", fontsize=8)
    fig.suptitle("Strict inner scaling: fixed nine-estimate family\nDots: dataset means; thick bars: 95% intervals; thin bars: Bonferroni family intervals", fontsize=12)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / ("focal_estimates." + ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for ax, field, label in zip(axes, ("F", "T"), ("Inner-CV BA", "Outer-test BA")):
        for protocol, name, color in (("shared", "Shared inner scaling", "#707070"), ("inner_fold", "Strict inner scaling", "#177e89")):
            rows = sorted([r for r in summary if r["protocol"] == protocol and r["estimand"] == "score_level" and
                           r["learner"] == "all" and r["field"] == field], key=lambda r: r["high_budget"])
            ax.plot([r["high_budget"] for r in rows], [r["mean_pp"] for r in rows], "o-", color=color, label=name)
        ax.set_xscale("log", base=4); ax.set_xticks(E.CPS); ax.set_xticklabels([str(c) for c in E.CPS])
        ax.set_xlabel("Wrapper requests"); ax.set_ylabel(label + " (%)")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / ("budget_curves." + ext), dpi=180)
    plt.close(fig)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path,"w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def check_original(shared):
    old_budget = list(csv.DictReader((ROOT/"results/tables/Q2b_honest_vs_budget.csv").open(encoding="utf-8-sig")))
    for field, original_field in (("F","fit"),("T","test_ba"),("B","gap_B")):
        row = next(r for r in old_budget if r["breakdown"]=="all" and r["outcome"]==original_field and
                   int(r["cp_lo"])==250 and int(r["cp_hi"])==4000)
        measured=t_summary(budget_values(shared,250,4000,field))
        for key, old_key in (("mean_pp","mean_diff"),("ci95_lo_pp","lo"),("ci95_hi_pp","hi")):
            if abs(measured[key]-100*float(row[old_key]))>1e-10:
                raise RuntimeError("Original budget aggregation mismatch")
    ga=t_summary(advantage_values(shared,"ridge","ga",4000,"B"))
    if abs(ga["mean_pp"]-6.053121970477)>1e-8:
        raise RuntimeError("Original GA/ridge headline mismatch")


def write_report(summary, validation):
    focal=[r for r in summary if r["focal"]]
    lines=["# R22 strict-inner-scaling follow-up", "", "Complete, fixed-design sensitivity experiment on the original datasets; not independent confirmation.", "",
           "All values below are BA percentage points. Dataset-equal summaries; no new p-values or equivalence tests.", "",
           "| Estimand | N datasets | Mean | Unadjusted 95% CI | Simultaneous family interval |", "|---|---:|---:|---|---|"]
    for row in focal:
        label="Budget-gap increase, 250 to 4000" if row["estimand"]=="budget_increment" else row["learner"]+" / "+row["algorithm"]+" vs RS, 4000"
        lines.append("| %s | %d | %.3f | [%.3f, %.3f] | [%.3f, %.3f] |"%(label,row["n_datasets"],row["mean_pp"],row["ci95_lo_pp"],row["ci95_hi_pp"],row["simultaneous_lo_pp"],row["simultaneous_hi_pp"]))
    lines += ["", "The simultaneous intervals use Bonferroni adjustment for the nine estimates fixed before this suite ran.",
              "Training-size and sampling differences remain in F minus T. Changing scaling may change the selected subset and search path.",
              "Coverage ends at 4000 requests and uses ordinary RS; no claim about longer strict-scaling budgets or alternative strict-scaling nulls follows.",
              "", "Validation: %d complete new jobs, %d paired checkpoints; %d overlapping X1 records reproduced exactly."%(validation["jobs"],validation["paired_checkpoints"],validation["x1_replay_jobs"])]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--check-original",action="store_true");args=parser.parse_args()
    jobs=E.job_specs()
    shared=load_shared(jobs);check_original(shared)
    if args.check_original:
        print("Original paired-budget means/intervals and GA-ridge headline reproduced; no output files written")
        return
    freeze=E.frozen(verify_inputs=True)
    strict,provenance,replays,execution=load_strict(jobs)
    if set(strict)!=set(shared):raise RuntimeError("Protocol coverage differs")
    changes={key:{field:strict[key][field]-shared[key][field] for field in ("F","T","B","k")} for key in shared}
    summary,ds_rows=make_summaries({"shared":shared,"inner_fold":strict,"inner_minus_shared":changes})
    if sum(r["focal"] for r in summary)!=FAMILY:raise RuntimeError("Focal family changed")
    paired=[]
    for key in sorted(shared):
        ds,learner,algo,rep,fold,cp=key
        row=dict(dataset=ds,learner=learner,algorithm=algo,repeat=rep,fold=fold,budget=cp)
        for protocol,records in (("shared",shared),("inner_fold",strict),("change",changes)):
            row.update({protocol+"_"+field:value for field,value in records[key].items()})
        paired.append(row)
    OUT.mkdir(parents=True,exist_ok=True)
    write_csv(OUT/"summary.csv",summary);write_csv(OUT/"dataset_estimates.csv",ds_rows);write_csv(OUT/"paired_checkpoints.csv",paired)
    write_csv(OUT/"search_properties.csv",search_properties(strict));write_csv(OUT/"execution.csv",execution)
    plot_results(summary,ds_rows)
    E.atomic_json(OUT/"job_hashes.json",provenance)
    validation=dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=len(jobs),paired_checkpoints=len(paired),
                    x1_replay_jobs=replays,focal_estimates=FAMILY,source_fingerprint=freeze["source_fingerprint"],
                    freeze_sha256=E.sha_file(E.FREEZE),all_original_comparators_unchanged=True,
                    original_summary_reproduced=True,split_indices_reconstructed_and_checked=True,
                    no_partial_results=True,new_p_values=0,new_equivalence_tests=0)
    write_report(summary,validation)
    validation["outputs"]={p.name:E.sha_file(p) for p in sorted(OUT.iterdir()) if p.suffix in (".csv",".md",".pdf",".png")}
    E.atomic_json(OUT/"validation.json",validation)
    print(json.dumps(validation,indent=2))


if __name__=="__main__":main()
