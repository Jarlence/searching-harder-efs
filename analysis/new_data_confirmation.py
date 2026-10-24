"""Fixed three-estimand new-data confirmation analysis; refuses partial coverage.

No search is performed here. Only the new suite's frozen jobs are read; original
outcomes are never pooled, read or changed. Pure aggregation functions can be
tested with synthetic scores before any new-data performance run exists.
"""
import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/new_data_confirmation"
CPS = (250, 1000, 4000)
ALGORITHMS = ("ga", "bpso", "bde", "nsga2", "rs")
LEARNERS = ("ridge", "knn")
N_DATASETS, N_REPEATS, N_FOLDS, FAMILY = 8, 10, 5, 3
PRIMARY = {
    "C1": "Budget-gap increase, 250 to 4000, five searches and two learners",
    "C2": "GA-ridge minus ordinary request-matched RS, 4000",
    "C3": "NSGA-II-5-NN minus ordinary request-matched RS, 4000",
}


def _sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def job_key(spec):
    return (spec["ds"], spec["learner"], spec["algo"], spec["rep"], spec["fold"])


def validate_design(jobs, datasets):
    """Check the exact factorial design, without loading any outcomes."""
    datasets = tuple(datasets)
    if len(datasets) != N_DATASETS or len(set(datasets)) != N_DATASETS:
        raise RuntimeError("Design must contain exactly eight distinct datasets")
    expected = {(ds, lr, algo, rep, fold) for ds in datasets for lr in LEARNERS
                for algo in ALGORITHMS for rep in range(N_REPEATS) for fold in range(N_FOLDS)}
    keys = [job_key(s) for s in jobs]
    if len(keys) != 4000 or len(keys) != len(set(keys)) or set(keys) != expected:
        raise RuntimeError("Incomplete, duplicate or changed fixed 4000-job design")
    if any(tuple(s["cps"]) != CPS for s in jobs):
        raise RuntimeError("Checkpoint design changed")
    return expected


def score_record(row, checkpoint):
    for field in ("inner_err", "test_ba"):
        value = row[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise RuntimeError("Invalid " + field)
    for field in ("calls", "distinct", "k"):
        if isinstance(row[field], bool) or not isinstance(row[field], int):
            raise RuntimeError("Noninteger " + field)
    if row["calls"] != checkpoint or not 1 <= row["distinct"] <= checkpoint or not 1 <= row["k"] <= 100:
        raise RuntimeError("Invalid request count, distinct count or selected size")
    F, T = 1.0 - row["inner_err"], row["test_ba"]
    return dict(F=F, T=T, B=F-T, k=row["k"], calls=row["calls"], distinct=row["distinct"])


def _index_list(values, label):
    if not isinstance(values, list) or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
        raise RuntimeError("Invalid indices: " + label)
    if len(values) != len(set(values)) or not values:
        raise RuntimeError("Empty or duplicate indices: " + label)
    return set(values)


def validate_splits(splits):
    tr = _index_list(splits["outer_train"], "outer training")
    te = _index_list(splits["outer_test"], "outer test")
    if tr & te or tr | te != set(range(len(tr) + len(te))):
        raise RuntimeError("Outer overlap or incomplete partition")
    inner = splits["inner_relative_to_outer_train"]
    if len(inner) != N_FOLDS:
        raise RuntimeError("Inner fold count changed")
    seen = []
    for fold in inner:
        fit = _index_list(fold["train"], "inner training")
        val = _index_list(fold["validation"], "inner validation")
        if fit & val or fit | val != set(range(len(tr))):
            raise RuntimeError("Inner overlap or incomplete partition")
        seen.extend(fold["validation"])
    if sorted(seen) != list(range(len(tr))):
        raise RuntimeError("Inner validation rows do not form one OOF partition")
    return len(tr) + len(te)


def load_complete(suite, jobs, datasets):
    """Validate every job before returning scores; no summaries are written."""
    validate_design(jobs, datasets)
    paths = [Path(suite.job_path(s)) for s in jobs]
    if len(set(paths)) != len(paths):
        raise RuntimeError("Job path collision")
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise RuntimeError("Incomplete suite: %d of 4000 files missing; first %s" % (len(missing), missing[0]))
    scores, hashes, executions = {}, {}, []
    outer, inner, lengths = {}, {}, {}
    for spec, path in zip(jobs, paths):
        raw = path.read_bytes()
        record = json.loads(raw.decode("utf-8"))
        suite.validate_record(record, spec)
        if record.get("status") != "ok" or record.get("spec") != spec:
            raise RuntimeError("Wrong job status or specification")
        if set(record["result"]) != {str(cp) for cp in CPS}:
            raise RuntimeError("Incomplete or unexpected checkpoint set")
        ds, lr, algo, rep, fold = job_key(spec)
        splits = record["splits"]
        n = validate_splits(splits)
        if ds in lengths and lengths[ds] != n:
            raise RuntimeError("Dataset row count changes between runs")
        lengths[ds] = n
        outer_key = ds, rep, fold
        outer_part = splits["outer_train"], splits["outer_test"]
        if outer_key in outer and outer[outer_key] != outer_part:
            raise RuntimeError("Outer folds differ across algorithms or learners")
        outer[outer_key] = outer_part
        inner_key = ds, lr, rep, fold
        if inner_key in inner and inner[inner_key] != splits["inner_relative_to_outer_train"]:
            raise RuntimeError("Inner folds differ between search and reference")
        inner[inner_key] = splits["inner_relative_to_outer_train"]
        previous_distinct = 0
        for cp in CPS:
            key = ds, lr, algo, rep, fold, cp
            score = score_record(record["result"][str(cp)], cp)
            if score["distinct"] < previous_distinct:
                raise RuntimeError("Distinct candidates decrease along one trajectory")
            previous_distinct = score["distinct"]
            scores[key] = score
        name = str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
        hashes[name] = hashlib.sha256(raw).hexdigest()
        seconds = record.get("seconds")
        if seconds is not None and (not math.isfinite(seconds) or seconds < 0):
            raise RuntimeError("Invalid elapsed seconds")
        executions.append(dict(dataset=ds, learner=lr, algorithm=algo, repeat=rep, fold=fold,
                               seconds=seconds, completed_utc=record.get("completed_utc", "")))
    for ds in datasets:
        for rep in range(N_REPEATS):
            tests = [i for fold in range(N_FOLDS) for i in outer[(ds, rep, fold)][1]]
            if sorted(tests) != list(range(lengths[ds])):
                raise RuntimeError("Outer test rows do not form one OOF partition")
    validate_score_coverage(scores, datasets)
    return scores, hashes, executions, lengths


def validate_score_coverage(scores, datasets):
    expected = {(ds, lr, algo, rep, fold, cp) for ds in datasets for lr in LEARNERS
                for algo in ALGORITHMS for rep in range(N_REPEATS) for fold in range(N_FOLDS) for cp in CPS}
    if len(expected) != 12000 or set(scores) != expected:
        raise RuntimeError("Complete 12000-checkpoint coverage is required")
    for value in scores.values():
        if any(not math.isfinite(value[field]) for field in ("F", "T", "B")):
            raise RuntimeError("Nonfinite checkpoint score")
        if not math.isclose(value["B"], value["F"]-value["T"], abs_tol=1e-12, rel_tol=0):
            raise RuntimeError("F-minus-T identity violated")


def t_summary(dataset_values, primary=False):
    values = np.asarray(list(dataset_values.values()), dtype=float) * 100.0
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Invalid dataset-level values")
    n, mean = len(values), float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / math.sqrt(n)
    half = float(stats.t.ppf(.975, n-1)) * se
    row = dict(n_datasets=n, df=n-1, mean_pp=mean, sd_pp=sd, se_pp=se,
               ci95_lo_pp=mean-half, ci95_hi_pp=mean+half,
               median_pp=float(np.median(values)), min_pp=float(values.min()), max_pp=float(values.max()),
               n_positive=int((values > 0).sum()))
    if primary:
        simultaneous = float(stats.t.ppf(1-.05/(2*FAMILY), n-1)) * se
        lo, hi = mean-simultaneous, mean+simultaneous
        decision = "supported_positive" if lo > 0 else "supported_opposite" if hi < 0 else "inconclusive"
        row.update(family_n=FAMILY, family_confidence=1-.05/FAMILY,
                   family_lo_pp=lo, family_hi_pp=hi, decision=decision)
    return row


def budget_values(scores, datasets, field="B", low=250, high=4000):
    # Pair checkpoints inside each run; then equal-weight the ten search/learner cells.
    if low not in CPS or high not in CPS or low >= high:
        raise ValueError("Invalid existing-checkpoint contrast")
    return {ds: float(np.mean([scores[(ds, lr, algo, rep, fold, high)][field]
                              - scores[(ds, lr, algo, rep, fold, low)][field]
                              for lr in LEARNERS for algo in ALGORITHMS
                              for rep in range(N_REPEATS) for fold in range(N_FOLDS)])) for ds in datasets}


def advantage_values(scores, datasets, learner, algorithm, field="B", checkpoint=4000):
    return {ds: float(np.mean([scores[(ds, learner, algorithm, rep, fold, checkpoint)][field]
                              - scores[(ds, learner, "rs", rep, fold, checkpoint)][field]
                              for rep in range(N_REPEATS) for fold in range(N_FOLDS)])) for ds in datasets}


def level_values(scores, datasets, checkpoint, field, learner="all", algorithm="all"):
    learners = LEARNERS if learner == "all" else (learner,)
    algorithms = ALGORITHMS if algorithm == "all" else (algorithm,)
    return {ds: float(np.mean([scores[(ds, lr, algo, rep, fold, checkpoint)][field]
                              for lr in learners for algo in algorithms
                              for rep in range(N_REPEATS) for fold in range(N_FOLDS)])) for ds in datasets}


def make_summaries(scores, datasets):
    """Three primary contrasts; remaining estimates are secondary descriptors."""
    validate_score_coverage(scores, datasets)
    contrasts = {
        "C1": budget_values(scores, datasets),
        "C2": advantage_values(scores, datasets, "ridge", "ga"),
        "C3": advantage_values(scores, datasets, "knn", "nsga2"),
    }
    primary, dataset_primary = [], []
    for contrast, values in contrasts.items():
        meta = dict(contrast=contrast, description=PRIMARY[contrast], status="primary", field="B")
        primary.append(dict(meta, **t_summary(values, primary=True)))
        dataset_primary.extend(dict(meta, dataset=ds, value_pp=100*value) for ds, value in values.items())
    increments, dataset_increments = [], []
    for low, high in ((250, 1000), (1000, 4000), (250, 4000)):
        for field in ("F", "T", "B"):
            values = budget_values(scores, datasets, field, low, high)
            meta = dict(low_budget=low, high_budget=high, learner="all", algorithm="all",
                        field=field, status="secondary_descriptive")
            increments.append(dict(meta, **t_summary(values)))
            dataset_increments.extend(dict(meta, dataset=ds, value_pp=100*value) for ds, value in values.items())
    levels, dataset_levels = [], []
    for cp in CPS:
        for lr in ("all",) + LEARNERS:
            for field in ("F", "T", "B"):
                values = level_values(scores, datasets, cp, field, lr)
                meta = dict(budget=cp, learner=lr, algorithm="all", field=field, status="secondary_descriptive")
                levels.append(dict(meta, **t_summary(values)))
                dataset_levels.extend(dict(meta, dataset=ds, value_pp=100*value) for ds, value in values.items())
    advantages, dataset_advantages = [], []
    for lr in LEARNERS:
        for algo in ALGORITHMS[:-1]:
            for field in ("F", "T", "B"):
                values = advantage_values(scores, datasets, lr, algo, field)
                meta = dict(budget=4000, learner=lr, algorithm=algo, reference="rs", field=field,
                            status="secondary_descriptive")
                advantages.append(dict(meta, **t_summary(values)))
                dataset_advantages.extend(dict(meta, dataset=ds, value_pp=100*value) for ds, value in values.items())
    return dict(primary=primary, dataset_primary=dataset_primary, budget_increments=increments,
                dataset_budget_increments=dataset_increments, budget_levels=levels,
                dataset_budget_levels=dataset_levels, advantage_components=advantages,
                dataset_advantage_components=dataset_advantages)


def _write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def report_text(primary, validation):
    decisions = [row["decision"] for row in primary]
    if all(d == "supported_positive" for d in decisions):
        conclusion = "All three prespecified positive effects are supported on this selected new benchmark collection."
    elif any(d == "supported_positive" for d in decisions):
        conclusion = "Partial support only; interpret each prespecified contrast separately."
    else:
        conclusion = "The simultaneous intervals do not support any prespecified positive effect; see each contrast's uncertainty and direction."
    lines = ["# New-data confirmation study", "", conclusion, "",
             "Same-team replication on eight previously unused project datasets. Not independent-team reproduction or representative sampling of EFS tasks.",
             "All scores are BA percentage points. Each dataset supplies one value per contrast and receives equal weight.", "",
             "| Contrast | Datasets | Mean | Ordinary 95% CI | Three-estimate family CI | Decision |",
             "|---|---:|---:|---|---|---|"]
    for row in primary:
        lines.append("| %s | %d | %.4f | [%.4f, %.4f] | [%.4f, %.4f] | %s |" %
                     (row["contrast"], row["n_datasets"], row["mean_pp"], row["ci95_lo_pp"], row["ci95_hi_pp"],
                      row["family_lo_pp"], row["family_hi_pp"], row["decision"]))
    lines += ["", "The fixed primary family is C1-C3; Bonferroni intervals each have confidence 98.333333%. No new p-value families or equivalence tests are introduced.",
              "C1 averages paired B(4000)-B(250) across all five searches and both learners; it does not isolate evolutionary operators.",
              "C2 and C3 use the ordinary request-matched RS reference at 4000 requests.",
              "A family lower bound greater than zero supports the specified positive contrast. An interval spanning zero is inconclusive, not absence or equivalence.",
              "Other algorithms' F/T/B contrasts and score levels are secondary descriptive outputs. Their unadjusted intervals carry no confirmation decision.",
              "Original and X10s data are not pooled with these results. Different significance statuses do not establish a difference between studies.",
              "F-T still contains training-size and sampling differences; results do not identify a pure causal search or leakage effect.", "",
              "Coverage: %d complete jobs, %d checkpoints, eight datasets, ten repeats x five outer folds, two learners, five searches." %
              (validation["jobs"], validation["checkpoints"]),
              "Registration status and timing are recorded by the frozen suite; this report alone is not external registration."]
    return "\n".join(lines)+"\n"


def analyse(suite):
    if tuple(suite.CPS) != CPS:
        raise RuntimeError("Suite checkpoint definition differs from analysis")
    datasets = tuple(suite.DATASETS)
    jobs = list(suite.job_specs())
    validate_design(jobs, datasets)
    if not Path(suite.FREEZE).is_file():
        raise RuntimeError("Frozen protocol is required before analysis")
    if not hasattr(suite, "frozen"):
        raise RuntimeError("Suite must expose frozen(verify_inputs=True) for source and data provenance checks")
    freeze = suite.frozen(verify_inputs=True)
    if freeze.get("jobs") != jobs:
        raise RuntimeError("Frozen job list differs from executed design")
    if not hasattr(suite, "verify_archive_receipt"):
        raise RuntimeError("Suite must verify the external archive receipt bound to this freeze")
    archive = suite.verify_archive_receipt()
    if not isinstance(archive, dict) or archive.get("freeze_sha256") != suite.sha_file(suite.FREEZE):
        raise RuntimeError("External archive receipt is not bound to the analysed freeze")
    scores, job_hashes, execution, row_counts = load_complete(suite, jobs, datasets)
    summaries = make_summaries(scores, datasets)
    if len(summaries["primary"]) != FAMILY:
        raise RuntimeError("Primary statistical family changed")
    validation = dict(created_utc=datetime.now(timezone.utc).isoformat(), jobs=len(jobs), checkpoints=len(scores),
                      datasets=list(datasets), row_counts=row_counts, repeats=N_REPEATS, outer_folds=N_FOLDS,
                      inner_folds=N_FOLDS, primary_family_n=FAMILY, family_confidence=1-.05/FAMILY,
                      source_fingerprint=freeze["source_fingerprint"], freeze_sha256=suite.sha_file(suite.FREEZE),
                      external_archive_receipt=archive,
                      no_partial_results=True, all_algorithm_learner_outer_splits_matched=True,
                      algorithm_reference_inner_splits_matched=True, outer_and_inner_oof_coverage_checked=True,
                      original_results_pooled=False, new_p_values=0, equivalence_tests=0)
    if OUT.exists():
        receipt = OUT / "validation.json"
        if not receipt.is_file() or not (OUT / "job_hashes.json").is_file():
            raise RuntimeError("Existing analysis directory is incomplete; it will not be overwritten")
        old = json.loads(receipt.read_text(encoding="utf-8"))
        if old.get("freeze_sha256") != validation["freeze_sha256"] or json.loads((OUT / "job_hashes.json").read_text(encoding="utf-8")) != job_hashes:
            raise RuntimeError("Existing analysis provenance differs; it will not be overwritten")
        if any(not (OUT / name).is_file() or _sha_file(OUT / name) != sha for name, sha in old["outputs"].items()):
            raise RuntimeError("Existing analysis output integrity failed")
        return old
    # A final directory is only published once every job and calculation passes.
    stage = OUT.with_name("."+OUT.name+".staging."+str(os.getpid()))
    stage.mkdir(parents=True, exist_ok=False)
    for name, rows in summaries.items():
        _write_csv(stage / (name+".csv"), rows)
    checkpoints = [dict(dataset=key[0], learner=key[1], algorithm=key[2], repeat=key[3], fold=key[4], budget=key[5], **score)
                   for key, score in sorted(scores.items())]
    _write_csv(stage / "checkpoints.csv", checkpoints)
    _write_csv(stage / "execution.csv", execution)
    _write_json(stage / "job_hashes.json", job_hashes)
    _write_json(stage / "primary.json", summaries["primary"])
    (stage / "REPORT.md").write_text(report_text(summaries["primary"], validation), encoding="utf-8")
    validation["outputs"] = {p.name: _sha_file(p) for p in sorted(stage.iterdir())}
    _write_json(stage / "validation.json", validation)
    os.replace(str(stage), str(OUT))
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyse", action="store_true", help="Analyse only the complete frozen 4000-job suite")
    args = parser.parse_args()
    if not args.analyse:
        parser.error("--analyse is required; no searches or partial summaries are available")
    sys.path.insert(0, str(ROOT / "extras"))
    suite = importlib.import_module("ex_suites7")
    print(json.dumps(analyse(suite), indent=2))


if __name__ == "__main__":
    main()
