"""Exploratory X12m same-size RS follow-up on the eight known datasets.

``make_summaries(records, datasets)`` is a pure, synthetic-testable aggregation
API. Its input is an iterable of 800 records containing ``spec``, ``result`` and
the frozen ``algorithm``/``ordinary_rs`` comparator payloads. It never opens a
file. ``analyse(suite=None, output=None)`` additionally verifies the freeze and
every record through ex_suites8, checks split/backend/coverage consistency, and
publishes the complete analysis atomically. No fitting is performed here.

F = 1-inner_err, T = test_ba, B = F-T. For each configuration, the four-estimate
family contains S = B_algorithm-B_same_size_RS and D = S-O, where the descriptive
ordinary-reference baseline is O = B_algorithm-B_ordinary_RS. Thus D is exactly
B_ordinary_RS-B_same_size_RS and contains no algorithm component.
"""
import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/new_data_samesize"
SUITE = "X12m"
KIND = "strict_samesize_rs"
CONFIGS = (("ga", "ridge"), ("nsga2", "knn"))
BUDGET = 4000
N_DATASETS, N_REPEATS, N_FOLDS, N_JOBS, FAMILY = 8, 10, 5, 800, 4
ROLES = ("algorithm", "ordinary_rs", "same_size_rs")
FIELDS = ("F", "T", "B")
SPEC_FIELDS = {"suite", "kind", "ds", "algo", "learner", "rep", "fold", "budget"}
CSV_TABLES = (
    "per_run_components", "dataset_effects", "family_summary",
    "component_levels", "dataset_component_levels",
    "paired_advantages", "dataset_paired_advantages",
    "ordinary_reference", "dataset_ordinary_reference",
)
OUTPUT_FILES = {name + ".csv" for name in CSV_TABLES} | {
    "execution.csv", "job_hashes.json", "comparator_hashes.json",
    "family_summary.json", "REPORT.md",
}


def _sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def job_key(spec):
    return spec["ds"], spec["algo"], spec["learner"], spec["rep"], spec["fold"]


def validate_design(jobs, datasets):
    """Require exactly eight datasets, two fixed configurations and 50 runs/cell."""
    datasets = tuple(datasets)
    if len(datasets) != N_DATASETS or len(set(datasets)) != N_DATASETS:
        raise RuntimeError("Exactly eight distinct datasets are required")
    expected = {(ds, algo, learner, rep, fold) for ds in datasets for algo, learner in CONFIGS
                for rep in range(N_REPEATS) for fold in range(N_FOLDS)}
    keys = []
    for spec in jobs:
        if not isinstance(spec, dict) or set(spec) != SPEC_FIELDS:
            raise RuntimeError("Changed job specification fields")
        if spec["suite"] != SUITE or spec["kind"] != KIND or spec["budget"] != BUDGET:
            raise RuntimeError("Wrong suite, kind or checkpoint budget")
        if any(isinstance(spec[f], bool) or not isinstance(spec[f], int) for f in ("rep", "fold", "budget")):
            raise RuntimeError("Repeat, fold and budget must be integers")
        keys.append(job_key(spec))
    if len(keys) != N_JOBS or len(set(keys)) != N_JOBS or set(keys) != expected:
        raise RuntimeError("Complete exact 800-job coverage is required; duplicate or changed configuration")
    return expected


def score_record(row):
    """Parse a single 4000-request score row; return proportion-scale F/T/B."""
    if not isinstance(row, dict):
        raise RuntimeError("Invalid checkpoint row")
    for field in ("inner_err", "test_ba"):
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise RuntimeError("Invalid " + field)
    for field in ("calls", "distinct", "k"):
        if isinstance(row.get(field), bool) or not isinstance(row.get(field), int):
            raise RuntimeError("Noninteger " + field)
    if row["calls"] != BUDGET or not 1 <= row["distinct"] <= BUDGET or not 1 <= row["k"] <= 100:
        raise RuntimeError("Invalid request count, distinct count or selected size")
    selected = row.get("sel")
    if (not isinstance(selected, list) or len(selected) != row["k"] or
            any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in selected) or
            len(set(selected)) != row["k"]):
        raise RuntimeError("Invalid selected predictor indices")
    for field in ("test_auc", "test_acc", "test_gm", "test_f1", "test_tpr", "test_tnr", "test_mcc"):
        if field in row:
            value = row[field]
            low = -1 if field == "test_mcc" else 0
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= 1:
                raise RuntimeError("Invalid " + field)
    f, t = 1.0-row["inner_err"], row["test_ba"]
    return dict(F=f, T=t, B=f-t, k=row["k"], calls=row["calls"], distinct=row["distinct"])


def _scores(record):
    if record.get("status") != "ok" or set(record.get("result", {})) != {str(BUDGET)}:
        raise RuntimeError("Wrong status or incomplete/unexpected checkpoint set")
    comparators = record.get("comparators", {})
    if set(comparators) != {"algorithm", "ordinary_rs"}:
        raise RuntimeError("Both frozen comparator payloads are required")
    scores = {"same_size_rs": score_record(record["result"][str(BUDGET)])}
    for role in ("algorithm", "ordinary_rs"):
        payload = comparators[role]
        if not isinstance(payload, dict) or set(payload) != {"path", "sha256", "result"}:
            raise RuntimeError("Changed comparator payload schema")
        path = payload["path"]
        if (not isinstance(path, str) or "\\" in path or not path or
                PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts or
                ":" in path or not _valid_sha(payload["sha256"])):
            raise RuntimeError("Invalid comparator path or SHA-256")
        scores[role] = score_record(payload["result"])
    if scores["same_size_rs"]["k"] != scores["algorithm"]["k"]:
        raise RuntimeError("Same-size RS does not match algorithm selected size")
    return scores


def t_summary(dataset_values, family=False):
    """Equal-dataset t interval in BA percentage points, always n=8 and df=7."""
    values = np.asarray(list(dataset_values.values()), dtype=float) * 100.0
    if len(values) != N_DATASETS or not np.isfinite(values).all():
        raise RuntimeError("Inference requires exactly eight finite dataset means")
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    se = sd / math.sqrt(N_DATASETS)
    half = float(stats.t.ppf(.975, 7)) * se
    result = dict(n_datasets=8, df=7, mean_pp=mean, sd_pp=sd, se_pp=se,
                  ci95_lo_pp=mean-half, ci95_hi_pp=mean+half,
                  median_pp=float(np.median(values)), min_pp=float(values.min()),
                  max_pp=float(values.max()), n_positive=int((values > 0).sum()))
    if family:
        family_half = float(stats.t.ppf(.99375, 7)) * se
        result.update(family_n=FAMILY, family_confidence=.9875, family_quantile=.99375,
                      family_lo_pp=mean-family_half, family_hi_pp=mean+family_half)
    return result


def make_summaries(records, datasets):
    """Pure aggregation of full record-shaped synthetic or validated inputs.

    This checks exact design, payload scores, same-size matching and algebraic
    identities. Disk provenance and GPU execution are checked by ``analyse``.
    All summary/effect values use BA percentage points. Per-run component F/T/B
    columns use proportions; explicit *_pp columns give the paired contrasts.
    """
    records, datasets = list(records), tuple(datasets)
    validate_design([record["spec"] for record in records], datasets)
    parsed = {job_key(record["spec"]): _scores(record) for record in records}
    runs, effects = [], defaultdict(list)
    levels, advantages, ordinary = defaultdict(list), defaultdict(list), defaultdict(list)
    for key in sorted(parsed):
        ds, algo, learner, rep, fold = key
        score = parsed[key]
        row = dict(dataset=ds, algorithm=algo, learner=learner, repeat=rep, fold=fold, budget=BUDGET)
        for role in ROLES:
            for field in FIELDS:
                row[role+"_"+field] = score[role][field]
                levels[(algo, learner, role, field, ds)].append(score[role][field])
            for field in ("k", "calls", "distinct"):
                row[role+"_"+field] = score[role][field]
        for left, right in (("algorithm", "same_size_rs"), ("algorithm", "ordinary_rs"),
                            ("ordinary_rs", "same_size_rs")):
            for field in FIELDS:
                value = score[left][field]-score[right][field]
                advantages[(algo, learner, left, right, field, ds)].append(value)
                row[left+"_minus_"+right+"_"+field+"_pp"] = 100*value
                if left == "algorithm" and right == "ordinary_rs":
                    ordinary[(algo, learner, field, ds)].append(value)
        s = ((score["algorithm"]["F"]-score["same_size_rs"]["F"]) -
             (score["algorithm"]["T"]-score["same_size_rs"]["T"]))
        o = ((score["algorithm"]["F"]-score["ordinary_rs"]["F"]) -
             (score["algorithm"]["T"]-score["ordinary_rs"]["T"]))
        d, cancelled = s-o, score["ordinary_rs"]["B"]-score["same_size_rs"]["B"]
        if not (math.isclose(s, score["algorithm"]["B"]-score["same_size_rs"]["B"], abs_tol=1e-12, rel_tol=0)
                and math.isclose(d, cancelled, abs_tol=1e-12, rel_tol=0)):
            raise RuntimeError("Contrast identity or algorithm cancellation failed")
        row.update(S_pp=100*s, O_pp=100*o, D_pp=100*d,
                   D_cancelled_pp=100*cancelled, cancellation_residual_pp=100*(d-cancelled))
        runs.append(row)
        effects[(algo, learner, "S", ds)].append(s)
        effects[(algo, learner, "D", ds)].append(d)

    def mean_for(group, prefix):
        result = {}
        for ds in datasets:
            values = group[prefix+(ds,)]
            if len(values) != N_REPEATS*N_FOLDS:
                raise RuntimeError("Each dataset/configuration requires exactly 50 paired runs")
            result[ds] = float(np.mean(values))
        return result

    output = {name: [] for name in CSV_TABLES}
    output["per_run_components"] = runs
    for algo, learner in CONFIGS:
        meta = dict(algorithm=algo, learner=learner, budget=BUDGET)
        for estimand in ("S", "D"):
            values = mean_for(effects, (algo, learner, estimand))
            definition = "B_algorithm-B_same_size_RS" if estimand == "S" else "S-O=B_ordinary_RS-B_same_size_RS"
            label = dict(meta, contrast=estimand+"_"+algo+"_"+learner,
                         estimand=estimand, definition=definition, status="exploratory_four_estimate_family")
            output["family_summary"].append(dict(label, **t_summary(values, family=True)))
            output["dataset_effects"].extend(dict(label, dataset=ds, runs=50, value_pp=100*value)
                                              for ds, value in values.items())
        for role in ROLES:
            for field in FIELDS:
                values = mean_for(levels, (algo, learner, role, field))
                label = dict(meta, role=role, field=field, status="descriptive")
                output["component_levels"].append(dict(label, **t_summary(values)))
                output["dataset_component_levels"].extend(dict(label, dataset=ds, runs=50, value_pp=100*value)
                                                           for ds, value in values.items())
        for left, right in (("algorithm", "same_size_rs"), ("algorithm", "ordinary_rs"),
                            ("ordinary_rs", "same_size_rs")):
            for field in FIELDS:
                values = mean_for(advantages, (algo, learner, left, right, field))
                label = dict(meta, left=left, reference=right, field=field, status="descriptive")
                output["paired_advantages"].append(dict(label, **t_summary(values)))
                output["dataset_paired_advantages"].extend(dict(label, dataset=ds, runs=50, value_pp=100*value)
                                                            for ds, value in values.items())
        for field in FIELDS:
            values = mean_for(ordinary, (algo, learner, field))
            label = dict(meta, field=field, estimand="O" if field == "B" else field+"_advantage",
                         definition="algorithm-minus-ordinary-RS", status="descriptive_ordinary_reference")
            output["ordinary_reference"].append(dict(label, **t_summary(values)))
            output["dataset_ordinary_reference"].extend(dict(label, dataset=ds, runs=50, value_pp=100*value)
                                                        for ds, value in values.items())
    if len(output["family_summary"]) != FAMILY or len(runs) != N_JOBS:
        raise RuntimeError("Fixed four-estimate family changed")
    return output


def _indices(values, label):
    if (not isinstance(values, list) or not values or
            any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values) or
            len(set(values)) != len(values)):
        raise RuntimeError("Invalid " + label + " indices")
    return set(values)


def validate_splits(splits):
    train, test = _indices(splits["outer_train"], "outer train"), _indices(splits["outer_test"], "outer test")
    if train & test or train | test != set(range(len(train)+len(test))):
        raise RuntimeError("Outer split overlap or incomplete partition")
    inner, seen = splits["inner_relative_to_outer_train"], []
    if len(inner) != N_FOLDS:
        raise RuntimeError("Inner fold count changed")
    for fold in inner:
        fit, val = _indices(fold["train"], "inner train"), _indices(fold["validation"], "inner validation")
        if fit & val or fit | val != set(range(len(train))):
            raise RuntimeError("Inner split overlap or incomplete partition")
        seen.extend(fold["validation"])
    if sorted(seen) != list(range(len(train))):
        raise RuntimeError("Inner validation rows do not form an OOF partition")
    return len(train)+len(test)


def load_complete(suite, jobs, datasets, freeze):
    """Validate every input before returning records; never write outcomes."""
    validate_design(jobs, datasets)
    root = Path(suite.ROOT).resolve()
    paths = [Path(suite.job_path(spec)).resolve() for spec in jobs]
    if len(set(paths)) != N_JOBS:
        raise RuntimeError("Job path collision")
    for path in paths:
        try:
            path.relative_to(root)
        except ValueError:
            raise RuntimeError("Job path escapes suite root")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("Incomplete suite: %d of 800 files missing; first %s" % (len(missing), missing[0]))
    discovered = {p.resolve() for p in Path(suite.OUTPUT).glob("*.json")}
    if discovered != set(paths):
        raise RuntimeError("Unexpected or missing records in the exact 800-job suite")
    records, job_hashes, comparator_hashes, execution = [], {}, {}, []
    outer, lengths = {}, {}
    for spec, path in zip(jobs, paths):
        raw = path.read_bytes()
        record = json.loads(raw.decode("utf-8"))
        suite.validate_record(record, spec)  # Full frozen comparator, source, split and GPU provenance.
        if record.get("spec") != spec:
            raise RuntimeError("Wrong job specification")
        _scores(record)
        if record.get("freeze_sha256") != _sha_file(suite.FREEZE) or record.get("source_fingerprint") != freeze["source_fingerprint"]:
            raise RuntimeError("Wrong freeze/source provenance")
        if not isinstance(record.get("gpu_backend"), dict) or record["gpu_backend"] != freeze["gpu_environment"]:
            raise RuntimeError("GPU backend provenance changed")
        diag = record.get("gpu_diagnostics", {})
        operation = "cuda_ridge_fits" if spec["learner"] == "ridge" else "cuda_knn_distance_matrices"
        if (diag.get("device") != "cuda:0" or diag.get("learner_dtype") != "float64" or
                any(isinstance(diag.get(k), bool) or not isinstance(diag.get(k), int) or diag[k] <= 0
                    for k in ("cuda_dispatches", operation))):
            raise RuntimeError("Expected actual CUDA learner execution is absent")
        splits = record["splits"]
        if record.get("split_sha256") != _json_sha(splits):
            raise RuntimeError("Split hash mismatch")
        n, ds = validate_splits(splits), spec["ds"]
        if ds in lengths and lengths[ds] != n:
            raise RuntimeError("Dataset row count changes between records")
        lengths[ds] = n
        key = ds, spec["rep"], spec["fold"]
        part = splits["outer_train"], splits["outer_test"]
        if key in outer and outer[key] != part:
            raise RuntimeError("Outer splits differ between configurations")
        outer[key] = part
        for payload in record["comparators"].values():
            name, digest = payload["path"], payload["sha256"]
            if name in comparator_hashes and comparator_hashes[name] != digest:
                raise RuntimeError("Comparator hash changes between records")
            comparator_hashes[name] = digest
        name = path.relative_to(root).as_posix()
        job_hashes[name] = hashlib.sha256(raw).hexdigest()
        seconds = record.get("seconds")
        if seconds is not None and (isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0):
            raise RuntimeError("Invalid elapsed seconds")
        execution.append(dict(dataset=ds, algorithm=spec["algo"], learner=spec["learner"],
                              repeat=spec["rep"], fold=spec["fold"], budget=BUDGET, seconds=seconds,
                              started_utc=record["started_utc"], completed_utc=record["completed_utc"],
                              split_sha256=record["split_sha256"],
                              gpu_backend=json.dumps(record["gpu_backend"], sort_keys=True),
                              gpu_diagnostics=json.dumps(diag, sort_keys=True)))
        records.append(record)
    for ds in datasets:
        for rep in range(N_REPEATS):
            tests = [i for fold in range(N_FOLDS) for i in outer[(ds, rep, fold)][1]]
            if sorted(tests) != list(range(lengths[ds])):
                raise RuntimeError("Outer test rows do not form one OOF partition")
    return records, job_hashes, comparator_hashes, execution, lengths


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def report_text(family, validation):
    lines = ["# X12m same-size RS exploratory follow-up", "",
             "This is an exploratory follow-up on eight known datasets whose existing GPU algorithm and ordinary RS outcomes were already available. It is not independent confirmation or publicly verified preregistration.", "",
             "F=1-inner_err, T=test_ba, B=F-T. Each configuration uses 4000 requests. Same-size RS uses the source algorithm's selected feature count on the same inner/outer split.",
             "S=(F_algorithm-F_same_RS)-(T_algorithm-T_same_RS)=B_algorithm-B_same_RS.",
             "O=(F_algorithm-F_ordinary_RS)-(T_algorithm-T_ordinary_RS)=B_algorithm-B_ordinary_RS is the descriptive ordinary-reference baseline.",
             "D=S-O=B_ordinary_RS-B_same_RS. The algorithm cancels exactly; D is not a second algorithm effect.", "",
             "The four-estimate family is S and D for GA-ridge and NSGA-II-5-NN at budget 4000. Each dataset first averages its 50 paired repeat/fold runs; all eight dataset means then receive equal weight. Inference uses n=8 and df=7, never n=800.",
             "Ordinary intervals use t(.975,7). Bonferroni family intervals have individual confidence 98.75%, using t(.99375,7), for a four-estimate 95% family. Repeated folds are not independent inferential units.", "",
             "| Contrast | Mean (BA pp) | Ordinary 95% CI | Four-estimate family CI |",
             "|---|---:|---|---|"]
    for row in family:
        lines.append("| %s | %.4f | [%.4f, %.4f] | [%.4f, %.4f] |" %
                     (row["contrast"], row["mean_pp"], row["ci95_lo_pp"], row["ci95_hi_pp"],
                      row["family_lo_pp"], row["family_hi_pp"]))
    lines += ["", "The ordinary-reference baseline and its F/T/B advantages, all three roles' F/T/B levels, and paired component advantages are descriptive and carry ordinary 95% intervals only. They are outside the four-estimate family.",
              "No p-values, equivalence tests or comparisons of significance statuses are made. An interval containing zero does not establish absence or equivalence. Compare S and O through the paired D estimate, not through whether their separate intervals cross zero.",
              "Dataset-level t intervals assume the eight dataset means supply an adequate independent unit approximation. This small, selected, known collection limits generalization; the report does not claim representative sampling of feature-selection tasks.",
              "F-T includes training-size and sampling differences. Positive B or S does not determine the sign of the outer-performance advantage or establish a causal search effect or a rank reversal.", "",
              "Coverage: %d complete new same-size jobs; eight datasets; two configurations; ten repeats x five folds; budget 4000. Frozen existing comparator rows are reused verbatim; no existing algorithm or ordinary RS outcomes are refitted or pooled as extra datasets." % validation["jobs"],
              "Every new record passed frozen source/comparator/GPU/split provenance checks, complete inner and outer OOF checks, and the D cancellation identity. No partial analysis was published."]
    return "\n".join(lines)+"\n"


def _existing(output, validation, job_hashes, comparator_hashes):
    receipt = output / "validation.json"
    if not receipt.is_file() or output.is_symlink():
        raise RuntimeError("Existing analysis directory is incomplete; no overwrite")
    old = json.loads(receipt.read_text(encoding="utf-8"))
    if any(old.get(key) != value for key, value in validation.items() if key != "created_utc"):
        raise RuntimeError("Existing analysis provenance differs; no overwrite")
    if set(old.get("outputs", {})) != OUTPUT_FILES or {p.name for p in output.iterdir()} != OUTPUT_FILES | {"validation.json"}:
        raise RuntimeError("Existing analysis output inventory failed")
    for name, digest in old["outputs"].items():
        path = output / name
        if path.is_symlink() or not path.is_file() or not _valid_sha(digest) or _sha_file(path) != digest:
            raise RuntimeError("Existing analysis output integrity failed")
    for name, expected in (("job_hashes.json", job_hashes), ("comparator_hashes.json", comparator_hashes)):
        if json.loads((output / name).read_text(encoding="utf-8")) != expected:
            raise RuntimeError("Existing analysis input hashes differ")
    return old


def analyse(suite=None, output=None):
    """Publish only the fully validated 800-job X12m suite; reuse intact output."""
    if suite is None:
        sys.path.insert(0, str(ROOT / "extras"))
        suite = importlib.import_module("ex_suites8")
    output = Path(OUT if output is None else output)
    if suite.SUITE != SUITE or tuple(map(tuple, suite.CONFIGS)) != CONFIGS or suite.BUDGET != BUDGET:
        raise RuntimeError("Suite definition differs from the fixed follow-up analysis")
    datasets, jobs = tuple(suite.DATASETS), list(suite.job_specs())
    validate_design(jobs, datasets)
    if not Path(suite.FREEZE).is_file():
        raise RuntimeError("Frozen protocol is required before analysis")
    freeze = suite.frozen(verify_inputs=True)
    if freeze.get("jobs") != jobs or not _valid_sha(freeze.get("source_fingerprint")):
        raise RuntimeError("Frozen job list or source fingerprint differs")
    records, job_hashes, comparator_hashes, execution, lengths = load_complete(suite, jobs, datasets, freeze)
    summaries = make_summaries(records, datasets)
    validation = dict(created_utc=datetime.now(timezone.utc).isoformat(), suite=SUITE,
                      jobs=N_JOBS, checkpoints=N_JOBS, datasets=list(datasets), row_counts=lengths,
                      configs=[list(config) for config in CONFIGS], budget=BUDGET,
                      repeats=N_REPEATS, outer_folds=N_FOLDS, inner_folds=N_FOLDS,
                      family_n=FAMILY, family_confidence=.9875, family_quantile=.99375,
                      n_inference_units=8, df=7, inference_unit="dataset", dataset_weighting="equal",
                      status="exploratory_known_eight_datasets", independent_confirmation=False,
                      public_preregistration_claim=False, no_partial_results=True,
                      frozen_comparators_reused_verbatim=True, outer_and_inner_oof_coverage_checked=True,
                      algorithm_cancellation_checked=True, new_p_values=0, equivalence_tests=0,
                      significance_status_comparisons=0, source_fingerprint=freeze["source_fingerprint"],
                      freeze_sha256=_sha_file(suite.FREEZE), gpu_environment=freeze["gpu_environment"])
    if output.exists():
        return _existing(output, validation, job_hashes, comparator_hashes)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="."+output.name+".staging.", dir=str(output.parent)))
    # Failed staging directories remain available for diagnostics; no partial
    # final directory is published and an existing destination is never removed.
    for name, rows in summaries.items():
        _write_csv(stage / (name+".csv"), rows)
    _write_csv(stage / "execution.csv", execution)
    _write_json(stage / "job_hashes.json", job_hashes)
    _write_json(stage / "comparator_hashes.json", comparator_hashes)
    _write_json(stage / "family_summary.json", summaries["family_summary"])
    (stage / "REPORT.md").write_text(report_text(summaries["family_summary"], validation), encoding="utf-8")
    validation["outputs"] = {name: _sha_file(stage / name) for name in sorted(OUTPUT_FILES)}
    _write_json(stage / "validation.json", validation)
    os.replace(str(stage), str(output))
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyse", action="store_true", required=True,
                        help="Analyse only the complete frozen 800-job same-size suite")
    parser.parse_args()
    print(json.dumps(analyse(), indent=2))


if __name__ == "__main__":
    main()
