"""Revision-2 exact distinct-budget comparison: X6d GA/NSGA-II versus X6r RS.

Only X6r_* tables are written. Old equal-request comparisons and the former
distinct-algorithm/request-RS comparison are retained as explicit controls.
The data set is the inference unit; primary intervals are Student-t intervals,
with dataset-level Wilcoxon and Holm within learner/target/comparison families.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "extras"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import ex_suites5 as E5  # noqa: E402
import statsx as SX  # noqa: E402

TAB = os.path.join(ROOT, "results", "tables")
KEY = ["ds", "learner", "rep", "fold"]


def _write_csv(frame, name):
    os.makedirs(TAB, exist_ok=True)
    path = os.path.join(TAB, name)
    tmp = path + ".tmp"
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def _write_json(value, name):
    os.makedirs(TAB, exist_ok=True)
    path = os.path.join(TAB, name)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=1, allow_nan=False)
    os.replace(path + ".tmp", path)


def load_main():
    rows = []
    for spec in E5.ST.E1_honest():
        if spec["algo"] not in ("ga", "nsga2", "rs"):
            continue
        with open(E5.ST.job_file(spec), encoding="utf-8") as f:
            rec = json.load(f)
        if rec["status"] != "ok" or rec["spec"] != spec:
            raise RuntimeError("Invalid main reference spec/status")
        for cp in E5.DISTINCT_CPS:
            r = rec["result"][str(cp)]
            rows.append(dict(ds=spec["ds"], learner=spec["learner"], algo=spec["algo"], rep=spec["rep"], fold=spec["fold"],
                             target=cp, test_ba=r["test_ba"], k=r["k"]))
    return pd.DataFrame(rows)


def load_distinct_algorithms():
    rows = []
    expected = [s for s in E5.ST.E1_honest() if s["algo"] in ("ga", "nsga2")]
    paths = glob.glob(os.path.join(ROOT, "results", "jobs_extra", "X6d", "*.json"))
    if len(paths) != len(expected):
        raise RuntimeError("X6d is incomplete or contains extra jobs")
    wanted = {(s["ds"], s["learner"], s["algo"], s["rep"], s["fold"]) for s in expected}
    seen = set()
    for path in sorted(paths):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        s = rec["spec"]
        key = (s["ds"], s["learner"], s["algo"], s["rep"], s["fold"])
        if rec["status"] != "ok" or key not in wanted or key in seen or s["dcps"] != list(E5.DISTINCT_CPS):
            raise RuntimeError("X6d spec/status/uniqueness check failed")
        seen.add(key)
        for cp in E5.DISTINCT_CPS:
            r = rec["result"][str(cp)]
            if r["reached"] is not True or r["distinct"] != cp or r["calls"] < cp:
                raise RuntimeError("X6d did not reach an exact distinct target")
            rows.append(dict(ds=s["ds"], learner=s["learner"], algo=s["algo"], rep=s["rep"], fold=s["fold"], target=cp,
                             test_ba=r["test_ba"], k=r["k"], calls=r["calls"], distinct=r["distinct"], reached=r["reached"]))
    if seen != wanted:
        raise RuntimeError("X6d keys do not cover every expected original outer run")
    return pd.DataFrame(rows)


def load_rs():
    specs = E5.X6_rs_distinct()
    actual = set(glob.glob(os.path.join(E5.JOBS, "X6r", "*.json")))
    expected = {E5.job_file(s) for s in specs}
    if actual != expected:
        raise RuntimeError("X6r missing %d jobs or has %d unexpected jobs" % (len(expected - actual), len(actual - expected)))
    rows = []
    check = dict(n_jobs=len(specs), n_distinct_records=0, n_replay_records=0, n_replay_numbers=0, n_different=0,
                 n_unreached=0, max_abs_diff_vs_original=0.0, code_sha=E5.CODE_SHA,
                 primary_interval="dataset-level Student-t", inference_unit="data set", comparator_mode="distinct")
    source_shas = Counter()
    for spec in specs:
        with open(E5.job_file(spec), encoding="utf-8") as f:
            rec = E5.validate_record(json.load(f), spec)
        with open(E5.ST.job_file(E5._original_spec(spec)), encoding="utf-8") as f:
            source = json.load(f)
        replay = E5._check_replay(rec["request_replay"], source)
        if rec["replay_check"] != replay:
            raise RuntimeError("Stored X6r replay metadata does not match an independent comparison")
        check["n_replay_records"] += replay["n_records"]
        check["n_replay_numbers"] += replay["n_numbers"]
        source_shas[source["code_sha"]] += 1
        for cp in E5.DISTINCT_CPS:
            r = rec["result"][str(cp)]
            old = rec["request_replay"][str(cp)]
            check["n_distinct_records"] += 1
            rows.append(dict(ds=spec["ds"], learner=spec["learner"], algo="rs", rep=spec["rep"], fold=spec["fold"], target=cp,
                             test_ba=r["test_ba"], k=r["k"], calls=r["calls"], distinct=r["distinct"], reached=r["reached"],
                             request_calls=old["calls"], request_distinct=old["distinct"], request_test_ba=old["test_ba"],
                             extra_calls=r["calls"] - cp, selected_mask_changed=r["sel"] != old["sel"], seconds=rec["seconds"]))
    check["source_code_shas"] = dict(source_shas)
    return pd.DataFrame(rows), check


def _paired(a, b, a_col="test_ba", b_col="test_ba"):
    if not a.index.is_unique or not b.index.is_unique or set(a.index) != set(b.index):
        raise RuntimeError("Comparators do not cover identical outer-run keys")
    return a[a_col] - b.reindex(a.index)[b_col]


def _summary(difference, seed, n_boot):
    per = {ds: [(g.to_numpy(dtype=float)[:, None], 1.0)] for ds, g in difference.groupby(level="ds")}
    bt = SX.double_boot(per, n_boot, seed)
    values = np.array(list(bt["ds_values"].values()))
    return dict(n_ds=len(values), mean_diff=bt["point"], lo=bt["lo"], hi=bt["hi"], lo_cb=bt["lo_cb"], hi_cb=bt["hi_cb"],
                lo_db=bt["lo_db"], hi_db=bt["hi_db"], n_pos=int((values > 0).sum()), p=SX.wilcoxon(values),
                ds_min=float(values.min()), ds_max=float(values.max())), bt["ds_values"]


def analyse(main, alg, rs, n_boot):
    rows, dataset_rows = [], []
    for lr in ("ridge", "knn"):
        for target in E5.DISTINCT_CPS:
            er = main[(main.learner == lr) & (main.algo == "rs") & (main.target == target)].set_index(KEY)
            xr = rs[(rs.learner == lr) & (rs.target == target)].set_index(KEY)
            for algo in ("ga", "nsga2"):
                xa = alg[(alg.learner == lr) & (alg.algo == algo) & (alg.target == target)].set_index(KEY)
                ea = main[(main.learner == lr) & (main.algo == algo) & (main.target == target)].set_index(KEY)
                equal_requests, _ = _summary(_paired(ea, er), 20261022, n_boot)
                legacy, _ = _summary(_paired(xa, er), 20261021, n_boot)
                for comp, comparator, mode in (("same_algo_same_requests", ea, "algorithm_requests"),
                                               ("vs_random_search", xr, "distinct")):
                    diff = _paired(xa, comparator)
                    summary, per_ds = _summary(diff, 20261021, n_boot)
                    row = dict(learner=lr, algo=algo, target=target, comparison=comp, comparator_mode=mode,
                               algorithm_budget_mode="distinct", comparator_budget_mode="distinct" if mode == "distinct" else "requests",
                               mean_calls=float(xa.calls.mean()), mean_distinct=float(xa.distinct.mean()), unreached=0,
                               rs_calls=float(xr.calls.mean()) if mode == "distinct" else np.nan,
                               rs_distinct=float(xr.distinct.mean()) if mode == "distinct" else np.nan,
                               n_pairs=len(diff), primary_interval="dataset-level Student-t", **summary,
                               **{"req_" + k: v for k, v in equal_requests.items()},
                               **({"legacy_" + k: v for k, v in legacy.items()} if mode == "distinct" else {}))
                    rows.append(row)
                    for ds, value in per_ds.items():
                        dataset_rows.append(dict(learner=lr, algo=algo, target=target, comparison=comp, comparator_mode=mode, ds=ds,
                                                 n_pairs=int((diff.index.get_level_values("ds") == ds).sum()), mean_diff=value))
    table = pd.DataFrame(rows)
    table["p_holm"] = np.nan
    for _, group in table.groupby(["learner", "target", "comparison"]):
        table.loc[group.index, "p_holm"] = SX.holm(group.p.to_numpy())
    extension = []
    for lr in ("ridge", "knn"):
        for target in E5.DISTINCT_CPS:
            xr = rs[(rs.learner == lr) & (rs.target == target)].set_index(KEY)
            summary, _ = _summary(xr.test_ba - xr.request_test_ba, 20261026, n_boot)
            extension.append(dict(learner=lr, target=target, comparison="rs_distinct_minus_requests", comparator_mode="rs_requests",
                                  mean_calls=float(xr.calls.mean()), mean_distinct=float(xr.distinct.mean()),
                                  mean_request_distinct=float(xr.request_distinct.mean()),
                                  mean_extra_calls=float(xr.extra_calls.mean()), max_extra_calls=int(xr.extra_calls.max()),
                                  n_mask_changed=int(xr.selected_mask_changed.sum()), n_ba_changed=int((xr.test_ba != xr.request_test_ba).sum()),
                                  n_runs=len(xr), **summary))
    extension = pd.DataFrame(extension)
    extension["p_holm"] = SX.holm(extension.p.to_numpy())
    return table, pd.DataFrame(dataset_rows), extension


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()
    if args.bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    old_main = load_main()
    algorithms = load_distinct_algorithms()
    rs, check = load_rs()
    table, dataset, extension = analyse(old_main, algorithms, rs, args.bootstrap)
    check["n_algorithm_jobs"] = int(len(algorithms) // len(E5.DISTINCT_CPS))
    check["n_algorithm_records"] = int(len(algorithms))
    check["bootstrap_draws_sensitivity"] = args.bootstrap
    _write_csv(rs, "X6r_runs.csv")
    _write_csv(table, "X6r_distinct_matched.csv")
    _write_csv(dataset, "X6r_by_dataset.csv")
    _write_csv(extension, "X6r_rs_extension.csv")
    _write_json(check, "X6r_check.json")
    print(json.dumps(check, indent=1), flush=True)
    fields = ["learner", "algo", "target", "comparator_mode", "mean_calls", "mean_distinct", "rs_calls", "rs_distinct",
              "n_ds", "mean_diff", "lo", "hi", "p", "p_holm", "req_mean_diff", "legacy_mean_diff"]
    print(table[table.comparison == "vs_random_search"][fields].to_string(index=False), flush=True)
    print(extension.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
