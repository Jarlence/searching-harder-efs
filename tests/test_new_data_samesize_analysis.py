"""Synthetic-only X12m analysis checks; no datasets or control results are read."""
import copy
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import new_data_samesize as A

DATASETS = tuple("synthetic_%d" % i for i in range(8))
SOURCE_FINGERPRINT = "a"*64
GPU = dict(device="synthetic CUDA device", dtype="float64")


def jobs():
    return [dict(suite="X12m", kind="strict_samesize_rs", ds=ds, algo=algo,
                 learner=learner, rep=rep, fold=fold, budget=4000)
            for ds in DATASETS for algo, learner in A.CONFIGS
            for rep in range(10) for fold in range(5)]


def splits(fold):
    test = list(range(3*fold, 3*fold+3))
    train = [i for i in range(15) if i not in test]
    return dict(outer_train=train, outer_test=test,
                inner_relative_to_outer_train=[
                    dict(train=[i for i in range(12) if i % 5 != f],
                         validation=[i for i in range(12) if i % 5 == f])
                    for f in range(5)])


def score(f, t, k=4):
    return dict(inner_err=1-f, test_ba=t, calls=4000, distinct=3900, k=k,
                sel=list(range(k)), test_auc=.73, test_acc=.72, test_gm=.71,
                test_f1=.70, test_tpr=.69, test_tnr=.68, test_mcc=.40)


def records():
    result = []
    for spec in jobs():
        i, c = DATASETS.index(spec["ds"]), A.CONFIGS.index((spec["algo"], spec["learner"]))
        r, f = spec["rep"], spec["fold"]
        alg = score(.80+c*.015+i*.001+r*.0001+f*.0001,
                    .65+c*.010+i*.0007+r*.0002+f*.00015)
        same = score(.72+c*.005+i*.0005+r*.0001+f*.0001,
                     .62+c*.005+i*.0008+r*.0002+f*.00015)
        ordinary = score(.70+c*.005+i*.0004+r*.0001+f*.0001,
                         .60+c*.005+i*.0006+r*.0002+f*.00015, k=3)
        name = "__".join(map(str, A.job_key(spec)))
        partition = splits(f)
        diag = dict(device="cuda:0", learner_dtype="float64", cuda_dispatches=4000,
                    cuda_ridge_fits=4000 if spec["learner"] == "ridge" else 0,
                    cuda_knn_distance_matrices=4000 if spec["learner"] == "knn" else 0)
        result.append(dict(status="ok", spec=spec, result={"4000": same},
                           comparators={
                               "algorithm": dict(path="synthetic_controls/algorithm/"+name+".json",
                                                 sha256=hashlib.sha256(("algorithm_"+name).encode()).hexdigest(), result=alg),
                               "ordinary_rs": dict(path="synthetic_controls/ordinary/"+name+".json",
                                                   sha256=hashlib.sha256(("ordinary_"+name).encode()).hexdigest(), result=ordinary)},
                           splits=partition, split_sha256=A._json_sha(partition),
                           gpu_backend=copy.deepcopy(GPU), gpu_diagnostics=diag,
                           seconds=.01, started_utc="2026-10-17T01:00:00+00:00",
                           completed_utc="2026-10-17T01:00:01+00:00",
                           freeze_sha256="b"*64, source_fingerprint=SOURCE_FINGERPRINT))
    return result


def fake_suite(base, payloads=None, write=True):
    base = Path(base)
    output = base / "jobs"
    output.mkdir()
    freeze = base / "freeze.json"
    freeze.write_text("synthetic freeze only\n", encoding="utf-8")
    digest = A._sha_file(freeze)
    design = jobs()
    payloads = records() if payloads is None else payloads
    for record in payloads:
        record["freeze_sha256"] = digest
    def path(spec):
        return output / ("__".join(map(str, A.job_key(spec)))+".json")
    if write:
        for record in payloads:
            path(record["spec"]).write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    called = []
    def validate(record, spec):
        called.append(A.job_key(spec))
        if record.get("core_rejection"):
            raise RuntimeError("Synthetic full provenance rejection")
    suite = SimpleNamespace(ROOT=base, OUTPUT=output, FREEZE=freeze,
                            SUITE="X12m", CONFIGS=A.CONFIGS, BUDGET=4000,
                            DATASETS=DATASETS, job_specs=lambda: design,
                            job_path=path, validate_record=validate,
                            frozen=lambda verify_inputs=True: dict(
                                jobs=design, source_fingerprint=SOURCE_FINGERPRINT,
                                gpu_environment=copy.deepcopy(GPU)))
    return suite, called


class SameSizeAnalysisChecks(unittest.TestCase):
    def test_exact_800_job_design_and_fixed_configurations(self):
        design = jobs()
        self.assertEqual(len(A.validate_design(design, DATASETS)), 800)
        for changed in (design[:-1], design[:-1]+[design[0]]):
            with self.assertRaises(RuntimeError):
                A.validate_design(changed, DATASETS)
        for field, value in (("algo", "bpso"), ("learner", "knn"), ("rep", True),
                             ("budget", 1000), ("suite", "X11c"), ("kind", "ordinary_rs")):
            changed = jobs()
            changed[0][field] = value
            with self.assertRaises(RuntimeError):
                A.validate_design(changed, DATASETS)
        with self.assertRaises(RuntimeError):
            A.validate_design(jobs(), DATASETS[:-1])

    def test_hand_calculated_four_family_uses_eight_dataset_means(self):
        payloads = records()
        before = copy.deepcopy(payloads)
        output = A.make_summaries(payloads, DATASETS)
        self.assertEqual(payloads, before)
        self.assertEqual(len(output["per_run_components"]), 800)
        self.assertEqual(len(output["family_summary"]), 4)
        self.assertEqual(len(output["dataset_effects"]), 32)
        self.assertEqual(len(output["component_levels"]), 18)
        self.assertEqual(len(output["paired_advantages"]), 18)
        self.assertEqual(len(output["ordinary_reference"]), 6)
        for row in output["family_summary"]:
            expected = .035 if row["estimand"] == "D" else 5.21+(.5 if row["algorithm"] == "nsga2" else 0)
            self.assertAlmostEqual(row["mean_pp"], expected, places=11)
            self.assertEqual((row["n_datasets"], row["df"], row["family_n"]), (8, 7, 4))
            self.assertEqual(row["family_confidence"], .9875)
            self.assertEqual(row["family_quantile"], .99375)
            self.assertAlmostEqual(row["mean_pp"]-row["ci95_lo_pp"], stats.t.ppf(.975, 7)*row["se_pp"], places=12)
            self.assertAlmostEqual(row["mean_pp"]-row["family_lo_pp"], stats.t.ppf(.99375, 7)*row["se_pp"], places=12)
            self.assertLess(row["family_lo_pp"], row["ci95_lo_pp"])
            self.assertGreater(row["family_hi_pp"], row["ci95_hi_pp"])
            self.assertNotIn("decision", row)
        for row in output["dataset_effects"]:
            i = DATASETS.index(row["dataset"])
            expected = .01*i if row["estimand"] == "D" else 5+.06*i+(.5 if row["algorithm"] == "nsga2" else 0)
            self.assertAlmostEqual(row["value_pp"], expected, places=11)
            self.assertEqual(row["runs"], 50)
        self.assertEqual(output, A.make_summaries(list(reversed(payloads)), DATASETS))

    def test_D_cancels_algorithm_and_ordinary_baseline_is_descriptive(self):
        payloads = records()
        first = A.make_summaries(payloads, DATASETS)
        for record in payloads:
            source = record["comparators"]["algorithm"]["result"]
            source["inner_err"] -= .02
            source["test_ba"] -= .01
        changed = A.make_summaries(payloads, DATASETS)
        for old, new in zip(first["family_summary"], changed["family_summary"]):
            difference = 0 if old["estimand"] == "D" else 3
            self.assertAlmostEqual(new["mean_pp"]-old["mean_pp"], difference, places=11)
        for row in changed["per_run_components"]:
            self.assertAlmostEqual(row["S_pp"]-row["O_pp"], row["D_pp"], places=12)
            self.assertAlmostEqual(row["D_pp"], row["D_cancelled_pp"], places=12)
            self.assertAlmostEqual(row["cancellation_residual_pp"], 0, places=12)
        for table in ("component_levels", "paired_advantages", "ordinary_reference"):
            for row in first[table]:
                self.assertNotIn("family_n", row)
                self.assertNotIn("family_lo_pp", row)
                self.assertEqual(row["n_datasets"], 8)
        baselines = [row for row in first["ordinary_reference"] if row["field"] == "B"]
        self.assertAlmostEqual(baselines[0]["mean_pp"], 5.175, places=11)
        self.assertAlmostEqual(baselines[1]["mean_pp"], 5.675, places=11)

    def test_partial_duplicate_bad_checkpoint_size_and_provenance_payload_fail(self):
        full = records()
        for payloads in (full[:-1], full[:-1]+[full[0]]):
            with self.assertRaises(RuntimeError):
                A.make_summaries(payloads, DATASETS)
        mutations = [
            lambda r: r["result"].update({"1000": r["result"]["4000"]}),
            lambda r: r["result"]["4000"].update(calls=3999),
            lambda r: r["result"]["4000"].update(k=3, sel=[0, 1, 2]),
            lambda r: r["comparators"]["algorithm"].update(path="../escape.json"),
            lambda r: r["comparators"]["ordinary_rs"].update(sha256="invalid"),
            lambda r: r["comparators"].pop("ordinary_rs"),
        ]
        for mutate in mutations:
            payloads = copy.deepcopy(full)
            mutate(payloads[0])
            with self.assertRaises(RuntimeError):
                A.make_summaries(payloads, DATASETS)
        with self.assertRaises(RuntimeError):
            A.t_summary({str(i): .1 for i in range(50)}, family=True)

    def test_score_parser_rejects_nonnumeric_nonfinite_counts_and_selection(self):
        good = score(.8, .7)
        self.assertAlmostEqual(A.score_record(good)["B"], .1)
        for field, value in (("inner_err", float("nan")), ("test_ba", 1.1),
                             ("test_auc", float("inf")), ("test_mcc", -1.1),
                             ("k", True), ("calls", 3999), ("distinct", 4001),
                             ("sel", [0, 0, 1, 2]), ("sel", [0, 1, 2, True])):
            with self.assertRaises(RuntimeError):
                A.score_record(dict(good, **{field: value}))

    def test_split_overlap_and_incomplete_inner_partition_fail(self):
        self.assertEqual(A.validate_splits(splits(0)), 15)
        bad = splits(0)
        bad["outer_test"] = [2, 3, 4]
        with self.assertRaises(RuntimeError):
            A.validate_splits(bad)
        bad = splits(0)
        bad["inner_relative_to_outer_train"][0]["train"].append(0)
        with self.assertRaises(RuntimeError):
            A.validate_splits(bad)
        bad = splits(0)
        bad["inner_relative_to_outer_train"][4] = copy.deepcopy(bad["inner_relative_to_outer_train"][0])
        with self.assertRaises(RuntimeError):
            A.validate_splits(bad)

    def test_missing_jobs_do_not_validate_records_or_create_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            suite, called = fake_suite(directory, write=False)
            output = Path(directory) / "analysis"
            with self.assertRaisesRegex(RuntimeError, "800 files missing"):
                A.analyse(suite, output)
            self.assertFalse(output.exists())
            self.assertFalse(called)

    def test_valid_full_synthetic_publish_reuse_and_output_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            suite, called = fake_suite(directory)
            output = Path(directory) / "analysis"
            validation = A.analyse(suite, output)
            self.assertEqual(len(called), 800)
            self.assertEqual(validation["jobs"], 800)
            self.assertEqual(validation["n_inference_units"], 8)
            self.assertEqual(validation["family_n"], 4)
            self.assertEqual(set(validation["outputs"]), A.OUTPUT_FILES)
            self.assertEqual({p.name for p in output.iterdir()}, A.OUTPUT_FILES | {"validation.json"})
            with (output / "per_run_components.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 800)
            self.assertEqual(A.analyse(suite, output), validation)
            report = (output / "REPORT.md").read_text(encoding="utf-8")
            for wording in ("exploratory", "not independent confirmation", "t(.99375,7)", "n=8 and df=7",
                            "not through whether their separate intervals cross zero", "does not establish absence or equivalence"):
                self.assertIn(wording, report)
            (output / "family_summary.csv").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "integrity"):
                A.analyse(suite, output)
            self.assertFalse(list(Path(directory).glob(".analysis.staging.*")))

    def test_full_validator_failure_blocks_publication(self):
        payloads = records()
        payloads[-1]["core_rejection"] = True
        with tempfile.TemporaryDirectory() as directory:
            suite, called = fake_suite(directory, payloads)
            output = Path(directory) / "analysis"
            with self.assertRaisesRegex(RuntimeError, "full provenance rejection"):
                A.analyse(suite, output)
            self.assertEqual(len(called), 800)
            self.assertFalse(output.exists())

    def test_wrong_backend_split_hash_and_outer_oof_fail_without_output(self):
        payloads = records()
        mutations = [
            lambda r: r.update(gpu_backend={"device": "CPU"}),
            lambda r: r["gpu_diagnostics"].update(cuda_dispatches=0),
            lambda r: r.update(split_sha256="0"*64),
            lambda r: r.update(source_fingerprint="0"*64),
        ]
        for mutate in mutations:
            changed = copy.deepcopy(payloads)
            mutate(changed[0])
            with tempfile.TemporaryDirectory() as directory:
                suite, _ = fake_suite(directory, changed)
                output = Path(directory) / "analysis"
                with self.assertRaises(RuntimeError):
                    A.analyse(suite, output)
                self.assertFalse(output.exists())
        changed = copy.deepcopy(payloads)
        changed[0]["splits"] = splits(1)
        changed[0]["split_sha256"] = A._json_sha(changed[0]["splits"])
        with tempfile.TemporaryDirectory() as directory:
            suite, _ = fake_suite(directory, changed)
            output = Path(directory) / "analysis"
            with self.assertRaisesRegex(RuntimeError, "Outer splits differ"):
                A.analyse(suite, output)
            self.assertFalse(output.exists())
        # Valid individual partitions can still fail the joint outer OOF check.
        changed = copy.deepcopy(payloads)
        for record in changed:
            if record["spec"]["ds"] == DATASETS[0] and record["spec"]["fold"] == 4:
                record["splits"] = splits(0)
                record["split_sha256"] = A._json_sha(record["splits"])
        with tempfile.TemporaryDirectory() as directory:
            suite, _ = fake_suite(directory, changed)
            output = Path(directory) / "analysis"
            with self.assertRaisesRegex(RuntimeError, "Outer test rows"):
                A.analyse(suite, output)
            self.assertFalse(output.exists())

    def test_unexpected_input_file_and_incomplete_existing_output_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            suite, _ = fake_suite(directory)
            output = Path(directory) / "analysis"
            extra = suite.OUTPUT / "unexpected.json"
            extra.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Unexpected"):
                A.analyse(suite, output)
            self.assertFalse(output.exists())
            extra.unlink()
            output.mkdir()
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                A.analyse(suite, output)
            self.assertEqual(list(output.iterdir()), [])

    def test_publication_failure_retains_staging_and_leaves_no_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            suite, _ = fake_suite(directory)
            output = Path(directory) / "analysis"
            with patch.object(A, "_write_csv", side_effect=RuntimeError("synthetic write failure")):
                with self.assertRaisesRegex(RuntimeError, "synthetic write failure"):
                    A.analyse(suite, output)
            self.assertFalse(output.exists())
            self.assertEqual(len(list(Path(directory).glob(".analysis.staging.*"))), 1)


if __name__ == "__main__":
    unittest.main()
