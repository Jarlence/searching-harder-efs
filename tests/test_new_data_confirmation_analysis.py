"""Synthetic-only confirmation analysis checks; never load new-data outcomes."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import new_data_confirmation as A

DATASETS = tuple("synthetic_%d" % index for index in range(8))


def jobs():
    return [dict(ds=ds, learner=lr, algo=algo, rep=rep, fold=fold, cps=list(A.CPS))
            for ds in DATASETS for lr in A.LEARNERS for algo in A.ALGORITHMS
            for rep in range(A.N_REPEATS) for fold in range(A.N_FOLDS)]


def scores():
    result = {}
    for index, ds in enumerate(DATASETS):
        for lr_index, lr in enumerate(A.LEARNERS):
            for algo in A.ALGORITHMS:
                for rep in range(A.N_REPEATS):
                    for fold in range(A.N_FOLDS):
                        for cp, weight in ((250, 0.0), (1000, 0.4), (4000, 1.0)):
                            F = .55 + index*.001 + lr_index*.02 + rep*.0001 + fold*.0001 + weight*(.08 + index*.001)
                            T = .50 + index*.002 + lr_index*.005 + rep*.0002 + fold*.0002 + weight*.01
                            if (lr, algo) == ("ridge", "ga"):
                                F += .10 + index*.002
                                T += .02
                            if (lr, algo) == ("knn", "nsga2"):
                                F += .07 + index*.003
                                T += .01
                            result[(ds, lr, algo, rep, fold, cp)] = dict(F=F, T=T, B=F-T, k=4, calls=cp, distinct=cp)
    return result


def valid_splits():
    return dict(outer_train=list(range(10)), outer_test=list(range(10, 15)),
                inner_relative_to_outer_train=[dict(train=[i for i in range(10) if i not in (2*f, 2*f+1)],
                                                    validation=[2*f, 2*f+1]) for f in range(5)])


class ConfirmationAnalysisChecks(unittest.TestCase):
    def test_exact_fixed_design_and_duplicate_rejection(self):
        design = jobs()
        self.assertEqual(len(A.validate_design(design, DATASETS)), 4000)
        for bad in (design[:-1], design[:-1] + [design[0]]):
            with self.assertRaises(RuntimeError):
                A.validate_design(bad, DATASETS)
        changed = jobs()
        changed[0]["cps"] = [250, 4000]
        with self.assertRaises(RuntimeError):
            A.validate_design(changed, DATASETS)

    def test_paired_effects_and_equal_dataset_inference(self):
        records = scores()
        output = A.make_summaries(records, DATASETS)
        self.assertEqual(len(output["primary"]), 3)
        self.assertEqual(len(output["dataset_primary"]), 24)
        expected = {"C1": 7.35, "C2": 8.70, "C3": 7.05}
        for row in output["primary"]:
            self.assertEqual(row["n_datasets"], 8)
            self.assertEqual(row["df"], 7)
            self.assertEqual(row["family_n"], 3)
            self.assertAlmostEqual(row["family_confidence"], 1-.05/3)
            self.assertAlmostEqual(row["mean_pp"], expected[row["contrast"]], places=12)
            self.assertLess(row["family_lo_pp"], row["ci95_lo_pp"])
            self.assertGreater(row["family_hi_pp"], row["ci95_hi_pp"])
            self.assertEqual(row["decision"], "supported_positive")
        dataset_rows = output["dataset_primary"]
        for index, ds in enumerate(DATASETS):
            for contrast, expected_value in (("C1", 7+.1*index), ("C2", 8+.2*index), ("C3", 6+.3*index)):
                actual = next(r["value_pp"] for r in dataset_rows if r["dataset"] == ds and r["contrast"] == contrast)
                self.assertAlmostEqual(actual, expected_value, places=12)

    def test_advantage_difference_equals_inner_minus_outer_advantage(self):
        records = scores()
        for lr in A.LEARNERS:
            for algo in A.ALGORITHMS[:-1]:
                b = A.advantage_values(records, DATASETS, lr, algo, "B")
                f = A.advantage_values(records, DATASETS, lr, algo, "F")
                t = A.advantage_values(records, DATASETS, lr, algo, "T")
                for ds in DATASETS:
                    self.assertAlmostEqual(b[ds], f[ds]-t[ds], places=13)

    def test_intermediate_budget_increments_remain_secondary_and_add_to_endpoint(self):
        records = scores()
        output = A.make_summaries(records, DATASETS)
        self.assertEqual(len(output["budget_increments"]), 9)
        self.assertEqual(len(output["dataset_budget_increments"]), 72)
        expected_means = {(250, 1000): {"F": 3.34, "T": .4, "B": 2.94},
                          (1000, 4000): {"F": 5.01, "T": .6, "B": 4.41},
                          (250, 4000): {"F": 8.35, "T": 1.0, "B": 7.35}}
        for row in output["budget_increments"]:
            self.assertEqual(row["status"], "secondary_descriptive")
            self.assertNotIn("family_n", row)
            expected = expected_means[(row["low_budget"], row["high_budget"])][row["field"]]
            self.assertAlmostEqual(row["mean_pp"], expected, places=12)
        for field in ("F", "T", "B"):
            first = A.budget_values(records, DATASETS, field, 250, 1000)
            second = A.budget_values(records, DATASETS, field, 1000, 4000)
            full = A.budget_values(records, DATASETS, field)
            for ds in DATASETS:
                self.assertAlmostEqual(first[ds]+second[ds], full[ds], places=13)

    def test_complete_records_required_before_summary(self):
        records = scores()
        del records[next(iter(records))]
        with self.assertRaises(RuntimeError):
            A.make_summaries(records, DATASETS)
        records = scores()
        records[next(iter(records))]["B"] += .03
        with self.assertRaises(RuntimeError):
            A.make_summaries(records, DATASETS)

    def test_signs_and_inconclusive_are_not_equivalence(self):
        positive = A.t_summary({str(i): .04 for i in range(8)}, primary=True)
        negative = A.t_summary({str(i): -.04 for i in range(8)}, primary=True)
        uncertain = A.t_summary({str(i): (-.05 if i < 4 else .05) for i in range(8)}, primary=True)
        self.assertEqual(positive["decision"], "supported_positive")
        self.assertEqual(negative["decision"], "supported_opposite")
        self.assertEqual(uncertain["decision"], "inconclusive")
        self.assertNotIn("family_n", A.t_summary({"a": .02, "b": .03}))

    def test_score_parser_rejects_invalid_values_and_budget(self):
        row = dict(inner_err=.2, test_ba=.7, calls=250, distinct=211, k=3)
        score = A.score_record(row, 250)
        self.assertAlmostEqual(score["B"], .1)
        for field, value in (("inner_err", float("nan")), ("test_ba", 1.1), ("calls", 249),
                             ("distinct", 251), ("k", 0), ("k", True)):
            bad = dict(row, **{field: value})
            with self.assertRaises(RuntimeError):
                A.score_record(bad, 250)

    def test_split_isolation_and_oof_partition(self):
        self.assertEqual(A.validate_splits(valid_splits()), 15)
        bad = valid_splits()
        bad["outer_test"] = [9, 10, 11, 12, 13]
        with self.assertRaises(RuntimeError):
            A.validate_splits(bad)
        bad = valid_splits()
        bad["inner_relative_to_outer_train"][0]["train"].append(0)
        with self.assertRaises(RuntimeError):
            A.validate_splits(bad)
        bad = valid_splits()
        bad["inner_relative_to_outer_train"][4] = bad["inner_relative_to_outer_train"][0]
        with self.assertRaises(RuntimeError):
            A.validate_splits(bad)

    def test_missing_jobs_do_not_create_analysis_output(self):
        design = jobs()
        prior_out = A.OUT
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            freeze = base / "freeze.json"
            freeze.write_text("{}", encoding="utf-8")
            A.OUT = base / "results"
            def unexpected_validate(*args):
                raise AssertionError("Missing jobs must be detected before loading outcomes")
            suite = SimpleNamespace(CPS=A.CPS, DATASETS=DATASETS, FREEZE=freeze, job_specs=lambda: design,
                                    frozen=lambda verify_inputs: {"jobs": design},
                                    sha_file=lambda path: "synthetic-freeze-sha",
                                    verify_archive_receipt=lambda: {"freeze_sha256": "synthetic-freeze-sha"},
                                    job_path=lambda s: base / ("__".join(map(str, A.job_key(s)))+".json"),
                                    validate_record=unexpected_validate)
            try:
                with self.assertRaisesRegex(RuntimeError, "4000 files missing"):
                    A.analyse(suite)
                self.assertFalse(A.OUT.exists())
            finally:
                A.OUT = prior_out

    def test_external_archive_receipt_must_match_freeze_before_outcomes(self):
        design = jobs()
        prior_out = A.OUT
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            freeze = base / "freeze.json"
            freeze.write_text("{}", encoding="utf-8")
            A.OUT = base / "results"
            def unexpected_path(*args):
                raise AssertionError("Bad archive binding must be rejected before outcome paths")
            suite = SimpleNamespace(CPS=A.CPS, DATASETS=DATASETS, FREEZE=freeze, job_specs=lambda: design,
                                    frozen=lambda verify_inputs: {"jobs": design}, sha_file=lambda path: "correct-sha",
                                    verify_archive_receipt=lambda: {"freeze_sha256": "different-sha"},
                                    job_path=unexpected_path)
            try:
                with self.assertRaisesRegex(RuntimeError, "not bound"):
                    A.analyse(suite)
                self.assertFalse(A.OUT.exists())
            finally:
                A.OUT = prior_out


if __name__ == "__main__":
    unittest.main()
