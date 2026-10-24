"""Implementation guards for the prospective new-data study.

Only synthetic arrays or tiny raw-format fixtures are evaluated here.  No final
confirmation dataset is loaded, searched, or used to choose acceptance limits.
These checks validate implementation boundaries, not scientific hypotheses.
"""
import copy
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "extras"))
import ex_suites6 as CORE
import confirmation_data as DATA
import ex_suites7 as PIPE
import numpy as np


def synthetic():
    rng = np.random.default_rng(172901)
    X = rng.normal(size=(90, 43))
    X[:, 0] = 7.0
    y = np.tile([0, 0, 1], 30).astype(np.int64)
    return X, y


class CountingFixture:
    """Deterministic mask objective; does not fit any classifier."""

    def __init__(self, p=127):
        self.p, self.calls, self.cache = p, 0, {}

    def evaluate(self, mask):
        self.calls += 1
        idx = np.flatnonzero(mask).astype(np.int32)
        if not 1 <= len(idx) <= min(self.p, 100):
            raise AssertionError("invalid mask reached the evaluation boundary")
        key = idx.tobytes()
        if key not in self.cache:
            self.cache[key] = float((idx.sum() % 97) / 100.0)
        return self.cache[key]


class ReusedStrictCoreChecks(unittest.TestCase):
    def test_scalers_ignore_their_validation_rows(self):
        X, y = synthetic()
        folds = CORE.S.make_folds(y, 5, CORE.S.rng_for("confirmation-synthetic-folds"))
        clean = CORE.LazyFoldScaledWrapper(X, y, folds, CORE.L.SPECS["ridge"])
        for number, (_, valid) in enumerate(folds):
            changed = X.copy()
            changed[valid] = changed[valid] * 911 + 27
            altered = CORE.LazyFoldScaledWrapper(changed, y, folds, CORE.L.SPECS["ridge"])
            for original, new in zip(clean.statistics[number], altered.statistics[number]):
                np.testing.assert_array_equal(original, new)

    def test_outer_test_values_and_labels_cannot_select_masks(self):
        X, y = synthetic()
        for algorithm in CORE.A.ALGOS:
            for learner in ("ridge", "knn"):
                keys = ("confirmation-synthetic", learner)
                clean, inner = CORE.strict_core(X[:60], y[:60], X[60:], y[60:], algorithm,
                                               learner, keys, keys + (algorithm,), (50, 75))
                changed, altered_inner = CORE.strict_core(
                    X[:60], y[:60], X[60:] * 500 + 30, 1 - y[60:], algorithm,
                    learner, keys, keys + (algorithm,), (50, 75))
                self.assertEqual(inner, altered_inner)
                for checkpoint in ("50", "75"):
                    for field in ("sel", "inner_err", "calls", "distinct"):
                        self.assertEqual(clean[checkpoint][field], changed[checkpoint][field])

    def test_all_five_algorithms_charge_exact_request_checkpoints(self):
        for algorithm in CORE.A.ALGOS:
            wrapper = CountingFixture()
            run = CORE.CountedRun(wrapper, (250, 1000, 4000))
            CORE.A.run_algo(algorithm, run, CORE.S.rng_for("confirmation-budget", algorithm))
            self.assertEqual(wrapper.calls, 4000)
            self.assertEqual(set(run.snap), {250, 1000, 4000})
            for checkpoint, snapshot in run.snap.items():
                self.assertEqual(snapshot["calls"], checkpoint)
                self.assertTrue(1 <= snapshot["distinct"] <= checkpoint)
                for field in ("wsum_idx", "lex_idx"):
                    indices = snapshot[field]
                    self.assertTrue(1 <= len(indices) <= 100)
                    self.assertEqual(len(indices), len(set(indices)))
                    self.assertTrue(np.all((indices >= 0) & (indices < wrapper.p)))

    def test_checkpoint_observation_does_not_change_search(self):
        for algorithm in CORE.A.ALGOS:
            full = CORE.CountedRun(CountingFixture(), (250, 1000, 4000))
            last = CORE.CountedRun(CountingFixture(), (4000,))
            CORE.A.run_algo(algorithm, full, CORE.S.rng_for("confirmation-passive", algorithm))
            CORE.A.run_algo(algorithm, last, CORE.S.rng_for("confirmation-passive", algorithm))
            for field in ("wsum_idx", "lex_idx"):
                np.testing.assert_array_equal(full.snap[4000][field], last.snap[4000][field])
            for field in ("wsum_err", "lex_err", "calls", "distinct"):
                self.assertEqual(full.snap[4000][field], last.snap[4000][field])

    def test_duplicate_requests_remain_charged(self):
        wrapper = CountingFixture(43)
        run = CORE.CountedRun(wrapper, (250, 1000, 4000))
        mask = np.zeros(43, dtype=bool)
        mask[:3] = True
        for _ in range(4000):
            run.eval(mask.copy())
        self.assertEqual(wrapper.calls, 4000)
        self.assertEqual(len(wrapper.cache), 1)
        for checkpoint in (250, 1000, 4000):
            self.assertEqual(run.snap[checkpoint]["calls"], checkpoint)
            self.assertEqual(run.snap[checkpoint]["distinct"], 1)


class OutcomeFreeAdapterChecks(unittest.TestCase):
    def test_sparse_and_arff_fixtures_preserve_numeric_schema(self):
        X = DATA.sparse_lines("1:2 4:3\n2:5\n", 4)
        np.testing.assert_array_equal(X, [[2, 0, 0, 3], [0, 5, 0, 0]])
        for invalid in ("0:1", "5:1", "1:2 1:4", "1:1\n\n2:3"):
            with self.assertRaises(ValueError):
                DATA.sparse_lines(invalid, 4)
        text = """@relation fixture
@attribute constant numeric
@attribute varying numeric
@attribute class {negative,positive}
@data
7,1,negative
7,2,positive
"""
        X, labels, columns = DATA.parse_dense_arff(text)
        np.testing.assert_array_equal(X, [[7, 1], [7, 2]])
        self.assertEqual(labels.tolist(), ["negative", "positive"])
        self.assertEqual(columns, ["constant", "varying"])

    def test_duplicate_rule_is_feature_only_and_keeps_first_official_row(self):
        X = np.array([[0.0, 2], [-0.0, 2], [1, 2], [1, 2]], dtype=float)
        first, quality = DATA.first_feature_rows(X, np.array([0, 1, 1, 1]))
        changed, changed_quality = DATA.first_feature_rows(X, np.array([1, 1, 0, 1]))
        np.testing.assert_array_equal(first, [0, 2])
        np.testing.assert_array_equal(changed, first)
        self.assertEqual(quality["duplicates_removed"], 2)
        self.assertEqual(quality["conflicting_feature_groups"], 1)
        self.assertEqual(changed_quality["conflicting_feature_groups"], 1)

    def test_fixed_cohort_is_deterministic_and_respects_exact_bound(self):
        y = np.r_[np.zeros(80, dtype=int), np.ones(20, dtype=int)]
        a = DATA.stratified_cohort(y, "synthetic_cohort", maximum=53)
        b = DATA.stratified_cohort(y, "synthetic_cohort", maximum=53)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(a), 53)
        self.assertEqual(len(set(a)), 53)
        self.assertTrue(np.all(np.diff(a) > 0))
        # Largest-remainder allocation: 42.4 negative, 10.6 positive.
        self.assertEqual(int(y[a].sum()), 11)
        np.testing.assert_array_equal(DATA.stratified_cohort(y[:20], "synthetic", maximum=53), np.arange(20))

    def test_climate_fixture_excludes_ids_retains_constants_and_rejects_nonfinite(self):
        rows = np.arange(80 * 21, dtype=float).reshape(80, 21)
        rows[:, 2] = 7.0
        rows[:, -1] = np.arange(80) % 2
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            path = target / "fixture.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("pop_failures.dat", "schema\n" + "\n".join(" ".join(map(str, row)) for row in rows))
            with mock.patch.object(DATA, "PROCESSED", target / "processed"), mock.patch.object(DATA, "fetch", return_value=(path, [])):
                card = DATA.build_one("nc_climate")
                with np.load(DATA.PROCESSED / "nc_climate.npz", allow_pickle=False) as data:
                    np.testing.assert_array_equal(data["X"], rows[:, 2:-1])
                    np.testing.assert_array_equal(data["source_indices"], np.arange(80))
                    self.assertEqual(data["X"].shape[1], 18)
                self.assertEqual(card["constant_predictors_retained"], 1)
                self.assertEqual(card["minority"], 40)
                with self.assertRaisesRegex(RuntimeError, "exists"):
                    DATA.build_one("nc_climate")
                with mock.patch.object(DATA, "parse", return_value=(np.full((80, 18), np.nan), rows[:, -1], list(range(18)), "fixture")):
                    with self.assertRaisesRegex(ValueError, "Nonfinite"):
                        DATA.build_one("nc_climate")
                missing_labels = rows[:, -1].copy()
                missing_labels[0] = np.nan
                with mock.patch.object(DATA, "parse", return_value=(rows[:, 2:-1], missing_labels, list(range(18)), "fixture")):
                    with self.assertRaisesRegex(ValueError, "Missing target"):
                        DATA.build_one("nc_climate")


class ArchiveAndExecutionChecks(unittest.TestCase):
    def test_exact_new_factorial_design_and_paired_partitions(self):
        jobs = PIPE.job_specs()
        self.assertEqual(len(jobs), 4000)
        self.assertEqual(len({PIPE.job_id(spec) for spec in jobs}), 4000)
        self.assertEqual(set(PIPE.DATASETS), set(DATA.DATASETS))
        self.assertEqual(len(PIPE.DATASETS), 8)
        self.assertTrue(all(spec["cps"] == [250, 1000, 4000] for spec in jobs))
        self.assertEqual(sum(len(spec["cps"]) for spec in jobs), 12000)
        y = synthetic()[1]
        with mock.patch.object(PIPE, "_SPLITS", {}):
            for repeat in range(10):
                testing = []
                for fold in range(5):
                    base = dict(jobs[0], rep=repeat, fold=fold)
                    original = PIPE.expected_splits(base, y)
                    train, test = original["outer_train"], original["outer_test"]
                    self.assertFalse(set(train) & set(test))
                    self.assertEqual(sorted(train + test), list(range(len(y))))
                    testing.extend(test)
                    for learner in PIPE.LEARNERS:
                        for algorithm in PIPE.ALGORITHMS:
                            other = PIPE.expected_splits(dict(base, algo=algorithm, learner=learner), y)
                            self.assertEqual(other["outer_train"], train)
                            self.assertEqual(other["outer_test"], test)
                            if learner == base["learner"]:
                                self.assertEqual(other, original)
                    inner_tests = []
                    for inner in original["inner_relative_to_outer_train"]:
                        self.assertFalse(set(inner["train"]) & set(inner["validation"]))
                        self.assertEqual(sorted(inner["train"] + inner["validation"]), list(range(len(train))))
                        inner_tests.extend(inner["validation"])
                    self.assertEqual(sorted(inner_tests), list(range(len(train))))
                self.assertEqual(sorted(testing), list(range(len(y))))

    def test_missing_public_archive_blocks_every_performance_entry_before_load(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(PIPE, "RECEIPT", Path(directory) / "absent.json"), \
                 mock.patch.object(PIPE, "verify_local_freeze", return_value={}), \
                 mock.patch.object(PIPE, "load_data") as loader, \
                 mock.patch.object(PIPE.CORE, "strict_core") as search, \
                 mock.patch.object(PIPE, "atomic_json") as writer:
                for function in (PIPE.compute, PIPE.run_job):
                    with self.assertRaisesRegex(RuntimeError, "Public archive not verified"):
                        function(PIPE.job_specs()[0])
                loader.assert_not_called()
                search.assert_not_called()
                writer.assert_not_called()

    def test_local_boolean_receipt_cannot_replace_remote_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            receipt = base / "receipt.json"
            receipt.write_text(json.dumps(dict(platform="zenodo", record_id="123", archived=True)), encoding="utf-8")
            freeze, info = base / "freeze.json", base / "bundle.json"
            freeze.write_text("{}", encoding="utf-8")
            info.write_text("{}", encoding="utf-8")
            with mock.patch.object(PIPE, "RECEIPT", receipt), mock.patch.object(PIPE, "FREEZE", freeze), \
                 mock.patch.object(PIPE, "BUNDLE_INFO", info), mock.patch.object(PIPE, "_ARCHIVE", None), \
                 mock.patch.object(PIPE, "_verify_public", side_effect=RuntimeError("remote proof required")) as verifier:
                with self.assertRaisesRegex(RuntimeError, "remote proof required"):
                    PIPE.verify_archive_receipt()
                verifier.assert_called_once_with("zenodo", "123")

    def test_public_archive_checks_bundle_bytes_and_each_frozen_member(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            freeze_file, info_file = base / "freeze.json", base / "bundle.json"
            freeze_file.write_text('{"fixture": true}', encoding="utf-8")
            good_source = b"# synthetic source, never a model run\n"
            freeze = dict(source_hashes={"fixture/source.py": hashlib.sha256(good_source).hexdigest()}, data_hashes={})
            def bundle(source):
                stream = io.BytesIO()
                with zipfile.ZipFile(stream, "w") as z:
                    z.writestr("fixture/source.py", source)
                    z.writestr("docs/NEW_DATA_CONFIRMATION_FREEZE.json", freeze_file.read_bytes())
                return stream.getvalue()
            raw = bundle(good_source)
            info_file.write_text(json.dumps(dict(bundle_sha256=hashlib.sha256(raw).hexdigest())), encoding="utf-8")
            with mock.patch.object(PIPE, "FREEZE", freeze_file), mock.patch.object(PIPE, "BUNDLE_INFO", info_file):
                PIPE._check_bundle_bytes(raw, freeze)
                with self.assertRaisesRegex(RuntimeError, "bundle hash mismatch"):
                    PIPE._check_bundle_bytes(raw + b"changed", freeze)
                changed = bundle(b"# changed source\n")
                info_file.write_text(json.dumps(dict(bundle_sha256=hashlib.sha256(changed).hexdigest())), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "Public content differs"):
                    PIPE._check_bundle_bytes(changed, freeze)

    def test_external_record_must_be_public_final_and_have_plausible_timestamp(self):
        for flag in ("public", "withdrawn", "embargoed", "archiving", "pending_registration_approval"):
            attributes = dict(public=True, withdrawn=False, embargoed=False, archiving=False, pending_registration_approval=False)
            attributes[flag] = False if flag == "public" else True
            with mock.patch.object(PIPE, "_public_json", return_value={"data": {"attributes": attributes}}):
                with self.assertRaisesRegex(RuntimeError, "not final and public"):
                    PIPE._archive_metadata("osf", "fixture", "fixture.zip")
        with mock.patch.object(PIPE, "_public_json", return_value={"metadata": {"access_right": "restricted"}}):
            with self.assertRaisesRegex(RuntimeError, "not public"):
                PIPE._archive_metadata("zenodo", "123", "fixture.zip")
        registration = {"data": {"attributes": dict(public=True, date_registered="2026-10-14T00:00:00")}}
        listing = dict(data=[dict(attributes={"name": "fixture.zip"}, links={"download": "https://files.osf.io/fixture"})], links={"next": None})
        with mock.patch.object(PIPE, "_public_json", side_effect=[registration, listing]):
            public = PIPE._archive_metadata("osf", "fixture", "fixture.zip")
            self.assertEqual(public["archive_utc"], "2026-10-14T00:00:00+00:00")
        with self.assertRaisesRegex(RuntimeError, "include timezone"):
            PIPE.timestamp("2026-10-14T00:00:00")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            freeze, info = base / "freeze.json", base / "bundle.json"
            freeze.write_text("synthetic freeze", encoding="utf-8")
            info.write_text(json.dumps(dict(freeze_sha256=PIPE.sha_file(freeze), bundle_name="fixture.zip")), encoding="utf-8")
            payload = dict(created_utc="2026-10-14T00:00:00+00:00")
            with mock.patch.object(PIPE, "FREEZE", freeze), mock.patch.object(PIPE, "BUNDLE_INFO", info), \
                 mock.patch.object(PIPE, "verify_local_freeze", return_value=payload), \
                 mock.patch.object(PIPE, "_archive_metadata") as metadata, mock.patch.object(PIPE, "_public_bytes") as download:
                for stamp in ("2026-10-13T00:00:00+00:00", "2999-01-01T00:00:00+00:00"):
                    metadata.return_value = dict(archive_utc=stamp)
                    with self.assertRaisesRegex(RuntimeError, "timestamp"):
                        PIPE._verify_public("zenodo", "123")
                download.assert_not_called()

    def test_receipt_binds_to_remotely_checked_bundle_and_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            receipt_file, freeze, info = base / "receipt.json", base / "freeze.json", base / "bundle.json"
            freeze.write_text("synthetic freeze", encoding="utf-8")
            info.write_text("synthetic bundle", encoding="utf-8")
            checked = dict(platform="zenodo", record_id="123", public_url="https://zenodo.org/records/123",
                           archive_utc="2026-10-14T00:00:00+00:00", freeze_sha256=PIPE.sha_file(freeze),
                           bundle_sha256="synthetic bundle sha", bundle_name="fixture.zip")
            receipt_file.write_text(json.dumps(checked), encoding="utf-8")
            with mock.patch.object(PIPE, "RECEIPT", receipt_file), mock.patch.object(PIPE, "FREEZE", freeze), \
                 mock.patch.object(PIPE, "BUNDLE_INFO", info), mock.patch.object(PIPE, "_ARCHIVE", None), \
                 mock.patch.object(PIPE, "_verify_public", return_value=checked) as verifier:
                self.assertEqual(PIPE.verify_archive_receipt(), checked)
                self.assertEqual(verifier.call_count, 1)
                for key in ("public_url", "archive_utc", "freeze_sha256", "bundle_sha256", "bundle_name"):
                    bad = dict(checked)
                    bad[key] = "wrong binding"
                    receipt_file.write_text(json.dumps(bad), encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "receipt binding changed"):
                        PIPE.verify_archive_receipt()

    def test_cached_local_freeze_rejects_same_size_restored_mtime_and_environment_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            processed = base / "processed"
            processed.mkdir()
            source, data, frozen_path = base / "source.py", processed / "fixture.npz", base / "freeze.json"
            source.write_text("synthetic source", encoding="utf-8")
            data.write_text("synthetic data", encoding="utf-8")
            source_hashes = {"source.py": PIPE.sha_file(source)}
            data_name = "processed/fixture.npz"
            payload = dict(source_hashes=source_hashes, data_hashes={data_name: PIPE.sha_file(data)},
                           environment={"synthetic": "v1"}, jobs=PIPE.job_specs(), primary_family=3,
                           source_fingerprint=PIPE.json_sha(source_hashes))
            frozen_path.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(PIPE, "ROOT", base), mock.patch.object(PIPE, "FREEZE", frozen_path), \
                 mock.patch.object(PIPE, "source_paths", return_value=["source.py"]), \
                 mock.patch.object(PIPE, "data_paths", return_value=[data_name]), \
                 mock.patch.object(PIPE, "environment", return_value={"synthetic": "v1"}) as env, \
                 mock.patch.object(PIPE, "assert_legacy_unchanged"), mock.patch.object(PIPE, "_LOCAL", None), \
                 mock.patch.object(PIPE, "_DATA", {}), mock.patch.object(DATA, "PROCESSED", processed), \
                 mock.patch.object(DATA, "load", return_value=synthetic()) as loader:
                self.assertEqual(PIPE.verify_local_freeze(), payload)
                # Warm both freeze and array caches without evaluating a learner.
                cached_arrays = PIPE.load_data("fixture")
                self.assertEqual(loader.call_count, 1)
                original_signature = PIPE._file_signature(sorted(["source.py", data_name]))
                saved = {path: (path.read_bytes(), path.stat()) for path in (source, data)}
                def replace_same_size_and_restore_mtime(path, blob):
                    original, stat = saved[path]
                    self.assertEqual(len(blob), len(original))
                    path.write_bytes(blob)
                    os.utime(str(path), ns=(stat.st_atime_ns, stat.st_mtime_ns))
                    self.assertEqual(PIPE._file_signature(sorted(["source.py", data_name])), original_signature)
                env.return_value = {"synthetic": "v2"}
                with self.assertRaisesRegex(RuntimeError, "environment changed"):
                    PIPE.verify_local_freeze()
                env.return_value = {"synthetic": "v1"}
                replace_same_size_and_restore_mtime(source, b"X" + saved[source][0][1:])
                # Scientific sources must be hashed even on the fast input path.
                with self.assertRaisesRegex(RuntimeError, "Frozen input changed"):
                    PIPE.verify_local_freeze(verify_inputs=False)
                replace_same_size_and_restore_mtime(source, saved[source][0])
                self.assertEqual(PIPE.verify_local_freeze(), payload)
                replace_same_size_and_restore_mtime(data, b"X" + saved[data][0][1:])
                with self.assertRaisesRegex(RuntimeError, "Frozen input changed"):
                    PIPE.verify_local_freeze(verify_inputs=True)
                # Previously loaded arrays cannot conceal a changed on-disk NPZ.
                with self.assertRaisesRegex(RuntimeError, "Frozen processed cohort changed"):
                    PIPE.load_data("fixture")
                self.assertEqual(loader.call_count, 1)
                replace_same_size_and_restore_mtime(data, saved[data][0])
                self.assertIs(PIPE.load_data("fixture"), cached_arrays)
                self.assertEqual(loader.call_count, 1)

    def test_local_freeze_cannot_overwrite_or_follow_outcome_files(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            existing = base / "freeze.json"
            existing.write_text("existing", encoding="utf-8")
            with mock.patch.object(PIPE, "FREEZE", existing):
                with self.assertRaisesRegex(RuntimeError, "must not be overwritten"):
                    PIPE.create_freeze()
            output = base / "jobs"
            output.mkdir()
            (output / "synthetic.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(PIPE, "FREEZE", base / "absent.json"), mock.patch.object(PIPE, "OUTPUT", output):
                with self.assertRaisesRegex(RuntimeError, "after any new outcomes"):
                    PIPE.create_freeze()

    def test_resume_revalidates_complete_provenance_budgets_and_incumbents(self):
        spec = PIPE.job_specs()[0]
        with mock.patch.object(PIPE, "_SPLITS", {}):
            splits = PIPE.expected_splits(spec, synthetic()[1])
        payload = dict(jobs=[spec], source_fingerprint="synthetic-fingerprint", split_hashes={PIPE.split_key(spec): PIPE.json_sha(splits)},
                       datasets={spec["ds"]: {"n": 90, "p": 43}})
        receipt = dict(archive_utc="2026-10-14T00:00:00+00:00")
        rows = {str(cp): dict(calls=cp, distinct=cp, k=2, sel=[1, 2], inner_err=.4,
                              test_ba=.6, test_auc=.6, test_acc=.6, test_gm=.6, test_f1=.6,
                              test_tpr=.6, test_tnr=.6, test_mcc=.2) for cp in PIPE.CPS}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            freeze_file, receipt_file = base / "freeze.json", base / "receipt.json"
            freeze_file.write_text("synthetic freeze", encoding="utf-8")
            receipt_file.write_text("synthetic receipt", encoding="utf-8")
            record = dict(status="ok", spec=spec, freeze_sha256=PIPE.sha_file(freeze_file),
                          source_fingerprint=payload["source_fingerprint"], archive_receipt_sha256=PIPE.sha_file(receipt_file),
                          started_utc="2026-10-14T01:00:00+00:00", completed_utc="2026-10-14T02:00:00+00:00",
                          seed_keys=dict(inner=list(PIPE.seed_keys(spec)[0]), algorithm=list(PIPE.seed_keys(spec)[1])),
                          result=rows, splits=splits, split_sha256=PIPE.json_sha(splits))
            with mock.patch.object(PIPE, "FREEZE", freeze_file), mock.patch.object(PIPE, "RECEIPT", receipt_file), \
                 mock.patch.object(PIPE, "OUTPUT", base / "jobs"), mock.patch.object(PIPE, "frozen", return_value=payload), \
                 mock.patch.object(PIPE, "verify_archive_receipt", return_value=receipt):
                self.assertFalse(PIPE.finished(spec))
                PIPE.atomic_json(PIPE.job_path(spec), record)
                self.assertTrue(PIPE.finished(spec))
                with mock.patch.object(PIPE, "compute") as compute:
                    self.assertEqual(PIPE.run_job(spec), "skip")
                    compute.assert_not_called()
                mutations = [
                    ("freeze_sha256", "wrong"), ("source_fingerprint", "wrong"),
                    ("archive_receipt_sha256", "wrong"), ("started_utc", "2026-10-13T00:00:00+00:00"),
                    ("completed_utc", "2026-10-13T00:00:00+00:00"), ("split_sha256", "wrong"), ("seed_keys", {})]
                for key, value in mutations:
                    bad = copy.deepcopy(record)
                    bad[key] = value
                    with self.assertRaises(RuntimeError):
                        PIPE.validate_record(bad, spec)
                for field, value in [("calls", 3999), ("distinct", True), ("sel", [1, 43]), ("k", 101), ("test_ba", float("nan"))]:
                    bad = copy.deepcopy(record)
                    bad["result"]["4000"][field] = value
                    with self.assertRaises(RuntimeError):
                        PIPE.validate_record(bad, spec)
                bad = copy.deepcopy(record)
                del bad["result"]["1000"]
                PIPE.atomic_json(PIPE.job_path(spec), bad)
                with self.assertRaisesRegex(RuntimeError, "Incomplete checkpoints"):
                    PIPE.finished(spec)


if __name__ == "__main__":
    unittest.main()
