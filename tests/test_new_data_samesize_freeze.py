"""Synthetic freeze/provenance checks; no benchmark data or learner is used."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "extras"))
import ex_suites8 as E


class SameSizeFreezeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="x12m_synthetic_freeze_")
        cls.root = Path(cls.temporary.name)
        cls.datasets = tuple("synthetic_freeze_%d" % i for i in range(8))
        cls.gpu = dict(backend_version="synthetic_cuda_float64", device="cuda:0",
                       learner_dtype="float64", cpu_learner_fallback=False)
        cls.new_sources = ["new/suite.py", "new/analysis.py"]
        for name in cls.new_sources + ["parent/source.py", "data/synthetic.bin"]:
            path = cls.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("synthetic fixture " + name).encode())
        cls.parent_freeze = cls.root / "parent/freeze.json"
        cls.parent_freeze.write_text('{"fixture":"parent freeze"}\n', encoding="utf-8")
        cls.freeze = cls.root / "new/freeze.json"
        cls.output = cls.root / "new_jobs"
        cls.hashes = cls.root / "parent/job_hashes.json"
        cls.validation = cls.root / "parent/validation.json"
        parent_sources = {"parent/source.py": E.sha_file(cls.root / "parent/source.py")}
        cls.parent_payload = dict(
            source_hashes=parent_sources, source_fingerprint=E.json_sha(parent_sources),
            data_hashes={"data/synthetic.bin": E.sha_file(cls.root / "data/synthetic.bin")},
            datasets={ds: dict(n=12, p=7) for ds in cls.datasets})
        cls.partition = dict(
            outer_train=list(range(10)), outer_test=[10, 11],
            inner_relative_to_outer_train=[
                dict(train=[i for i in range(10) if i % 5 != fold],
                     validation=[i for i in range(10) if i % 5 == fold])
                for fold in range(5)])
        parent_sha = E.sha_file(cls.parent_freeze)
        inventory = {}
        cls.source_raw = {}
        for ds in cls.datasets:
            for algo, learner in E.CONFIGS:
                for rep in range(10):
                    for fold in range(5):
                        spec = dict(suite=E.SUITE, kind="strict_samesize_rs", ds=ds,
                                    algo=algo, learner=learner, rep=rep, fold=fold, budget=4000)
                        for ordinary in (False, True):
                            source = E.source_spec(spec, ordinary)
                            k = 2 if ordinary else 3
                            row = dict(k=k, sel=list(range(k)), calls=4000, distinct=3900,
                                       inner_err=.25, test_ba=.70, test_auc=.71,
                                       test_acc=.72, test_gm=.69, test_f1=.68,
                                       test_tpr=.67, test_tnr=.73, test_mcc=.40)
                            record = dict(status="ok", spec=source, freeze_sha256=parent_sha,
                                          gpu_backend=cls.gpu, result={"4000": row},
                                          splits=cls.partition, split_sha256=E.json_sha(cls.partition),
                                          seed_keys=dict(inner=list(cls.seed_keys(source)[0])))
                            path = cls.source_path(source)
                            path.parent.mkdir(parents=True, exist_ok=True)
                            raw = json.dumps(record, sort_keys=True).encode()
                            path.write_bytes(raw)
                            name = path.relative_to(cls.root).as_posix()
                            inventory[name] = hashlib.sha256(raw).hexdigest()
                            cls.source_raw[name] = raw
        cls.hashes.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")
        cls.validation.write_text(json.dumps(dict(
            jobs=4000, checkpoints=12000,
            outputs={"job_hashes.json": E.sha_file(cls.hashes)})), encoding="utf-8")
        with cls.patches():
            cls.baseline = E.create_freeze()
        cls.baseline_raw = cls.freeze.read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def source_path(cls, spec):
        name = "%s__%s__%s__rep%d__fol%d.json" % (
            spec["ds"], spec["algo"], spec["learner"], spec["rep"], spec["fold"])
        return cls.root / "parent_jobs" / name

    @staticmethod
    def seed_keys(spec):
        return (("inner", spec["ds"], spec["learner"], spec["rep"], spec["fold"],
                 None, False, False), ("unused_synthetic_algorithm_seed",))

    @classmethod
    def patches(cls):
        stack = ExitStack()
        parent = SimpleNamespace(
            FREEZE=cls.parent_freeze,
            frozen=mock.Mock(return_value=copy.deepcopy(cls.parent_payload)),
            job_path=cls.source_path, seed_keys=cls.seed_keys,
            load_data=mock.Mock(side_effect=AssertionError("Benchmark data access is forbidden")),
            expected_splits=mock.Mock(side_effect=AssertionError("Search must not start")))
        for name, value in dict(ROOT=cls.root, DATASETS=cls.datasets, PARENT=parent,
                                FREEZE=cls.freeze, OUTPUT=cls.output,
                                REFERENCE_HASHES=cls.hashes, REFERENCE_VALIDATION=cls.validation,
                                _FREEZE=None, _VALIDATED_FREEZE_SHA=None).items():
            stack.enter_context(mock.patch.object(E, name, value))
        stack.enter_context(mock.patch.object(E, "source_paths", return_value=cls.new_sources))
        stack.enter_context(mock.patch.object(E.GPU, "device_metadata", return_value=copy.deepcopy(cls.gpu)))
        stack.enter_context(mock.patch.object(E.GPU, "require_cuda", side_effect=AssertionError("No CUDA fitting")))
        stack.enter_context(mock.patch.object(E, "same_size_core_gpu", side_effect=AssertionError("No learner fitting")))
        return stack

    def setUp(self):
        self.freeze.write_bytes(self.baseline_raw)
        self.stack = self.patches()
        self.stack.__enter__()
        self.addCleanup(self.stack.close)
        self.spec = E.job_specs()[0]
        self.identifier = E.job_id(self.spec)

    def change_freeze(self, mutate):
        payload = json.loads(self.freeze.read_text(encoding="utf-8"))
        mutate(payload)
        self.freeze.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def assert_rejected_before_fitting(self, message):
        with self.assertRaisesRegex(RuntimeError, message):
            E.compute(self.spec)
        E.PARENT.load_data.assert_not_called()
        E.PARENT.expected_splits.assert_not_called()
        E.same_size_core_gpu.assert_not_called()
        E.GPU.require_cuda.assert_not_called()

    def test_create_freeze_and_full_validation_bind_exact_800_and_1600(self):
        self.freeze.unlink()
        freeze = E.create_freeze()
        validated = E.frozen(verify_inputs=True)
        self.assertEqual(freeze, validated)
        self.assertEqual(len(freeze["jobs"]), 800)
        self.assertEqual(len(freeze["targets"]), 800)
        self.assertEqual(len(freeze["comparator_hashes"]), 1600)
        for spec in freeze["jobs"]:
            target = freeze["targets"][E.job_id(spec)]
            self.assertEqual(target["k"], 3)
            self.assertEqual(target["inner_seed_key"], list(self.seed_keys(E.source_spec(spec))[0]))
            self.assertEqual(target["sampling_seed_key"], list(E.sampling_key(spec, 3)))
            self.assertEqual(target["split_sha256"], E.json_sha(self.partition))
        for name, expected in freeze["comparator_hashes"].items():
            self.assertEqual((self.root / name).read_bytes(), self.source_raw[name])
            self.assertEqual(E.sha_file(self.root / name), expected)
        self.assertFalse(self.output.exists())
        E.same_size_core_gpu.assert_not_called()
        E.PARENT.load_data.assert_not_called()

    def test_freeze_is_not_overwritten(self):
        with self.assertRaisesRegex(RuntimeError, "Freeze must precede"):
            E.create_freeze()
        self.assertEqual(self.freeze.read_bytes(), self.baseline_raw)

    def test_changed_target_k_fails_even_when_sampling_key_is_consistent(self):
        def mutate(payload):
            target = payload["targets"][self.identifier]
            target["k"] = 4
            target["sampling_seed_key"] = list(E.sampling_key(self.spec, 4))
        self.change_freeze(mutate)
        self.assert_rejected_before_fitting("Target size/seed/split changed")

    def test_bool_target_k_is_rejected(self):
        self.change_freeze(lambda p: p["targets"][self.identifier].update(k=True))
        self.assert_rejected_before_fitting("Target size/seed/split changed")

    def test_changed_inner_seed_is_rejected_before_loading_predictors(self):
        self.change_freeze(lambda p: p["targets"][self.identifier].update(inner_seed_key=["changed"]))
        self.assert_rejected_before_fitting("Target size/seed/split changed")

    def test_changed_sampling_seed_invalidates_warm_validation_cache(self):
        E.frozen(verify_inputs=False)
        self.assertEqual(E._VALIDATED_FREEZE_SHA, E.sha_file(self.freeze))
        self.change_freeze(lambda p: p["targets"][self.identifier].update(sampling_seed_key=["changed"]))
        self.assert_rejected_before_fitting("Target size/seed/split changed")

    def test_missing_target_is_rejected(self):
        self.change_freeze(lambda p: p["targets"].pop(self.identifier))
        self.assert_rejected_before_fitting("Target job inventory changed")

    def test_extra_target_is_rejected(self):
        self.change_freeze(lambda p: p["targets"].update(unexpected=copy.deepcopy(p["targets"][self.identifier])))
        self.assert_rejected_before_fitting("Target job inventory changed")

    def test_missing_source_inventory_entry_is_rejected(self):
        name = self.source_path(E.source_spec(self.spec)).relative_to(self.root).as_posix()
        self.change_freeze(lambda p: p["comparator_hashes"].pop(name))
        self.assert_rejected_before_fitting("Exact comparator inventory changed")

    def test_extra_source_inventory_entry_is_rejected(self):
        self.change_freeze(lambda p: p["comparator_hashes"].update({"parent_jobs/extra.json": "0"*64}))
        self.assert_rejected_before_fitting("Exact comparator inventory changed")

    def test_changed_source_inventory_digest_is_rejected(self):
        name = self.source_path(E.source_spec(self.spec, True)).relative_to(self.root).as_posix()
        self.change_freeze(lambda p: p["comparator_hashes"].update({name: "0"*64}))
        self.assert_rejected_before_fitting("Exact comparator inventory changed")

    def test_changed_source_bytes_are_rejected_before_fitting(self):
        path = self.source_path(E.source_spec(self.spec))
        original = path.read_bytes()
        self.addCleanup(path.write_bytes, original)
        path.write_bytes(original + b" ")
        self.assert_rejected_before_fitting("Frozen comparator bytes changed")

    def test_compute_rechecks_source_size_even_after_validated_freeze(self):
        payload = E.frozen(verify_inputs=False)
        payload = copy.deepcopy(payload)
        payload["targets"][self.identifier]["k"] = 4
        with mock.patch.object(E, "frozen", return_value=payload):
            self.assert_rejected_before_fitting("Conditional size or source split/seed")


if __name__ == "__main__":
    unittest.main()
