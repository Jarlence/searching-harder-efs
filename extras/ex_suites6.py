"""R22/X10s: full core-budget sensitivity to inner-training-only scaling.

New suite and output tree; original sources and jobs are read-only.
"""
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np
from efsbench import algos as A, datasets as D, learners as L, metrics as M, protocols as P, splits as S
from efsbench.wrapper import Wrapper
import suites as ST

SUITE = "X10s"
CPS = [250, 1000, 4000]
OUTPUT = ROOT / "results" / "jobs_extra" / SUITE
FREEZE = ROOT / "docs" / "STRICT_SCALING_R22_FREEZE.json"
PLAN = ROOT / "docs" / "STRICT_SCALING_R22_PLAN.md"
_DATA = {}
_FROZEN = None


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_paths():
    paths = ["docs/STRICT_SCALING_R22_PLAN.md", "extras/ex_suites6.py", "extras/run_extra6.py",
             "analysis/strict_scaling_r22.py", "tests/test_strict_scaling_r22.py",
             "experiments/suites.py", "extras/ex_suites.py", "requirements-lock.txt", "data/data_cards.json"]
    paths += [str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "src/efsbench").glob("*.py")]
    return sorted(paths)


def sources():
    return {name: sha_file(ROOT / name) for name in source_paths()}


def original_spec(spec):
    return dict(suite="E1h", kind="honest", ds=spec["ds"], algo=spec["algo"], learner=spec["learner"],
                rep=spec["rep"], fold=spec["fold"], cps=list(ST.cps_for(spec["ds"], spec["learner"])), metric="ba")


def comparator_path(spec):
    return Path(ST.job_file(original_spec(spec)))


def job_specs():
    jobs = []
    for old in ST.E1_honest():
        spec = dict(old)
        spec.update(suite=SUITE, kind="strict_scaler", cps=CPS[:])
        jobs.append(spec)
    assert len(jobs) == 5350
    assert len({job_id(s) for s in jobs}) == 5350
    return jobs


def job_id(spec):
    return "%s__%s__%s__%s__rep%d__fol%d" % (SUITE, spec["ds"], spec["algo"], spec["learner"], spec["rep"], spec["fold"])


def job_path(spec):
    return OUTPUT / (job_id(spec) + ".json")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".%d.tmp" % os.getpid())
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(tmp), str(path))


def environment():
    import scipy, pandas, sklearn, joblib, matplotlib
    return dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                pandas=pandas.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__,
                matplotlib=matplotlib.__version__)


def create_freeze():
    if FREEZE.exists():
        raise RuntimeError("Existing R22 freeze must not be overwritten")
    if OUTPUT.exists() and list(OUTPUT.glob("*.json")):
        raise RuntimeError("Cannot create a first freeze after suite outcomes exist")
    jobs = job_specs()
    comparison = {}
    for spec in jobs:
        path = comparator_path(spec)
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["status"] != "ok" or old["spec"] != original_spec(spec):
            raise RuntimeError("Invalid comparator: " + str(path))
        comparison[str(path.relative_to(ROOT)).replace("\\", "/")] = sha_file(path)
    data_paths = {"data/processed/%s.npz" % ds for ds in ST.HD + ST.IMB}
    payload = dict(created_utc=datetime.now(timezone.utc).isoformat(), client_date="2026-10-13",
                   client_timezone="Asia/Shanghai", status="internal follow-up specification, not external preregistration",
                   source_hashes=sources(), data_hashes={p: sha_file(ROOT / p) for p in sorted(data_paths)},
                   comparator_hashes=comparison, jobs=jobs,
                   environment=environment())
    payload["replay_hashes"] = {str(p.relative_to(ROOT)).replace("\\", "/"): sha_file(p)
                                for p in sorted((ROOT / "results/jobs_extra/X1s").glob("*.json"))}
    payload["source_fingerprint"] = json_sha(payload["source_hashes"])
    atomic_json(FREEZE, payload)
    return payload


def frozen(verify_inputs=False):
    global _FROZEN
    if _FROZEN is None:
        _FROZEN = json.loads(FREEZE.read_text(encoding="utf-8"))
        if sources() != _FROZEN["source_hashes"]:
            raise RuntimeError("Frozen implementation or analysis source changed")
        if environment() != _FROZEN["environment"]:
            raise RuntimeError("Frozen package environment changed")
    if verify_inputs:
        for group in ("data_hashes", "comparator_hashes", "replay_hashes"):
            for name, expected in _FROZEN[group].items():
                if sha_file(ROOT / name) != expected:
                    raise RuntimeError("Frozen input changed: " + name)
        if _FROZEN["jobs"] != job_specs():
            raise RuntimeError("Frozen job design changed")
    return _FROZEN


class LazyFoldScaledWrapper(Wrapper):
    """Same inner-fold transformations as X1, storing statistics instead of five full matrices."""

    def __init__(self, Xraw, y, folds, spec, metric="ba"):
        self.Xraw, self.y, self.folds, self.spec, self.metric = Xraw, y, folds, spec, metric
        self.thr = L.threshold(spec)
        self.calls, self.cache, self.p = 0, {}, Xraw.shape[1]
        self.statistics = []
        for tr, _ in folds:
            train = Xraw[tr]
            mu, sd = train.mean(0), train.std(0)
            self.statistics.append((mu, np.where(sd > 0, sd, 1.0)))

    def scores(self, mask):
        idx = np.flatnonzero(mask)
        selected = self.Xraw[:, idx]
        scores = np.empty(len(self.y))
        for (tr, te), (mu, sd) in zip(self.folds, self.statistics):
            train = np.asfortranarray((selected[tr] - mu[idx]) / sd[idx])
            valid = np.asfortranarray((selected[te] - mu[idx]) / sd[idx])
            scores[te] = L.fit_predict(train, self.y[tr], valid, self.spec)
        return scores

    def evaluate(self, mask):
        self.calls += 1
        idx = np.flatnonzero(mask)
        key = idx.astype(np.int32).tobytes()
        if key not in self.cache:
            self.cache[key] = self._err_from_scores(self.scores(mask))
        return self.cache[key]

    def oof_metrics(self, mask):
        scores = self.scores(mask)
        return M.all_metrics(self.y, (scores > self.thr).astype(np.int64), scores)


class CountedRun(A.Run):
    def eval(self, mask):
        answer = super().eval(mask)
        if self.w.calls in self.snap:
            self.snap[self.w.calls].update(calls=int(self.w.calls), distinct=int(len(self.w.cache)))
        return answer


def strict_core(Xtr, ytr, Xte, yte, algo, learner, inner_key, algo_key, checkpoints=CPS):
    inner = S.make_folds(ytr, 5, S.rng_for(*inner_key))
    spec = L.SPECS[learner]
    wrapper = LazyFoldScaledWrapper(Xtr, ytr, inner, spec)
    run = CountedRun(wrapper, checkpoints)
    A.run_algo(algo, run, S.rng_for(*algo_key))
    # No outer test value is used until the search has completed.
    train_scaled, test_scaled = P.standardize(Xtr, Xte)
    out = {}
    for cp in checkpoints:
        snap = run.snap[cp]
        idx, err = P._sel(algo, snap)
        scores = L.fit_predict(train_scaled[:, idx], ytr, test_scaled[:, idx], spec)
        metrics = M.all_metrics(yte, (scores > L.threshold(spec)).astype(np.int64), scores)
        out[str(cp)] = dict(k=int(len(idx)), inner_err=float(err), sel=[int(i) for i in idx],
                            calls=snap["calls"], distinct=snap["distinct"],
                            **{"test_" + key: float(value) for key, value in metrics.items()})
    return out, [{"train": tr.tolist(), "validation": te.tolist()} for tr, te in inner]


def compute(spec):
    ds = spec["ds"]
    if ds not in _DATA:
        _DATA.clear()  # bound memory when workers move between large datasets
        _DATA[ds] = D.load(ds)
    X, y, _ = _DATA[ds]
    rep, fold, algo, learner = spec["rep"], spec["fold"], spec["algo"], spec["learner"]
    tr, te = ST.outer_split(y, ds, rep, fold)
    inner_key = ("inner", ds, learner, rep, fold, None, False, False)
    algo_key = ("algo", ds, algo, learner, rep, fold, None, 0, False, False)
    result, inner = strict_core(X[tr], y[tr], X[te], y[te], algo, learner, inner_key, algo_key, spec["cps"])
    split = dict(outer_train=tr.tolist(), outer_test=te.tolist(), inner_relative_to_outer_train=inner)
    replay = dict(available=False)
    old_path = ROOT / "results/jobs_extra/X1s" / ("X1s__%s__%s__%s__rep%d__fol%d.json" % (ds, algo, learner, rep, fold))
    if old_path.exists():
        old = json.loads(old_path.read_text(encoding="utf-8"))
        if old["status"] != "ok":
            raise RuntimeError("Invalid X1 replay source")
        for cp, previous in old["result"].items():
            for key, value in previous.items():
                if result[cp][key] != value:
                    raise RuntimeError("X1 replay mismatch %s/%s" % (cp, key))
        replay = dict(available=True, path=str(old_path.relative_to(ROOT)).replace("\\", "/"),
                      sha256=sha_file(old_path), checkpoints=len(old["result"]), exact=True)
    return dict(result=result, splits=split, split_sha256=json_sha(split), x1_replay=replay)


def validate_record(record, spec):
    freeze = frozen()
    if record["status"] != "ok" or record["spec"] != spec:
        raise RuntimeError("Wrong job record")
    if record["source_fingerprint"] != freeze["source_fingerprint"] or record["freeze_sha256"] != sha_file(FREEZE):
        raise RuntimeError("Wrong provenance")
    comp = str(comparator_path(spec).relative_to(ROOT)).replace("\\", "/")
    if record["comparator_sha256"] != freeze["comparator_hashes"][comp]:
        raise RuntimeError("Wrong comparator hash")
    if set(record["result"]) != set(map(str, CPS)) or record["split_sha256"] != json_sha(record["splits"]):
        raise RuntimeError("Incomplete checkpoints or split corruption")
    for cp in CPS:
        row = record["result"][str(cp)]
        if row["calls"] != cp or not 0 < row["distinct"] <= cp:
            raise RuntimeError("Budget violation")
        if row["k"] != len(row["sel"]) or not 1 <= row["k"] <= 100 or len(set(row["sel"])) != row["k"]:
            raise RuntimeError("Selected mask invalid")
        for field in ("inner_err", "test_ba", "test_auc", "test_acc", "test_gm", "test_f1", "test_tpr", "test_tnr"):
            if not np.isfinite(row[field]) or not 0 <= row[field] <= 1:
                raise RuntimeError("Invalid performance field")
        if not np.isfinite(row["test_mcc"]) or not -1 <= row["test_mcc"] <= 1:
            raise RuntimeError("Invalid MCC")


def finished(spec):
    path = job_path(spec)
    if not path.exists():
        return False
    validate_record(json.loads(path.read_text(encoding="utf-8")), spec)
    return True


def run_job(spec):
    if finished(spec):
        return "skip"
    start = time.monotonic()
    freeze = frozen()
    try:
        payload = compute(spec)
        record = dict(status="ok", spec=spec, seconds=round(time.monotonic() - start, 3),
                      completed_utc=datetime.now(timezone.utc).isoformat(), source_fingerprint=freeze["source_fingerprint"],
                      freeze_sha256=sha_file(FREEZE), comparator_sha256=sha_file(comparator_path(spec)), **payload)
        validate_record(record, spec)
        atomic_json(job_path(spec), record)
        return "ok"
    except Exception:
        atomic_json(OUTPUT / "errors" / (job_id(spec) + ".json"), dict(status="error", spec=spec,
                    seconds=time.monotonic() - start, error=traceback.format_exc()))
        return "error"
