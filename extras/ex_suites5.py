"""Revision 2: original random-search streams at equal DISTINCT-mask budgets.

X6r supplements X6d without changing any existing job. Every job also records
the original 250/1,000/4,000-request checkpoints and checks them against E1h.
The extra bookkeeping consumes no random numbers.
"""
import ctypes
import hashlib
import json
import os
import sys
import time
import traceback

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import numpy as np  # noqa: E402
from efsbench import algos as A, learners as L, metrics as M, protocols as P, splits as S, tracking as T  # noqa: E402
from efsbench.wrapper import Wrapper  # noqa: E402
import suites as ST  # noqa: E402

JOBS = os.path.join(ROOT, "results", "jobs_extra")
DISTINCT_CPS = (1000, 4000)
REQUEST_CPS = (250, 1000, 4000)
CALL_CAP = 150000
FIELDS = ("k", "inner_err", "test_ba", "test_gm", "test_acc", "test_f1", "test_mcc", "test_auc", "test_tpr", "test_tnr")


def code_sha():
    """Full SHA-256 covers the suite, driver and every local runtime dependency."""
    paths = ["extras/ex_suites5.py", "extras/run_extra5.py", "experiments/suites.py",
             "requirements-lock.txt", "data/data_cards.json"]
    paths.extend("src/efsbench/" + f for f in os.listdir(os.path.join(ROOT, "src", "efsbench")) if f.endswith(".py"))
    h = hashlib.sha256()
    for path in sorted(paths):
        with open(os.path.join(ROOT, path), "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        h.update(path.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\0")
    return h.hexdigest()


CODE_SHA = code_sha()


class ReplayRun(T.TrackRun):
    """Distinct stopping rule plus passive snapshots at original request counts."""
    def __init__(self, wrapper, checkpoints, request_checkpoints):
        super().__init__(wrapper, checkpoints, mode="distinct", archive=False, call_cap=CALL_CAP)
        self.request_checkpoints = set(request_checkpoints)
        self.request_snap = {}

    def eval(self, mask):
        value = super().eval(mask)
        if self.w.calls in self.request_checkpoints:
            self.request_snap[self.w.calls] = self.snapshot()
        return value


def _record(snapshot, Xtr, ytr, Xte, yte, spec):
    idx, err = T._sel("rs", snapshot)
    scores = L.fit_predict(Xtr[:, idx], ytr, Xte[:, idx], spec)
    metrics = M.all_metrics(yte, (scores > L.threshold(spec)).astype(np.int64), scores)
    return dict(k=int(len(idx)), inner_err=float(err), sel=[int(i) for i in idx],
                calls=int(snapshot["calls"]), distinct=int(snapshot["distinct"]),
                **{"test_" + name: float(value) for name, value in metrics.items()})


def _original_spec(s):
    return dict(suite="E1h", kind="honest", ds=s["ds"], algo="rs", learner=s["learner"],
                rep=s["rep"], fold=s["fold"], cps=list(ST.cps_for(s["ds"], s["learner"])), metric="ba")


def _check_replay(replay, original):
    n_numbers = 0
    for cp in REQUEST_CPS:
        new, old = replay[str(cp)], original["result"][str(cp)]
        for field in FIELDS:
            n_numbers += 1
            if new[field] != old[field]:
                raise RuntimeError("RS request replay differs at %s/%s: %r vs %r" % (cp, field, new[field], old[field]))
        if new["sel"] != old["sel"]:
            raise RuntimeError("RS selected mask differs at request checkpoint %s" % cp)
        n_numbers += len(old["sel"])
        if new["calls"] != cp or not 0 < new["distinct"] <= cp:
            raise RuntimeError("Invalid request-checkpoint bookkeeping")
    return dict(n_records=len(REQUEST_CPS), n_numbers=n_numbers, n_different=0, max_abs_diff=0.0,
                source_code_sha=original["code_sha"])


def k_rs_distinct(s):
    ds, lr, rep, fold = s["ds"], s["learner"], s["rep"], s["fold"]
    source = ST.job_file(_original_spec(s))
    with open(source, encoding="utf-8") as f:
        original = json.load(f)
    if original["status"] != "ok" or original["spec"] != _original_spec(s):
        raise RuntimeError("E1h RS reference is invalid: " + source)
    X, y, _ = ST.data(ds)
    tr, te = ST.outer_split(y, ds, rep, fold)
    Xtr, Xte = P.standardize(X[tr], X[te])
    ytr, yte = y[tr], y[te]
    spec = L.SPECS[lr]
    inner_key = ("inner", ds, lr, rep, fold, None, False, False)
    algo_key = ("algo", ds, "rs", lr, rep, fold, None, 0, False, False)
    inner = S.make_folds(ytr, 5, S.rng_for(*inner_key), None)
    wrapper = Wrapper(np.asfortranarray(Xtr), ytr, inner, spec, "ba")
    run = ReplayRun(wrapper, s["dcps"], s["request_cps"])
    A.run_algo("rs", run, S.rng_for(*algo_key))
    if set(run.tsnap) != set(s["dcps"]):
        raise RuntimeError("RS did not reach every distinct-mask target before call cap")
    if set(run.request_snap) != set(s["request_cps"]):
        raise RuntimeError("RS request replay is incomplete")
    result, replay = {}, {}
    for cp in s["dcps"]:
        rec = _record(run.tsnap[cp], Xtr, ytr, Xte, yte, spec)
        rec["reached"] = True
        if rec["distinct"] != cp:
            raise RuntimeError("RS distinct count does not equal target")
        result[str(cp)] = rec
    for cp in s["request_cps"]:
        replay[str(cp)] = _record(run.request_snap[cp], Xtr, ytr, Xte, yte, spec)
    check = _check_replay(replay, original)
    return dict(result=result, request_replay=replay, replay_check=check)


def X6_rs_distinct():
    return [dict(suite="X6r", kind="distinct_rs", ds=s["ds"], algo="rs", learner=s["learner"],
                 rep=s["rep"], fold=s["fold"], dcps=list(DISTINCT_CPS), request_cps=list(REQUEST_CPS))
            for s in ST.E1_honest() if s["algo"] == "rs"]


def job_id(s):
    return "__".join([s["suite"], s["ds"], "rs", s["learner"], "rep%d" % s["rep"], "fol%d" % s["fold"], "distinct_v1"])


def job_file(s):
    return os.path.join(JOBS, s["suite"], job_id(s) + ".json")


def validate_record(rec, spec):
    if rec.get("status") != "ok" or rec.get("spec") != spec or rec.get("code_sha") != CODE_SHA:
        raise RuntimeError("Existing X6r job does not match the requested spec and code hash: " + job_id(spec))
    if set(rec.get("result", {})) != {str(cp) for cp in spec["dcps"]}:
        raise RuntimeError("X6r distinct checkpoints are incomplete")
    for cp in spec["dcps"]:
        row = rec["result"][str(cp)]
        if row.get("reached") is not True or row.get("distinct") != cp or row.get("calls", 0) < cp:
            raise RuntimeError("X6r target was not reached exactly")
    check = rec.get("replay_check", {})
    if check.get("n_records") != len(REQUEST_CPS) or check.get("n_different") != 0 or check.get("max_abs_diff") != 0.0:
        raise RuntimeError("X6r original-request replay check failed")
    if set(rec.get("request_replay", {})) != {str(cp) for cp in spec["request_cps"]}:
        raise RuntimeError("X6r original-request replay checkpoints are incomplete")
    return rec


def finished(s):
    path = job_file(s)
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        validate_record(json.load(f), s)
    return True


def _atomic_write(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".%d.tmp" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _peak_memory_mb():
    if os.name != "nt":
        return None
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                                         "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return round(counters.PeakWorkingSetSize / (1024.0 ** 2), 2)
    return None


def run_job(s):
    if finished(s):
        return "skip"
    t0 = time.time()
    try:
        payload = k_rs_distinct(s)
        rec = dict(spec=s, status="ok", seconds=round(time.time() - t0, 2), code_sha=CODE_SHA,
                   worker_peak_memory_mb=_peak_memory_mb(), **payload)
        validate_record(rec, s)
        _atomic_write(job_file(s), rec)
        return "ok"
    except Exception:
        rec = dict(spec=s, status="error", seconds=round(time.time() - t0, 2), code_sha=CODE_SHA,
                   error=traceback.format_exc())
        _atomic_write(os.path.join(JOBS, "X6r", "errors", job_id(s) + ".json"), rec)
        return "error"
