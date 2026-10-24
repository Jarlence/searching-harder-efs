"""Resumable X10s execution; no partial performance summaries."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ex_suites6 as E
from joblib import Parallel, delayed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="fixed prefix for execution diagnostics, never outcome filtering")
    args = parser.parse_args()
    if args.freeze:
        record = E.create_freeze()
        print("Frozen %d jobs; fingerprint %s" % (len(record["jobs"]), record["source_fingerprint"]), flush=True)
        return
    if args.workers < 1:
        raise ValueError("workers must be positive")
    freeze = E.frozen(verify_inputs=True)
    jobs = freeze["jobs"]
    todo = [s for s in jobs if not E.finished(s)]
    valid_before = len(jobs) - len(todo)
    print("X10s total=%d valid_done=%d todo=%d workers=%d" % (len(jobs), len(jobs)-len(todo), len(todo), args.workers), flush=True)
    if args.dry or not todo:
        return
    E.OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = E.OUTPUT / ".runner.lock"
    descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    try:
        if args.limit:
            todo = todo[:args.limit]
        start = time.monotonic()
        done = errors = 0
        for status in Parallel(n_jobs=args.workers, return_as="generator_unordered", batch_size=1, pre_dispatch=args.workers)(
                delayed(E.run_job)(s) for s in todo):
            done += 1
            errors += status == "error"
            if done % 25 == 0 or status == "error" or done == len(todo):
                elapsed = time.monotonic()-start
                payload = dict(total=len(jobs), valid_before=valid_before, valid_now=valid_before+done-errors,
                               batch_total=len(todo), completed=done, errors=errors, elapsed_seconds=round(elapsed,1))
                E.atomic_json(E.ROOT / "results/r22_strict_scaling_progress.json", payload)
                print("%s X10s completed=%d/%d errors=%d elapsed=%.0fs" % (time.strftime("%H:%M:%S"), done, len(todo), errors, elapsed), flush=True)
        if errors:
            raise RuntimeError("%d jobs failed; results are not ready for analysis" % errors)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
