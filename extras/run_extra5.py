"""Run only revision-2 RS distinct-mask jobs; validated resume, single BLAS thread."""
import argparse
import os
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ex_suites5 as E5  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--datasets", default="")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    jobs = E5.X6_rs_distinct()
    if args.datasets:
        keep = set(args.datasets.split(","))
        jobs = [s for s in jobs if s["ds"] in keep]
    if len({E5.job_file(s) for s in jobs}) != len(jobs):
        raise RuntimeError("X6r job ID collision")
    todo = [s for s in jobs if not E5.finished(s)]
    print("X6r total %d valid_done %d todo %d code_sha %s" % (len(jobs), len(jobs) - len(todo), len(todo), E5.CODE_SHA), flush=True)
    if args.limit:
        todo = todo[:args.limit]
    if args.dry or not todo:
        return
    if args.workers < 1:
        raise ValueError("workers must be positive")
    start = time.time()
    done = errors = 0
    results = Parallel(n_jobs=args.workers, return_as="generator", batch_size=1, pre_dispatch=args.workers)(delayed(E5.run_job)(s) for s in todo)
    for status in results:
        done += 1
        errors += status == "error"
        if done % 20 == 0 or status == "error":
            print("%s X6r done %d/%d errors %d elapsed %.0fs" % (time.strftime("%H:%M:%S"), done, len(todo), errors, time.time() - start), flush=True)
    print("X6r finished %d jobs errors %d elapsed %.0fs" % (done, errors, time.time() - start), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
