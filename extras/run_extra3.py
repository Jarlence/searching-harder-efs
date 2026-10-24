"""Resumable driver for the third batch of extra studies (see ex_suites3.py):
    python extras/run_extra3.py --suites X8p,X6d,X5m,X7d --workers 20"""
import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ex_suites3 as E3  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    jobs = []
    for name in a.suites.split(","):
        js = E3.SUITES[name]()
        ids = {E3.job_file(j) for j in js}
        assert len(ids) == len(js), "job id collision in %s" % name
        todo = [j for j in js if not os.path.exists(E3.job_file(j))]
        print("%-5s total %5d   todo %5d" % (name, len(js), len(todo)), flush=True)
        jobs += todo
    if a.limit:
        jobs = jobs[: a.limit]
    if a.dry or not jobs:
        return
    t0 = time.time(); done = err = 0
    gen = Parallel(n_jobs=a.workers, return_as="generator", batch_size=1)(delayed(E3.run_job)(j) for j in jobs)
    for st in gen:
        done += 1; err += (st == "error")
        if done % 50 == 0 or st == "error":
            print("%s done %d/%d err %d elapsed %.0fs" % (time.strftime("%H:%M:%S"), done, len(jobs), err, time.time() - t0), flush=True)
    print("finished %d jobs, %d errors, %.0fs" % (done, err, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
