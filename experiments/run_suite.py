"""Resumable parallel driver.  Examples (evolve env, project root):
    python experiments/run_suite.py --suites E1h --workers 14
    python experiments/run_suite.py --suites E1h,E1f,Eperm --workers 14 --dry
Each finished job is a JSON file; reruns skip them.  Errors go to results/jobs/<suite>/errors/ and are retried."""
import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import suites as ST  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", required=True)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    jobs = []
    for name in a.suites.split(","):
        js = ST.SUITES[name]()
        todo = [j for j in js if not os.path.exists(ST.job_file(j))]
        print("%-6s total %6d   todo %6d" % (name, len(js), len(todo)), flush=True)
        jobs += todo
    if a.limit:
        jobs = jobs[: a.limit]
    if a.dry or not jobs:
        return
    log_dir = os.path.join(ST.ROOT, "results", "logs"); os.makedirs(log_dir, exist_ok=True)
    log = open(os.path.join(log_dir, "run_%s.log" % time.strftime("%Y%m%d_%H%M%S")), "a")
    t0 = time.time(); done = err = 0
    gen = Parallel(n_jobs=a.workers, return_as="generator", batch_size=1)(delayed(ST.run_job)(j) for j in jobs)
    for st in gen:
        done += 1; err += (st == "error")
        if done % 50 == 0 or st == "error":
            msg = "%s done %d/%d err %d  elapsed %.0fs" % (time.strftime("%H:%M:%S"), done, len(jobs), err, time.time() - t0)
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
    print("finished %d jobs, %d errors, %.0fs" % (done, err, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
