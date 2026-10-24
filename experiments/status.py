"""Progress of all suites: finished / total jobs, errors, CPU seconds spent, rough ETA."""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import suites as ST  # noqa: E402

tot_done = tot_all = 0
cpu_done = 0.0
rows = []
for name, fn in ST.SUITES.items():
    js = fn()
    d = os.path.join(ST.JOBS, js[0]["suite"]) if js else None
    suites_here = sorted({j["suite"] for j in js})
    done = 0; err = 0; cpu = 0.0
    for sname in suites_here:
        fs = glob.glob(os.path.join(ST.JOBS, sname, "*.json"))
        done += len(fs); cpu += sum(json.load(open(f))["seconds"] for f in fs[:400]) * max(1, len(fs) / max(1, min(len(fs), 400)))
        err += len(glob.glob(os.path.join(ST.JOBS, sname, "errors", "*.json")))
    rows.append((name, done, len(js), err, cpu))
    tot_done += done; tot_all += len(js); cpu_done += cpu
for name, done, n, err, cpu in rows:
    print("%-6s %6d / %6d  (%5.1f%%)  errors %d  cpu %7.0fs" % (name, done, n, 100.0 * done / max(n, 1), err, cpu))
print("TOTAL  %6d / %6d  (%.1f%%)   cpu-seconds so far %.0f   %s" % (tot_done, tot_all, 100.0 * tot_done / tot_all, cpu_done,
                                                                time.strftime("%H:%M:%S")))
