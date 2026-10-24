"""Determinism audit: re-run a seeded random sample of finished jobs from scratch (fresh process, same code) and compare every number with the stored result.
    python analysis/repro_check.py [N=12]   ->   results/REPRODUCIBILITY.md"""
import glob
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments")); sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np  # noqa: E402
import suites as ST  # noqa: E402
from efsbench.splits import rng_for  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
MAX_SECONDS = float(os.environ.get("REPRO_MAX_SECONDS", 25.0))        # only re-run jobs that were cheap the first time
ONLY = [x for x in os.environ.get("REPRO_SUITES", "").split(",") if x]               # optional: restrict to some suites (supplementary pass)
SUPP = bool(ONLY)


def flatten(x, prefix=""):
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(flatten(v, prefix + "/" + str(k)))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            out.update(flatten(v, prefix + "[%d]" % i))
    else:
        out[prefix] = x
    return out


files = sorted(f for f in glob.glob(os.path.join(ST.JOBS, "*", "*.json")) if os.sep + "errors" + os.sep not in f and (not ONLY or os.path.basename(os.path.dirname(f)) in ONLY))
rng = rng_for("repro_check", 1)
order = rng.permutation(len(files))
if SUPP:
    cost = {}
    for i, f in enumerate(files):
        suite = os.path.basename(os.path.dirname(f)); sec = json.load(open(f))["seconds"]
        cost.setdefault(suite, []).append((sec, i))
    order = [i for suite in ONLY for _, i in sorted(cost.get(suite, []))[:max(N // max(len(ONLY), 1), 1)]]
rows = []
for i in order:
    if len(rows) >= N:
        break
    rec = json.load(open(files[int(i)]))
    if rec["seconds"] > MAX_SECONDS:
        continue
    t0 = time.time()
    res = ST.KINDS[rec["spec"]["kind"]](rec["spec"])
    a = flatten(json.loads(json.dumps(res)))                      # round-trip through json exactly like the stored file
    b = flatten(rec["result"])
    keys = set(a) | set(b)
    worst = 0.0; ndiff = 0
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            d = abs(va - vb)
            if d > 0:
                ndiff += 1; worst = max(worst, d)
        elif va != vb:
            ndiff += 1; worst = float("inf")
    rows.append((os.path.basename(files[int(i)]), rec["code_sha"], len(keys), ndiff, worst, time.time() - t0))
    print("%-70s sha=%s numbers=%4d differing=%d max|diff|=%.3g" % (rows[-1][0][:70], rows[-1][1], rows[-1][2], rows[-1][3], rows[-1][4]), flush=True)
ok = all(r[3] == 0 for r in rows)
L = ["# 可复现性检查（自动生成）", "",
     "随机抽取 %d 个已完成作业（耗时 ≤ %.0f 秒的），在新进程里用同一份代码从头重跑，逐个数字与保存的结果比较。" % (len(rows), MAX_SECONDS), "",
     "| 作业文件 | code_sha | 数字个数 | 不同的个数 | 最大绝对差 |", "|---|---|---:|---:|---:|"]
for r in rows:
    L.append("| %s | %s | %d | %d | %.3g |" % (r[0], r[1], r[2], r[3], r[4]))
L += ["", "**结论：%s**" % ("全部逐位一致（0 个数字不同）" if ok else "存在不一致，需要排查（线程数、BLAS、浮点非结合性）")]
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
if SUPP:                      # supplementary pass (suites that the random sample may miss): APPEND to the main report
    L[0] = "\n## 补充抽样：%s 各自最便宜的作业（耗时上限 %.0f 秒）" % (",".join(ONLY), MAX_SECONDS)
    L[2] = "在新进程里从头重跑，逐个数字与保存的结果比较（共 %d 个作业）。" % len(rows)
    open(os.path.join(ROOT, "results", "REPRODUCIBILITY.md"), "a", encoding="utf-8").write("\n".join(L) + "\n")
else:
    open(os.path.join(ROOT, "results", "REPRODUCIBILITY.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L[-2:]))
