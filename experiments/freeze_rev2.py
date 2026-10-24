"""Record the local revision-2 state and verify the preserved original job set.

Run after analysis, manuscript compilation and validation documentation, before
analysis/manifest.py. This is a local revision record, not a pre-registration.
"""
import glob
import hashlib
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    import numpy, scipy, pandas, sklearn
    previous_path = os.path.join(ROOT, "results", "superseded", "before_revision2", "MANIFEST_before_revision2.json")
    with open(previous_path, encoding="utf-8") as f:
        previous = json.load(f)
    out = dict(created_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
               note="local revision-2 record; public release pending; not a pre-registration",
               python=sys.version.split()[0], files={},
               packages=dict(numpy=numpy.__version__, scipy=scipy.__version__, pandas=pandas.__version__, sklearn=sklearn.__version__))
    patterns = ["src/efsbench/*.py", "experiments/*.py", "extras/*.py", "analysis/*.py", "tests/*.py", "audit/*.py", "paper/*.py",
                "docs/*.md", "*.md", "requirements-lock.txt", "paper/main.tex", "paper/sections/*.tex", "paper/refs.bib",
                "paper/generated/*", "paper/main.pdf", "novelty/external/li2026/provenance.md", "results/REVISION2_VALIDATION.md"]
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(ROOT, pattern))):
            if os.path.isfile(path):
                key = os.path.relpath(path, ROOT).replace("\\", "/")
                out["files"][key] = sha(path)
    out["data"] = {os.path.basename(p): sha(p) for p in sorted(glob.glob(os.path.join(ROOT, "data", "processed", "*.npz")))}
    with open(os.path.join(ROOT, "docs", "FREEZE_dev4.json"), encoding="utf-8") as f:
        dev4 = json.load(f)
    out["changed_since_dev4"] = [k for k, value in dev4["files"].items() if out["files"].get(k) != value]
    out["data_unchanged_since_dev4"] = out["data"] == dev4["data"]
    if out["changed_since_dev4"] or not out["data_unchanged_since_dev4"]:
        raise RuntimeError("Frozen main implementation or processed data changed")
    hashes = {"original": hashlib.sha256(), "all": hashlib.sha256()}
    counts = dict(original=0, all=0)
    for base in ("results/jobs", "results/jobs_extra"):
        for path in sorted(glob.glob(os.path.join(ROOT, base, "*", "*.json"))):
            key = os.path.relpath(path, ROOT).replace("\\", "/")
            digest = sha(path)
            for kind in (("all",) if key.startswith("results/jobs_extra/X6r/") else ("original", "all")):
                hashes[kind].update(key.encode())
                hashes[kind].update(digest.encode())
                counts[kind] += 1
    out["original_job_set"] = dict(n=counts["original"], sha256=hashes["original"].hexdigest())
    out["all_job_files"] = dict(n=counts["all"], sha256=hashes["all"].hexdigest())
    if out["original_job_set"] != previous["all_job_files"]:
        raise RuntimeError("Original job collection no longer matches the pre-revision snapshot")
    if counts != dict(original=37290, all=38360):
        raise RuntimeError("Unexpected job counts: %s" % counts)
    with open(os.path.join(ROOT, "docs", "FREEZE_rev2.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("revision-2 freeze: %d files; original jobs %d unchanged; total jobs %d; data unchanged" % (len(out["files"]), counts["original"], counts["all"]))


if __name__ == "__main__":
    main()
