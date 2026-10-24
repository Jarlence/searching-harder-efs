"""Write docs/FREEZE_dev5.json: the state of code, documents and data AFTER the external-review revision (docs/DEVIATIONS.md #7-#11).
This is a record made at revision time.  It documents what the released materials contain and which frozen files are unchanged since FREEZE_dev4.json;
it cannot prove that anything existed earlier (the hashes are produced on the authors' machine; there is no external time stamp).
    python experiments/freeze_rev1.py"""
import hashlib
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    out = dict(frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"), python=sys.version.split()[0], note="revision-time record, not a pre-registration", files={})
    for sub in ("src/efsbench", "experiments", "tests", "extras", "analysis", "audit", "paper"):
        d = os.path.join(ROOT, sub)
        for f in sorted(os.listdir(d)):
            if f.endswith(".py"):
                out["files"]["%s/%s" % (sub, f)] = sha(os.path.join(d, f))
    for f in ("docs/PROTOCOL.md", "docs/ANALYSIS_PLAN.md", "docs/DEVIATIONS.md", "docs/REVISION_1.md", "REPRODUCE.md", "requirements-lock.txt"):
        if os.path.exists(os.path.join(ROOT, f)):
            out["files"][f] = sha(os.path.join(ROOT, f))
    proc = os.path.join(ROOT, "data", "processed")
    out["data"] = {f: sha(os.path.join(proc, f)) for f in sorted(os.listdir(proc)) if f.endswith(".npz")}
    import numpy, scipy, sklearn, pandas
    out["packages"] = dict(numpy=numpy.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, pandas=pandas.__version__)
    prev = json.load(open(os.path.join(ROOT, "docs", "FREEZE_dev4.json"), encoding="utf-8"))
    out["unchanged_since_dev4"] = sorted(k for k, v in prev["files"].items() if out["files"].get(k) == v)
    out["changed_since_dev4"] = sorted(k for k, v in prev["files"].items() if k in out["files"] and out["files"][k] != v)
    out["new_since_dev4"] = sorted(k for k in out["files"] if k not in prev["files"])
    out["data_unchanged_since_dev4"] = bool(prev["data"] == out["data"])
    with open(os.path.join(ROOT, "docs", "FREEZE_dev5.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("files:", len(out["files"]), "| unchanged since dev4:", len(out["unchanged_since_dev4"]), "| changed:", out["changed_since_dev4"],
          "| data unchanged:", out["data_unchanged_since_dev4"])


if __name__ == "__main__":
    main()
