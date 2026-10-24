"""Write docs/FREEZE.json: hashes of code, protocol, data cards and processed data, taken BEFORE the main runs."""
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
    out = dict(frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"), python=sys.version.split()[0], files={})
    for sub in ("src/efsbench", "experiments", "tests"):
        d = os.path.join(ROOT, sub)
        for f in sorted(os.listdir(d)):
            if f.endswith(".py"):
                out["files"]["%s/%s" % (sub, f)] = sha(os.path.join(d, f))
    out["files"]["docs/PROTOCOL.md"] = sha(os.path.join(ROOT, "docs", "PROTOCOL.md"))
    proc = os.path.join(ROOT, "data", "processed")
    out["data"] = {f: sha(os.path.join(proc, f)) for f in sorted(os.listdir(proc)) if f.endswith(".npz")}
    import numpy, scipy, sklearn, pandas
    out["packages"] = dict(numpy=numpy.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, pandas=pandas.__version__)
    name = sys.argv[1] if len(sys.argv) > 1 else "FREEZE.json"
    with open(os.path.join(ROOT, "docs", name), "w") as f:
        json.dump(out, f, indent=1)
    print("frozen %d files, %d datasets" % (len(out["files"]), len(out["data"])))


if __name__ == "__main__":
    main()
