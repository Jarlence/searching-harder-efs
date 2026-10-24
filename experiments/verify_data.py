"""Offline integrity check of the data: recompute the MD5 of every raw file and the SHA-256 of every processed file and compare them with
data/data_cards.json (written when the data were first downloaded).  No network access.
    python experiments/verify_data.py        -> exit code 0 if everything matches"""
import hashlib
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from efsbench import datasets as D  # noqa: E402


def digest(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    cards = json.load(open(os.path.join(ROOT, "data", "data_cards.json"), encoding="utf-8"))
    did_of = {v: k for k, v in D.OPENML.items()}
    bad = 0
    for c in cards:
        name = c["name"]
        proc = os.path.join(D.PROC, name + ".npz")
        ok_proc = os.path.exists(proc) and digest(proc, "sha256") == c["processed_sha256"]
        if name == "parkinson":
            raw = os.path.join(D.RAW, "uci", "parkinson_470.zip")
        else:
            raw = os.path.join(D.RAW, "openml", "%d_%s.arff" % (did_of[name], name))
        ok_raw = os.path.exists(raw) and digest(raw, "md5") == c["raw_md5"]
        print("%-10s raw md5 %-4s  processed sha256 %-4s  (n=%d, p=%d)" % (name, "ok" if ok_raw else "FAIL", "ok" if ok_proc else "FAIL", c["n"], c["p"]))
        bad += (not ok_raw) + (not ok_proc)
    print("all data files match their data cards" if not bad else "%d MISMATCHES" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
