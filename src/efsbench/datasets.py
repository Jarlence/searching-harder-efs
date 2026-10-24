"""Public-data acquisition (md5-verified) and loading for efsbench.  Python 3.8; numpy / scipy / pandas only."""
import hashlib
import io
import json
import os
import urllib.request
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
UA = {"User-Agent": "t1-efsbench/1.0"}

# did -> short name.  All OpenML records carry licence "Public".
OPENML = {
    45087: "colon", 45086: "cns", 45088: "dlbcl", 45090: "leukemia", 45099: "prostate", 45089: "gli",
    45100: "smk", 45098: "ovarian", 45085: "breast",                 # high-dimensional, small sample
    311: "oil_spill", 1487: "ozone", 40910: "speech", 1065: "kc3",   # strongly imbalanced
}
UCI_PARKINSON = "https://archive.ics.uci.edu/static/public/470/parkinson+s+disease+classification.zip"


def _md5(b):
    return hashlib.md5(b).hexdigest()


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def fetch_openml(did):
    """Download one OpenML dataset (ARFF), verify md5, return the local path."""
    name = OPENML[did]
    os.makedirs(os.path.join(RAW, "openml"), exist_ok=True)
    path = os.path.join(RAW, "openml", "%d_%s.arff" % (did, name))
    meta = json.load(urllib.request.urlopen(urllib.request.Request(
        "https://www.openml.org/api/v1/json/data/%d" % did, headers=UA), timeout=60))["data_set_description"]
    if os.path.exists(path) and _md5(open(path, "rb").read()) == meta["md5_checksum"]:
        return path, meta
    blob = urllib.request.urlopen(urllib.request.Request(meta["url"], headers=UA), timeout=300).read()
    if _md5(blob) != meta["md5_checksum"]:
        raise RuntimeError("md5 mismatch for OpenML %d" % did)
    with open(path, "wb") as f:
        f.write(blob)
    return path, meta


def _decode(v):
    return v.decode("utf-8") if isinstance(v, bytes) else v


def load_arff(path, target):
    """Small ARFF reader that also accepts STRING attributes (scipy's reader does not).  Dense format only."""
    import re
    import pandas as pd
    names, data_lines, in_data = [], [], False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            if in_data:
                data_lines.append(s)
                continue
            low = s.lower()
            if low.startswith("@attribute"):
                m = re.match(r"@attribute\s+('(?:[^']|\\')*'|\"[^\"]*\"|\S+)\s+(.*)$", s, flags=re.I)
                names.append(m.group(1).strip("'\""))
            elif low.startswith("@data"):
                in_data = True
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), header=None, names=names, quotechar="'",
                     skipinitialspace=True, na_values=["?"])
    ti = [n.lower() for n in names].index(target.lower())
    tname = names[ti]
    y_raw = df[tname].astype(str).str.strip().str.strip("'\"").values
    feats = df.drop(columns=[tname])
    if not all(np.issubdtype(t, np.number) for t in feats.dtypes):
        raise ValueError("non-numeric feature column in %s" % path)
    return feats.values.astype(np.float64), y_raw, list(feats.columns)


def binarize(y_raw):
    """minority class -> 1, majority -> 0 (ties broken by sorted label)."""
    labs, cnt = np.unique(y_raw, return_counts=True)
    if len(labs) != 2:
        raise ValueError("expected 2 classes, got %s" % labs)
    pos = labs[np.argmin(cnt)] if cnt[0] != cnt[1] else labs[1]
    return (y_raw == pos).astype(np.int8), str(pos)


def _finalize(name, X, y, groups, source, raw_md5, extra=None):
    """drop columns that are constant over ALL rows (label-free, benign), store float32 npz + data card."""
    keep = X.std(axis=0) > 0
    X = X[:, keep].astype(np.float32)
    os.makedirs(PROC, exist_ok=True)
    out = os.path.join(PROC, name + ".npz")
    np.savez_compressed(out, X=X, y=y, groups=(groups if groups is not None else np.array([], dtype=np.int32)))
    n1 = int(y.sum())
    card = dict(name=name, n=int(X.shape[0]), p=int(X.shape[1]), p_dropped_constant=int((~keep).sum()),
                n_minority=n1, n_majority=int(len(y) - n1), minority_fraction=round(n1 / len(y), 4),
                n_groups=int(len(np.unique(groups))) if groups is not None else 0, source=source,
                raw_md5=raw_md5, processed_sha256=_sha256_file(out), has_nan=bool(np.isnan(X).any()))
    if extra:
        card.update(extra)
    with open(os.path.join(PROC, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(card, f, indent=1, ensure_ascii=False)
    return card


def build_openml(did):
    path, meta = fetch_openml(did)
    X, y_raw, _ = load_arff(path, meta["default_target_attribute"].split(",")[0])
    y, pos = binarize(y_raw)
    return _finalize(OPENML[did], X, y, None, "OpenML %d (%s)" % (did, meta["name"]), meta["md5_checksum"],
                     dict(positive_label=pos))


def build_parkinson():
    """UCI 470: 252 subjects x 3 recordings, 754 columns incl. id / gender / class."""
    import pandas as pd
    os.makedirs(os.path.join(RAW, "uci"), exist_ok=True)
    zpath = os.path.join(RAW, "uci", "parkinson_470.zip")
    if not os.path.exists(zpath):
        with open(zpath, "wb") as f:
            f.write(urllib.request.urlopen(urllib.request.Request(UCI_PARKINSON, headers=UA), timeout=300).read())
    # The UCI zip contains pd_speech_features.rar, which holds the single file pd_speech_features.csv.
    # Python cannot read RAR; extract once with 7-Zip:  7z x pd_speech_features.rar -o<RAW>/uci/pd
    member = os.path.join(RAW, "uci", "pd", "pd_speech_features.csv")
    if not os.path.exists(member):
        raise FileNotFoundError("extract pd_speech_features.rar first (see comment above): " + member)
    df = pd.read_csv(member, header=1)
    ids = df["id"].values
    y = df["class"].values.astype(np.int8)                  # 1 = Parkinson's (majority here)
    feats = df.drop(columns=["id", "class"])
    X = feats.values.astype(np.float64)
    _, groups = np.unique(ids, return_inverse=True)
    # for 'minority = positive' convention keep class labels as given but record who is minority
    y_min = (1 - y) if y.mean() > 0.5 else y
    return _finalize("parkinson", X, y_min.astype(np.int8), groups.astype(np.int32),
                     "UCI 470 Parkinson's Disease Classification", _md5(open(zpath, "rb").read()),
                     dict(member="pd_speech_features.csv", minority_is_healthy=bool(y.mean() > 0.5)))


def load(name):
    z = np.load(os.path.join(PROC, name + ".npz"))
    g = z["groups"]
    return z["X"].astype(np.float64), z["y"].astype(np.int64), (g if len(g) else None)


def all_names():
    return [OPENML[d] for d in OPENML] + ["parkinson"]
