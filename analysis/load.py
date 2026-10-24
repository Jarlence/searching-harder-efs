"""Load every finished job file into tidy pandas tables (cached).  No statistics here."""
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
JOBS = os.environ.get("EFS_JOBS", os.path.join(ROOT, "results", "jobs"))            # EFS_JOBS / EFS_OUT: debugging on a fixture
OUT = os.environ.get("EFS_OUT", os.path.join(ROOT, "results"))
CACHE = os.path.join(OUT, "cache")
TAB = os.path.join(OUT, "tables")
FIG = os.path.join(OUT, "figures")
for _d in (TAB, FIG):
    os.makedirs(_d, exist_ok=True)
METRICS = ("ba", "gm", "acc", "f1", "mcc", "auc", "tpr", "tnr")
HD = ["colon", "cns", "dlbcl", "leukemia", "prostate", "gli", "smk", "ovarian", "breast"]
IMB = ["oil_spill", "ozone", "speech", "kc3"]
CORE = HD + IMB
ALGOS = ["ga", "bpso", "bde", "nsga2", "rs"]


def _base(spec):
    return dict(suite=spec["suite"], ds=spec["ds"], algo=spec.get("algo", spec.get("method", "")), learner=spec.get("learner", ""),
                rep=spec.get("rep", -1), fold=spec.get("fold", -1), seed=spec.get("seed", 0),
                perm=spec.get("perm", -1) if spec.get("perm") is not None else -1,
                metric=spec.get("metric", "ba"), outer_group=bool(spec.get("outer_group", False)),
                inner_group=bool(spec.get("inner_group", False)), group=bool(spec.get("group", False)),
                mode=spec.get("mode", ""), protocol=spec.get("protocol", ""))


def _read_suite(suite):
    out = []
    for f in glob.glob(os.path.join(JOBS, suite, "*.json")):
        try:
            out.append(json.load(open(f)))
        except Exception:  # noqa: BLE001  half-written file cannot exist (atomic replace) but be safe
            pass
    return out


def load_all(use_cache=True):
    os.makedirs(CACHE, exist_ok=True)
    stamp = sum(len(os.listdir(os.path.join(JOBS, d))) for d in os.listdir(JOBS) if os.path.isdir(os.path.join(JOBS, d)))
    cpath = os.path.join(CACHE, "tables.pkl")
    if use_cache and os.path.exists(cpath):
        c = pickle.load(open(cpath, "rb"))
        if c["stamp"] == stamp:
            return c["tables"]
    honest, full, base, so, si = [], [], [], [], []
    secs = []
    for suite in sorted(d for d in os.listdir(JOBS) if os.path.isdir(os.path.join(JOBS, d))):
        for rec in _read_suite(suite):
            if rec.get("status") != "ok":
                continue
            s = rec["spec"]; res = rec["result"]; b = _base(s)
            secs.append(dict(suite=suite, ds=s["ds"], learner=s.get("learner", ""), algo=b["algo"], kind=s["kind"], seconds=rec["seconds"]))
            kind = s["kind"]
            if kind == "honest":
                for cp, r in res.items():
                    row = dict(b, cp=int(cp), k=r["k"], inner_err=r["inner_err"], sel=np.array(r["sel"], dtype=np.int32))
                    for m in METRICS:
                        row["test_" + m] = r["test_" + m]
                    if "front_test_ba" in r:
                        row["front_test_ba"] = np.array(r["front_test_ba"]); row["front_k"] = np.array(r["front_k"])
                        row["front_inner_err"] = np.array(r["front_inner_err"])
                    honest.append(row)
            elif kind == "full":
                for cp, r in res.items():
                    full.append(dict(b, cp=int(cp), k=r["k"], reported_err=r["reported_err"], leaky_cv_ba=r["leaky_cv_ba"],
                                     leaky_cv_auc=r["leaky_cv_auc"]))
            elif kind == "baseline":
                row = dict(b, k=res["k"])
                for m in METRICS:
                    row["test_" + m] = res["test_" + m]
                base.append(row)
            elif kind == "smote_outer":
                so.append(dict(b, **{m: res[m] for m in METRICS}))
            elif kind == "smote_inner":
                for cp, r in res.items():
                    row = dict(b, cp=int(cp), k=r["k"], inner_err=r["inner_err"])
                    for m in METRICS:
                        row["test_" + m] = r["test_" + m]
                    si.append(row)
    bcols = list(_base(dict(suite="", ds="")).keys())
    tables = dict(honest=pd.DataFrame(honest, columns=None if honest else bcols + ["cp", "k", "inner_err", "sel"] + ["test_" + m for m in METRICS]),
                  full=pd.DataFrame(full, columns=None if full else bcols + ["cp", "k", "reported_err", "leaky_cv_ba", "leaky_cv_auc"]),
                  base=pd.DataFrame(base, columns=None if base else bcols + ["k"] + ["test_" + m for m in METRICS]),
                  smote_outer=pd.DataFrame(so, columns=None if so else bcols + list(METRICS)),
                  smote_inner=pd.DataFrame(si, columns=None if si else bcols + ["cp", "k", "inner_err"] + ["test_" + m for m in METRICS]),
                  secs=pd.DataFrame(secs, columns=None if secs else ["suite", "ds", "learner", "algo", "kind", "seconds"]))
    pickle.dump(dict(stamp=stamp, tables=tables), open(cpath, "wb"))
    return tables


EXTRA = os.environ.get("EFS_JOBS_EXTRA", os.path.join(ROOT, "results", "jobs_extra"))


def load_pareto():
    """X8p (docs/DEVIATIONS.md #7): the NSGA-II honest runs of E1h replayed with the Pareto ARCHIVE.  One row per (ds, learner, rep, fold, cp).
    d_merged: D with one representative per objective vector (primary); d_ties: D when every mask tied at a non-dominated objective vector may be picked."""
    rows = []
    for f in glob.glob(os.path.join(EXTRA, "X8p", "*.json")):
        rec = json.load(open(f))
        if rec.get("status") != "ok":
            continue
        s = rec["spec"]
        for cp, r in rec["result"].items():
            rows.append(dict(ds=s["ds"], learner=s["learner"], rep=s["rep"], fold=s["fold"], cp=int(cp), k=r["k"], inner_err=r["inner_err"],
                             test_ba=r["test_ba"], d_merged=max(r["arch_test_ba"]) - r["test_ba"], d_ties=max(r["arch_tie_max_ba"]) - r["test_ba"],
                             n_front=len(r["arch_k"]), n_tied=int(sum(r["arch_tie_n"])), calls=r["calls"], distinct=r["distinct"]))
    return pd.DataFrame(rows, columns=["ds", "learner", "rep", "fold", "cp", "k", "inner_err", "test_ba", "d_merged", "d_ties", "n_front", "n_tied",
                                       "calls", "distinct"])


def cards():
    c = json.load(open(os.path.join(ROOT, "data", "data_cards.json")))
    return {r["name"]: r for r in c}


if __name__ == "__main__":
    t = load_all(use_cache=False)
    for k, v in t.items():
        print(k, v.shape)
