"""Independent arithmetic QA of the complete R22 outputs.

Does not import the R22 summariser or generate/select additional scientific tests.
Written after the execution plan was frozen; it verifies the nine fixed estimands.
"""
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/r22_strict_scaling"
FREEZE = ROOT / "docs/STRICT_SCALING_R22_FREEZE.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, name):
    if not math.isclose(a, b, abs_tol=1e-9, rel_tol=1e-10):
        raise AssertionError("Arithmetic mismatch: " + name)


def main():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    freeze_digest = digest(FREEZE)
    assert len(freeze["jobs"]) == 5350
    summary = list(csv.DictReader((OUT / "summary.csv").open(encoding="utf-8")))
    cards = {r["name"]: r for r in json.loads((ROOT / "data/data_cards.json").read_text(encoding="utf-8"))}
    scores, fingerprints = {}, {}
    for spec in freeze["jobs"]:
        name = "X10s__%s__%s__%s__rep%d__fol%d.json" % (spec["ds"], spec["algo"], spec["learner"], spec["rep"], spec["fold"])
        path = ROOT / "results/jobs_extra/X10s" / name
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["status"] == "ok" and raw["spec"] == spec
        assert raw["freeze_sha256"] == freeze_digest
        assert raw["source_fingerprint"] == freeze["source_fingerprint"]
        assert set(raw["result"]) == {"250", "1000", "4000"}
        fingerprints[str(path.relative_to(ROOT)).replace("\\", "/")] = digest(path)
        n, p = cards[spec["ds"]]["n"], cards[spec["ds"]]["p"]
        splits = raw["splits"]
        assert sorted(splits["outer_train"] + splits["outer_test"]) == list(range(n))
        last_distinct = 0
        for cp in (250, 1000, 4000):
            r = raw["result"][str(cp)]
            assert r["k"] == len(set(r["sel"])) == len(r["sel"])
            assert 1 <= r["k"] <= min(100, p)
            assert all(isinstance(x, int) and 0 <= x < p for x in r["sel"])
            assert last_distinct <= r["distinct"] <= cp and r["calls"] == cp
            last_distinct = r["distinct"]
            key = (spec["ds"], spec["learner"], spec["algo"], spec["rep"], spec["fold"], cp)
            assert key not in scores
            scores[key] = 1-r["inner_err"]-r["test_ba"]
    assert len(scores) == 16050
    assert fingerprints == json.loads((OUT / "job_hashes.json").read_text(encoding="utf-8"))
    focal = [r for r in summary if r["focal"] == "True"]
    assert len(focal) == 9
    checks = []
    for row in focal:
        dataset_values = defaultdict(list)
        if row["estimand"] == "budget_increment":
            cell_values = defaultdict(list)
            for (ds, learner, algo, repeat, fold, cp), value in scores.items():
                if cp == 4000:
                    cell_values[(ds, learner, algo)].append(value-scores[(ds, learner, algo, repeat, fold, 250)])
            for (ds, learner, algo), values in cell_values.items():
                dataset_values[ds].append(statistics.mean(values))
        else:
            for (ds, learner, algo, repeat, fold, cp), value in scores.items():
                if (learner, algo, cp) == (row["learner"], row["algorithm"], 4000):
                    dataset_values[ds].append(value-scores[(ds, learner, "rs", repeat, fold, cp)])
        values = [100*statistics.mean(v) for v in dataset_values.values()]
        n = len(values)
        assert n == int(row["n_datasets"])
        assert n == (11 if row["learner"] == "knn" else 13)
        mean = statistics.mean(values)
        se = statistics.stdev(values)/math.sqrt(n)
        close(mean, float(row["mean_pp"]), "focal mean")
        for alpha, lo, hi in ((.05, "ci95_lo_pp", "ci95_hi_pp"), (.05/9, "simultaneous_lo_pp", "simultaneous_hi_pp")):
            half = float(t.ppf(1-alpha/2, n-1))*se
            close(mean-half, float(row[lo]), lo)
            close(mean+half, float(row[hi]), hi)
        checks.append(dict(estimand=row["estimand"], learner=row["learner"], algorithm=row["algorithm"], n_datasets=n))
    payload = dict(created_utc=datetime.now(timezone.utc).isoformat(), jobs=5350, checkpoints=16050,
                   verifier_sha256=digest(Path(__file__)), freeze_sha256=freeze_digest,
                   focal_family_size=9, independent_arithmetic=True, raw_job_hashes_match=True,
                   masks_and_outer_partitions_valid=True, checks=checks)
    (OUT / "independent_verification.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    print("Independent QA passed: 5350 jobs, 16050 checkpoints, all nine fixed means and both interval types.")


if __name__ == "__main__":
    main()
