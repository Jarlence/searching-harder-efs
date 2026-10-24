"""Tests for efsbench.tracking (added after the main experiments; docs/DEVIATIONS.md #7).
Run:  python -m unittest discover -s tests -v   (from the project root, evolve env)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from efsbench import algos as A, learners as L, protocols as P, splits as S  # noqa: E402
from efsbench import tracking as T  # noqa: E402
from efsbench.wrapper import Wrapper  # noqa: E402


def toy(n=80, p=60, seed=0, pos_frac=0.3, signal=1.0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_frac).astype(np.int64)
    y[:3] = 1; y[3:6] = 0
    X = rng.standard_normal((n, p)); X[:, :4] += signal * y[:, None]
    return X, y


class TestParetoArchive(unittest.TestCase):
    def test_against_brute_force_with_ties(self):
        rng = np.random.default_rng(0)
        for trial in range(200):
            n = int(rng.integers(1, 60))
            cands = [(float(rng.integers(0, 8)) / 10.0, int(rng.integers(1, 9)), np.array([i], np.int32)) for i in range(n)]
            keep = T.pareto_archive(cands)
            vecs = {(c[0], c[1]) for c in cands}
            brute = {v for v in vecs if not any((u[0] <= v[0] and u[1] <= v[1]) and (u[0] < v[0] or u[1] < v[1]) for u in vecs)}
            self.assertEqual({(cands[i][0], cands[i][1]) for i in keep}, brute)
            self.assertEqual(len(keep), len(brute))                                         # one representative per objective vector
            for i in keep:                                                                   # ... and it is the first one evaluated
                first = min(j for j, c in enumerate(cands) if (c[0], c[1]) == (cands[i][0], cands[i][1]))
                self.assertEqual(i, first)
            ks = [cands[i][1] for i in keep]
            es = [cands[i][0] for i in keep]
            self.assertEqual(ks, sorted(ks)); self.assertTrue(all(a > b for a, b in zip(es, es[1:])))   # k up, error strictly down

    def test_lexicographic_incumbent_is_in_the_archive(self):
        rng = np.random.default_rng(1)
        cands = [(float(rng.integers(0, 6)) / 10.0, int(rng.integers(1, 7)), np.array([i], np.int32)) for i in range(50)]
        keep = T.pareto_archive(cands)
        lex = None
        for i, c in enumerate(cands):                                                        # the incumbent rule of algos.Run (strict <)
            if lex is None or (c[0], c[1]) < (cands[lex][0], cands[lex][1]):
                lex = i
        self.assertIn(lex, keep)


class TestTrackRun(unittest.TestCase):
    def setUp(self):
        self.X, self.y = toy(70, 40, seed=10)
        self.Xs, = P.standardize(self.X)
        self.folds = S.stratified_kfold(self.y, 5, np.random.default_rng(11))

    def test_identical_to_run_in_calls_mode(self):
        for name in A.ALGOS:
            w1 = Wrapper(self.Xs, self.y, self.folds, L.SPECS["ridge"]); r1 = A.Run(w1, (30, 120, 300))
            A.run_algo(name, r1, np.random.default_rng(5))
            w2 = Wrapper(self.Xs, self.y, self.folds, L.SPECS["ridge"]); r2 = T.TrackRun(w2, (30, 120, 300), "calls", archive=True)
            A.run_algo(name, r2, np.random.default_rng(5))
            self.assertEqual(w1.calls, w2.calls, name)
            for c in (30, 120, 300):
                a, b = r1.snap[c], r2.tsnap[c]
                for key in ("wsum_idx", "lex_idx"):
                    self.assertTrue(np.array_equal(a[key], b[key]), (name, c, key))
                for key in ("wsum_err", "lex_err"):
                    self.assertEqual(a[key], b[key], (name, c, key))
                self.assertEqual(b["calls"], c)
                self.assertLessEqual(b["distinct"], c)

    def test_archive_contains_incumbent_and_only_grows_in_quality(self):
        w = Wrapper(self.Xs, self.y, self.folds, L.SPECS["ridge"]); run = T.TrackRun(w, (50, 150, 300), "calls", archive=True)
        A.run_algo("nsga2", run, np.random.default_rng(6))
        prev = 1.0
        for c in (50, 150, 300):
            sn = run.tsnap[c]
            self.assertEqual(min(a[0] for a in sn["arch"]), sn["lex_err"])
            inc = [a for a in sn["arch"] if a[0] == sn["lex_err"]]
            self.assertEqual(len(inc), 1)
            self.assertTrue(np.array_equal(inc[0][2], sn["lex_idx"]))                       # the archive's best member IS the incumbent
            self.assertLessEqual(sn["lex_err"], prev); prev = sn["lex_err"]

    def test_distinct_mode_stops_at_the_target(self):
        X, y = toy(70, 14, seed=12)                                                          # few features -> many duplicate masks
        Xs, = P.standardize(X)
        folds = S.stratified_kfold(y, 5, np.random.default_rng(13))
        w = Wrapper(Xs, y, folds, L.SPECS["ridge"]); run = T.TrackRun(w, (40, 120), "distinct", call_cap=20000)
        A.run_algo("ga", run, np.random.default_rng(7))
        self.assertEqual(run.tsnap[40]["distinct"], 40); self.assertEqual(run.tsnap[120]["distinct"], 120)
        self.assertGreater(run.tsnap[120]["calls"], 120)                                     # duplicates were requested but not re-evaluated
        self.assertEqual(len(w.cache), 120)

    def test_cache_hits_are_counted_in_calls_mode(self):
        w = Wrapper(self.Xs, self.y, self.folds, L.SPECS["ridge"]); run = T.TrackRun(w, (300,), "calls")
        A.run_algo("ga", run, np.random.default_rng(8))
        self.assertEqual(run.tsnap[300]["calls"], 300)
        self.assertEqual(run.tsnap[300]["distinct"], len(w.cache))


class TestTrackedProtocol(unittest.TestCase):
    def test_same_numbers_as_honest_core(self):
        X, y = toy(90, 50, seed=14)
        tr, te = np.arange(60), np.arange(60, 90)
        for algo in ("ga", "nsga2", "rs"):
            a = P.honest_core(X[tr], y[tr], X[te], y[te], algo, "ridge", ("t", 3), ("a", 3), (60, 240))
            b = T.honest_core_tracked(X[tr], y[tr], X[te], y[te], algo, "ridge", ("t", 3), ("a", 3), (60, 240), archive=True)
            for cp in (60, 240):
                for key in ("k", "inner_err", "test_ba", "test_auc", "test_mcc"):
                    self.assertEqual(a[cp][key], b[cp][key], (algo, cp, key))

    def test_D_is_nonnegative_by_construction(self):
        X, y = toy(90, 50, seed=15)
        tr, te = np.arange(60), np.arange(60, 90)
        for seed in range(6):
            r = T.honest_core_tracked(X[tr], y[tr], X[te], y[te], "nsga2", "ridge", ("t", seed), ("a", seed), (50, 200, 400), archive=True)
            for cp in (50, 200, 400):
                d = max(r[cp]["arch_test_ba"]) - r[cp]["test_ba"]
                d_ties = max(r[cp]["arch_tie_max_ba"]) - r[cp]["test_ba"]
                self.assertGreaterEqual(d, 0.0)
                self.assertGreaterEqual(d_ties, d)                                      # keeping all masks tied in objective space can only help the test-set pick
                self.assertEqual(min(r[cp]["arch_err"]), r[cp]["inner_err"])
                self.assertTrue(all(n >= 1 for n in r[cp]["arch_tie_n"]))
                self.assertTrue(all(m >= b for m, b in zip(r[cp]["arch_tie_max_ba"], r[cp]["arch_test_ba"])))

    def test_archive_ignores_heldout_part(self):
        """leak detector for the archive: the candidates and their inner errors cannot depend on the outer-test rows."""
        X, y = toy(90, 50, seed=16)
        tr, te = np.arange(60), np.arange(60, 90)
        rng = np.random.default_rng(17)
        for mode in ("calls", "distinct"):
            real = T.honest_core_tracked(X[tr], y[tr], X[te], y[te], "nsga2", "knn", ("t", 4), ("a", 4), (60, 200), mode=mode, archive=True)
            junk = T.honest_core_tracked(X[tr], y[tr], rng.standard_normal((30, 50)) * 50, rng.integers(0, 2, 30), "nsga2", "knn",
                                         ("t", 4), ("a", 4), (60, 200), mode=mode, archive=True)
            for cp in (60, 200):
                self.assertEqual(real[cp]["arch_k"], junk[cp]["arch_k"]); self.assertEqual(real[cp]["arch_err"], junk[cp]["arch_err"])
                self.assertEqual(real[cp]["calls"], junk[cp]["calls"])


if __name__ == "__main__":
    unittest.main()
