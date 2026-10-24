"""Correctness tests for efsbench.  Run:  python -m unittest discover -s tests -v   (from the project root, evolve env)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from efsbench import algos as A, learners as L, metrics as M, protocols as P, smote as SM, splits as S  # noqa: E402
from efsbench.wrapper import Wrapper  # noqa: E402


def toy(n=80, p=60, seed=0, pos_frac=0.3, signal=1.0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_frac).astype(np.int64)
    y[:3] = 1; y[3:6] = 0
    X = rng.standard_normal((n, p)); X[:, :4] += signal * y[:, None]
    return X, y


class TestSplits(unittest.TestCase):
    def test_stratified(self):
        X, y = toy(97)
        folds = S.stratified_kfold(y, 5, np.random.default_rng(1))
        te = np.concatenate([f[1] for f in folds])
        self.assertEqual(sorted(te.tolist()), list(range(len(y))))              # a partition
        sizes = [len(f[1]) for f in folds]
        self.assertLessEqual(max(sizes) - min(sizes), 1)
        pos = [int(y[f[1]].sum()) for f in folds]
        self.assertLessEqual(max(pos) - min(pos), 1)
        for tr, t in folds:
            self.assertFalse(set(tr) & set(t))

    def test_group(self):
        rng = np.random.default_rng(0)
        g = np.repeat(np.arange(40), 3); y = np.repeat((rng.random(40) < 0.3).astype(int), 3)
        folds = S.stratified_group_kfold(y, g, 5, np.random.default_rng(2))
        for tr, te in folds:
            self.assertFalse(set(g[tr]) & set(g[te]))                           # no subject on both sides
        self.assertEqual(sum(len(f[1]) for f in folds), len(y))
        pos = [int(y[f[1]].sum()) for f in folds]
        self.assertLessEqual(max(pos) - min(pos), 6)                            # roughly balanced (units of 3)

    def test_rng_stable(self):
        a = S.rng_for("x", 1).integers(0, 10 ** 9); b = S.rng_for("x", 1).integers(0, 10 ** 9)
        self.assertEqual(a, b)
        self.assertNotEqual(a, S.rng_for("x", 2).integers(0, 10 ** 9))


class TestMetrics(unittest.TestCase):
    def test_against_sklearn(self):
        from sklearn import metrics as sk
        rng = np.random.default_rng(3)
        y = (rng.random(200) < 0.2).astype(int); s = rng.random(200) + 0.5 * y; pred = (s > 0.8).astype(int)
        self.assertAlmostEqual(M.ba(y, pred), sk.balanced_accuracy_score(y, pred))
        self.assertAlmostEqual(M.f1(y, pred), sk.f1_score(y, pred))
        self.assertAlmostEqual(M.mcc(y, pred), sk.matthews_corrcoef(y, pred))
        self.assertAlmostEqual(M.auc(y, s), sk.roc_auc_score(y, s))
        s2 = np.round(s, 1)                                                      # ties
        self.assertAlmostEqual(M.auc(y, s2), sk.roc_auc_score(y, s2))


class TestLearners(unittest.TestCase):
    def test_knn_oof_matches_sklearn(self):
        from sklearn.neighbors import KNeighborsClassifier
        X, y = toy(70, 12, seed=4)
        folds = S.stratified_kfold(y, 5, np.random.default_rng(5))
        s = L.knn_oof(X, y, folds, 5)
        ref = np.empty(len(y))
        for tr, te in folds:
            ref[te] = KNeighborsClassifier(5, algorithm="brute").fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        self.assertTrue(np.allclose(s, ref))

    def test_ridge_matches_sklearn_primal_and_dual(self):
        from sklearn.linear_model import Ridge
        for n, p in ((60, 8), (30, 50)):                                         # primal / dual branch
            X, y = toy(n, p, seed=6)
            w = L.balanced_weights(y)
            mu, mt, beta = L._ridge_fit(X, 2.0 * y - 1.0, w, 1.0)
            ref = Ridge(alpha=1.0).fit(X, 2.0 * y - 1.0, sample_weight=w)
            Z = np.random.default_rng(7).standard_normal((15, p))
            self.assertTrue(np.allclose((Z - mu) @ beta + mt, ref.predict(Z), atol=1e-8))

    def test_oof_ridge_runs(self):
        X, y = toy(60, 10, seed=8)
        folds = S.stratified_kfold(y, 5, np.random.default_rng(9))
        aux = L.prepare_cv(y, folds, L.SPECS["ridge"])
        s = L.oof_scores(X, y, folds, L.SPECS["ridge"], aux)
        self.assertTrue(M.auc(y, s) > 0.8)                                       # signal is learnable


class TestWrapperAndAlgos(unittest.TestCase):
    def setUp(self):
        self.X, self.y = toy(70, 40, seed=10)
        self.Xs, = P.standardize(self.X)
        self.folds = S.stratified_kfold(self.y, 5, np.random.default_rng(11))

    def test_call_counting_and_cache(self):
        w = Wrapper(self.Xs, self.y, self.folds, L.SPECS["ridge"])
        m = np.zeros(40, bool); m[:5] = True
        a = w.evaluate(m); b = w.evaluate(m)
        self.assertEqual(w.calls, 2); self.assertEqual(len(w.cache), 1); self.assertEqual(a, b)
        self.assertAlmostEqual(w.oof_metrics(m)["ba"], 1 - a)

    def test_every_algorithm_hits_budget_exactly(self):
        for name in A.ALGOS:
            for learner in ("ridge", "knn"):
                w = Wrapper(self.Xs, self.y, self.folds, L.SPECS[learner])
                run = A.Run(w, (30, 120, 300))
                A.run_algo(name, run, np.random.default_rng(0))
                self.assertEqual(w.calls, 300, name)
                self.assertEqual(sorted(run.snap), [30, 120, 300], name)
                lex = [run.snap[c]["lex_err"] for c in (30, 120, 300)]
                self.assertTrue(lex[0] >= lex[1] >= lex[2], name)               # incumbents never get worse
                for c in (30, 120, 300):
                    self.assertGreaterEqual(len(run.snap[c]["wsum_idx"]), 1)
                if name == "nsga2":
                    self.assertIsNotNone(run.snap[300]["front"])

    def test_bde_stays_sparse_in_high_dimension(self):
        """regression test for DEVIATIONS #1: the threshold-based BDE densified the masks (hundreds of features)."""
        X, y = toy(70, 600, seed=21)
        Xs, = P.standardize(X)
        folds = S.stratified_kfold(y, 5, np.random.default_rng(22))
        w = Wrapper(Xs, y, folds, L.SPECS["ridge"])
        run = A.Run(w, (3000,))
        A.run_algo("bde", run, np.random.default_rng(3))
        self.assertLessEqual(len(run.snap[3000]["wsum_idx"]), A.KMAX)
        self.assertEqual(w.calls, 3000)

    def test_cardinality_cap_repairs_in_place(self):
        X, y = toy(70, 300, seed=23)
        Xs, = P.standardize(X)
        w = Wrapper(Xs, y, S.stratified_kfold(y, 5, np.random.default_rng(24)), L.SPECS["ridge"])
        run = A.Run(w, (10,))
        m = np.zeros(300, bool); m[:180] = True
        _, k, _ = run.eval(m)
        self.assertEqual(k, A.KMAX)
        self.assertEqual(int(m.sum()), A.KMAX)                                   # the caller's array was repaired
        self.assertEqual(len(run.best_idx), A.KMAX)

    def test_algorithms_beat_chance_on_signal(self):
        for name in ("ga", "nsga2", "bpso", "bde"):
            w = Wrapper(self.Xs, self.y, self.folds, L.SPECS["ridge"])
            run = A.Run(w, (600,)); A.run_algo(name, run, np.random.default_rng(1))
            self.assertLess(run.snap[600]["lex_err"], 0.25, name)


class TestNoLeak(unittest.TestCase):
    """The honest protocol must be a function of the training rows only."""

    def test_selection_ignores_heldout_part(self):
        X, y = toy(90, 50, seed=12)
        tr, te = np.arange(60), np.arange(60, 90)
        rng = np.random.default_rng(13)
        for algo in A.ALGOS:
            for learner in ("ridge", "knn"):
                real = P.honest_core(X[tr], y[tr], X[te], y[te], algo, learner, ("t", 1), ("a", 1), (60, 240))
                junk = P.honest_core(X[tr], y[tr], rng.standard_normal((30, 50)) * 50, rng.integers(0, 2, 30),
                                     algo, learner, ("t", 1), ("a", 1), (60, 240))
                for cp in (60, 240):
                    self.assertEqual(real[cp]["sel"], junk[cp]["sel"], (algo, learner))
                    self.assertEqual(real[cp]["inner_err"], junk[cp]["inner_err"])

    def test_determinism(self):
        X, y = toy(90, 50, seed=14)
        tr, te = np.arange(60), np.arange(60, 90)
        a = P.honest_core(X[tr], y[tr], X[te], y[te], "ga", "ridge", ("t", 2), ("a", 2), (200,))
        b = P.honest_core(X[tr], y[tr], X[te], y[te], "ga", "ridge", ("t", 2), ("a", 2), (200,))
        self.assertEqual(a, b)

    def test_flawed_protocol_is_optimistic_on_pure_noise(self):
        """No signal at all: honest ~ 0.5, selection-before-split looks clearly better than chance."""
        rng = np.random.default_rng(15)
        n, p = 60, 400
        X = rng.standard_normal((n, p)); y = np.zeros(n, np.int64); y[rng.permutation(n)[:20]] = 1
        flaw = []; hon = []
        for s in range(6):
            r = P.fulldata_core(X, y, "ga", "ridge", ("n", s), ("a", s), (800,))
            flaw.append(r[800]["leaky_cv_ba"])
            folds = S.stratified_kfold(y, 5, np.random.default_rng(100 + s))
            for i, (tr, te) in enumerate(folds[:3]):
                h = P.honest_core(X[tr], y[tr], X[te], y[te], "ga", "ridge", ("n", s, i), ("a", s, i), (800,))
                hon.append(h[800]["test_ba"])
        self.assertGreater(np.mean(flaw), 0.62)
        self.assertLess(abs(np.mean(hon) - 0.5), 0.08)


class TestSmote(unittest.TestCase):
    def test_points_on_segments_and_counts(self):
        X, y = toy(100, 6, seed=16, pos_frac=0.15)
        rng = np.random.default_rng(17)
        plan = SM.smote_plan(X, y, rng)
        Xn = SM.smote_apply(X, plan)
        self.assertEqual(len(Xn), int((y == 0).sum() - (y == 1).sum()))
        i, j, lam = plan
        self.assertTrue(((lam >= 0) & (lam <= 1)).all())
        self.assertTrue((y[i] == 1).all() and (y[j] == 1).all())
        self.assertTrue(np.allclose(Xn, X[i] + lam[:, None] * (X[j] - X[i])))
        Xa, ya = SM.smote(X, y, np.random.default_rng(17))
        self.assertEqual(int((ya == 1).sum()), int((ya == 0).sum()))


if __name__ == "__main__":
    unittest.main()
