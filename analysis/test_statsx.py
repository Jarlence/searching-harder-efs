"""Unit tests of the analysis helpers (run:  python analysis/test_statsx.py)."""
import os
import sys
import unittest

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import statsx as SX  # noqa: E402


class T(unittest.TestCase):
    def test_holm_matches_reference(self):
        p = [0.01, 0.04, 0.03, 0.005]
        adj = SX.holm(p)
        # reference (R p.adjust(method='holm')): 0.03 0.06 0.06 0.02
        np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02], atol=1e-12)
        self.assertTrue(np.all(adj >= np.array(p)))

    def test_clopper_pearson(self):
        lo, hi = SX.clopper_pearson(7, 10)
        self.assertAlmostEqual(lo, 0.3475471698, places=6)          # reference values (R binom.test)
        self.assertAlmostEqual(hi, 0.9332605, places=5)
        self.assertEqual(SX.clopper_pearson(0, 5)[0], 0.0)
        self.assertEqual(SX.clopper_pearson(5, 5)[1], 1.0)

    def test_exact_expected_max_vs_monte_carlo(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=20)
        for s in (1, 3, 5, 10, 20):
            mc = np.mean([rng.choice(x, s, replace=False).max() for _ in range(40000)])
            self.assertAlmostEqual(SX.exact_expected_max(x, s), mc, delta=0.02)
        self.assertAlmostEqual(SX.exact_expected_max(x, 1), x.mean(), places=12)
        self.assertAlmostEqual(SX.exact_expected_max(x, 20), x.max(), places=12)
        self.assertTrue(np.isnan(SX.exact_expected_max(x, 21)))

    def test_sign_delta_and_wilcoxon(self):
        self.assertEqual(SX.sign_delta([1, 2, 3, -1]), 0.5)
        self.assertEqual(SX.wilcoxon([0, 0, 0]), 1.0)
        d = np.arange(1, 14, dtype=float)
        self.assertAlmostEqual(SX.wilcoxon(d), 2.0 / 2 ** 13, places=10)   # exact two-sided p, all positive, n = 13

    def test_double_boot_recovers_known_mean_and_coverage(self):
        rng = np.random.default_rng(1)
        per = {}
        true = []
        for d in range(13):
            mu = rng.normal(0.1, 0.03)
            true.append(mu)
            M = mu + rng.normal(0, 0.05, size=(40, 6))          # 40 runs x 6 cells
            per[d] = [(M, 1.0)]
        bt = SX.double_boot(per, n_boot=3000, seed=5)
        self.assertAlmostEqual(bt["point"], np.mean([SX.ds_value(per[d]) for d in per]), places=12)
        self.assertLess(abs(bt["point"] - np.mean(true)), 0.03)
        self.assertLess(bt["lo"], bt["point"]); self.assertGreater(bt["hi"], bt["point"])
        # CI width should be close to the data-set-level standard error (1.96 * sd / sqrt(13)), up to the within-data-set noise
        sd = np.std([bt["ds_values"][k] for k in bt["ds_values"]], ddof=1)
        self.assertGreater(bt["hi"] - bt["lo"], 0.8 * 2 * 1.96 * sd / np.sqrt(13))
        self.assertLess(bt["hi"] - bt["lo"], 1.6 * 2 * 1.96 * sd / np.sqrt(13))

    def test_two_source_difference_and_nan_handling(self):
        A = np.full((30, 3), 0.8); B = np.full((50, 3), 0.7)
        B[:, 2] = np.nan                                          # a cell without honest runs is dropped (all-NaN column)
        v = SX.ds_value([(A, 1.0), (B, -1.0)])
        self.assertAlmostEqual(v, 0.1, places=12)
        bt = SX.double_boot({"a": [(A, 1.0), (B, -1.0)], "b": [(A + 0.02, 1.0), (B, -1.0)]}, n_boot=500, seed=1)
        self.assertAlmostEqual(bt["point"], 0.11, places=12)

    def test_double_boot_best_known_values(self):
        # 3 runs x 4 algos x 2 cells; run-means per algo = [0.6, 0.7, 0.8, 0.9] in both cells -> max - mean = 0.9 - 0.75 = 0.15
        base = np.array([0.6, 0.7, 0.8, 0.9])
        A = np.stack([np.tile(base[:, None], (1, 2)) for _ in range(3)])            # [3, 4, 2]
        bt = SX.double_boot_best({"a": A, "b": A}, n_boot=200, seed=1)
        self.assertAlmostEqual(bt["point"], 0.15, places=12)
        self.assertAlmostEqual(bt["lo"], 0.15, places=12); self.assertAlmostEqual(bt["hi"], 0.15, places=12)   # no noise -> degenerate interval
        # identical algorithms plus noise: the best-of statistic is positive and shrinks with more runs
        rng = np.random.default_rng(0)
        small = {"d": 0.7 + rng.normal(0, 0.05, size=(10, 5, 3))}
        large = {"d": 0.7 + rng.normal(0, 0.05, size=(200, 5, 3))}
        self.assertGreater(SX.double_boot_best(small, 200, 1)["point"], SX.double_boot_best(large, 200, 1)["point"])
        self.assertGreater(SX.double_boot_best(large, 200, 1)["point"], 0.0)

    def test_cluster_boot_and_t_interval(self):
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
        lo, hi = SX.cluster_boot(v, 20000, 1)
        self.assertLess(lo, v.mean()); self.assertGreater(hi, v.mean())
        se = v.std(ddof=1) / np.sqrt(len(v))
        self.assertAlmostEqual((hi - lo) / 2, 1.96 * se * np.sqrt((len(v) - 1) / len(v)), delta=0.15)       # bootstrap SE of the mean uses ddof=0
        tl, th = SX.t_interval(v)
        self.assertAlmostEqual(tl, v.mean() - stats.t.ppf(0.975, 12) * se, places=10)
        self.assertAlmostEqual(th, v.mean() + stats.t.ppf(0.975, 12) * se, places=10)
        c = SX.cluster_boot(np.full(7, 3.0), 1000, 2)
        self.assertEqual(c, (3.0, 3.0))
        self.assertTrue(np.isnan(SX.cluster_boot([1.0], 100, 0)[0]))

    def test_double_boot_reports_t_interval_as_primary(self):
        rng = np.random.default_rng(5)
        per = {"d%d" % i: [(rng.normal(0.1 * i, 0.05, size=(50, 1)), 1.0)] for i in range(10)}
        bt = SX.double_boot(per, 4000, seed=3)
        dv = np.array(list(bt["ds_values"].values()))
        tl, th = SX.t_interval(dv)
        self.assertEqual((bt["lo"], bt["hi"]), (tl, th))
        cl, ch = SX.cluster_boot(dv, 4000, 3)
        self.assertEqual((bt["lo_cb"], bt["hi_cb"]), (cl, ch))
        self.assertLess(bt["lo_db"], bt["point"]); self.assertGreater(bt["hi_db"], bt["point"])
        self.assertLess(bt["hi"] - bt["lo"], 3.0)
        self.assertGreater(bt["hi"] - bt["lo"], bt["hi_cb"] - bt["lo_cb"])          # the t interval is the wider (more honest) one for 10 data sets

    def test_spearman_cell(self):
        self.assertAlmostEqual(SX.spearman_cell([250, 1000, 4000, 16000], [0.1, 0.2, 0.3, 0.4]), 1.0)
        self.assertAlmostEqual(SX.spearman_cell([250, 1000, 4000, 16000], [0.4, 0.3, 0.2, 0.1]), -1.0)
        self.assertTrue(np.isnan(SX.spearman_cell([250, 1000, 4000], [0.1, 0.1, 0.1])))


if __name__ == "__main__":
    unittest.main()
