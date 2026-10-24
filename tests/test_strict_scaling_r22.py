"""Implementation checks for strict inner scaling; no scientific acceptance gates."""
import os
import sys
import unittest
import tempfile
from pathlib import Path

for var in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[var]="1"
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"extras"))
import ex_suites6 as E
import ex_suites as OLD
import numpy as np
sys.path.insert(0,str(ROOT/"analysis"))
import strict_scaling_r22 as ANALYSIS


def toy():
    rng=np.random.default_rng(581)
    X=rng.normal(size=(90,43));X[:,0]=1
    y=np.tile([0,0,1],30).astype(np.int64)
    return X,y


class StrictScalingChecks(unittest.TestCase):
    def test_lazy_scores_equal_existing_x1(self):
        X,y=toy();folds=E.S.make_folds(y,5,E.S.rng_for("R22-test"))
        for learner in ("ridge","knn"):
            old=OLD.FoldScaledWrapper(X,y,folds,E.L.SPECS[learner])
            new=E.LazyFoldScaledWrapper(X,y,folds,E.L.SPECS[learner])
            rng=np.random.default_rng(17)
            for _ in range(40):
                mask=rng.random(X.shape[1])<.35;mask[1]=True
                self.assertEqual(old.evaluate(mask),new.evaluate(mask))
            self.assertEqual(old.calls,new.calls)
            self.assertEqual(old.cache,new.cache)

    def test_validation_rows_do_not_fit_their_fold_scaler(self):
        X,y=toy();folds=E.S.make_folds(y,5,E.S.rng_for("R22-isolation"))
        clean=E.LazyFoldScaledWrapper(X,y,folds,E.L.SPECS["ridge"])
        for index,(_,valid) in enumerate(folds):
            changed=X.copy();changed[valid]=changed[valid]*700+31
            new=E.LazyFoldScaledWrapper(changed,y,folds,E.L.SPECS["ridge"])
            np.testing.assert_array_equal(clean.statistics[index][0],new.statistics[index][0])
            np.testing.assert_array_equal(clean.statistics[index][1],new.statistics[index][1])

    def test_all_algorithms_ignore_outer_test(self):
        X,y=toy();rng=np.random.default_rng(21)
        for algo in E.A.ALGOS:
            for learner in ("ridge","knn"):
                clean,indices=E.strict_core(X[:60],y[:60],X[60:],y[60:],algo,learner,("r22",1),("r22",2),(60,150))
                junk,_=E.strict_core(X[:60],y[:60],rng.normal(size=X[60:].shape)*99,1-y[60:],algo,learner,("r22",1),("r22",2),(60,150))
                for cp in ("60","150"):
                    for field in ("sel","inner_err","calls","distinct"):
                        self.assertEqual(clean[cp][field],junk[cp][field])
                    self.assertEqual(clean[cp]["calls"],int(cp))

    def test_same_trajectory_as_existing_wrapper(self):
        X,y=toy();folds=E.S.make_folds(y[:60],5,E.S.rng_for("r22-trajectory"))
        for algo in E.A.ALGOS:
            for learner in ("ridge","knn"):
                old=E.A.Run(OLD.FoldScaledWrapper(X[:60],y[:60],folds,E.L.SPECS[learner]),[50,200])
                new=E.CountedRun(E.LazyFoldScaledWrapper(X[:60],y[:60],folds,E.L.SPECS[learner]),[50,200])
                E.A.run_algo(algo,old,E.S.rng_for("r22-trajectory-search"))
                E.A.run_algo(algo,new,E.S.rng_for("r22-trajectory-search"))
                for cp in (50,200):
                    oi,oe=E.P._sel(algo,old.snap[cp]);ni,ne=E.P._sel(algo,new.snap[cp])
                    self.assertEqual(oe,ne);np.testing.assert_array_equal(oi,ni)

    def test_checkpoint_recording_is_passive(self):
        X,y=toy()
        for algo in E.A.ALGOS:
            a,_=E.strict_core(X[:60],y[:60],X[60:],y[60:],algo,"ridge",("passive",),("passive-a",),(50,200))
            b,_=E.strict_core(X[:60],y[:60],X[60:],y[60:],algo,"ridge",("passive",),("passive-a",),(200,))
            self.assertEqual(a["200"],b["200"])

    def test_design_matches_all_original_keys(self):
        specs=E.job_specs();self.assertEqual(len(specs),5350)
        self.assertEqual(len({E.job_id(s) for s in specs}),5350)
        self.assertTrue(all(s["cps"]==[250,1000,4000] for s in specs))

    def test_dataset_weighting_and_paired_identity(self):
        records={}
        for ds,runs in (("small",1),("large",7)):
            for rep in range(runs):
                for algo in ("ga","rs"):
                    for cp in E.CPS:
                        F=.55+cp/40000+(.1 if algo=="ga" else 0)
                        T=.5+cp/80000+(.02 if algo=="ga" else 0)
                        records[(ds,"ridge",algo,rep,0,cp)]=dict(F=F,T=T,B=F-T,k=3)
        values=ANALYSIS.advantage_values(records,"ridge","ga",4000,"B")
        for ds in ("small","large"):
            self.assertAlmostEqual(values[ds],.08)
        means=ANALYSIS.t_summary(values)
        self.assertEqual(means["n_datasets"],2)
        self.assertAlmostEqual(means["mean_pp"],8)
        cells={("small","ridge","ga"):[.4],("large","ridge","ga"):[.1]*7}
        self.assertAlmostEqual(np.mean(list(ANALYSIS.equal_dataset_means(cells).values())),.25)

    def test_complete_focal_family_and_plot_outputs(self):
        records={}
        for ds in ("a","b","c"):
            for learner in ("ridge","knn"):
                for algo in E.A.ALGOS:
                    for cp in E.CPS:
                        F=.5+cp/40000;T=.5+cp/80000
                        records[(ds,learner,algo,0,0,cp)]=dict(F=F,T=T,B=F-T,k=4,distinct=cp)
        summary,rows=ANALYSIS.make_summaries({"shared":records,"inner_fold":records})
        self.assertEqual(sum(r["focal"] for r in summary),9)
        old_output=ANALYSIS.OUT
        try:
            with tempfile.TemporaryDirectory() as directory:
                ANALYSIS.OUT=Path(directory)
                ANALYSIS.plot_results(summary,rows)
                for name in ("focal_estimates.pdf","focal_estimates.png","budget_curves.pdf","budget_curves.png"):
                    self.assertGreater((ANALYSIS.OUT/name).stat().st_size,1000)
        finally:
            ANALYSIS.OUT=old_output


if __name__=="__main__":unittest.main()
