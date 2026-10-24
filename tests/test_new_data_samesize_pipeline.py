"""Synthetic contract fixtures only; no new benchmark outcome is generated."""
import hashlib
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'extras'))
import numpy as np
import ex_suites8 as E
import confirmation_gpu_backend as GPU


class FixedKContract(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(172318)
        X=rng.normal(size=(70,9))
        y=np.array([0,1]*35,dtype=int)
        return X[:60],y[:60],X[60:],y[60:]

    def test_exact_design_and_independent_seed_keys(self):
        jobs=E.job_specs()
        self.assertEqual(len(jobs),800)
        self.assertEqual(len({E.job_id(s) for s in jobs}),800)
        self.assertEqual({(s['algo'],s['learner']) for s in jobs},set(E.CONFIGS))
        self.assertTrue(all(s['budget']==4000 for s in jobs))
        self.assertNotEqual(E.sampling_key(jobs[0],3),E.sampling_key(jobs[1],3))
        self.assertNotEqual(E.sampling_key(jobs[0],3),E.sampling_key(jobs[0],4))

    def test_sampler_preserves_uniform_fixed_k_rng_and_ties(self):
        class W:
            p=7
            def __init__(self):self.calls=0;self.cache={}
            def evaluate(self,mask):
                self.calls+=1
                self.cache.setdefault(mask.tobytes(),.3)
                return .3
        wrapper=W();run=GPU.CountedRun(wrapper,[25])
        digest=E._sample_fixed_k(run,np.random.default_rng(813),3)
        rng=np.random.default_rng(813);stream=hashlib.sha256();first=None
        for _ in range(25):
            mask=np.zeros(7,dtype=bool);mask[rng.choice(7,3,replace=False)]=True
            if first is None:first=np.flatnonzero(mask)
            stream.update(np.flatnonzero(mask).astype('<i4').tobytes())
        self.assertEqual(digest,stream.hexdigest())
        np.testing.assert_array_equal(run.best_idx,first)
        self.assertEqual(wrapper.calls,25)

    def test_invalid_sizes_are_rejected_without_clamping(self):
        class W:p=5
        class R:
            p=5;w=W()
        for k in (0,6,101,True):
            with self.assertRaises(ValueError):E._sample_fixed_k(R(),np.random.default_rng(0),k)

    def test_cuda_replay_and_outer_test_isolation(self):
        X,y,T,t=self.fixture()
        for learner in ('ridge','knn'):
            GPU.reset_diagnostics()
            a,inner,digest=E.same_size_core_gpu(X,y,T,t,learner,3,('fixture',learner),('random',learner),budget=35)
            b,inner2,digest2=E.same_size_core_gpu(X,y,T*19+7,1-t,learner,3,('fixture',learner),('random',learner),budget=35)
            self.assertEqual(inner,inner2);self.assertEqual(digest,digest2)
            self.assertEqual(a['35']['sel'],b['35']['sel'])
            self.assertEqual(a['35']['inner_err'],b['35']['inner_err'])
            self.assertEqual(a['35']['k'],3);self.assertEqual(a['35']['calls'],35)
            diag=GPU.diagnostics()
            self.assertEqual(diag['device'],'cuda:0');self.assertEqual(diag['learner_dtype'],'float64')
            self.assertGreater(diag['cuda_ridge_fits' if learner=='ridge' else 'cuda_knn_distance_matrices'],0)

    def test_full_size_keeps_request_budget_and_one_distinct_mask(self):
        X,y,T,t=self.fixture()
        out,_,_=E.same_size_core_gpu(X,y,T,t,'ridge',9,('full-k',),('full-k-sampler',),budget=4000)
        self.assertEqual(out['4000']['calls'],4000)
        self.assertEqual(out['4000']['distinct'],1)
        self.assertEqual(out['4000']['sel'],list(range(9)))

    def test_outer_prediction_only_after_search_and_train_only_scaler(self):
        X,y,T,t=self.fixture()
        completed=[]
        original=GPU.fit_predict_gpu
        def outer(*args,**kwargs):
            self.assertEqual(completed[-1],20)
            return original(*args,**kwargs)
        with mock.patch.object(GPU,'fit_predict_gpu',side_effect=outer) as call:
            E.same_size_core_gpu(X,y,T,t,'ridge',4,('inner-only',),('search-only',),budget=20,
                progress=lambda n,d:completed.append(n))
            self.assertEqual(call.call_count,1)
        folds=GPU.S.make_folds(y,5,GPU.S.rng_for('scaler-only'))
        a=GPU.CudaFoldScaledWrapper(X,y,folds,('ridge',1.))
        changed=X.copy();changed[folds[0][1]]*=101
        b=GPU.CudaFoldScaledWrapper(changed,y,folds,('ridge',1.))
        for x,z in zip(a.statistics[0],b.statistics[0]):np.testing.assert_array_equal(x,z)


if __name__=='__main__':unittest.main()
