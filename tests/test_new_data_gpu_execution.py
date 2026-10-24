"""Synthetic-only CUDA contract checks; never fit a new benchmark."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'extras'))
sys.path.insert(0,str(ROOT/'analysis'))
import numpy as np
import torch
import ex_suites6 as CPU
import confirmation_gpu_backend as GPU
import ex_suites7_gpu as E
import new_data_confirmation_gpu as ANALYSIS


def synthetic():
    rng=np.random.default_rng(7317)
    X=rng.normal(size=(72,16))
    y=(X[:,0]+.4*X[:,1]>0).astype(np.int64)
    return X,y


class CudaContractChecks(unittest.TestCase):
    def test_runtime_is_cuda_float64_deterministic(self):
        meta=GPU.device_metadata()
        self.assertEqual(meta['device'],'cuda:0')
        self.assertEqual(meta['learner_dtype'],'float64')
        self.assertTrue(meta['deterministic_algorithms'])
        self.assertFalse(meta['tf32'])
        self.assertFalse(meta['cudnn_allow_tf32'])
        self.assertFalse(meta['cpu_learner_fallback'])
        E.assert_backend_contract()

    def test_ridge_primal_dual_weights_and_constant_columns(self):
        rng=np.random.default_rng(12)
        for n,p in [(72,16),(18,37)]:
            X=rng.normal(size=(n,p));X[:,1]=4
            y=np.r_[np.zeros(n-5,dtype=int),np.ones(5,dtype=int)]
            T=rng.normal(size=(9,p));T[:,1]=4
            expected=CPU.L.fit_predict(X,y,T,('ridge',1.))
            GPU.reset_diagnostics()
            actual=GPU.fit_predict_gpu(X,y,T,('ridge',1.))
            np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-9)
            np.testing.assert_array_equal(actual>0,expected>0)
            diag=GPU.diagnostics()
            self.assertGreater(diag['cuda_ridge_fits'],0)
            self.assertGreater(diag['cuda_primal_solves' if p<=n else 'cuda_dual_solves'],0)

    def test_knn_random_and_exact_ties_keep_partition_rule(self):
        rng=np.random.default_rng(8)
        X=rng.normal(size=(29,7));T=rng.normal(size=(8,7));y=np.arange(29)%2
        for A,B in [(X,T),(np.repeat([[0.,0.],[1.,0.],[0.,1.]],10,axis=0),np.zeros((4,2)))]:
            labels=np.arange(len(A))%2
            expected=CPU.L.fit_predict(A,labels,B,('knn',5))
            actual=GPU.fit_predict_gpu(A,labels,B,('knn',5))
            np.testing.assert_array_equal(actual,expected)
        self.assertGreater(GPU.diagnostics()['cuda_knn_distance_matrices'],0)
        self.assertGreater(GPU.diagnostics()['cpu_knn_argpartitions'],0)

    def test_fold_scaler_uses_training_rows_and_cached_counts_match(self):
        X,y=synthetic();folds=CPU.S.make_folds(y,5,CPU.S.rng_for('gpu-scaler-test'))
        for learner in ('ridge','knn'):
            original=CPU.LazyFoldScaledWrapper(X,y,folds,CPU.L.SPECS[learner])
            cuda=GPU.CudaFoldScaledWrapper(X,y,folds,CPU.L.SPECS[learner])
            for (a,b),(c,d) in zip(original.statistics,cuda.statistics):
                np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
            for cols in ([0,2,3],[0,1,4,8,12]):
                mask=np.zeros(X.shape[1],dtype=bool);mask[cols]=True
                np.testing.assert_allclose(cuda.scores(mask),original.scores(mask),atol=1e-10,rtol=1e-9)
                self.assertEqual(cuda.evaluate(mask),original.evaluate(mask))
                self.assertEqual(cuda.evaluate(mask),original.evaluate(mask))
            self.assertEqual(cuda.calls,4);self.assertEqual(len(cuda.cache),2)
            changed=X.copy();changed[folds[0][1]]*=31
            altered=GPU.CudaFoldScaledWrapper(changed,y,folds,CPU.L.SPECS[learner])
            for a,b in zip(cuda.statistics[0],altered.statistics[0]):np.testing.assert_array_equal(a,b)

    def test_all_searches_match_synthetic_cpu_and_gpu_replay(self):
        X,y=synthetic()
        for algo in CPU.A.ALGOS:
            for learner in ('ridge','knn'):
                ik=('gpu-contract',learner);ak=('gpu-contract',learner,algo)
                cpu,ci=CPU.strict_core(X[:60],y[:60],X[60:],y[60:],algo,learner,ik,ak,(12,25))
                gpu,gi=GPU.strict_core_gpu(X[:60],y[:60],X[60:],y[60:],algo,learner,ik,ak,(12,25))
                replay,ri=GPU.strict_core_gpu(X[:60],y[:60],X[60:],y[60:],algo,learner,ik,ak,(12,25))
                self.assertEqual(ci,gi);self.assertEqual(gi,ri)
                self.assertEqual(cpu,gpu);self.assertEqual(gpu,replay)

    def test_outer_test_data_cannot_change_gpu_selection(self):
        X,y=synthetic()
        for learner in ('ridge','knn'):
            args=('ga',learner,('gpu-isolation',learner),('gpu-isolation-algo',learner),(12,25))
            first,folds=GPU.strict_core_gpu(X[:60],y[:60],X[60:],y[60:],*args)
            changed,other=GPU.strict_core_gpu(X[:60],y[:60],X[60:]*49,1-y[60:],*args)
            self.assertEqual(folds,other)
            for cp in ('12','25'):
                for key in ('sel','inner_err','calls','distinct'):self.assertEqual(first[cp][key],changed[cp][key])

    def test_float32_and_missing_cuda_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'float64'):
            GPU._fit_predict_tensors(torch.ones((8,3),device='cuda',dtype=torch.float32),
                np.arange(8)%2,torch.ones((2,3),device='cuda',dtype=torch.float32),('knn',5))
        with mock.patch.object(torch.cuda,'is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'CPU fallback is forbidden'):
                GPU.fit_predict_gpu(np.ones((8,3)),np.arange(8)%2,np.ones((2,3)),('knn',5))

    def test_missing_gpu_freeze_blocks_before_benchmark_loading(self):
        with tempfile.TemporaryDirectory() as td,mock.patch.object(E,'FREEZE',Path(td)/'missing'),mock.patch.object(E,'load_data') as load:
            with self.assertRaises(RuntimeError):E.compute(E.job_specs()[0])
            load.assert_not_called()

    def test_no_partial_gpu_analysis_or_cpu_result_pooling(self):
        with mock.patch.object(E,'frozen',return_value={'jobs':E.job_specs()}), \
             mock.patch.object(ANALYSIS.A,'load_complete',side_effect=RuntimeError('Incomplete GPU suite')), \
             mock.patch.object(ANALYSIS.A,'make_summaries') as summarize:
            with self.assertRaisesRegex(RuntimeError,'Incomplete GPU suite'):ANALYSIS.analyse()
            summarize.assert_not_called()
        self.assertNotEqual(E.OUTPUT,E.BASE.OUTPUT)

    def test_gpu_records_require_real_device_and_learner_diagnostics(self):
        spec=E.job_specs()[0]
        splits=dict(outer_train=[0,1,2,3],outer_test=[4,5],inner_relative_to_outer_train=[])
        backend=GPU.device_metadata()
        fixture=dict(jobs=[spec],created_utc='2026-01-01T00:00:00+00:00',source_fingerprint='source',
            parent_freeze_sha256='parent',amendment_sha256='amend',gpu_environment=backend,
            split_hashes={E.split_key(spec):E.json_sha(splits)},datasets={spec['ds']:{'n':6,'p':18}})
        row=dict(calls=0,distinct=1,k=1,sel=[0],inner_err=.3,test_ba=.6,test_auc=.5,
                 test_acc=.5,test_gm=.5,test_f1=.5,test_tpr=.5,test_tnr=.5,test_mcc=0.)
        record=dict(status='ok',spec=spec,execution_mode=E.MODE,freeze_sha256='gpu',source_fingerprint='source',
            parent_freeze_sha256='parent',amendment_sha256='amend',gpu_backend=backend,
            gpu_diagnostics=dict(device='cuda:0',learner_dtype='float64',cuda_dispatches=1,cuda_ridge_fits=1),
            started_utc='2026-01-02T00:00:00+00:00',completed_utc='2026-01-02T00:01:00+00:00',
            splits=splits,split_sha256=E.json_sha(splits),
            seed_keys=dict(zip(('inner','algorithm'),[list(k) for k in E.seed_keys(spec)])),
            result={str(cp):dict(row,calls=cp) for cp in E.CPS})
        with mock.patch.object(E,'frozen',return_value=fixture),mock.patch.object(E,'sha_file',return_value='gpu'):
            E.validate_record(record,spec)
            for key,value in [('device','cpu'),('learner_dtype','float32'),('cuda_dispatches',0),('cuda_ridge_fits',0)]:
                bad=copy.deepcopy(record);bad['gpu_diagnostics'][key]=value
                with self.assertRaises(RuntimeError):E.validate_record(bad,spec)

    def test_gpu_freeze_cannot_overwrite_or_follow_gpu_outcomes(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);f=base/'freeze';f.write_text('{}',encoding='utf-8')
            with mock.patch.object(E,'FREEZE',f):
                with self.assertRaises(RuntimeError):E.create_gpu_freeze()
            f.unlink();out=base/'gpu';out.mkdir();(out/'record.json').write_text('{}',encoding='utf-8')
            with mock.patch.object(E,'FREEZE',f),mock.patch.object(E,'OUTPUT',out):
                with self.assertRaises(RuntimeError):E.create_gpu_freeze()


if __name__=='__main__':unittest.main()
