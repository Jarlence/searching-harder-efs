"""Synthetic provenance tests for the explicitly amended local execution route."""
import ast
import copy
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'extras'))
sys.path.insert(0,str(ROOT/'analysis'))
import ex_suites7 as P
import ex_suites7_local as L
import new_data_confirmation_local as ANALYSIS


class LocalExecutionChecks(unittest.TestCase):
    def test_computation_ast_is_exactly_preserved(self):
        L.assert_computation_identical()
        self.assertIs(L.CORE,P.CORE)
        for name in ('job_specs','seed_keys','expected_splits','load_data'):
            self.assertIs(getattr(L,name),getattr(P,name))

    def test_missing_local_freeze_blocks_before_predictors(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(L,'FREEZE',Path(td)/'missing'), \
                mock.patch.object(L,'load_data') as load:
            with self.assertRaisesRegex(RuntimeError,'local execution freeze'):
                L.compute(L.job_specs()[0])
            load.assert_not_called()

    def test_original_public_authority_is_not_repurposed(self):
        self.assertNotEqual(L.FREEZE,P.FREEZE)
        self.assertNotEqual(L.OUTPUT,P.OUTPUT)
        text=inspect.getsource(L)
        self.assertNotIn('def verify_archive_receipt',text)
        self.assertNotIn('_public_json(',text)
        self.assertNotIn('_public_bytes(',text)
        self.assertNotIn('register_archive(',text)

    def test_no_partial_local_analysis(self):
        with mock.patch.object(L,'frozen',return_value={'jobs':L.job_specs()}), \
             mock.patch.object(ANALYSIS.A,'load_complete',side_effect=RuntimeError('Incomplete suite')), \
             mock.patch.object(ANALYSIS.A,'make_summaries') as summarize:
            with self.assertRaisesRegex(RuntimeError,'Incomplete suite'):ANALYSIS.analyse()
            summarize.assert_not_called()

    def test_local_manifest_checks_added_source_and_keeps_parent_design(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);source=base/'local.py';source.write_text('original',encoding='utf-8')
            parentfile=base/'parent.json';parentfile.write_text('{}',encoding='utf-8')
            amendment=base/'amend.md';amendment.write_text('local authority',encoding='utf-8')
            manifest=base/'local.json'
            parent=dict(source_hashes={'parent.py':'parent-hash'},created_utc='2026-01-01T00:00:00+00:00',
                jobs=L.job_specs(),data_hashes={},datasets={},split_hashes={},environment={},primary_family=3,legacy={})
            added={'local.py':L.sha_file(source)};all_sources=dict(parent['source_hashes'],**added)
            freeze=dict(parent,execution_mode=L.MODE,parent_freeze_sha256=L.sha_file(parentfile),
                amendment_sha256=L.sha_file(amendment),added_source_hashes=added,source_hashes=all_sources,
                source_fingerprint=L.json_sha(all_sources),created_utc='2026-01-02T00:00:00+00:00')
            with mock.patch.object(L,'ROOT',base),mock.patch.object(L,'FREEZE',manifest), \
                 mock.patch.object(L,'AMENDMENT',amendment),mock.patch.object(P,'FREEZE',parentfile), \
                 mock.patch.object(P,'verify_local_freeze',return_value=parent), \
                 mock.patch.object(L,'local_source_paths',return_value=['local.py']):
                manifest.write_text(json.dumps(freeze),encoding='utf-8')
                self.assertEqual(L.frozen()['primary_family'],3)
                source.write_text('modified',encoding='utf-8')
                with self.assertRaisesRegex(RuntimeError,'input changed'):L.frozen()
                source.write_text('original',encoding='utf-8')
                changed=copy.deepcopy(freeze);changed['primary_family']=4
                manifest.write_text(json.dumps(changed),encoding='utf-8')
                with self.assertRaisesRegex(RuntimeError,'Scientific design changed'):L.frozen()

    def test_record_rejects_local_provenance_time_budget_and_mask_changes(self):
        spec=L.job_specs()[0]
        splits=dict(outer_train=[0,1,2,3],outer_test=[4,5],inner_relative_to_outer_train=[])
        fixture=dict(jobs=[spec],created_utc='2026-01-01T00:00:00+00:00',source_fingerprint='source',
            parent_freeze_sha256='parent',amendment_sha256='amend',
            split_hashes={L.split_key(spec):L.json_sha(splits)},datasets={spec['ds']:{'n':6,'p':18}})
        row=dict(calls=0,distinct=1,k=1,sel=[0],inner_err=.3,test_ba=.6,test_auc=.5,
                 test_acc=.5,test_gm=.5,test_f1=.5,test_tpr=.5,test_tnr=.5,test_mcc=0.)
        record=dict(status='ok',spec=spec,execution_mode=L.MODE,freeze_sha256='local',source_fingerprint='source',
            parent_freeze_sha256='parent',amendment_sha256='amend',started_utc='2026-01-02T00:00:00+00:00',
            completed_utc='2026-01-02T00:01:00+00:00',splits=splits,split_sha256=L.json_sha(splits),
            seed_keys=dict(zip(('inner','algorithm'),[list(k) for k in L.seed_keys(spec)])),
            result={str(cp):dict(row,calls=cp) for cp in L.CPS})
        with mock.patch.object(L,'frozen',return_value=fixture),mock.patch.object(L,'sha_file',return_value='local'):
            L.validate_record(record,spec)
            for field,value in [('execution_mode','public'),('parent_freeze_sha256','wrong'),
                ('freeze_sha256','wrong'),('started_utc','2025-01-01T00:00:00+00:00'),
                ('archive_receipt_sha256','fabricated')]:
                bad=copy.deepcopy(record);bad[field]=value
                with self.assertRaises(RuntimeError):L.validate_record(bad,spec)
            for field,value in [('calls',3999),('sel',[18]),('inner_err',float('nan'))]:
                bad=copy.deepcopy(record);bad['result']['4000'][field]=value
                with self.assertRaises(RuntimeError):L.validate_record(bad,spec)

    def test_freeze_cannot_overwrite_or_follow_outcomes(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);f=base/'freeze';f.write_text('{}',encoding='utf-8')
            with mock.patch.object(L,'FREEZE',f):
                with self.assertRaisesRegex(RuntimeError,'cannot be overwritten'):L.create_local_freeze()
            f.unlink();out=base/'jobs';out.mkdir();(out/'record.json').write_text('{}',encoding='utf-8')
            with mock.patch.object(L,'FREEZE',f),mock.patch.object(L,'OUTPUT',out):
                with self.assertRaisesRegex(RuntimeError,'must precede outcomes'):L.create_local_freeze()


if __name__=='__main__':unittest.main()
