"""Versioned CUDA learner execution; preserved local design and analysis."""
import ast
import inspect
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ex_suites7 as E
import ex_suites7_local as BASE
import confirmation_gpu_backend as GPU

ROOT = E.ROOT
CORE = E.CORE
CPS = E.CPS
DATASETS = E.DATASETS
MODE = 'locally_frozen_cuda_float64_not_publicly_preregistered'
FREEZE = ROOT/'docs/NEW_DATA_CONFIRMATION_GPU_EXECUTION_FREEZE.json'
OUTPUT = ROOT/'results/jobs_extra/X11c_gpu'
AMENDMENT = ROOT/'docs/NEW_DATA_CONFIRMATION_GPU_EXECUTION_AMENDMENT.md'
sha_file = E.sha_file
json_sha = E.json_sha
atomic_json = E.atomic_json
utc_now = E.utc_now
timestamp = E.timestamp
job_specs = E.job_specs
job_id = E.job_id
split_key = E.split_key
seed_keys = E.seed_keys
load_data = E.load_data
expected_splits = E.expected_splits


def gpu_source_paths():
    return [
        'extras/confirmation_gpu_backend.py', 'extras/ex_suites7_gpu.py',
        'extras/run_extra7_gpu.py', 'analysis/new_data_confirmation_gpu.py',
        'tests/test_new_data_gpu_execution.py',
        'docs/NEW_DATA_CONFIRMATION_GPU_EXECUTION_AMENDMENT.md',
        'docs/NEW_DATA_CONFIRMATION_GPU_TEST_REPORT.json',
    ]


def compute(spec):
    frozen()  # Must precede loading new predictors and every learner/search.
    if spec not in job_specs():raise RuntimeError('Job is outside frozen design')
    X,y=load_data(spec['ds'])
    split=expected_splits(spec,y)
    tr,te=split['outer_train'],split['outer_test']
    ikey,akey=seed_keys(spec)
    result,inner=GPU.strict_core_gpu(X[tr],y[tr],X[te],y[te],spec['algo'],spec['learner'],ikey,akey,CPS)
    if inner!=split['inner_relative_to_outer_train']:raise RuntimeError('Inner partition drift')
    return dict(result=result,splits=split,split_sha256=json_sha(split),seed_keys=dict(inner=list(ikey),algorithm=list(akey)))


def assert_backend_contract():
    expected = ast.parse(inspect.getsource(E.compute))
    actual = ast.parse(inspect.getsource(compute))
    class NormaliseBackend(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id=='GPU' and node.attr=='strict_core_gpu':
                return ast.copy_location(ast.Attribute(value=ast.Name(id='CORE',ctx=ast.Load()),attr='strict_core',ctx=ast.Load()),node)
            return self.generic_visit(node)
    if ast.dump(NormaliseBackend().visit(actual),include_attributes=False)!=ast.dump(expected,include_attributes=False):
        raise RuntimeError('GPU orchestration differs beyond the declared numeric backend')
    if job_specs is not E.job_specs or seed_keys is not E.seed_keys or expected_splits is not E.expected_splits:
        raise RuntimeError('Search design or seed/fold dependencies changed')
    GPU.require_cuda()



def create_gpu_freeze():
    if FREEZE.exists():
        raise RuntimeError('Existing local execution freeze cannot be overwritten')
    for output in (OUTPUT,):
        if output.exists() and any(output.rglob('*.json')):
            raise RuntimeError('Local execution amendment must precede outcomes')
    parent = BASE.frozen()
    assert_backend_contract()
    added = {name: sha_file(ROOT/name) for name in gpu_source_paths()}
    all_sources = dict(parent['source_hashes'], **added)
    freeze = dict(parent)
    freeze.update(created_utc=utc_now(), status='GPU technical amendment frozen before GPU execution; prior CPU records retained',
        execution_mode=MODE, parent_freeze_sha256=sha_file(BASE.FREEZE),
        amendment_sha256=sha_file(AMENDMENT), added_source_hashes=added,
        source_hashes=all_sources, source_fingerprint=json_sha(all_sources),
        gpu_environment=GPU.device_metadata(), prior_cpu_completed_records=len(list(BASE.OUTPUT.glob('*.json'))),
        authority='Direct author instruction: use GPU rather than CPU for learner computation')
    atomic_json(FREEZE, freeze)
    return freeze


def frozen(verify_inputs=True):
    if not FREEZE.is_file():
        raise RuntimeError('Create the outcome-free local execution freeze first')
    parent = BASE.frozen(verify_inputs=verify_inputs)
    freeze = json.loads(FREEZE.read_text(encoding='utf-8'))
    if freeze['execution_mode'] != MODE or freeze['parent_freeze_sha256'] != sha_file(BASE.FREEZE):
        raise RuntimeError('Wrong execution mode or parent freeze')
    if set(freeze['added_source_hashes']) != set(gpu_source_paths()):
        raise RuntimeError('Local source inventory changed')
    for name, expected in freeze['added_source_hashes'].items():
        if sha_file(ROOT/name) != expected:
            raise RuntimeError('Local execution input changed: '+name)
    wanted = dict(parent['source_hashes'], **freeze['added_source_hashes'])
    if freeze['source_hashes'] != wanted or freeze['source_fingerprint'] != json_sha(wanted):
        raise RuntimeError('Local source provenance changed')
    for key in ('jobs','data_hashes','datasets','split_hashes','environment','primary_family','legacy'):
        if freeze[key] != parent[key]:
            raise RuntimeError('Scientific design changed: '+key)
    if freeze['amendment_sha256'] != sha_file(AMENDMENT):
        raise RuntimeError('Execution amendment changed')
    if not timestamp(parent['created_utc']) <= timestamp(freeze['created_utc']) <= E.datetime.now(E.timezone.utc):
        raise RuntimeError('Invalid local freeze chronology')
    if GPU.device_metadata()!=freeze['gpu_environment']:
        raise RuntimeError('Frozen GPU runtime/device configuration changed')
    assert_backend_contract()
    return freeze


def job_path(spec):
    return OUTPUT/(job_id(spec)+'.json')


def validate_record(record, spec):
    freeze = frozen()
    if spec not in freeze['jobs']:raise RuntimeError('Job is outside frozen design')
    if record['status']!='ok' or record['spec']!=spec:raise RuntimeError('Wrong job record')
    if record['execution_mode']!=MODE:raise RuntimeError('Wrong execution mode')
    if record['gpu_backend']!=freeze['gpu_environment']:raise RuntimeError('GPU backend provenance changed')
    diag=record['gpu_diagnostics']
    if diag['device']!='cuda:0' or diag['learner_dtype']!='float64' or diag['cuda_dispatches']<=0:
        raise RuntimeError('Missing actual CUDA learner execution')
    required='cuda_ridge_fits' if spec['learner']=='ridge' else 'cuda_knn_distance_matrices'
    if diag[required]<=0:raise RuntimeError('Expected CUDA learner operations absent')

    if record['freeze_sha256']!=sha_file(FREEZE) or record['source_fingerprint']!=freeze['source_fingerprint']:
        raise RuntimeError('Wrong local provenance')
    if record['parent_freeze_sha256']!=freeze['parent_freeze_sha256'] or record['amendment_sha256']!=freeze['amendment_sha256']:
        raise RuntimeError('Wrong parent or amendment provenance')
    if any(key.startswith('archive_receipt') for key in record):raise RuntimeError('Public archive provenance is inapplicable')
    ikey,akey=seed_keys(spec)
    if record['seed_keys']!=dict(inner=list(ikey),algorithm=list(akey)):raise RuntimeError('Seed provenance changed')
    if timestamp(record['started_utc'])<timestamp(freeze['created_utc']):raise RuntimeError('Performance began before local freeze')
    if timestamp(record['completed_utc'])<timestamp(record['started_utc']):raise RuntimeError('Job clock reversed')
    if set(record['result'])!=set(map(str,CPS)):raise RuntimeError('Incomplete checkpoints')
    if record['split_sha256']!=json_sha(record['splits']) or record['split_sha256']!=freeze['split_hashes'][split_key(spec)]:
        raise RuntimeError('Frozen partition mismatch')
    n,p=freeze['datasets'][spec['ds']]['n'],freeze['datasets'][spec['ds']]['p']
    tr,te=record['splits']['outer_train'],record['splits']['outer_test']
    if sorted(tr+te)!=list(range(n)) or len(set(tr)&set(te)):raise RuntimeError('Invalid outer split')
    previous=0
    for cp in CPS:
        row=record['result'][str(cp)]
        for field in ('calls','distinct','k'):
            if isinstance(row[field],bool) or not isinstance(row[field],int):raise RuntimeError('Noninteger count')
        if row['calls']!=cp or not previous<=row['distinct']<=cp or row['distinct']<1:raise RuntimeError('Budget violation')
        previous=row['distinct']
        selected=row['sel']
        if any(isinstance(i,bool) or not isinstance(i,int) or not 0<=i<p for i in selected):raise RuntimeError('Invalid selected predictor')
        if row['k']!=len(selected) or len(set(selected))!=row['k'] or not 1<=row['k']<=min(100,p):
            raise RuntimeError('Invalid incumbent size')
        for field in ('inner_err','test_ba','test_auc','test_acc','test_gm','test_f1','test_tpr','test_tnr','test_mcc'):
            v=row[field];low=-1 if field=='test_mcc' else 0
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not low<=v<=1:
                raise RuntimeError('Invalid '+field)


def finished(spec):
    path = job_path(spec)
    if not path.exists():return False
    validate_record(json.loads(path.read_text(encoding='utf-8')), spec)
    return True


def run_job(spec):
    freeze = frozen()
    if finished(spec):return 'skip'
    started=utc_now();clock=time.monotonic()
    try:
        GPU.reset_diagnostics()
        payload=compute(spec)
        record=dict(status='ok',spec=spec,started_utc=started,completed_utc=utc_now(),
            seconds=round(time.monotonic()-clock,3),freeze_sha256=sha_file(FREEZE),
            source_fingerprint=freeze['source_fingerprint'], execution_mode=MODE,
            gpu_backend=freeze['gpu_environment'],gpu_diagnostics=GPU.diagnostics(),
            parent_freeze_sha256=freeze['parent_freeze_sha256'], amendment_sha256=freeze['amendment_sha256'],**payload)
        validate_record(record,spec)
        atomic_json(job_path(spec),record)
        return 'ok'
    except Exception:
        attempt=started.replace(':','').replace('+','_')
        atomic_json(OUTPUT/'errors'/(job_id(spec)+'__'+attempt+'.json'),
            dict(status='error',spec=spec,started_utc=started,completed_utc=utc_now(),
                freeze_sha256=sha_file(FREEZE),execution_mode=MODE,error=traceback.format_exc()))
        return 'error'
