"""Explicit local X11c execution amendment; no public-registration authority."""
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

ROOT = E.ROOT
CORE = E.CORE
CPS = E.CPS
DATASETS = E.DATASETS
MODE = 'locally_frozen_not_publicly_preregistered'
FREEZE = ROOT/'docs/NEW_DATA_CONFIRMATION_LOCAL_EXECUTION_FREEZE.json'
OUTPUT = ROOT/'results/jobs_extra/X11c_local'
AMENDMENT = ROOT/'docs/NEW_DATA_CONFIRMATION_LOCAL_EXECUTION_AMENDMENT.md'
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


def local_source_paths():
    return [
        'extras/ex_suites7_local.py', 'extras/run_extra7_local.py',
        'analysis/new_data_confirmation_local.py',
        'tests/test_new_data_local_execution.py',
        'docs/NEW_DATA_CONFIRMATION_LOCAL_EXECUTION_AMENDMENT.md',
        'docs/NEW_DATA_CONFIRMATION_CANDIDATE_RECORD.csv',
        'docs/NEW_DATA_CONFIRMATION_LOCAL_TEST_REPORT.json',
    ]


def compute(spec):
    frozen()  # Must precede loading new predictors and every learner/search.
    if spec not in job_specs():raise RuntimeError('Job is outside frozen design')
    X,y=load_data(spec['ds'])
    split=expected_splits(spec,y)
    tr,te=split['outer_train'],split['outer_test']
    ikey,akey=seed_keys(spec)
    result,inner=CORE.strict_core(X[tr],y[tr],X[te],y[te],spec['algo'],spec['learner'],ikey,akey,CPS)
    if inner!=split['inner_relative_to_outer_train']:raise RuntimeError('Inner partition drift')
    return dict(result=result,splits=split,split_sha256=json_sha(split),seed_keys=dict(inner=list(ikey),algorithm=list(akey)))


def assert_computation_identical():
    node = lambda fn: ast.dump(ast.parse(inspect.getsource(fn)), include_attributes=False)
    if node(compute) != node(E.compute):
        raise RuntimeError('Local scientific computation differs from the preserved original')
    if CORE is not E.CORE or load_data is not E.load_data or job_specs is not E.job_specs:
        raise RuntimeError('Scientific dependencies changed')


def create_local_freeze():
    if FREEZE.exists():
        raise RuntimeError('Existing local execution freeze cannot be overwritten')
    for output in (OUTPUT, E.OUTPUT):
        if output.exists() and any(output.rglob('*.json')):
            raise RuntimeError('Local execution amendment must precede outcomes')
    parent = E.verify_local_freeze()
    assert_computation_identical()
    added = {name: sha_file(ROOT/name) for name in local_source_paths()}
    all_sources = dict(parent['source_hashes'], **added)
    freeze = dict(parent)
    freeze.update(created_utc=utc_now(), status='locally frozen before execution; no public preregistration claim',
        execution_mode=MODE, parent_freeze_sha256=sha_file(E.FREEZE),
        amendment_sha256=sha_file(AMENDMENT), added_source_hashes=added,
        source_hashes=all_sources, source_fingerprint=json_sha(all_sources),
        authority='Direct author instruction: stop OSF registration and start experiments locally')
    atomic_json(FREEZE, freeze)
    return freeze


def frozen(verify_inputs=True):
    if not FREEZE.is_file():
        raise RuntimeError('Create the outcome-free local execution freeze first')
    parent = E.verify_local_freeze(verify_inputs=verify_inputs)
    freeze = json.loads(FREEZE.read_text(encoding='utf-8'))
    if freeze['execution_mode'] != MODE or freeze['parent_freeze_sha256'] != sha_file(E.FREEZE):
        raise RuntimeError('Wrong execution mode or parent freeze')
    if set(freeze['added_source_hashes']) != set(local_source_paths()):
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
    assert_computation_identical()
    return freeze


def job_path(spec):
    return OUTPUT/(job_id(spec)+'.json')


def validate_record(record, spec):
    freeze = frozen()
    if spec not in freeze['jobs']:raise RuntimeError('Job is outside frozen design')
    if record['status']!='ok' or record['spec']!=spec:raise RuntimeError('Wrong job record')
    if record['execution_mode']!=MODE:raise RuntimeError('Wrong execution mode')
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
        payload=compute(spec)
        record=dict(status='ok',spec=spec,started_utc=started,completed_utc=utc_now(),
            seconds=round(time.monotonic()-clock,3),freeze_sha256=sha_file(FREEZE),
            source_fingerprint=freeze['source_fingerprint'], execution_mode=MODE,
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
