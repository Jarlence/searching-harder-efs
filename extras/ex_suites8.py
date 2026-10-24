"""X12m: targeted exploratory same-size RS under the preserved X11c CUDA protocol.

Only new paths are written. The parent design, code, data and completed records
remain immutable. A local outcome-free execution freeze is required; no OSF
operation or publicly verified preregistration is implied.
"""
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import ex_suites7_gpu as PARENT
import confirmation_gpu_backend as GPU

ROOT = PARENT.ROOT
SUITE = 'X12m'
CONFIGS = (('ga', 'ridge'), ('nsga2', 'knn'))
DATASETS = tuple(PARENT.DATASETS)
BUDGET = 4000
FAMILY = 4
OUTPUT = ROOT / 'results/jobs_extra/X12m'
FREEZE = ROOT / 'docs/NEW_DATA_SAMESIZE_FREEZE.json'
PLAN = ROOT / 'docs/NEW_DATA_SAMESIZE_PLAN.md'
MODE = 'targeted_exploratory_strict_samesize_cuda_float64'
REFERENCE_HASHES = ROOT / 'results/new_data_confirmation_gpu/job_hashes.json'
REFERENCE_VALIDATION = ROOT / 'results/new_data_confirmation_gpu/validation.json'
sha_file, json_sha, atomic_json = PARENT.sha_file, PARENT.json_sha, PARENT.atomic_json
utc_now, timestamp = PARENT.utc_now, PARENT.timestamp
_FREEZE = None
_VALIDATED_FREEZE_SHA = None


def job_specs():
    return [dict(suite=SUITE, kind='strict_samesize_rs', ds=ds, algo=algo,
                 learner=lr, rep=rep, fold=fold, budget=BUDGET)
            for ds in DATASETS for algo, lr in CONFIGS for rep in range(10) for fold in range(5)]


def job_id(spec):
    return '%s__%s__%s__%s__rep%d__fol%d' % (SUITE, spec['ds'], spec['algo'],
        spec['learner'], spec['rep'], spec['fold'])


def job_path(spec):
    return OUTPUT / (job_id(spec) + '.json')


def source_spec(spec, ordinary=False):
    return dict(suite='X11c', kind='new_data_strict', ds=spec['ds'],
                algo='rs' if ordinary else spec['algo'], learner=spec['learner'],
                rep=spec['rep'], fold=spec['fold'], cps=[250, 1000, 4000], metric='ba')


def source_paths():
    return ['extras/ex_suites8.py', 'extras/run_extra8.py',
            'analysis/new_data_samesize.py', 'tests/test_new_data_samesize_pipeline.py',
            'tests/test_new_data_samesize_analysis.py', 'tests/test_new_data_samesize_freeze.py',
            'docs/NEW_DATA_SAMESIZE_PLAN.md',
            'docs/NEW_DATA_SAMESIZE_TEST_REPORT.json']


def sampling_key(spec, k):
    return ('X12m', 'conditional_fixed_final_k_v1', spec['ds'], spec['algo'],
            spec['learner'], spec['rep'], spec['fold'], BUDGET, int(k))


def _read_source(spec, ordinary, expected_hash):
    old_spec = source_spec(spec, ordinary)
    path = PARENT.job_path(old_spec)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise RuntimeError('Frozen comparator bytes changed: ' + str(path))
    rec = json.loads(raw)
    if rec['status'] != 'ok' or rec['spec'] != old_spec:
        raise RuntimeError('Wrong source record')
    if rec['freeze_sha256'] != sha_file(PARENT.FREEZE) or rec['gpu_backend'] != GPU.device_metadata():
        raise RuntimeError('Source execution provenance changed')
    row = rec['result'][str(BUDGET)]
    if row['calls'] != BUDGET or row['k'] != len(row['sel']):
        raise RuntimeError('Invalid comparator checkpoint')
    return rec, dict(path=path.relative_to(ROOT).as_posix(), sha256=expected_hash, result=row)


def create_freeze():
    if FREEZE.exists() or OUTPUT.exists():
        raise RuntimeError('Freeze must precede every new reference outcome; no overwrite')
    parent = PARENT.frozen(verify_inputs=True)
    receipt = json.loads(REFERENCE_VALIDATION.read_text())
    if receipt['jobs'] != 4000 or receipt['checkpoints'] != 12000:
        raise RuntimeError('Parent validation is incomplete')
    if sha_file(REFERENCE_HASHES) != receipt['outputs']['job_hashes.json']:
        raise RuntimeError('Parent job inventory changed')
    inventory = json.loads(REFERENCE_HASHES.read_text())
    jobs = job_specs()
    if len(jobs) != 800 or len({job_id(s) for s in jobs}) != 800:
        raise RuntimeError('Wrong follow-up design')
    comparator_hashes, targets = {}, {}
    for spec in jobs:
        records = []
        for ordinary in (False, True):
            path = PARENT.job_path(source_spec(spec, ordinary))
            name = path.relative_to(ROOT).as_posix()
            rec, meta = _read_source(spec, ordinary, inventory[name])
            comparator_hashes[name] = meta['sha256']
            records.append(rec)
        a, r = records
        if a['splits'] != r['splits'] or a['split_sha256'] != r['split_sha256']:
            raise RuntimeError('Source algorithm and ordinary RS are not paired')
        k = a['result'][str(BUDGET)]['k']
        p = parent['datasets'][spec['ds']]['p']
        if not 1 <= k <= min(p, 100):
            raise RuntimeError('Invalid source feature count')
        targets[job_id(spec)] = dict(k=k, split_sha256=a['split_sha256'],
            inner_seed_key=a['seed_keys']['inner'], sampling_seed_key=list(sampling_key(spec, k)))
    if len(comparator_hashes) != 1600:
        raise RuntimeError('Wrong comparator inventory')
    added = {name:sha_file(ROOT / name) for name in source_paths()}
    sources = dict(parent['source_hashes'], **added)
    freeze = dict(suite=SUITE, created_utc=utc_now(), execution_mode=MODE,
        status='locally fixed targeted exploratory follow-up before new RS outcomes',
        parent_freeze_sha256=sha_file(PARENT.FREEZE), parent_source_fingerprint=parent['source_fingerprint'],
        source_hashes=sources, added_source_hashes=added, source_fingerprint=json_sha(sources),
        data_hashes=parent['data_hashes'], gpu_environment=GPU.device_metadata(),
        reference_validation_sha256=sha_file(REFERENCE_VALIDATION),
        reference_hash_inventory_sha256=sha_file(REFERENCE_HASHES),
        comparator_hashes=comparator_hashes, targets=targets, datasets=parent['datasets'],
        jobs=jobs, budget=BUDGET, family_n=FAMILY, family_confidence=1-.05/FAMILY,
        independent_units=8, run_repeats=10, outer_folds=5,
        public_preregistration_claim=False, original_three_contrast_family_modified=False,
        outcome_access='Original X11c results already known; new same-size outcomes not yet generated')
    atomic_json(FREEZE, freeze)
    return freeze


def frozen(verify_inputs=True):
    global _FREEZE, _VALIDATED_FREEZE_SHA
    parent = PARENT.frozen(verify_inputs=verify_inputs)
    payload = json.loads(FREEZE.read_text())
    if payload['parent_freeze_sha256'] != sha_file(PARENT.FREEZE):
        raise RuntimeError('Parent freeze changed')
    if (payload['parent_source_fingerprint'] != parent['source_fingerprint'] or
            payload['datasets'] != parent['datasets'] or payload['data_hashes'] != parent['data_hashes']):
        raise RuntimeError('Parent scientific inputs changed')
    if payload['execution_mode'] != MODE or payload['jobs'] != job_specs():
        raise RuntimeError('Follow-up design changed')
    wanted = dict(parent['source_hashes'], **{name:sha_file(ROOT/name) for name in source_paths()})
    if payload['source_hashes'] != wanted or payload['source_fingerprint'] != json_sha(wanted):
        raise RuntimeError('Frozen source or plan changed')
    if payload['added_source_hashes'] != {name:wanted[name] for name in source_paths()}:
        raise RuntimeError('New-source inventory changed')
    if payload['gpu_environment'] != GPU.device_metadata():
        raise RuntimeError('Frozen GPU numerical backend changed')
    if (payload['family_n'], payload['budget'], payload['independent_units']) != (4, 4000, 8):
        raise RuntimeError('Analysis family changed')
    if sha_file(REFERENCE_VALIDATION) != payload['reference_validation_sha256'] or sha_file(REFERENCE_HASHES) != payload['reference_hash_inventory_sha256']:
        raise RuntimeError('Parent analysis receipt changed')
    current_sha = sha_file(FREEZE)
    if _VALIDATED_FREEZE_SHA != current_sha:
        inventory=json.loads(REFERENCE_HASHES.read_text())
        expected_paths={PARENT.job_path(source_spec(s,ordinary)).relative_to(ROOT).as_posix()
                        for s in payload['jobs'] for ordinary in (False,True)}
        expected_hashes={name:inventory[name] for name in expected_paths}
        if len(expected_paths)!=1600 or payload['comparator_hashes']!=expected_hashes:
            raise RuntimeError('Exact comparator inventory changed')
        if set(payload['targets'])!={job_id(s) for s in payload['jobs']}:
            raise RuntimeError('Target job inventory changed')
        for spec in payload['jobs']:
            name=PARENT.job_path(source_spec(spec)).relative_to(ROOT).as_posix()
            rec,_=_read_source(spec,False,expected_hashes[name])
            k=rec['result'][str(BUDGET)]['k']
            expected_target=dict(k=k,split_sha256=rec['split_sha256'],
                inner_seed_key=rec['seed_keys']['inner'],sampling_seed_key=list(sampling_key(spec,k)))
            target=payload['targets'][job_id(spec)]
            expected_inner=list(PARENT.seed_keys(source_spec(spec))[0])
            if (target!=expected_target or type(target['k'])!=int or
                    not 1<=k<=min(parent['datasets'][spec['ds']]['p'],100) or
                    rec['seed_keys']['inner']!=expected_inner):
                raise RuntimeError('Target size/seed/split changed')
            if json_sha(rec['splits'])!=rec['split_sha256']:
                raise RuntimeError('Comparator split hash corrupted')
        _VALIDATED_FREEZE_SHA=current_sha
    if verify_inputs:
        for name, expected in payload['comparator_hashes'].items():
            if sha_file(ROOT/name) != expected: raise RuntimeError('Comparator changed: '+name)
        for name, expected in payload['data_hashes'].items():
            if sha_file(ROOT/name) != expected: raise RuntimeError('Data changed: '+name)
    _FREEZE = payload
    return payload


def _sample_fixed_k(run, rng, k, progress=None):
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= min(run.p, 100):
        raise ValueError('Conditional subset size is outside the preserved search space')
    stream = hashlib.sha256()
    while not run.done():
        mask = np.zeros(run.p, dtype=bool)
        mask[rng.choice(run.p, int(k), replace=False)] = True
        if int(mask.sum()) != k: raise RuntimeError('Cardinality matching failed')
        stream.update(np.flatnonzero(mask).astype('<i4').tobytes())
        run.eval(mask)
        if progress and (run.w.calls % 250 == 0 or run.done()):
            progress(run.w.calls, len(run.w.cache))
    return stream.hexdigest()


def same_size_core_gpu(Xtr, ytr, Xte, yte, learner, k, inner_key, random_key, budget=BUDGET, progress=None):
    """Use only the learned k; no source score or outer outcome enters the search."""
    GPU.require_cuda()
    inner = GPU.S.make_folds(ytr, 5, GPU.S.rng_for(*inner_key))
    learner_spec = GPU.L.SPECS[learner]
    wrapper = GPU.CudaFoldScaledWrapper(Xtr, ytr, inner, learner_spec)
    run = GPU.CountedRun(wrapper, [budget])
    digest = _sample_fixed_k(run, GPU.S.rng_for(*random_key), k, progress)
    # Search is complete before any outer standardisation or prediction.
    idx, err = GPU.P._sel('rs', run.snap[budget])
    train, test = GPU.P.standardize(Xtr, Xte)
    scores = GPU.fit_predict_gpu(train[:,idx], ytr, test[:,idx], learner_spec)
    metrics = GPU.M.all_metrics(yte, (scores > GPU.L.threshold(learner_spec)).astype(np.int64), scores)
    GPU.synchronize()
    row = dict(k=int(len(idx)), inner_err=float(err), sel=[int(i) for i in idx],
        calls=int(wrapper.calls), distinct=int(len(wrapper.cache)),
        **{'test_'+name:float(value) for name,value in metrics.items()})
    splits = [dict(train=a.tolist(), validation=b.tolist()) for a,b in inner]
    return {str(budget):row}, splits, digest


def compute(spec, progress=None):
    freeze = frozen(verify_inputs=False)
    if spec not in freeze['jobs']: raise RuntimeError('Job outside frozen design')
    comps, originals = {}, []
    for name, ordinary in (('algorithm',False),('ordinary_rs',True)):
        path = PARENT.job_path(source_spec(spec, ordinary)).relative_to(ROOT).as_posix()
        rec, comps[name] = _read_source(spec, ordinary, freeze['comparator_hashes'][path])
        originals.append(rec)
    a, r = originals
    target = freeze['targets'][job_id(spec)]
    if a['splits'] != r['splits'] or a['split_sha256'] != target['split_sha256']:
        raise RuntimeError('Paired source split changed')
    if (target['k']!=a['result'][str(BUDGET)]['k'] or
            target['inner_seed_key']!=a['seed_keys']['inner'] or
            json_sha(a['splits'])!=target['split_sha256']):
        raise RuntimeError('Conditional size or source split/seed does not match algorithm')
    X,y = PARENT.load_data(spec['ds'])
    split = PARENT.expected_splits(source_spec(spec), y)
    if split != a['splits']: raise RuntimeError('Outer or inner fold changed')
    tr,te = split['outer_train'],split['outer_test']
    result,inner,digest = same_size_core_gpu(X[tr],y[tr],X[te],y[te],spec['learner'],target['k'],
        target['inner_seed_key'],target['sampling_seed_key'],BUDGET,progress)
    if inner != split['inner_relative_to_outer_train']:
        raise RuntimeError('Inner fold drift')
    return dict(result=result, comparators=comps, splits=split, split_sha256=json_sha(split),
        target_k=target['k'], sampling_seed_key=target['sampling_seed_key'],
        inner_seed_key=target['inner_seed_key'], request_mask_sha256=digest)


def validate_record(record, spec):
    freeze = frozen(verify_inputs=False)
    if spec not in freeze['jobs'] or record['status'] != 'ok' or record['spec'] != spec:
        raise RuntimeError('Wrong job spec/status')
    if record['freeze_sha256'] != sha_file(FREEZE) or record['source_fingerprint'] != freeze['source_fingerprint']:
        raise RuntimeError('Wrong frozen provenance')
    if record['execution_mode'] != MODE or record['gpu_backend'] != freeze['gpu_environment']:
        raise RuntimeError('Wrong GPU backend')
    if timestamp(record['started_utc']) < timestamp(freeze['created_utc']) or timestamp(record['completed_utc']) < timestamp(record['started_utc']):
        raise RuntimeError('Invalid outcome chronology')
    target = freeze['targets'][job_id(spec)]
    if type(record['target_k'])!=int or record['target_k'] != target['k'] or record['sampling_seed_key'] != target['sampling_seed_key'] or record['inner_seed_key'] != target['inner_seed_key']:
        raise RuntimeError('Wrong size or seed')
    if record['split_sha256'] != target['split_sha256'] or json_sha(record['splits']) != target['split_sha256']:
        raise RuntimeError('Wrong split')
    if set(record['comparators']) != {'algorithm','ordinary_rs'} or set(record['result']) != {'4000'}:
        raise RuntimeError('Incomplete comparator or checkpoint')
    for name, ordinary in (('algorithm',False),('ordinary_rs',True)):
        expected_path = PARENT.job_path(source_spec(spec,ordinary)).relative_to(ROOT).as_posix()
        meta = record['comparators'][name]
        expected_sha = freeze['comparator_hashes'][expected_path]
        original, expected = _read_source(spec, ordinary, expected_sha)
        if meta != expected or original['splits'] != record['splits']:
            raise RuntimeError('Comparator payload or split drift')
    row = record['result']['4000']
    n,p = freeze['datasets'][spec['ds']]['n'],freeze['datasets'][spec['ds']]['p']
    sel = row['sel']
    if type(row['k'])!=int or row['k'] != target['k'] or len(sel) != target['k'] or len(set(sel)) != target['k'] or any(type(i)!=int or not 0<=i<p for i in sel):
        raise RuntimeError('Selected subset does not match learned cardinality')
    if type(row['calls']) != int or row['calls'] != BUDGET or type(row['distinct']) != int or not 1 <= row['distinct'] <= BUDGET:
        raise RuntimeError('Request/distinct accounting violation')
    for name in ('inner_err','test_ba','test_auc','test_acc','test_gm','test_f1','test_tpr','test_tnr','test_mcc'):
        value=row[name];low=-1 if name=='test_mcc' else 0
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=1:
            raise RuntimeError('Invalid score: '+name)
    diag=record['gpu_diagnostics']
    field='cuda_ridge_fits' if spec['learner']=='ridge' else 'cuda_knn_distance_matrices'
    if diag['device']!='cuda:0' or diag['learner_dtype']!='float64' or diag[field]<=0 or diag['cuda_dispatches']<=0:
        raise RuntimeError('Actual CUDA learner execution missing')
    digest=record['request_mask_sha256']
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
        raise RuntimeError('Invalid sampled-mask stream digest')


def finished(spec):
    path=job_path(spec)
    if not path.exists():return False
    validate_record(json.loads(path.read_text()),spec)
    return True


def run_job(spec, progress=None):
    if finished(spec):return 'skip'
    started=utc_now();clock=time.monotonic()
    try:
        freeze=frozen(verify_inputs=False)
        GPU.reset_diagnostics()
        payload=compute(spec,progress)
        rec=dict(status='ok',spec=spec,started_utc=started,completed_utc=utc_now(),
            seconds=round(time.monotonic()-clock,3),execution_mode=MODE,
            freeze_sha256=sha_file(FREEZE),source_fingerprint=freeze['source_fingerprint'],
            gpu_backend=GPU.device_metadata(),gpu_diagnostics=GPU.diagnostics(),**payload)
        validate_record(rec,spec)
        atomic_json(job_path(spec),rec)
        return 'ok'
    except Exception:
        attempt=started.replace(':','').replace('+','_')
        atomic_json(OUTPUT/'errors'/(job_id(spec)+'__'+attempt+'.json'),
            dict(status='error',spec=spec,started_utc=started,completed_utc=utc_now(),error=traceback.format_exc()))
        return 'error'
