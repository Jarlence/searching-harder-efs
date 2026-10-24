"""X11c: prospective new-data replication with an enforced public archive gate.

The strict R22 science is imported without modification. Preparation is outcome
free. Every performance entry point requires a remotely verified public archive
bound to this exact local freeze and bundle. No local approval flag is accepted.
"""
import hashlib
import io
import json
import math
import os
import re
import sys
import time
import traceback
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'extras'))
import numpy as np
import confirmation_data as DATA
import ex_suites6 as CORE

SUITE='X11c'
CPS=[250,1000,4000]
DATASETS=DATA.DATASETS
ALGORITHMS=('ga','bpso','bde','nsga2','rs')
LEARNERS=('ridge','knn')
OUTPUT=ROOT/'results/jobs_extra/X11c'
FREEZE=ROOT/'docs/NEW_DATA_CONFIRMATION_FREEZE.json'
PLAN=ROOT/'docs/NEW_DATA_CONFIRMATION_PLAN.md'
RECEIPT=ROOT/'docs/NEW_DATA_CONFIRMATION_ARCHIVE_RECEIPT.json'
BUNDLE_INFO=ROOT/'submission/asc/confirmation/bundle.json'
LEGACY_FREEZE=ROOT/'docs/STRICT_SCALING_R22_FREEZE.json'
_LOCAL=None
_ARCHIVE=None
_DATA={}
_SPLITS={}
sha_file=CORE.sha_file
json_sha=CORE.json_sha
atomic_json=CORE.atomic_json
environment=CORE.environment


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
    if parsed.tzinfo is None:raise RuntimeError('Timestamp must include timezone')
    return parsed


def job_specs():
    jobs=[dict(suite=SUITE,kind='new_data_strict',ds=ds,algo=a,learner=l,rep=r,fold=f,cps=CPS[:],metric='ba')
          for ds in DATASETS for a in ALGORITHMS for l in LEARNERS for r in range(10) for f in range(5)]
    if len(jobs)!=4000 or len({job_id(s) for s in jobs})!=4000:raise RuntimeError('Fixed design changed')
    return jobs


def job_id(spec):
    return '%s__%s__%s__%s__rep%d__fol%d'%(SUITE,spec['ds'],spec['algo'],spec['learner'],spec['rep'],spec['fold'])


def job_path(spec):
    return OUTPUT/(job_id(spec)+'.json')


def source_paths():
    legacy=json.loads(LEGACY_FREEZE.read_text(encoding='utf-8'))['source_hashes']
    new=['extras/confirmation_data.py','extras/ex_suites7.py','extras/run_extra7.py',
         'extras/package_confirmation.py',
         'analysis/new_data_confirmation.py','tests/test_new_data_confirmation_analysis.py',
         'tests/test_new_data_confirmation_pipeline.py','docs/NEW_DATA_CONFIRMATION_PLAN.md',
         'docs/NEW_DATA_CONFIRMATION_TEST_REPORT.json',
         'docs/STRICT_SCALING_R22_FREEZE.json']
    return sorted(set(legacy)|set(new))


def data_paths():
    paths=['data/new_confirmation/registry.json']
    for ds in DATASETS:
        paths.extend('data/new_confirmation/'+folder+'/'+ds+ext for folder,ext in
                     [('raw','.zip'),('raw','.access.json'),('processed','.npz'),('processed','.json')])
    return sorted(paths)


def assert_legacy_unchanged():
    old=json.loads(LEGACY_FREEZE.read_text(encoding='utf-8'))
    for name,expected in old['source_hashes'].items():
        if sha_file(ROOT/name)!=expected:raise RuntimeError('Protected R22 source changed: '+name)
    return dict(freeze_sha256=sha_file(LEGACY_FREEZE),source_fingerprint=old['source_fingerprint'])


def split_key(spec):
    return '%s__%s__rep%d__fol%d'%(spec['ds'],spec['learner'],spec['rep'],spec['fold'])


def seed_keys(spec):
    ds,a,l,r,f=(spec[k] for k in ('ds','algo','learner','rep','fold'))
    return (('inner',ds,l,r,f,None,False,False),('algo',ds,a,l,r,f,None,0,False,False))


def expected_splits(spec,y):
    key=split_key(spec)
    if key not in _SPLITS:
        tr,te=CORE.ST.outer_split(y,spec['ds'],spec['rep'],spec['fold'])
        ikey,_=seed_keys(spec)
        folds=CORE.S.make_folds(y[tr],5,CORE.S.rng_for(*ikey))
        _SPLITS[key]=dict(outer_train=tr.tolist(),outer_test=te.tolist(),
            inner_relative_to_outer_train=[dict(train=a.tolist(),validation=b.tolist()) for a,b in folds])
    return _SPLITS[key]


def load_data(ds):
    freeze=verify_local_freeze(verify_inputs=False)
    path=DATA.PROCESSED/(ds+'.npz')
    expected=freeze['data_hashes'][str(path.relative_to(ROOT)).replace('\\','/')]
    if sha_file(path)!=expected:raise RuntimeError('Frozen processed cohort changed: '+ds)
    if ds not in _DATA:
        _DATA.clear()
        _DATA[ds]=DATA.load(ds)
    return _DATA[ds]


def create_freeze():
    if FREEZE.exists():raise RuntimeError('Existing X11c freeze must not be overwritten')
    if OUTPUT.exists() and any(OUTPUT.rglob('*.json')):raise RuntimeError('Cannot freeze after any new outcomes exist')
    if RECEIPT.exists():raise RuntimeError('Archive receipt precedes local freeze')
    legacy=assert_legacy_unchanged()
    registry=json.loads(DATA.REGISTRY.read_text(encoding='utf-8'))
    if set(registry['datasets'])!=set(DATASETS):raise RuntimeError('Dataset qualification changed')
    jobs=job_specs()
    split_hashes={}
    for spec in jobs:
        key=split_key(spec)
        if key not in split_hashes:
            # Qualification/fold checks use labels only and never fit a learner.
            with np.load(DATA.PROCESSED/(spec['ds']+'.npz'),allow_pickle=False) as z:y=z['y']
            splits=expected_splits(spec,y)
            for indices in [splits['outer_test'],splits['outer_train']]:
                if len(np.unique(y[indices]))!=2:raise RuntimeError('One-class outer partition')
            yt=y[splits['outer_train']]
            for fold in splits['inner_relative_to_outer_train']:
                if len(np.unique(yt[fold['train']]))!=2 or len(np.unique(yt[fold['validation']]))!=2:
                    raise RuntimeError('One-class inner partition')
            split_hashes[key]=json_sha(splits)
    source_hashes={p:sha_file(ROOT/p) for p in source_paths()}
    payload=dict(suite=SUITE,created_utc=utc_now(),status='outcome-free local freeze; public archive required before execution',
        source_hashes=source_hashes,data_hashes={p:sha_file(ROOT/p) for p in data_paths()},
        source_fingerprint=json_sha(source_hashes),environment=environment(),legacy=legacy,
        datasets={ds:{k:registry['datasets'][ds][k] for k in ('n','p','minority','processed_sha256','source_sha256')} for ds in DATASETS},
        jobs=jobs,split_hashes=split_hashes,primary_family=3)
    atomic_json(FREEZE,payload)
    return payload


def _file_signature(paths):
    return tuple((p,(ROOT/p).stat().st_size,(ROOT/p).stat().st_mtime_ns) for p in paths)


def verify_local_freeze(verify_inputs=True):
    global _LOCAL
    if not FREEZE.is_file():raise RuntimeError('Create the outcome-free freeze first')
    payload=json.loads(FREEZE.read_text(encoding='utf-8'))
    if environment()!=payload['environment']:raise RuntimeError('Frozen numerical environment changed')
    # Never use size/mtime to bypass scientific-source verification. A restored
    # mtime is not evidence that file bytes still match the registered code.
    for name,expected in payload['source_hashes'].items():
        if sha_file(ROOT/name)!=expected:raise RuntimeError('Frozen input changed: '+name)
    if verify_inputs:
        for name,expected in payload['data_hashes'].items():
            if sha_file(ROOT/name)!=expected:raise RuntimeError('Frozen input changed: '+name)
    inputs=dict(payload['source_hashes'],**payload['data_hashes'])
    key=(sha_file(FREEZE),_file_signature(sorted(inputs)))
    if _LOCAL is None or _LOCAL[0]!=key:
        if set(payload['source_hashes'])!=set(source_paths()) or set(payload['data_hashes'])!=set(data_paths()):
            raise RuntimeError('Frozen source/input inventory changed')
        for name,expected in inputs.items():
            if sha_file(ROOT/name)!=expected:raise RuntimeError('Frozen input changed: '+name)
        if payload['jobs']!=job_specs() or payload['primary_family']!=3:raise RuntimeError('Frozen design changed')
        if json_sha(payload['source_hashes'])!=payload['source_fingerprint']:raise RuntimeError('Source fingerprint corrupted')
        assert_legacy_unchanged()
        _LOCAL=(key,payload)
    return _LOCAL[1]


def frozen(verify_inputs=True):
    payload=verify_local_freeze(verify_inputs)
    verify_archive_receipt()
    return payload


def _public_json(url):
    req=urllib.request.Request(url,headers={'User-Agent':'X11c-archive-verification/1.0','Accept':'application/json'})
    with urllib.request.urlopen(req,timeout=60) as response:return json.load(response)


def _public_bytes(url):
    req=urllib.request.Request(url,headers={'User-Agent':'X11c-archive-verification/1.0'})
    with urllib.request.urlopen(req,timeout=60) as response:
        stream=io.BytesIO()
        size=0
        for block in iter(lambda:response.read(1<<20),b''):
            size+=len(block)
            if size>512*(1<<20):raise RuntimeError('Archive exceeds fixed safety size')
            stream.write(block)
        return stream.getvalue()


def _archive_metadata(platform,record_id,bundle_name):
    if platform=='zenodo':
        if not re.fullmatch(r'[0-9]+',str(record_id)):raise RuntimeError('Invalid Zenodo record ID')
        doc=_public_json('https://zenodo.org/api/records/'+str(record_id))
        if doc.get('metadata',{}).get('access_right')!='open':raise RuntimeError('Zenodo files are not public')
        files=[f for f in doc.get('files',[]) if f['key']==bundle_name]
        if len(files)!=1:raise RuntimeError('Frozen bundle is not in the published Zenodo record')
        return dict(public_url='https://zenodo.org/records/'+str(record_id),archive_utc=doc['created'],
                    bundle_url=files[0]['links']['self'],metadata_sha256=json_sha(doc))
    if platform=='osf':
        if not re.fullmatch(r'[a-z0-9]+',str(record_id)):raise RuntimeError('Invalid OSF registration ID')
        url='https://api.osf.io/v2/registrations/'+str(record_id)+'/'
        doc=_public_json(url)['data'];attr=doc['attributes']
        if (not attr.get('public') or attr.get('withdrawn') or attr.get('embargoed') or attr.get('archiving')
            or attr.get('pending_registration_approval')):raise RuntimeError('OSF registration is not final and public')
        listing='https://api.osf.io/v2/registrations/'+str(record_id)+'/files/osfstorage/?page[size]=100'
        files=[]
        while listing:
            page=_public_json(listing)
            files.extend(f for f in page['data'] if f['attributes']['name']==bundle_name)
            listing=page.get('links',{}).get('next')
        if len(files)!=1:raise RuntimeError('Bundle must be at registered OSF storage root')
        # OSF's public API also emits legacy naive ISO strings; those API
        # timestamps are UTC. Normalize explicitly, never use local timezone.
        registered=datetime.fromisoformat(attr['date_registered'].replace('Z','+00:00'))
        if registered.tzinfo is None:registered=registered.replace(tzinfo=timezone.utc)
        return dict(public_url='https://osf.io/'+str(record_id)+'/',archive_utc=registered.isoformat(),
                    bundle_url=files[0]['links']['download'],metadata_sha256=json_sha(doc))
    raise RuntimeError('Only public OSF registrations or published Zenodo records are accepted')


def _check_bundle_bytes(raw,freeze):
    info=json.loads(BUNDLE_INFO.read_text(encoding='utf-8'))
    if hashlib.sha256(raw).hexdigest()!=info['bundle_sha256']:raise RuntimeError('Public bundle hash mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        if len(z.namelist())!=len(set(z.namelist())):raise RuntimeError('Duplicate archive entries')
        names=dict(freeze['source_hashes'],**freeze['data_hashes'])
        names['docs/NEW_DATA_CONFIRMATION_FREEZE.json']=sha_file(FREEZE)
        for name,expected in names.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=expected:raise RuntimeError('Public content differs: '+name)
    return info


def _verify_public(platform,record_id):
    freeze=verify_local_freeze()
    info=json.loads(BUNDLE_INFO.read_text(encoding='utf-8'))
    if info['freeze_sha256']!=sha_file(FREEZE):raise RuntimeError('Bundle is bound to a different freeze')
    meta=_archive_metadata(platform,record_id,info['bundle_name'])
    if timestamp(meta['archive_utc'])<timestamp(freeze['created_utc']):raise RuntimeError('Archive timestamp precedes frozen content')
    if timestamp(meta['archive_utc'])>datetime.now(timezone.utc):raise RuntimeError('Archive timestamp is in the future')
    parsed=urllib.parse.urlparse(meta['bundle_url'])
    if parsed.scheme!='https' or parsed.hostname not in ('zenodo.org','osf.io','files.osf.io','storage.googleapis.com'):
        raise RuntimeError('Unexpected public archive file endpoint')
    _check_bundle_bytes(_public_bytes(meta['bundle_url']),freeze)
    return dict(platform=platform,record_id=str(record_id),freeze_sha256=sha_file(FREEZE),
                bundle_name=info['bundle_name'],bundle_sha256=info['bundle_sha256'],verified_utc=utc_now(),**meta)


def register_archive(platform,record_id):
    if RECEIPT.exists():raise RuntimeError('Existing verified archive receipt must not be overwritten')
    if OUTPUT.exists() and any(OUTPUT.rglob('*.json')):raise RuntimeError('Cannot establish first archive gate after outcomes')
    receipt=_verify_public(platform,record_id)
    atomic_json(RECEIPT,receipt)
    return receipt


def verify_archive_receipt():
    global _ARCHIVE
    if not RECEIPT.is_file():raise RuntimeError('Public archive not verified; new-data performance is blocked')
    receipt=json.loads(RECEIPT.read_text(encoding='utf-8'))
    key=(sha_file(RECEIPT),sha_file(FREEZE),sha_file(BUNDLE_INFO))
    if _ARCHIVE is None or _ARCHIVE[0]!=key:
        checked=_verify_public(receipt['platform'],receipt['record_id'])
        for name in ('public_url','archive_utc','freeze_sha256','bundle_sha256','bundle_name'):
            if receipt[name]!=checked[name]:raise RuntimeError('Archive receipt binding changed: '+name)
        _ARCHIVE=(key,receipt)
    return _ARCHIVE[1]


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


def validate_record(record,spec):
    freeze=frozen();receipt=verify_archive_receipt()
    if spec not in freeze['jobs']:raise RuntimeError('Job is outside frozen design')
    if record['status']!='ok' or record['spec']!=spec:raise RuntimeError('Wrong job record')
    if record['freeze_sha256']!=sha_file(FREEZE) or record['source_fingerprint']!=freeze['source_fingerprint']:
        raise RuntimeError('Wrong provenance')
    if record['archive_receipt_sha256']!=sha_file(RECEIPT):raise RuntimeError('Wrong archive receipt provenance')
    ikey,akey=seed_keys(spec)
    if record['seed_keys']!=dict(inner=list(ikey),algorithm=list(akey)):raise RuntimeError('Seed provenance changed')
    if timestamp(record['started_utc'])<timestamp(receipt['archive_utc']):raise RuntimeError('Performance began before public archive')
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
    path=job_path(spec)
    if not path.exists():return False
    validate_record(json.loads(path.read_text(encoding='utf-8')),spec)
    return True


def run_job(spec):
    freeze=frozen();receipt=verify_archive_receipt()  # Gate also applies to resume.
    if finished(spec):return 'skip'
    started=utc_now();clock=time.monotonic()
    try:
        payload=compute(spec)
        record=dict(status='ok',spec=spec,started_utc=started,completed_utc=utc_now(),
            seconds=round(time.monotonic()-clock,3),freeze_sha256=sha_file(FREEZE),
            source_fingerprint=freeze['source_fingerprint'],archive_receipt_sha256=sha_file(RECEIPT),**payload)
        validate_record(record,spec)
        atomic_json(job_path(spec),record)
        return 'ok'
    except Exception:
        atomic_json(OUTPUT/'errors'/(job_id(spec)+'.json'),dict(status='error',spec=spec,started_utc=started,
            completed_utc=utc_now(),freeze_sha256=sha_file(FREEZE),error=traceback.format_exc()))
        return 'error'
