"""Outcome-free acquisition and fixed cohort construction for X11c.

Only structural/data-quality information is inspected. No learner is imported.
Official fixed predictors are retained, including constant columns. Frozen
algorithm code is not edited and the legacy dataset registry is not used.
"""
import hashlib
import json
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/new_confirmation"
RAW = BASE / "raw"
PROCESSED = BASE / "processed"
REGISTRY = BASE / "registry.json"
COHORT_MAX = 1200
COHORT_SALT = "X11c-fixed-cohort-20261014-v1"
SOURCES = {
    "nc_climate": (252, "climate+model+simulation+crashes"),
    "nc_credit": (144, "statlog+german+credit+data"),
    "nc_dbworld": (219, "dbworld+e+mails"),
    "nc_dexter": (168, "dexter"),
    "nc_gisette": (170, "gisette"),
    "nc_ionosphere": (52, "ionosphere"),
    "nc_qsar": (254, "qsar+biodegradation"),
    "nc_spambase": (94, "spambase"),
}
DATASETS = tuple(sorted(SOURCES))

def sha_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def stable_rng(name):
    h=hashlib.sha256((COHORT_SALT+"|"+name).encode()).hexdigest()
    return np.random.default_rng(int(h[:16],16))

def fetch(name):
    did,slug=SOURCES[name]
    path=RAW/(name+".zip")
    url="https://archive.ics.uci.edu/static/public/%d/%s.zip"%(did,slug)
    RAW.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        req=urllib.request.Request(url,headers={"User-Agent":"t1-confirmation/1.0"})
        with urllib.request.urlopen(req,timeout=180) as response:
            tmp=path.with_suffix(".download")
            with open(tmp,"wb") as f:
                for block in iter(lambda:response.read(1<<20),b""):f.write(block)
        tmp.replace(path)
        (RAW/(name+".access.json")).write_text(json.dumps({"url":url,"download_utc":datetime.now(timezone.utc).isoformat(),"sha256":sha_file(path),"access":"official public feature/label files; no model evaluation"},indent=2))
    with zipfile.ZipFile(path) as archive:
        return path,archive.namelist()

def member(z,suffix):
    matches=[n for n in z.namelist() if n.lower().endswith(suffix.lower())]
    if len(matches)!=1:raise ValueError("Expected one %s, got %s"%(suffix,matches))
    return z.read(matches[0]).decode("utf-8-sig")

def sparse_lines(text,p):
    lines=text.splitlines()
    if any(not v.strip() for v in lines):
        raise ValueError("Blank sparse record would change source/label alignment")
    X=np.zeros((len(lines),p),dtype=np.float64)
    for row,line in enumerate(lines):
        seen=set()
        for token in line.split():
            feature,value=token.split(":")
            col=int(feature)-1
            if not 0<=col<p:raise ValueError("Sparse feature out of range")
            if col in seen:raise ValueError("Duplicate sparse feature index")
            seen.add(col)
            X[row,col]=float(value)
    return X

def parse_dense_arff(text):
    import re,io,pandas as pd
    attrs=[];lines=[];active=False
    for line in text.splitlines():
        s=line.strip()
        if not s or s.startswith("%"):continue
        if active:
            if s.startswith("{"):raise ValueError("Unexpected sparse ARFF")
            lines.append(s)
        elif s.lower().startswith("@attribute"):
            m=re.match(r"@attribute\s+('.*?'|\".*?\"|\S+)\s+(.*)",s,re.I)
            if not m:raise ValueError("Invalid ARFF attribute")
            attrs.append(m[1].strip("'\""))
        elif s.lower().startswith("@data"):active=True
    df=pd.read_csv(io.StringIO("\n".join(lines)),header=None,quotechar="'",skipinitialspace=True,na_values=["?"])
    if df.shape[1]!=len(attrs):raise ValueError("ARFF width mismatch")
    # UCI DBWorld official ARFF files put the fixed class column last.
    X=df.iloc[:,:-1].values.astype(np.float64)
    y=df.iloc[:,-1].astype(str).values
    return X,y,attrs[:-1]

def parse(name,path):
    import io
    with zipfile.ZipFile(path) as z:
        if name=="nc_credit":
            a=np.loadtxt(io.StringIO(member(z,"german.data-numeric")))
            if a.ndim!=2 or a.shape[1]!=25:raise ValueError("German numeric schema must contain 24 predictors and target")
            return a[:,:-1],a[:,-1].astype(int),["numeric_%d"%(j+1) for j in range(a.shape[1]-1)],"official german.data-numeric"
        if name=="nc_ionosphere":
            rows=[line.split(",") for line in member(z,"ionosphere.data").splitlines() if line.strip()]
            return np.array([r[:-1] for r in rows],float),np.array([r[-1] for r in rows]),["radar_%d"%(j+1) for j in range(34)],"official ionosphere.data"
        if name=="nc_qsar":
            rows=[line.split(";") for line in member(z,"biodeg.csv").splitlines() if line.strip()]
            return np.array([r[:-1] for r in rows],float),np.array([r[-1] for r in rows]),["descriptor_%d"%(j+1) for j in range(41)],"official biodeg.csv"
        if name=="nc_spambase":
            a=np.loadtxt(io.StringIO(member(z,"spambase.data")),delimiter=",")
            return a[:,:-1],a[:,-1].astype(int),["spam_feature_%d"%(j+1) for j in range(57)],"official spambase.data"
        if name=="nc_climate":
            text=member(z,"pop_failures.dat")
            a=np.loadtxt(io.StringIO(text),skiprows=1)
            if a.shape[1]!=21:raise ValueError("Climate schema should have 2 IDs, 18 predictors, target")
            return a[:,2:-1],a[:,-1].astype(int),["parameter_%d"%(j+1) for j in range(18)],"pop_failures.dat; study/simulation IDs excluded"
        if name=="nc_dbworld":
            candidates=[v for v in z.namelist() if v.lower().endswith('.arff') and 'bodies' in v.lower() and 'stem' not in v.lower()]
            if len(candidates)!=1:raise ValueError("DBWorld bodies non-stem ARFF ambiguous: "+str(candidates))
            X,y,cols=parse_dense_arff(z.read(candidates[0]).decode('utf-8-sig'))
            return X,y,cols,"official fixed non-stem bodies representation: "+candidates[0]
        if name in ("nc_dexter","nc_gisette"):
            stem='dexter' if name=='nc_dexter' else 'gisette'
            arrays=[];labels=[]
            for part in ['train','valid']:
                txt=member(z,stem+'_'+part+'.data')
                X=sparse_lines(txt,20000) if name=='nc_dexter' else np.loadtxt(io.StringIO(txt))
                y=np.loadtxt(io.StringIO(member(z,stem+'_'+part+'.labels')),dtype=int)
                if len(X)!=len(y):raise ValueError("Public labels do not match features")
                arrays.append(X);labels.append(y)
            X=np.vstack(arrays);y=np.concatenate(labels)
            return X,y,["fixed_%d"%(j+1) for j in range(X.shape[1])],"official train+valid only; unlabeled test excluded"
    raise ValueError(name)

def first_feature_rows(X,y):
    """One row per exact fixed feature vector, first in official source order.

    Selection is independent of labels. Contradictory-label groups are counted,
    not used to choose which row survives. This defines a feature-unique cohort,
    not a guarantee that every collision is a repeated physical observation.
    """
    X=np.ascontiguousarray(X,dtype=np.float64);X[X==0]=0.0
    seen={};keep=[];groups={}
    for i,row in enumerate(X):
        key=row.tobytes()
        if key not in seen:seen[key]=i;keep.append(i);groups[i]=[]
        groups[seen[key]].append(i)
    conflicts=[v for v in groups.values() if len(set(str(y[i]) for i in v))>1]
    return np.array(keep,dtype=np.int64),dict(duplicates_removed=len(X)-len(keep),
        conflicting_feature_groups=len(conflicts),rows_in_conflicting_groups=sum(map(len,conflicts)),
        conflict_groups_first_row=[int(v[0]) for v in conflicts])

def stratified_cohort(y,name,maximum=COHORT_MAX):
    if len(y)<=maximum:return np.arange(len(y),dtype=np.int64)
    labels,counts=np.unique(y,return_counts=True)
    ideal=counts.astype(float)*maximum/len(y)
    take=np.floor(ideal).astype(int)
    remainder=maximum-int(take.sum())
    # Stable sorted labels break fractional remainders, not any measured outcome.
    order=np.argsort(-(ideal-take),kind='stable')
    for j in order[:remainder]:take[j]+=1
    rng=stable_rng(name);idx=[]
    for label,n in zip(labels,take):
        candidates=np.flatnonzero(y==label);rng.shuffle(candidates);idx.extend(candidates[:n])
    return np.array(sorted(idx),dtype=np.int64)

def build_one(name):
    path,_=fetch(name)
    X,y,cols,description=parse(name,path)
    X=np.asarray(X,dtype=np.float64);y=np.asarray(y)
    if X.ndim!=2 or len(y)!=len(X) or not np.isfinite(X).all():raise ValueError("Nonfinite or invalid fixed predictors")
    if any(str(v).strip().lower() in ('','?','nan','none','null') for v in y):raise ValueError("Missing target label")
    if len(cols)!=X.shape[1]:raise ValueError("Predictor schema width mismatch")
    if len(np.unique(y))!=2:raise ValueError("Not a binary task")
    original_n=len(X)
    keep,quality=first_feature_rows(X,y)
    sub=stratified_cohort(y[keep],name)
    source_indices=keep[sub];X=X[source_indices];y=y[source_indices]
    labs,counts=np.unique(y,return_counts=True)
    positive=labs[np.argmin(counts)] if counts[0]!=counts[1] else labs[1]
    binary=(y==positive).astype(np.int8)
    if len(X)<50 or X.shape[1]<16 or min(counts)<25:raise ValueError("Prespecified eligibility bound failed")
    PROCESSED.mkdir(parents=True,exist_ok=True)
    target=PROCESSED/(name+'.npz')
    if target.exists():raise RuntimeError("Processed cohort exists; do not silently replace it")
    np.savez_compressed(target,X=X,y=binary,source_indices=source_indices)
    card=dict(name=name,uci_id=SOURCES[name][0],n=int(len(X)),p=int(X.shape[1]),original_labelled_n=original_n,
        feature_unique_n=int(len(keep)),cohort_cap=COHORT_MAX,source_indices=source_indices.tolist(),
        counts={str(k):int(v) for k,v in zip(labs,counts)},positive_label=str(positive),minority=int(binary.sum()),
        minority_fraction=float(binary.mean()),source_description=description,constant_predictors_retained=int(np.sum(X.std(0)==0)),
        columns=cols,source_sha256=sha_file(path),processed_sha256=sha_file(target),quality=quality,
        cohort_rule=COHORT_SALT,representation_dtype='float64',access='structure/class-count checks only; no learner evaluation',
        built_utc=datetime.now(timezone.utc).isoformat())
    (PROCESSED/(name+'.json')).write_text(json.dumps(card,ensure_ascii=False,indent=2),encoding='utf-8')
    return card

def prepare_all():
    if REGISTRY.exists():raise RuntimeError("Existing registry cannot be replaced")
    cards={}
    for name in DATASETS:
        cardpath=PROCESSED/(name+'.json')
        if cardpath.exists():
            card=json.loads(cardpath.read_text(encoding='utf-8'))
            if sha_file(PROCESSED/(name+'.npz'))!=card['processed_sha256']:raise RuntimeError("Cohort hash changed")
        else:card=build_one(name)
        cards[name]=card
        print('%s n=%d p=%d minority=%d source_n=%d duplicate_vectors=%d'%(name,card['n'],card['p'],card['minority'],card['original_labelled_n'],card['quality']['duplicates_removed']),flush=True)
    REGISTRY.write_text(json.dumps(dict(status='qualified before new-study model evaluation',created_utc=datetime.now(timezone.utc).isoformat(),datasets=cards),indent=2),encoding='utf-8')
    return cards

def load(name):
    with np.load(PROCESSED/(name+'.npz'),allow_pickle=False) as z:return z['X'].astype(np.float64),z['y'].astype(np.int64)

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--fetch-only',action='store_true');args=a.parse_args()
    if args.fetch_only:
        for name in DATASETS:
            p,names=fetch(name);print(name,p.stat().st_size,names[:16],flush=True)
    else:prepare_all()
