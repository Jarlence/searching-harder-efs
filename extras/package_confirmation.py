"""Create an immutable, self-contained outcome-free registration ZIP."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ex_suites7 as E

README='''X11c prospective new-data replication\n\nThis package contains the final protocol, exact 4,000-job freeze, source code,\nfrozen numerical dependencies, eight official UCI raw archives and fixed\nprocessed cohorts with provenance cards. It contains no new-study outcomes.\nLocal freezing is not public registration. Publish this exact ZIP as a public\nOSF registration attachment or a published Zenodo record before any new-study\nperformance run. OSF: store the ZIP at the root of OSF Storage before registering\nthe project. Include all files in the registration; wait for archiving to finish.\n\nRead docs/NEW_DATA_CONFIRMATION_PLAN.md first. Run the synthetic tests with the\nrequirements-lock.txt scientific environment. Use extras/run_extra7.py --check\nfor outcome-free local verification. After the external record is public:\n  python extras/run_extra7.py --register-archive osf --record-id <registration-id>\n(or --register-archive zenodo --record-id <published-record-id>). The verifier\nreads the official public API and downloads the archived ZIP without credentials\nto check its exact hash and frozen content. It writes a verified archive receipt.\nOnly then run:\n  python extras/run_extra7.py --run --workers 2\n  python analysis/new_data_confirmation.py --analyse\n\nThe sealed R22 sources are imported unchanged. All eight new tasks have complete\ncoverage of both learners. Retaining the original fixed predictor schema and\nusing float64 are stated in the protocol. The source archives are public UCI\ndata under CC BY 4.0; source credits and URLs appear in the protocol and cards.\nThe included code is provided as study material; repository authorship and any\ncode license must be supplied by the actual authors, not invented by automation.\nA public registration identity is the account holder's real identity; it does not\nreplace the final manuscript author list.\n'''


def package():
    freeze=E.verify_local_freeze()
    if E.OUTPUT.exists() and any(E.OUTPUT.rglob('*.json')):raise RuntimeError('Registration bundle must precede outcomes')
    target=E.BUNDLE_INFO.parent/('X11c_preregistration_'+E.sha_file(E.FREEZE)[:12]+'.zip')
    if target.exists() or E.BUNDLE_INFO.exists():raise RuntimeError('Existing registration bundle must not be overwritten')
    files=dict(freeze['source_hashes'],**freeze['data_hashes'])
    files['docs/NEW_DATA_CONFIRMATION_FREEZE.json']=E.sha_file(E.FREEZE)
    if any('audit/' in p.lower() or p.startswith('results/') for p in files):raise RuntimeError('Private files or outcomes cannot enter this archive')
    manifest=dict(suite=E.SUITE,freeze_sha256=E.sha_file(E.FREEZE),contains_new_study_outcomes=False,
                  total_jobs=4000,total_checkpoints=12000,files=files,
                  readme_sha256=hashlib.sha256(README.encode('utf-8')).hexdigest())
    target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,expected in sorted(files.items()):
            if E.sha_file(E.ROOT/name)!=expected:raise RuntimeError('Frozen content changed: '+name)
            z.write(E.ROOT/name,name)
        z.writestr('README.txt',README)
        z.writestr('ARCHIVE_MANIFEST.json',json.dumps(manifest,indent=2,sort_keys=True))
    verify_zip(target,files)
    info=dict(bundle_name=target.name,bundle_path=str(target.relative_to(E.ROOT)).replace('\\','/'),
              bundle_sha256=E.sha_file(target),freeze_sha256=E.sha_file(E.FREEZE),
              bytes=target.stat().st_size,members=len(files)+2,created_utc=E.utc_now(),status='ready for external archive; not registered')
    E.atomic_json(E.BUNDLE_INFO,info)
    print(json.dumps(info,indent=2))
    return info


def verify_zip(path,files):
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=len(files)+2 or len(set(z.namelist()))!=len(z.namelist()):raise RuntimeError('Unexpected registration archive inventory')
        for name,expected in files.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=expected:raise RuntimeError('Bad ZIP member: '+name)


if __name__=='__main__':package()
