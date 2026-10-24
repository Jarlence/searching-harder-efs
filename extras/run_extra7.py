"""Preparation, public archive verification and fixed-coverage X11c runner."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ex_suites7 as E
from joblib import Parallel,delayed


def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze',action='store_true')
    mode.add_argument('--check',action='store_true',help='Outcome-free local freeze validation; no public claim')
    mode.add_argument('--register-archive',choices=['osf','zenodo'])
    mode.add_argument('--run',action='store_true')
    parser.add_argument('--record-id',help='Public OSF registration ID or published Zenodo record ID')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.freeze:
        freeze=E.create_freeze()
        print('Local freeze: %d jobs / 12000 checkpoints; no public registration yet'%len(freeze['jobs']))
        return
    if args.check:
        freeze=E.verify_local_freeze()
        print('Local freeze valid: %d jobs. Public gate: %s'%(len(freeze['jobs']),'receipt present, not checked by this command' if E.RECEIPT.exists() else 'not registered'))
        return
    if args.register_archive:
        if not args.record_id:parser.error('--record-id is required')
        receipt=E.register_archive(args.register_archive,args.record_id)
        print('Public bundle verified: '+receipt['public_url'])
        return
    if not 1<=args.workers<=16:parser.error('workers must be in 1..16')
    freeze=E.frozen(verify_inputs=True)
    todo=[s for s in freeze['jobs'] if not E.finished(s)]
    valid_before=len(freeze['jobs'])-len(todo)
    print('X11c total=4000 valid_done=%d todo=%d workers=%d'%(valid_before,len(todo),args.workers),flush=True)
    if not todo:return
    E.OUTPUT.mkdir(parents=True,exist_ok=True)
    lock=E.OUTPUT/'.runner.lock'
    fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        start=time.monotonic();done=errors=0
        for status in Parallel(n_jobs=args.workers,return_as='generator_unordered',batch_size=1,pre_dispatch=args.workers)(delayed(E.run_job)(s) for s in todo):
            done+=1;errors+=status=='error'
            if done%20==0 or status=='error' or done==len(todo):
                payload=dict(total=4000,valid_before=valid_before,valid_now=valid_before+done-errors,
                    submitted=len(todo),completed=done,errors=errors,elapsed_seconds=round(time.monotonic()-start,1))
                E.atomic_json(E.ROOT/'results/new_data_confirmation_progress.json',payload)
                print('X11c completed=%d/%d errors=%d elapsed=%.0fs'%(done,len(todo),errors,time.monotonic()-start),flush=True)
        if errors:raise RuntimeError('%d jobs failed; no partial performance analysis is permitted'%errors)
    finally:
        lock.unlink()


if __name__=='__main__':main()
