"""Locally frozen X11c: fixed execution followed by complete-coverage analysis."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ex_suites7_gpu as E
from joblib import Parallel, delayed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze',action='store_true')
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--run',action='store_true')
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    if args.freeze:
        freeze=E.create_gpu_freeze()
        print(json.dumps(dict(status=freeze['status'],jobs=len(freeze['jobs']),
            freeze_sha256=E.sha_file(E.FREEZE),execution_mode=E.MODE)),flush=True)
        return
    freeze=E.frozen()
    if args.check:
        print(json.dumps(dict(status='local_freeze_valid',jobs=len(freeze['jobs']),
            freeze_sha256=E.sha_file(E.FREEZE),execution_mode=E.MODE)),flush=True)
        return
    if args.workers!=1:parser.error('this single-GPU version uses one execution process')
    E.OUTPUT.mkdir(parents=True,exist_ok=True)
    lock=E.OUTPUT/'.runner.lock'
    fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode('ascii'));os.close(fd)
    progress=E.ROOT/'results/new_data_confirmation_gpu_progress.json'
    start=time.monotonic()
    def record(status,**more):
        payload=dict(status=status,updated_utc=E.utc_now(),runner_pid=os.getpid(),
            execution_mode=E.MODE,freeze_sha256=E.sha_file(E.FREEZE),
            total_jobs=4000,total_checkpoints=12000,workers=args.workers,
            elapsed_seconds=round(time.monotonic()-start,1),**more)
        E.atomic_json(progress,payload)
        print(json.dumps(payload),flush=True)
    try:
        todo=[s for s in freeze['jobs'] if not E.finished(s)]
        before=len(freeze['jobs'])-len(todo)
        record('running',valid_jobs=before,valid_before=before,submitted=len(todo),completed_this_run=0,errors=0)
        done=0
        with Parallel(n_jobs=args.workers,return_as='generator_unordered',batch_size=1,
                      pre_dispatch=args.workers) as executor:
            for status in executor(delayed(E.run_job)(s) for s in todo):
                done+=1
                if status=='error':
                    record('stopped_for_job_error',valid_jobs=before+done-1,
                           completed_this_run=done,errors=1)
                    raise RuntimeError('A fixed job failed; preserve errors and review before any retry')
                if done<=5 or done%20==0 or done==len(todo):
                    record('running',valid_jobs=before+done,valid_before=before,
                           submitted=len(todo),completed_this_run=done,errors=0)
        record('validating_complete_coverage',valid_jobs=4000,errors=0)
        sys.path.insert(0,str(E.ROOT/'analysis'))
        import new_data_confirmation_gpu as A
        validation=A.analyse()
        record('complete_results_ready_for_manuscript_integration',valid_jobs=4000,
               validated_checkpoints=validation['checkpoints'],errors=0,
               result_directory=str(A.OUT.relative_to(E.ROOT)))
    except BaseException as exc:
        record('stopped_for_error',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        lock.unlink()


if __name__=='__main__':main()
