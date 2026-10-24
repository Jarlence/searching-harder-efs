"""Freeze, execute and analyse the fixed 800-job X12m sensitivity follow-up."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ex_suites8 as E


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze',action='store_true')
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.freeze:
        freeze=E.create_freeze()
        print(json.dumps(dict(status=freeze['status'],jobs=len(freeze['jobs']),freeze_sha256=E.sha_file(E.FREEZE))),flush=True)
        return
    freeze=E.frozen(verify_inputs=True)
    if args.check:
        print(json.dumps(dict(status='fixed_design_and_inputs_valid',jobs=len(freeze['jobs']),
            comparators=len(freeze['comparator_hashes']),freeze_sha256=E.sha_file(E.FREEZE),
            gpu_environment=freeze['gpu_environment'])),flush=True)
        return
    E.OUTPUT.mkdir(parents=True,exist_ok=True)
    lock=E.OUTPUT/'.runner.lock'
    fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    clock=time.monotonic()
    progress_path=E.ROOT/'results/new_data_samesize_progress.json'
    completed=0
    def update(status,**kwargs):
        payload=dict(status=status,updated_utc=E.utc_now(),runner_pid=os.getpid(),
            execution_mode=E.MODE,freeze_sha256=E.sha_file(E.FREEZE),total_jobs=800,
            valid_jobs=completed,total_requests=3200000,
            elapsed_seconds=round(time.monotonic()-clock,1),**kwargs)
        E.atomic_json(progress_path,payload)
        return payload
    try:
        todo=[]
        for spec in freeze['jobs']:
            if E.finished(spec):completed+=1
            else:todo.append(spec)
        print(json.dumps(update('running',submitted=len(todo))),flush=True)
        for spec in todo:
            job_clock=time.monotonic()
            def heartbeat(calls,distinct):
                update('running',active_job=E.job_id(spec),active_requests=calls,
                    active_distinct=distinct,active_elapsed_seconds=round(time.monotonic()-job_clock,1))
            update('running',active_job=E.job_id(spec),active_requests=0)
            status=E.run_job(spec,heartbeat)
            if status=='error':
                raise RuntimeError('Fixed job failed; error retained; no automatic retry or exclusion')
            completed+=1
            payload=update('running',last_completed=E.job_id(spec),errors=0)
            if completed<=5 or completed%10==0 or completed==800:
                print(json.dumps(payload),flush=True)
        print(json.dumps(update('validating_complete_coverage')),flush=True)
        sys.path.insert(0,str(E.ROOT/'analysis'))
        import new_data_samesize as A
        validation=A.analyse()
        print(json.dumps(update('complete_results_ready_for_integration',
            analysis_directory='results/new_data_samesize',validation=validation)),flush=True)
    except BaseException as exc:
        print(json.dumps(update('stopped_for_error',error=type(exc).__name__+': '+str(exc))),flush=True)
        raise
    finally:
        lock.unlink()


if __name__=='__main__':main()
