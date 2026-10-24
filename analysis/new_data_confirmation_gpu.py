"""Local execution provenance; unchanged three-estimand analysis functions."""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
sys.path.insert(0,str(ROOT/'extras'))
import new_data_confirmation as A
import ex_suites7_gpu as E
OUT=ROOT/'results/new_data_confirmation_gpu'


def analyse():
    freeze=E.frozen(verify_inputs=True)
    jobs=list(E.job_specs());datasets=tuple(E.DATASETS)
    A.validate_design(jobs,datasets)
    if freeze['jobs']!=jobs or tuple(E.CPS)!=A.CPS:
        raise RuntimeError('Local freeze and unchanged statistical design disagree')
    scores,job_hashes,execution,row_counts=A.load_complete(E,jobs,datasets)
    summaries=A.make_summaries(scores,datasets)
    if len(summaries['primary'])!=3:raise RuntimeError('Primary family changed')
    validation=dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=len(jobs),
        checkpoints=len(scores),datasets=list(datasets),row_counts=row_counts,repeats=A.N_REPEATS,
        outer_folds=A.N_FOLDS,inner_folds=A.N_FOLDS,primary_family_n=A.FAMILY,
        family_confidence=1-.05/A.FAMILY,source_fingerprint=freeze['source_fingerprint'],
        freeze_sha256=E.sha_file(E.FREEZE),parent_freeze_sha256=freeze['parent_freeze_sha256'],
        amendment_sha256=freeze['amendment_sha256'],execution_mode=E.MODE,
        gpu_environment=freeze['gpu_environment'],prior_cpu_results_pooled=False,
        public_preregistration_claim=False,no_partial_results=True,
        all_algorithm_learner_outer_splits_matched=True,algorithm_reference_inner_splits_matched=True,
        outer_and_inner_oof_coverage_checked=True,original_results_pooled=False,
        new_p_values=0,equivalence_tests=0)
    if OUT.exists():
        receipt=OUT/'validation.json'
        if not receipt.is_file() or not (OUT/'job_hashes.json').is_file():
            raise RuntimeError('Existing local analysis directory is incomplete; no overwrite')
        old=json.loads(receipt.read_text(encoding='utf-8'))
        if old.get('freeze_sha256')!=validation['freeze_sha256'] or json.loads((OUT/'job_hashes.json').read_text(encoding='utf-8'))!=job_hashes:
            raise RuntimeError('Existing local analysis provenance differs; no overwrite')
        if any(not (OUT/name).is_file() or A._sha_file(OUT/name)!=h for name,h in old['outputs'].items()):
            raise RuntimeError('Existing local analysis integrity failed')
        return old
    stage=OUT.with_name('.'+OUT.name+'.staging.'+str(os.getpid()))
    stage.mkdir(parents=True,exist_ok=False)
    for name,rows in summaries.items():A._write_csv(stage/(name+'.csv'),rows)
    checkpoints=[dict(dataset=k[0],learner=k[1],algorithm=k[2],repeat=k[3],fold=k[4],budget=k[5],**score)
                 for k,score in sorted(scores.items())]
    A._write_csv(stage/'checkpoints.csv',checkpoints)
    A._write_csv(stage/'execution.csv',execution)
    A._write_json(stage/'job_hashes.json',job_hashes)
    A._write_json(stage/'primary.json',summaries['primary'])
    report=A.report_text(summaries['primary'],validation)
    report=report.replace('# New-data confirmation study','# Prospectively specified local new-data validation',1)
    report=report.replace('Registration status and timing are recorded by the frozen suite; this report alone is not external registration.',
        'The original design was locally frozen before CPU execution. After the interrupted CPU run, the CUDA technical version was frozen before GPU execution. No publicly verified preregistration is claimed.')
    report+='\nLearner linear algebra and distances used CUDA float64. NumPy control, scaler statistics, metrics and kNN partition were retained. The interrupted CPU records are not pooled. CPU/GPU bitwise equivalence is not claimed.\n'
    report+='\nInternal advantage (F), outer performance difference (T) and advantage discrepancy (B) are separate components. Positive B does not determine the sign of T or establish a rank reversal.\n'
    (stage/'REPORT.md').write_text(report,encoding='utf-8')
    validation['outputs']={p.name:A._sha_file(p) for p in sorted(stage.iterdir())}
    A._write_json(stage/'validation.json',validation)
    os.replace(str(stage),str(OUT))
    return validation


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analyse',action='store_true',required=True)
    parser.parse_args()
    print(json.dumps(analyse(),indent=2))
