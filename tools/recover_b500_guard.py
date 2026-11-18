"""CPU-only recovery for the known B500 pre-update guard failure; never train here."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from b500_run import ROOT,PACKAGE,read,write,digest,prepare,verify_launch_code,accepted_cpu_receipt

OLD_IMPL='7076dbbadea1064ed4d96d880afe2f989d9b2f8f059aee1f2efbc826867406c1'
OLD_ERROR='B500 configuration differs from supported frozen Actor contract'


def rows(path):
    return [json.loads(l)for l in path.read_text().splitlines()if l.strip()]if path.exists()else []


def zero_update_proof(old):
    spec=read(old/'run.json')
    if spec['protocol_sha256']!=digest(PACKAGE/'protocol.json')or spec['implementation_manifest_sha256']!=OLD_IMPL:
        raise ValueError('Not the supported original B500 guard failure')
    if not spec.get('training_started')or not spec.get('launch_attempted')or not(old/'stopped.json').is_file():
        raise ValueError('Original supervisor must have stopped after attempting training')
    if OLD_ERROR not in (old/'train_to_350.log').read_text():raise ValueError('Expected pre-update guard error missing')
    for n in ['accepted_step_1.json','phase350_complete.json','accepted_step_500.json','training_complete.json','budget_truncated.json']:
        if(old/n).exists():raise ValueError('Cannot prove zero updates: '+n)
    if rows(old/'weights.jsonl'):raise ValueError('Actor update may have started; automatic recovery forbidden')
    diagnostics=rows(old/'training_diagnostics.jsonl')
    if any(e.get('event')=='optimizer_step'or e.get('metrics',{}).get('training/optimizer_updated')==1 for e in diagnostics):
        raise ValueError('Optimizer update evidence present; automatic recovery forbidden')
    if list((old/'train').glob('global_step_*')):raise ValueError('Checkpoint evidence present; automatic recovery forbidden')
    budget=read(old/'generation_budget.json')
    if any(type(budget.get(k))is not int or budget[k]<0 for k in ('train','validation_physical')):
        raise ValueError('Missing/invalid generation counters')
    if not(8<=budget['train']<24000)or budget['validation_physical']>200:
        raise ValueError('Prior generation cost cannot fit the unchanged complete-run budget')
    # Repeat the original initial Dev with the fresh seeded process, charging
    # its cost too. 11 scheduled Dev passes need 2200 physical generations.
    if budget['validation_physical']+2200>2400:raise ValueError('Insufficient validation budget')
    deadline=spec.get('deadline');started=spec.get('started_at')
    if not isinstance(deadline,(float,int))or not math.isfinite(deadline)or not isinstance(started,(float,int))or not math.isfinite(started):
        raise ValueError('Missing original wall-clock accounting')
    if not(0<deadline-started<=20*3600+1):raise ValueError('Original wall budget differs')
    if time.time()>=deadline-120:raise ValueError('Original wall budget expired; cannot reset it')
    return {'zero_updates':True,'no_actor_update_events':True,'no_optimizer_steps':True,'no_checkpoints':True,
        'old_run':str(old),'original_started_at':started,'inherited_deadline':deadline,'carried_budget':budget,
        'initial_Dev_repeated_and_charged':True}


def assert_no_processes(old):
    for p in Path('/proc').glob('[0-9]*/environ'):
        try:entries=p.read_bytes().split(b'\0')
        except (OSError,PermissionError):continue
        if b'B500_RUN_DIR='+old.as_posix().encode()in entries:
            raise ValueError('Original run still has an owned process: '+p.parent.name)


def prepare_recovery(old,out,receipt,proof):
    # prepare verifies repaired source hashes, the native CPU receipt, all
    # historical identities and baseline runtime before creating the directory.
    prepare(Path(read(old/'run.json')['project']),out,cpu_receipt=receipt)
    shutil.copy2(old/'generation_budget.json',out/'generation_budget.json')
    if(old/'generation_tokens.jsonl').is_file():shutil.copy2(old/'generation_tokens.jsonl',out/'generation_tokens.jsonl')
    for p in old.iterdir():
        if p.is_file()and p.suffix in ('.json','.jsonl','.log'):shutil.copy2(p,out/('prior_attempt__'+p.name))
    for p in (old/'configs').glob('*.yaml'):shutil.copy2(p,out/'configs'/('prior_attempt__'+p.name))
    for p in (old/'validation').glob('*.jsonl'):shutil.copy2(p,out/('prior_attempt__Dev_'+p.name))
    shutil.copy2(receipt,out/'repaired_cpu_acceptance.json')
    spec=read(out/'run.json');spec.update(inherited_deadline=proof['inherited_deadline'],
        original_started_at=proof['original_started_at'],prior_attempt=str(old),cpu_receipt=str(out/'repaired_cpu_acceptance.json'))
    write(out/'run.json',spec)
    write(out/'recovery_receipt.json',{**proof,'new_run':str(out),'method_protocol_changed':False,
        'old_implementation_sha256':OLD_IMPL,'new_implementation_sha256':digest(PACKAGE/'implementation_manifest.json'),
        'training_started':False,'cpu_acceptance_sha256':digest(receipt)})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--old-run',type=Path,required=True)
    p.add_argument('--active-pointer',type=Path,required=True);a=p.parse_args();old=a.old_run.resolve()
    if sys.platform!='linux':raise ValueError('Recovery runs in the original AutoDL Linux environment')
    import fcntl
    lock=(old/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_launch_code();assert_no_processes(old);proof=zero_update_proof(old)
    pointer=a.active_pointer.resolve()
    if pointer.exists():raise ValueError('Active-run pointer already exists; preserve it and use status/return')
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    receipt=ROOT/'outputs'/('b500-guard-native-cpu-'+stamp+'.json');receipt.parent.mkdir(exist_ok=True)
    print('NATIVE_CPU_LOG='+str(receipt.with_suffix('.log')),flush=True)
    with (receipt.with_suffix('.log')).open('w')as log:
        subprocess.run([sys.executable,str(ROOT/'tools/run_b500_cpu_acceptance.py'),'--asset-project',read(old/'run.json')['project'],
            '--output',str(receipt)],check=True,stdout=log,stderr=subprocess.STDOUT)
    accepted_cpu_receipt(receipt)
    # Recheck after CPU tests, before writing any new run/pointer.
    assert_no_processes(old);proof=zero_update_proof(old)
    out=old.parent/('b500_group_weight_v1_recovery_'+stamp+'_'+os.urandom(3).hex())
    prepare_recovery(old,out,receipt,proof)
    pointer.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=pointer.parent,delete=False)as f:
        f.write(str(out)+'\n');temporary=Path(f.name)
    temporary.replace(pointer)
    print('RECOVERY_PREPARED=true\nNATIVE_CPU_PASSED=true\nRUN='+str(out)+'\nOLD_RUN_PRESERVED='+str(old))
    print('CARRIED_BUDGET='+json.dumps(proof['carried_budget'])+'\nINHERITED_DEADLINE='+str(proof['inherited_deadline']))
    print('No GPU initialized, no models loaded, no training started. Use run only after RECOVERY_PREPARED=true.')


if __name__=='__main__':main()
