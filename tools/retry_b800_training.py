"""One user-authorized B800 retry from unchanged complete B650, preserving failed work."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import b800_training as continuation
from b500_run import ROOT,PACKAGE,read,write,digest,verify_launch_code
from recover_b500_guard import rows,assert_no_processes

SKIP_ERROR='shopping dynamic sampling reached max_consecutive_skipped_updates=10 without an optimizer update; stopping a persistently no-signal run'


def evidence(old):
    spec=read(old/'run.json')
    if spec.get('resume_step')!=650 or spec.get('target_step')!=800 or not spec.get('training_started'):
        raise ValueError('Expected attempted B650->800 run')
    if not(old/'stopped.json').is_file()or(old/'training_complete.json').exists():raise ValueError('Expected stopped, incomplete B800')
    if spec.get('retry_of'):raise ValueError('This entry permits only one retry; return this failure for diagnosis')
    if (spec['protocol_sha256']!=digest(PACKAGE/'protocol.json')or spec['implementation_manifest_sha256']!=digest(PACKAGE/'implementation_manifest.json')):
        raise ValueError('Original B method/protocol differs')
    if digest(old/'configs/phase800.yaml')!=spec['config_hashes']['800']:raise ValueError('Failed config changed')
    if list((old/'train').glob('global_step_*'))or(old/'train/latest_checkpointed_iteration.txt').exists():
        raise ValueError('New checkpoint evidence present; inspect it rather than rolling back blindly')
    if SKIP_ERROR not in(old/'train_to_800.log').read_text(errors='replace'):
        raise ValueError('Training log does not confirm the ten-skip guard failure; return logs before retry')
    parent=Path(spec['prior_B650']);parent_spec,prefix,parent_budget=continuation.source_evidence(parent)
    if spec['source_checkpoint']!=str(parent/'train/global_step_650')or spec['source_model_sha256']!=continuation.MODEL_SHA:
        raise ValueError('Failed run source is not evaluated B650')
    events=rows(old/'training_diagnostics.jsonl')
    if events[:len(prefix)]!=prefix:raise ValueError('Failed run parent diagnostics prefix changed')
    steps=[e['global_step']for e in events if e.get('event')=='optimizer_step']
    if steps!=list(range(1,654)):raise ValueError('Expected completed chain1..653 with no654 update')
    budget=read(old/'generation_budget.json')
    if any(type(budget.get(k))is not int or budget[k]<v for k,v in parent_budget.items()):raise ValueError('Missing/decreasing generation ledger')
    return spec,parent,budget


def stage(old,out):
    spec,parent,budget=evidence(old)
    # Native CPU acceptance and complete650 CRC/SHA are rechecked by the existing entry.
    continuation.prepare(parent,out)
    for p in old.iterdir():
        if p.is_file()and p.suffix in ('.json','.jsonl','.log'):shutil.copy2(p,out/('prior_attempt__'+p.name))
    for p in(old/'configs').glob('*.yaml'):shutil.copy2(p,out/'configs'/('prior_attempt__'+p.name))
    if(old/'generation_tokens.jsonl').exists():shutil.copy2(old/'generation_tokens.jsonl',out/'generation_tokens.jsonl')
    # Roll back model updates651..653, but never subtract their physical sampling cost.
    write(out/'generation_budget.json',budget)
    new=read(out/'run.json');new.update(retry_of=str(old),code_branch='codex/b800-retry650',
        retry_number=1,retry_updates_rolled_back=[651,652,653],budget_record_only=True)
    write(out/'run.json',new)
    receipt={'passed':True,'old_run':str(old),'source_checkpoint':new['source_checkpoint'],
        'resume_step':650,'target_step':800,'new_updates':150,'rolled_back_updates':[651,652,653],
        'carried_generation_budget':budget,'old_failure_confirmed':SKIP_ERROR,'method_changed':False,
        'seed_changed':False,'skip_limit_changed':False,'old_files_preserved':True,'training_started':False,
        'retry_may_reproduce_failure':'Same full650 policy/optimizer/RNG/data restoration; no success guarantee',
        'authorization':'User explicitly requested one unchanged retry before considering a repair'}
    write(out/'retry650_receipt.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--old-run',type=Path,required=True);p.add_argument('--active-pointer',type=Path,required=True);a=p.parse_args()
    if sys.platform!='linux':raise ValueError('Run in original AutoDL Linux environment')
    import fcntl
    old=a.old_run.resolve();pointer=a.active_pointer.resolve()
    if pointer.exists():raise ValueError('Retry already prepared; use status/return, do not prepare again')
    with(old/'supervisor.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify_launch_code();assert_no_processes(old);evidence(old)
        out=old.parent/('b800_retry650_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+os.urandom(3).hex())
        stage(old,out);assert_no_processes(old)
        pointer.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w',dir=pointer.parent,delete=False)as f:f.write(str(out)+'\n');temp=Path(f.name)
        temp.replace(pointer)
    print('B800_RETRY_PREPARED=true\nNATIVE_CPU_PASSED=true\nRUN='+str(out)+'\nNo training started. Source650, new updates651..800; original attempt preserved.')


if __name__=='__main__':main()
