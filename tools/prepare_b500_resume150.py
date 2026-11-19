"""Stage full-state B150->350->500 recovery without GPU or checkpoint copies."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from b500_run import ROOT,PACKAGE,read,write,digest,prepare,verify_launch_code,accepted_cpu_receipt
sys.path.insert(0,str(PACKAGE/'overlay'))
from b500_run_control import checkpoint,checked_folder
from recover_b500_guard import assert_no_processes,rows


def evidence(old):
    spec=read(old/'run.json')
    if spec['protocol_sha256']!=digest(PACKAGE/'protocol.json')or spec['implementation_manifest_sha256']!=digest(PACKAGE/'implementation_manifest.json'):
        raise ValueError('Original B method/protocol differs')
    if not(old/'stopped.json').is_file()or not spec.get('training_started'):raise ValueError('Original supervisor must be stopped')
    if(old/'training_complete.json').exists()or(old/'phase350_complete.json').exists():raise ValueError('Unexpected later completion')
    for n,h in spec['config_hashes'].items():
        if digest(old/'configs'/f'phase{n}.yaml')!=h:raise ValueError('Original configuration drift')
    checkpoint(old/'train',150)
    events=rows(old/'training_diagnostics.jsonl')
    steps=[e['global_step']for e in events if e.get('event')=='optimizer_step']
    if not steps or steps!=list(range(1,steps[-1]+1))or not(150<=steps[-1]<200):raise ValueError('Expected continuous original update chain through150, before200')
    budget=read(old/'generation_budget.json')
    if any(type(budget.get(k))is not int or budget[k]<v for k,v in {'train':6392,'validation_physical':1000}.items()):
        raise ValueError('Generation ledger is missing or below confirmed pre-failure cost')
    return spec,events,budget


def stage(old,out,receipt):
    spec,events,budget=evidence(old)
    print('VERIFY_CHECKPOINT_150_CRC (reads all files; no tensor loading)',flush=True)
    source=checked_folder(old/'train',150,crc=True)
    size=sum(p.stat().st_size for p in source.rglob('*')if p.is_file())
    required=4*size+16*2**30
    free=shutil.disk_usage(old).free
    print(f'DISK_FREE_GiB={free/2**30:.2f} REQUIRED_GiB={required/2**30:.2f}',flush=True)
    if free<required:raise ValueError('Insufficient free space for new checkpoint peak and export/log margin')
    prepare(Path(spec['project']),out,cpu_receipt=receipt)
    import yaml
    path=out/'configs/phase350.yaml';cfg=yaml.safe_load(path.read_text())
    cfg['trainer'].update(resume_mode='resume_path',resume_from_path=str(source),val_before_train=False,del_local_ckpt_after_load=False)
    if set(cfg['actor_rollout_ref']['actor']['checkpoint']['load_contents'])!={'model','optimizer','extra'}:
        raise ValueError('Full model/optimizer/extra restore required')
    path.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
    # Only the committed chain through150 counts toward the final endpoint.
    prefix=[e for e in events if e.get('global_step',0)<=150]
    (out/'training_diagnostics.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n'for e in prefix))
    for p in old.iterdir():
        if p.is_file()and p.suffix in ('.json','.jsonl','.log'):shutil.copy2(p,out/('prior_attempt__'+p.name))
    for p in (old/'configs').glob('*.yaml'):shutil.copy2(p,out/'configs'/('prior_attempt__'+p.name))
    for p in (old/'validation').glob('*.jsonl'):shutil.copy2(p,out/('prior_attempt__Dev_'+p.name))
    if(old/'generation_tokens.jsonl').exists():shutil.copy2(old/'generation_tokens.jsonl',out/'generation_tokens.jsonl')
    # Preserve the complete old weights log separately, including rolled-back updates.
    if(old/'weights.jsonl').exists():shutil.copy2(old/'weights.jsonl',out/'weights_before_rollback.jsonl')
    shutil.copy2(receipt,out/'resume_cpu_acceptance.json');write(out/'generation_budget.json',budget)
    new=read(out/'run.json');new.update(resume_step=150,prior_attempt=str(old),source_checkpoint=str(source),
        checkpoint_size_bytes=size,required_free_bytes=required,budget_record_only=True,
        budget_instruction='User manages budget; retain generation counts, disable time/generation budget cutoffs',
        cpu_receipt=str(out/'resume_cpu_acceptance.json'),config_hashes={str(n):digest(out/'configs'/f'phase{n}.yaml')for n in (350,500)})
    write(out/'run.json',new)
    write(out/'resume150_receipt.json',{'source':str(source),'resume_step':150,'first_new_update':151,
        'source_crc_verified':True,'checkpoint_copied':False,'retention':'latest2_plus350_in_new_train_only',
        'committed_prefix':150,'last_old_diagnostic_update':max(e['global_step']for e in events if e.get('event')=='optimizer_step'),
        'carried_generation_budget':budget,'budget_record_only':True,'training_started':False,
        'unplanned_restart':True,'config_sha256':new['config_hashes']})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--old-run',type=Path,required=True)
    p.add_argument('--active-pointer',type=Path,required=True);a=p.parse_args();old=a.old_run.resolve();pointer=a.active_pointer.resolve()
    if sys.platform!='linux':raise ValueError('Use original AutoDL Linux environment')
    import fcntl
    lock=(old/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_launch_code();assert_no_processes(old);evidence(old)
    if pointer.exists():raise ValueError('Resume pointer already exists; use status, never prepare again')
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());receipt=ROOT/'outputs'/f'resume150_cpu_{stamp}.json'
    receipt.parent.mkdir(exist_ok=True);print('NATIVE_CPU_LOG='+str(receipt.with_suffix('.log')),flush=True)
    with receipt.with_suffix('.log').open('w')as log:
        subprocess.run([sys.executable,str(ROOT/'tools/run_b500_cpu_acceptance.py'),'--asset-project',read(old/'run.json')['project'],
            '--output',str(receipt)],check=True,stdout=log,stderr=subprocess.STDOUT)
    accepted_cpu_receipt(receipt);assert_no_processes(old)
    out=old.parent/('b500_resume150_'+stamp+'_'+os.urandom(3).hex());stage(old,out,receipt)
    pointer.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=pointer.parent,delete=False)as f:
        f.write(str(out)+'\n');temporary=Path(f.name)
    temporary.replace(pointer)
    print('RESUME150_PREPARED=true\nNATIVE_CPU_PASSED=true\nRUN='+str(out)+'\nSOURCE='+str(old/'train/global_step_150'))
    print('No training started. First new update151; scheduled full restart350; endpoint500. Budget is user-managed.')


if __name__=='__main__':main()
