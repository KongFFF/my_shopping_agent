"""Prepare user-authorized fixed B500->650 continuation; never evaluate endpoints."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import b500_run as runner
from b500_run import ROOT,PACKAGE,read,write,digest,verify_runtime_snapshot,accepted_cpu_receipt
from recover_b500_guard import rows,assert_no_processes
sys.path.insert(0,str(PACKAGE/'overlay'))
from b500_run_control import checkpoint,checked_folder

MODEL_SHA='ffb6a3f4994ad0b1fa72b4c0dbf8db8e57adf430ea0d4e7b97975619b2125275'


def source_evidence(parent):
    spec=read(parent/'run.json');done=read(parent/'training_complete.json')
    if done['completed_updates']!=500 or (parent/'stopped.json').exists():raise ValueError('Parent is not a completed B500')
    if spec['implementation_manifest_sha256']!=digest(PACKAGE/'implementation_manifest.json')or spec['protocol_sha256']!=digest(PACKAGE/'protocol.json'):raise ValueError('B500 method identity drift')
    if digest(parent/'configs/phase500.yaml')!=spec['config_hashes']['500']:raise ValueError('Parent phase500 configuration drift')
    events=rows(parent/'training_diagnostics.jsonl')
    if [e['global_step']for e in events if e.get('event')=='optimizer_step']!=list(range(1,501)):raise ValueError('Parent chain not exactly1..500')
    budget=read(parent/'generation_budget.json')
    if budget!=done['generation_budget']:raise ValueError('Parent generation ledger mismatch')
    return spec,events,budget


def derive_config(before,old_spec,parent,out):
    cfg=copy.deepcopy(before);env=dict(old_spec['env'])
    env.update(GRPO_OUTPUT_DIR=str(out/'train'),SHOPPING_GRPO_DIAGNOSTICS_PATH=str(out/'training_diagnostics.jsonl'),
        B500_WEIGHT_LOG=str(out/'weights.jsonl'),B500_RUN_DIR=str(out),B500_CONTROL_FILE=str(out/'control.json'),
        OMP_NUM_THREADS='1',PYTHONPATH=os.pathsep.join([str(PACKAGE/'overlay'),str(Path(old_spec['runtime'])/'src'),str(ROOT/'tools')]))
    cfg['trainer'].update(total_training_steps=650,resume_mode='resume_path',resume_from_path=str(parent/'train/global_step_500'),
        del_local_ckpt_after_load=False,val_before_train=False,default_local_dir=str(out/'train'),
        validation_data_dir=str(out/'validation'),experiment_name='b650-group-weight-v1')
    cfg['actor_rollout_ref']['rollout']['trace']['experiment_name']='b650-group-weight-v1'
    cfg['ray_kwargs']['ray_init']['_temp_dir']='/tmp/b650-'+out.name[-12:]
    cfg['ray_kwargs']['ray_init']['runtime_env']['env_vars']=env
    actor=cfg['actor_rollout_ref']['actor'];optim=actor['optim']
    if optim['lr']!=1e-6 or optim['lr_scheduler_type']!='constant' or set(actor['checkpoint']['load_contents'])!={'model','optimizer','extra'}:
        raise ValueError('Full optimizer/scheduler restore and constant LR required')
    if actor['checkpoint']['async_save']or cfg['trainer']['save_freq']!=50 or cfg['trainer']['test_freq']!=50:raise ValueError('Checkpoint/validation schedule differs')
    return cfg,env


def stage(parent,out,cpu):
    runner.verify_launch_code();accepted_cpu_receipt(cpu)
    if out.exists():raise ValueError('New B650 output exists; preserve it')
    old,events,budget=source_evidence(parent);asset=Path(old['project']);runtime=Path(old['runtime']);verify_runtime_snapshot(asset,runtime)
    source=Path(checkpoint(parent/'train',500));print('VERIFY_B500_FULL_STATE_CRC',flush=True);checked_folder(parent/'train',500,crc=True)
    if digest(source/'actor/model_world_size_1_rank_0.pt')!=MODEL_SHA:raise ValueError('Checkpoint is not the B500 previously exported and evaluated')
    size=sum(p.stat().st_size for p in source.rglob('*')if p.is_file());required=3*size+4*2**30;free=shutil.disk_usage(parent).free
    print(f'DISK_FREE_GiB={free/2**30:.2f} REQUIRED_GiB={required/2**30:.2f}',flush=True)
    if free<required:raise ValueError('Insufficient disk for three transient B650 checkpoints plus logs; free space before opening GPU')
    import yaml
    before=yaml.safe_load((parent/'configs/phase500.yaml').read_text());cfg,env=derive_config(before,old,parent,out)
    # Recheck original SFT/data/engine identity without loading tensor weights.
    (ROOT/'outputs').mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix='.json')as f:
        subprocess.run([sys.executable,str(ROOT/'tools/check_b500_history.py'),'--asset-project',str(asset),'--runtime-project',str(runtime),'--output',f.name],check=True)
    (out/'configs').mkdir(parents=True);(out/'train').mkdir()
    (out/'configs/phase650.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
    shutil.copy2(parent/'configs/phase500.yaml',out/'configs/parent_phase500.yaml')
    (out/'training_diagnostics.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n'for e in events))
    for name in ['run.json','training_complete.json','generation_budget.json','weights.jsonl','weights_before_rollback.jsonl','resume150_receipt.json']:
        if(parent/name).exists():shutil.copy2(parent/name,out/('parent_B500__'+name))
    shutil.copy2(cpu,out/'native_cpu_acceptance.json');write(out/'generation_budget.json',budget)
    if(parent/'generation_tokens.jsonl').exists():shutil.copy2(parent/'generation_tokens.jsonl',out/'generation_tokens.jsonl')
    spec={'version':'B650-user-continuation-v1','project':str(asset),'runtime':str(runtime),'output':str(out),'repo':str(ROOT),
        'protocol_sha256':old['protocol_sha256'],'implementation_manifest_sha256':old['implementation_manifest_sha256'],
        'cpu_receipt':str(out/'native_cpu_acceptance.json'),'env':env,'config_hashes':{'650':digest(out/'configs/phase650.yaml')},
        'training_started':False,'launch_attempted':False,'resume_step':500,'target_step':650,'phase_stops':[650],
        'source_checkpoint':str(source),'source_model_sha256':MODEL_SHA,'checkpoint_size_bytes':size,'required_free_bytes':required,
        'budget_record_only':True,'code_branch':'codex/b650-training','return_root':'analysis/b650_training_return',
        'prior_B500':str(parent),'no_automatic_evaluation':True}
    write(out/'run.json',spec)
    write(out/'b650_preparation.json',{'passed':True,'resume_step':500,'target_step':650,'new_updates':150,'source_crc_verified':True,
        'source_model_sha256':MODEL_SHA,'model_optimizer_extra_restore':True,'data_checkpoint_required':True,
        'dataloader_restore_policy':'Original pinned veRL epoch-boundary rule;500 may skip saved dataloader state and begin a fresh epoch, as historical A650',
        'lr_after_restore':1e-6,'first_new_update':501,
        'checkpoint_steps':[550,600,650],'validation_steps':[550,600,650],'parent_B500_untouched':True,
        'method':'F0.5/S1; global effective Actor-token normalization','budget_record_only':True,
        'authorization':'User explicitly requested B650 after observing B500 Dev and supplemental Final',
        'comparison_limit':'Historical single-seed continuation; A650 had a600 fault recovery; no forced recovery in B650',
        'evaluation_started':False,'training_started':False})


def prepare(parent,out):
    if sys.platform!='linux':raise ValueError('Run preparation in original AutoDL Linux environment')
    import fcntl
    with (parent/'supervisor.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert_no_processes(parent);source_evidence(parent)
        (ROOT/'outputs').mkdir(exist_ok=True);cpu=ROOT/'outputs'/('b650_cpu_'+str(time.time_ns())+'.json')
        print('NATIVE_CPU_LOG='+str(cpu.with_suffix('.log')),flush=True)
        with cpu.with_suffix('.log').open('w')as log:
            subprocess.run([sys.executable,str(ROOT/'tools/run_b500_cpu_acceptance.py'),'--asset-project',read(parent/'run.json')['project'],'--output',str(cpu)],check=True,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1'))
        assert_no_processes(parent);stage(parent,out,cpu)
    print('B650_PREPARED=true\nNATIVE_CPU_PASSED=true\nRUN='+str(out)+'\nNo training started; new updates501..650 only.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();prepare(a.parent.resolve(),a.out.resolve())
