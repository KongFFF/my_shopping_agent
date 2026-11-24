"""Prepare user-authorized fixed C650->800 continuation; never evaluate endpoints."""
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

import c800_run as runner
from c800_run import ROOT,PACKAGE,read,write,digest,verify_runtime_snapshot,accepted_cpu_receipt
def rows(path):return [json.loads(line)for line in path.read_text().splitlines()if line.strip()]if path.exists()else []

def assert_no_processes(old):
    for p in Path('/proc').glob('[0-9]*/environ'):
        try:entries=p.read_bytes().split(b'\0')
        except (OSError,PermissionError):continue
        if b'C500_RUN_DIR='+old.as_posix().encode()in entries:raise ValueError('Parent still has owned process: '+p.parent.name)
sys.path.insert(0,str(PACKAGE/'overlay'))
from c500_run_control import checkpoint,checked_folder

MODEL_SHA='ca4524e7217dcc64d002421d913e9dbf13561b01aec20df390e1e6b065617eac'


def source_evidence(parent):
    spec=read(parent/'run.json');done=read(parent/'training_complete.json')
    if done['completed_updates']!=650 or (parent/'stopped.json').exists():raise ValueError('Parent is not a completed C650')
    if spec['implementation_manifest_sha256']!=digest(PACKAGE/'implementation_manifest.json')or spec['protocol_sha256']!=digest(PACKAGE/'protocol.json'):raise ValueError('C500 method identity drift')
    if digest(parent/'configs/phase650.yaml')!=spec['config_hashes']['650']:raise ValueError('Parent phase650 configuration drift')
    expected=digest(ROOT/'experiments/group_weight_c650_v1/protocol.json')
    if spec.get('continuation_protocol_sha256')!=expected or done.get('continuation_protocol_sha256')!=expected:
        raise ValueError('Parent C650 continuation protocol drift')
    if done.get('experiment')!='C650' or done.get('gamma')!=4.0 or done.get('protocol_sha256')!=spec['protocol_sha256']:
        raise ValueError('Parent C650 completion identity drift')
    if spec.get('resume_step')!=500 or spec.get('target_step')!=650 or spec.get('phase_stops')!=[650]:
        raise ValueError('Parent C650 phase contract drift')
    receipt=read(parent/'accepted_step_650.json')
    if receipt.get('step')!=650 or Path(receipt.get('checkpoint','')).resolve()!=(parent/'train/global_step_650').resolve():
        raise ValueError('Parent C650 final checkpoint receipt differs')
    events=rows(parent/'training_diagnostics.jsonl')
    if [e['global_step']for e in events if e.get('event')=='optimizer_step']!=list(range(1,651)):raise ValueError('Parent chain not exactly1..650')
    budget=read(parent/'generation_budget.json')
    if budget!=done['generation_budget']:raise ValueError('Parent generation ledger mismatch')
    if spec.get('experiment')!='C650' or spec.get('gamma')!=4.0:raise ValueError('Parent is not the evaluated C650 gamma4 arm')
    return spec,events,budget


def derive_config(before,old_spec,parent,out):
    cfg=copy.deepcopy(before);env=dict(old_spec['env'])
    env.update(GRPO_OUTPUT_DIR=str(out/'train'),SHOPPING_GRPO_DIAGNOSTICS_PATH=str(out/'training_diagnostics.jsonl'),
        C500_WEIGHT_LOG=str(out/'weights.jsonl'),C500_RUN_DIR=str(out),C500_CONTROL_FILE=str(out/'control.json'),
        OMP_NUM_THREADS='1',PYTHONPATH=os.pathsep.join([str(PACKAGE/'overlay'),str(Path(old_spec['runtime'])/'src'),str(ROOT/'tools')]))
    cfg['trainer'].update(total_training_steps=800,resume_mode='resume_path',resume_from_path=str(parent/'train/global_step_650'),
        del_local_ckpt_after_load=False,val_before_train=False,default_local_dir=str(out/'train'),
        validation_data_dir=str(out/'validation'),experiment_name='c800-group-weight-v1')
    cfg['actor_rollout_ref']['rollout']['trace']['experiment_name']='c800-group-weight-v1'
    cfg['ray_kwargs']['ray_init']['_temp_dir']='/tmp/c800-'+out.name[-12:]
    cfg['ray_kwargs']['ray_init']['runtime_env']['env_vars']=env
    actor=cfg['actor_rollout_ref']['actor'];optim=actor['optim']
    if optim['lr']!=1e-6 or optim['lr_scheduler_type']!='constant' or set(actor['checkpoint']['load_contents'])!={'model','optimizer','extra'}:
        raise ValueError('Full optimizer/scheduler restore and constant LR required')
    if actor['checkpoint']['async_save']or cfg['trainer']['save_freq']!=50 or cfg['trainer']['test_freq']!=50:raise ValueError('Checkpoint/validation schedule differs')
    if cfg['ray_kwargs']['ray_init']['runtime_env']['worker_process_setup_hook']!='c500_run_control.setup_worker':raise ValueError('C hook identity differs')
    return cfg,env


def stage(parent,out,cpu):
    runner.verify_launch_code();accepted_cpu_receipt(cpu)
    if out.exists():raise ValueError('New C800 output exists; preserve it')
    old,events,budget=source_evidence(parent);asset=Path(old['project']);runtime=Path(old['runtime']);verify_runtime_snapshot(asset,runtime)
    source=Path(checkpoint(parent/'train',650));print('VERIFY_C650_FULL_STATE_CRC',flush=True);checked_folder(parent/'train',650,crc=True)
    if digest(source/'actor/model_world_size_1_rank_0.pt')!=MODEL_SHA:raise ValueError('Checkpoint is not the C650 previously exported and evaluated')
    size=sum(p.stat().st_size for p in source.rglob('*')if p.is_file());required=3*size+4*2**30;free=shutil.disk_usage(parent).free
    print(f'DISK_FREE_GiB={free/2**30:.2f} REQUIRED_GiB={required/2**30:.2f}',flush=True)
    if free<required:raise ValueError('Insufficient disk for three transient C800 checkpoints plus logs; free space before opening GPU')
    import yaml
    before=yaml.safe_load((parent/'configs/phase650.yaml').read_text());cfg,env=derive_config(before,old,parent,out)
    # Recheck original SFT/data/engine identity without loading tensor weights.
    (ROOT/'outputs').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='c800-history-',dir=ROOT/'outputs')as tmp:
        receipt=Path(tmp)/'identity.json'
        subprocess.run([sys.executable,str(ROOT/'tools/check_b500_history.py'),'--asset-project',str(asset),'--runtime-project',str(runtime),'--output',str(receipt)],check=True)
        history=read(receipt)
        if not history.get('passed'):raise ValueError('Historical asset identity check failed')
    (out/'configs').mkdir(parents=True);(out/'train').mkdir()
    write(out/'baseline_identity_preflight.json',history)
    (out/'configs/phase800.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
    shutil.copy2(parent/'configs/phase650.yaml',out/'configs/parent_phase650.yaml')
    (out/'training_diagnostics.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n'for e in events))
    for name in ['run.json','training_complete.json','generation_budget.json','weights.jsonl']:
        if(parent/name).exists():shutil.copy2(parent/name,out/('parent_C650__'+name))
    shutil.copy2(ROOT/'experiments/group_weight_c800_v1/protocol.json',out/'continuation_protocol.json')
    if cpu.with_suffix('.log').is_file():shutil.copy2(cpu.with_suffix('.log'),out/'native_cpu_acceptance.log')
    shutil.copy2(cpu,out/'native_cpu_acceptance.json');write(out/'generation_budget.json',budget)
    if(parent/'generation_tokens.jsonl').exists():shutil.copy2(parent/'generation_tokens.jsonl',out/'generation_tokens.jsonl')
    spec={'version':'C800-user-continuation-v1','project':str(asset),'runtime':str(runtime),'output':str(out),'repo':str(ROOT),
        'protocol_sha256':old['protocol_sha256'],'implementation_manifest_sha256':old['implementation_manifest_sha256'],
        'cpu_receipt':str(out/'native_cpu_acceptance.json'),'env':env,'config_hashes':{'800':digest(out/'configs/phase800.yaml')},
        'training_started':False,'launch_attempted':False,'resume_step':650,'target_step':800,'phase_stops':[800],
        'experiment':'C800','gamma':4.0,'continuation_protocol_sha256':digest(ROOT/'experiments/group_weight_c800_v1/protocol.json'),'source_checkpoint':str(source),'source_model_sha256':MODEL_SHA,'checkpoint_size_bytes':size,'required_free_bytes':required,
        'budget_record_only':True,'code_branch':'codex/c800-training','return_root':'analysis/c800_training_return',
        'prior_C650':str(parent),'no_automatic_evaluation':True}
    write(out/'run.json',spec)
    write(out/'c800_preparation.json',{'passed':True,'resume_step':650,'target_step':800,'new_updates':150,'source_crc_verified':True,
        'source_model_sha256':MODEL_SHA,'model_optimizer_extra_restore':True,'data_checkpoint_required':True,
        'dataloader_restore_policy':'Original pinned veRL epoch-boundary rule;650 may skip saved dataloader state and begin a fresh epoch, as historical A800/B800',
        'lr_after_restore':1e-6,'first_new_update':651,
        'checkpoint_steps':[700,750,800],'validation_steps':[700,750,800],'parent_C650_untouched':True,
        'method':'F0.25/S1 (gamma4); global effective Actor-token normalization','budget_record_only':True,
        'authorization':'User explicitly requested C800 after C650 independent Dev24/50 and Final136/200',
        'comparison_limit':'Historical single-seed continuation; A/B have different recovery and sampling costs; C keeps skip threshold10 unlike adjusted B800 tail',
        'evaluation_started':False,'training_started':False})


def prepare(parent,out):
    if sys.platform!='linux':raise ValueError('Run preparation in original AutoDL Linux environment')
    import fcntl
    with (parent/'supervisor.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert_no_processes(parent);source_evidence(parent)
        (ROOT/'outputs').mkdir(exist_ok=True);cpu=ROOT/'outputs'/('c800_cpu_'+str(time.time_ns())+'.json')
        print('NATIVE_CPU_LOG='+str(cpu.with_suffix('.log')),flush=True)
        with cpu.with_suffix('.log').open('w')as log:
            subprocess.run([sys.executable,str(ROOT/'tools/run_c500_cpu_acceptance.py'),'--asset-project',read(parent/'run.json')['project'],'--output',str(cpu)],check=True,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1'))
        with cpu.with_suffix('.log').open('a')as log:
            for pattern in ['test_c800_training.py','test_watch_c500_shutdown.py','test_watch_c800_shutdown.py']:
                subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tools'),'-p',pattern],check=True,stdout=log,stderr=subprocess.STDOUT)
        assert_no_processes(parent);stage(parent,out,cpu)
    print('C800_PREPARED=true\nNATIVE_CPU_PASSED=true\nRUN='+str(out)+'\nNo training started; new updates651..800 only.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();prepare(a.parent.resolve(),a.out.resolve())
