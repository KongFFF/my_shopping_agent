"""Restore verified B750 and finish fixed800 with bounded skip-limit50, CPU preparation only."""
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

from b500_run import ROOT,PACKAGE,read,write,digest,verify_launch_code,verify_runtime_snapshot,accepted_cpu_receipt
from recover_b500_guard import rows,assert_no_processes
sys.path.insert(0,str(PACKAGE/'overlay'))
from b500_run_control import checkpoint,checked_folder

AUDIT=ROOT/'analysis/b800_interruption_audit_20261122/audit.json'
SUPPLEMENT=PACKAGE/'finish750_protocol.json'
FAILURE='shopping dynamic sampling reached max_consecutive_skipped_updates=10 without an optimizer update; stopping a persistently no-signal run'


def evidence(old):
    expected=read(AUDIT);spec=read(old/'run.json')
    if not expected['audit_passed']or str(old)!=expected['source_run']:raise ValueError('Expected the returned, audited failed B800 run')
    for rel,key in [('run.json','source_run_sha256'),('training_diagnostics.jsonl','source_diagnostics_sha256'),('weights.jsonl','source_weights_sha256'),('configs/phase800.yaml','source_config_sha256')]:
        if digest(old/rel)!=expected[key]:raise ValueError('Audited source changed: '+rel)
    if not(old/'stopped.json').exists()or(old/'training_complete.json').exists():raise ValueError('Expected stopped incomplete source')
    if spec['protocol_sha256']!=digest(PACKAGE/'protocol.json')or spec['implementation_manifest_sha256']!=digest(PACKAGE/'implementation_manifest.json'):raise ValueError('Frozen method identity differs')
    if FAILURE not in(old/'train_to_800.log').read_text(errors='replace'):raise ValueError('Expected ten-skip failure missing')
    events=rows(old/'training_diagnostics.jsonl')
    if [e['global_step']for e in events if e['event']=='optimizer_step']!=list(range(1,788)):raise ValueError('Source update chain differs from1..787')
    weights=rows(old/'weights.jsonl')
    if [w['driver_global_step_before_update']for w in weights if w['event']=='actor_update_returned']!=list(range(650,787)):raise ValueError('Source Actor return chain changed')
    budget=read(old/'generation_budget.json')
    if budget!=expected['cumulative_generation']:raise ValueError('Audited physical generation ledger changed')
    plans=[r for r in rows(old/'train/checkpoint_retention.jsonl')if r['event']=='retention_plan']
    if not plans or plans[-1]['new_step']!=750 or plans[-1]['keep']!=[700,750]:raise ValueError('Missing750 save/retention proof')
    source=Path(checkpoint(old/'train',750))
    return spec,events,weights,budget,source


def derive_config(before,spec,old,out):
    supplement=read(SUPPLEMENT)
    if (before['trainer']['total_training_steps']!=800 or before['shopping_dynamic_sampling']['max_consecutive_skipped_updates']!=10
        or supplement['base_protocol_sha256']!=digest(PACKAGE/'protocol.json')or supplement['new_skip_limit']!=50):raise ValueError('Expected fixed800/base10 and frozen supplement50')
    cfg=copy.deepcopy(before);env=dict(spec['env'])
    env.update(GRPO_OUTPUT_DIR=str(out/'train'),SHOPPING_GRPO_DIAGNOSTICS_PATH=str(out/'training_diagnostics.jsonl'),
        B500_WEIGHT_LOG=str(out/'weights.jsonl'),B500_RUN_DIR=str(out),B500_CONTROL_FILE=str(out/'control.json'),
        OMP_NUM_THREADS='1',PYTHONPATH=os.pathsep.join([str(PACKAGE/'overlay'),str(Path(spec['runtime'])/'src'),str(ROOT/'tools')]))
    cfg['trainer'].update(resume_mode='resume_path',resume_from_path=str(old/'train/global_step_750'),del_local_ckpt_after_load=False,
        val_before_train=False,default_local_dir=str(out/'train'),validation_data_dir=str(out/'validation'),experiment_name='b800-finish750-skip50-v1')
    cfg['shopping_dynamic_sampling']['max_consecutive_skipped_updates']=50
    cfg['actor_rollout_ref']['rollout']['trace']['experiment_name']='b800-finish750-skip50-v1'
    cfg['ray_kwargs']['ray_init']['_temp_dir']='/tmp/b8f-'+out.name[-12:]
    cfg['ray_kwargs']['ray_init']['runtime_env']['env_vars']=env
    actor=cfg['actor_rollout_ref']['actor'];optim=actor['optim']
    if optim['lr']!=1e-6 or optim['lr_scheduler_type']!='constant'or set(actor['checkpoint']['load_contents'])!={'model','optimizer','extra'}:raise ValueError('Full optimizer/scheduler restore required')
    if actor['checkpoint']['async_save']or cfg['trainer']['save_freq']!=50 or cfg['trainer']['test_freq']!=50:raise ValueError('Original synchronous save/validation schedule required')
    return cfg,env


def stage(old,out,cpu):
    verify_launch_code();accepted_cpu_receipt(cpu)
    if out.exists():raise ValueError('New recovery directory exists; preserve it')
    spec,events,weights,budget,source=evidence(old);asset=Path(spec['project']);runtime=Path(spec['runtime']);verify_runtime_snapshot(asset,runtime)
    print('VERIFY_FULL750_CRC (reads all files; no tensor loading)',flush=True);checked_folder(old/'train',750,crc=True)
    model_sha=digest(source/'actor/model_world_size_1_rank_0.pt')
    size=sum(p.stat().st_size for p in source.rglob('*')if p.is_file());required=int(size*1.1)+4*2**30;free=shutil.disk_usage(old).free
    print(f'DISK_FREE_GiB={free/2**30:.2f} REQUIRED_GiB={required/2**30:.2f}',flush=True)
    if free<required:raise ValueError('Insufficient disk for new800 checkpoint plus logs; source750 is never deleted')
    import yaml
    before=yaml.safe_load((old/'configs/phase800.yaml').read_text());cfg,env=derive_config(before,spec,old,out)
    with tempfile.NamedTemporaryFile(suffix='.json')as f:
        subprocess.run([sys.executable,str(ROOT/'tools/check_b500_history.py'),'--asset-project',str(asset),'--runtime-project',str(runtime),'--output',f.name],check=True)
    (out/'configs').mkdir(parents=True);(out/'train').mkdir();(out/'validation').mkdir()
    (out/'configs/phase800.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
    for p in old.iterdir():
        if p.is_file()and p.suffix in ('.json','.jsonl','.log'):shutil.copy2(p,out/('interrupted__'+p.name))
    for p in(old/'configs').glob('*.yaml'):shutil.copy2(p,out/'configs'/('interrupted__'+p.name))
    for p in(old/'validation').glob('*.jsonl'):shutil.copy2(p,out/'validation'/p.name)
    prefix=[e for e in events if e.get('global_step',0)<=750]
    (out/'training_diagnostics.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n'for e in prefix))
    committed=[w for w in weights if w['driver_global_step_before_update']<750]
    (out/'weights.jsonl').write_text(''.join(json.dumps(w,ensure_ascii=False)+'\n'for w in committed))
    shutil.copy2(old/'generation_tokens.jsonl',out/'generation_tokens.jsonl');write(out/'generation_budget.json',budget)
    shutil.copy2(cpu,out/'native_cpu_acceptance.json');shutil.copy2(SUPPLEMENT,out/'effective_protocol_supplement.json')
    new={'version':'B800-finish750-skip50-v1','project':str(asset),'runtime':str(runtime),'output':str(out),'repo':str(ROOT),
        'protocol_sha256':spec['protocol_sha256'],'implementation_manifest_sha256':spec['implementation_manifest_sha256'],
        'effective_protocol_supplement_sha256':digest(SUPPLEMENT),'cpu_receipt':str(out/'native_cpu_acceptance.json'),'env':env,
        'config_hashes':{'800':digest(out/'configs/phase800.yaml')},'training_started':False,'launch_attempted':False,
        'resume_step':750,'target_step':800,'phase_stops':[800],'source_checkpoint':str(source),'source_model_sha256':model_sha,
        'checkpoint_size_bytes':size,'required_free_bytes':required,'budget_record_only':True,'code_branch':'codex/b800-finish750',
        'return_root':'analysis/b800_training_return','prior_B800':str(old),'prior_B650':spec['prior_B650'],'no_automatic_evaluation':True}
    write(out/'run.json',new)
    receipt={'passed':True,'source':str(source),'source_crc_verified':True,'source_model_sha256':model_sha,'checkpoint_copied':False,
        'resume_step':750,'target_step':800,'new_updates':50,'first_new_update':751,'canonical_prefix':750,
        'old_last_completed_update':787,'rolled_back_updates':list(range(751,788)),'carried_generation_budget':budget,
        'skip_limit_before':10,'skip_limit_after':50,'effective_protocol_supplement_sha256':digest(SUPPLEMENT),
        'learning_objective_changed':False,'sampling_filter_changed':False,'per_attempt_generation_limit_changed':False,
        'no_rewarmup':True,'parent_runs_untouched':True,'budget_record_only':True,'training_started':False,
        'comparability':'Tail stopping policy and recovery differ; disclose extra sampling/replayed updates, no same-compute claim'}
    write(out/'finish750_preparation.json',receipt)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--old-run',type=Path,required=True);p.add_argument('--active-pointer',type=Path,required=True);a=p.parse_args()
    if sys.platform!='linux':raise ValueError('Run in original AutoDL Linux environment')
    import fcntl
    old=a.old_run.resolve();pointer=a.active_pointer.resolve()
    if pointer.exists():raise ValueError('Recovery already prepared; use status/return')
    with(old/'supervisor.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify_launch_code();assert_no_processes(old);evidence(old)
        cpu=ROOT/'outputs'/('finish750_cpu_'+str(time.time_ns())+'.json');cpu.parent.mkdir(exist_ok=True)
        print('NATIVE_CPU_LOG='+str(cpu.with_suffix('.log')),flush=True)
        with cpu.with_suffix('.log').open('w')as log:
            subprocess.run([sys.executable,str(ROOT/'tools/run_b500_cpu_acceptance.py'),'--asset-project',read(old/'run.json')['project'],'--output',str(cpu)],check=True,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1'))
        assert_no_processes(old)
        out=old.parent/('b800_finish750_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+os.urandom(3).hex());stage(old,out,cpu)
        with tempfile.NamedTemporaryFile(mode='w',dir=pointer.parent,delete=False)as f:f.write(str(out)+'\n');temp=Path(f.name)
        temp.replace(pointer)
    print('B800_FINISH750_PREPARED=true\nNATIVE_CPU_PASSED=true\nRUN='+str(out)+'\nNo training started. Full750 restore; new updates751..800; skip limit50.')


if __name__=='__main__':main()
