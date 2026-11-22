"""Fixed B800 export, original smoke3/Dev50/Final200. Explicitly user authorized."""
import argparse
import copy
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import b500_dev50 as base
from b500_run import ROOT,PACKAGE,read,write,digest,verify_runtime_snapshot
legacy=base.legacy
POST='preparation/posttrain_step800_v1'
SUITES=('smoke3','dev50','final200_clean')


def check_code():
    # Own manifest pins current shared dependencies; no change to original harness.
    for n,h in read(PACKAGE/'b800_eval_code_manifest.json')['files_sha256'].items():
        if digest(ROOT/n)!=h:raise ValueError('B800 evaluation dependency drift: '+n)


def derive(reference,asset,runtime,out,source):
    spec={k:copy.deepcopy(reference[k])for k in ('protocols','benchmarks')}
    spec['protocols']={s:spec['protocols'][s]for s in SUITES};spec['benchmarks']={s:spec['benchmarks'][s]for s in SUITES}
    sources={}
    for n,h in reference['source_sha256'].items():
        p=Path(n)
        if p.is_relative_to(asset)and p.relative_to(asset).parts[0]in ('src','scripts','configs'):p=runtime/p.relative_to(asset)
        if digest(p)!=h:raise ValueError('Original A800 source differs: '+str(p))
        sources[str(p)]=h
    tag='shopping-agent-b800-group-weight-v1';commands={}
    for s in SUITES:
        a=list(reference['commands'][s]);a[0]=sys.executable;a[1]=str(runtime/'scripts/evaluate_shop_benchmark.py')
        for k,v in [('--model',tag),('--output',out/'evaluation'/s/'trajectories.jsonl'),('--summary',out/'evaluation'/s/'summary.json')]:a=legacy.replace_option(a,k,v)
        if a[a.index('--benchmark')+1]!=spec['benchmarks'][s]['path']:raise ValueError('Command/benchmark binding differs')
        for k,v in [('--base-url','http://127.0.0.1:5700'),('--llm-base-url','http://127.0.0.1:8000/v1')]:
            if k in a and a[a.index(k)+1].rstrip('/')!=v:raise ValueError('Service URL differs: '+k)
        commands[s]=a
    spec.update(project=str(runtime),asset_project=str(asset),checkpoint=str(source),actor=str(source/'actor'),
        raw_export=str(out/'export/hf_unmerged'),merged_model=str(out/'export/merged'),evaluation_root=str(out/'evaluation'),
        model_tag=tag,source_sha256=sources,commands=commands,evaluated_step=800)
    return spec


def verify_completed_training(training):
    import math
    import yaml
    from collections import Counter
    from recover_b500_guard import rows
    run=read(training/'run.json');done=read(training/'training_complete.json');prep=read(training/'finish750_preparation.json')
    if ((training/'stopped.json').exists()or(training/'budget_truncated.json').exists()
        or done['completed_updates']!=800 or done['new_updates']!=50):raise ValueError('Expected completed B800 from750,50 new updates')
    if run.get('version')!='B800-finish750-skip50-v1'or run['resume_step']!=750 or run['target_step']!=800:raise ValueError('Expected fixed750->800 completion')
    if run['protocol_sha256']!=done['protocol_sha256']or run['protocol_sha256']!=digest(PACKAGE/'protocol.json'):raise ValueError('Base research protocol identity differs')
    if run['implementation_manifest_sha256']!=digest(PACKAGE/'implementation_manifest.json'):raise ValueError('Core weighting identity changed')
    supplement=digest(PACKAGE/'finish750_protocol.json')
    if (run['effective_protocol_supplement_sha256']!=supplement or digest(training/'effective_protocol_supplement.json')!=supplement
        or not prep['passed']or prep['resume_step']!=750 or prep['skip_limit_after']!=50):raise ValueError('Recorded750/skip50 supplement differs')
    if digest(training/'configs/phase800.yaml')!=run['config_hashes']['800']:raise ValueError('Completed800 config changed')
    cfg=yaml.safe_load((training/'configs/phase800.yaml').read_text())
    if cfg['trainer']['total_training_steps']!=800 or cfg['shopping_dynamic_sampling']['max_consecutive_skipped_updates']!=50:raise ValueError('Unexpected target/stopping bound')
    accepted=read(training/'accepted_step_800.json')
    if (accepted['step']!=800 or accepted['checkpoint']!=str(training/'train/global_step_800')or not accepted['diagnostics_written']
        or not math.isclose(accepted['lr'],1e-6,rel_tol=1e-5)or not math.isfinite(accepted['grad_norm'])or accepted['grad_norm']<=0):raise ValueError('Missing/invalid accepted800 update')
    es=rows(training/'training_diagnostics.jsonl');updates=[e for e in es if e['event']=='optimizer_step']
    if [e['global_step']for e in updates]!=list(range(1,801)):raise ValueError('Effective update chain must be exactly1..800')
    parent=[e for e in rows(training/'interrupted__training_diagnostics.jsonl')if e.get('global_step',0)<=750]
    if es[:len(parent)]!=parent:raise ValueError('Recovered750 diagnostic prefix changed')
    for e in updates[750:]:
        m=e['metrics']
        if (m['training/optimizer_updated']!=1 or not math.isclose(m['actor/lr'],1e-6,rel_tol=1e-5)
            or m['actor/grad_norm']<=0 or not all(math.isfinite(m[k])for k in ['actor/lr','actor/grad_norm','actor/loss'])
            or m['b500/actor_tokens']<=0):raise ValueError('Invalid resumed Actor update: '+str(e['global_step']))
    ws=rows(training/'weights.jsonl');returns=[w for w in ws if w['event']=='actor_update_returned']
    if (Counter(w['event']for w in ws)!=Counter(actor_update_started=150,actor_update_returned=150)
        or [w['driver_global_step_before_update']for w in returns]!=list(range(650,800))):raise ValueError('B800 weighting return chain differs')
    gen=rows(training/'generation_tokens.jsonl');budget={'train':sum(g['reserved_trajectories']for g in gen if not g['validation']),
        'validation_physical':sum(g['reserved_trajectories']for g in gen if g['validation'])}
    if budget!=read(training/'generation_budget.json')or budget!=done['generation_budget']:raise ValueError('Physical generation ledger mismatch')
    if budget['train']<prep['carried_generation_budget']['train']or budget['validation_physical']!=prep['carried_generation_budget']['validation_physical']+200:raise ValueError('Recovery generation cost or final native validation differs')
    log=(training/'train_to_800.log').read_text(errors='replace')
    if 'Setting global step to 750'not in log:raise ValueError('Actual750 restore log missing')
    return {'passed':True,'completed_updates':800,'new_updates':50,'resume_step':750,'skip_limit':50,
        'rolled_back_updates':prep['rolled_back_updates'],'physical_generation_budget':budget,
        'core_implementation_sha256':run['implementation_manifest_sha256'],'base_protocol_sha256':run['protocol_sha256'],
        'supplement_sha256':supplement,'recovery_config_sha256':run['config_hashes']['800'],
        'diagnostics_sha256':digest(training/'training_diagnostics.jsonl'),'weights_sha256':digest(training/'weights.jsonl'),
        'source800':accepted['checkpoint'],'training_status':'Runtime completion checked before evaluation; full evidence included on return'}


def prepare(training,out):
    check_code()
    if out.exists():raise ValueError('Output exists; preserve it and use status/return')
    (ROOT/'outputs').mkdir(exist_ok=True);log=ROOT/'outputs/b800_eval_cpu_tests.log';print('EVAL_CPU_LOG='+str(log),flush=True)
    with log.open('w')as f:subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tools'),'-p','test_b500_b800_evaluation.py'],check=True,stdout=f,stderr=subprocess.STDOUT)
    training_check=verify_completed_training(training);run=read(training/'run.json')
    asset=Path(run['project']);runtime=Path(run['runtime']);verify_runtime_snapshot(asset,runtime)
    sys.path.insert(0,str(PACKAGE/'overlay'))
    from b500_run_control import checkpoint,checked_folder
    source=Path(checkpoint(training/'train',800));base.require_export_space(source/'actor')
    print('CHECK800_CRC: reading checkpoint, no tensor loading',flush=True);checked_folder(training/'train',800,crc=True)
    meta=read(source/'actor/lora_train_meta.json');fsdp=read(source/'actor/fsdp_config.json')
    if int(meta['r'])!=16 or int(meta['lora_alpha'])!=32 or int(fsdp['world_size'])!=1:raise ValueError('B800 LoRA/FSDP metadata differs')
    reference_path=asset/legacy.RUN/POST/'evaluation_spec.json';ref=read(reference_path)
    if ref.get('evaluated_step',800)!=800 or not ref['checkpoint'].endswith('/global_step_800')or ref['model_tag']!='shopping-agent-grpo-step800':raise ValueError('Expected original A800 evaluation specification')
    spec=derive(ref,asset,runtime,out,source);frozen=read(PACKAGE/'protocol.json')['evaluation']['protocol'];refs={}
    for suite,n,score in [('smoke3',3,None),('dev50',50,29),('final200_clean',200,143)]:
        b=spec['benchmarks'][suite];rs=[json.loads(l)for l in Path(b['path']).read_text().splitlines()if l.strip()];ids=[int(r['task_id'])for r in rs]
        if digest(b['path'])!=b['sha256']or ids!=b['task_ids']or len(ids)!=len(set(ids))or len(ids)!=n:raise ValueError('Original benchmark identity differs: '+suite)
        if suite=='final200_clean'and b['sha256']!=legacy.FINAL_HASH:raise ValueError('Final Clean fingerprint differs')
        if score is not None:
            a=read(Path(ref['evaluation_root'])/suite/'summary.json')
            if a['expected_tasks']!=n or a['strict_successes']!=score or len(set(a['strict_success_task_ids']))!=score or not set(a['strict_success_task_ids']).issubset(ids):raise ValueError('Historical A800 result differs: '+suite)
            if a['protocol'].get('model')!=ref['model_tag']or a['protocol'].get('benchmark')!=b['path']:raise ValueError('Historical A800 summary identity differs: '+suite)
            for k,v in frozen.items():
                if k!='benchmark'and a['protocol'].get(k)!=v:raise ValueError('Original A800 protocol differs: '+k)
            refs[suite]=a
    post=out/POST;post.mkdir(parents=True);write(post/'evaluation_spec.json',spec)
    for suite,a in refs.items():write(post/f'A800_{suite}_summary.json',a)
    write(post/'training_completion_check.json',training_check)
    shutil.copy2(training/'effective_protocol_supplement.json',post/'effective_protocol_supplement.json')
    shutil.copy2(log,post/'preparation_cpu_tests.log')
    write(out/'posttrain.json',{'repo':str(ROOT),'out':str(out),'training':str(training),'source_spec_sha256':digest(reference_path),
        'checkpoint_model_sha256':digest(source/'actor/model_world_size_1_rank_0.pt'),'asset_project':str(asset),
        'original_A800_evaluation_root':ref['evaluation_root'],'training_completion_check_sha256':digest(post/'training_completion_check.json')})
    write(post/'preparation_receipt.json',{'passed':True,'fixed_step':800,'source_crc_verified':True,'A800_dev':29,'A800_Final':143,
        'user_authorized_suites':list(SUITES),'Final_runs_regardless_of_Dev_score':True,'Final_is_unseen':False,
        'checkpoint_selection':'User-fixed800 after observing B650 Dev/Final;750 recovery and tail skip50 disclosed, no best-checkpoint substitution',
        'training_started':False,'export_started':False,'evaluation_started':False})
    print('B800_EVAL_PREPARED=true\nOUT='+str(out)+'\nNo export or evaluation started.')


def pipeline(out):
    import fcntl
    check_code();post=out/POST
    fresh=verify_completed_training(Path(read(out/'posttrain.json')['training']))
    if fresh!=read(post/'training_completion_check.json'):raise ValueError('Training evidence changed since evaluation preparation')
    with (post/'pipeline.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if(post/'pipeline_started.json').exists():raise ValueError('Already attempted; preserve it and return evidence')
        write(post/'pipeline_started.json',{'at':time.time()});spec=read(post/'evaluation_spec.json');code=1;old_run=legacy.RUN
        old_env={k:os.environ.get(k)for k in ('OMP_NUM_THREADS','PYTHONPATH')}
        try:
            verify_runtime_snapshot(Path(spec['asset_project']),Path(spec['project']))
            for port in (5700,8000):
                with socket.socket()as sock:sock.bind(('127.0.0.1',port))
            legacy.RUN=out
            # Keep CPU export and native worker imports on the verified runtime.
            os.environ.update(OMP_NUM_THREADS='1',PYTHONPATH=str(Path(spec['project'])/'src'))
            legacy.export_model(Path(spec['asset_project']),800)
            exported=read(post/'export_receipt.json')
            if exported['checkpoint']!=spec['checkpoint']or exported['actor_model_sha256']!=read(out/'posttrain.json')['checkpoint_model_sha256']:raise ValueError('Exported source changed since preparation')
            write(post/'export_exit_status.json',{'exit_code':0})
            original=subprocess.Popen
            def tagged(*args,**kw):
                kw['env']=dict(kw.get('env',os.environ),B500_DEV_RUN=str(out),OMP_NUM_THREADS='1',
                    SHOP_ENV_CONFIG=str(Path(spec['asset_project'])/'environments/ShopSimulator/shop_env/configs/environment.json'),
                    SHOP_SEARCH_INDEX=str(Path(spec['asset_project'])/'environments/ShopSimulator/shop_env/search_engine/products.sqlite3'))
                return original(*args,**kw)
            legacy.subprocess.Popen=tagged
            try:legacy.serve(Path(spec['asset_project']),800)
            finally:legacy.subprocess.Popen=original
            from shopping_grpo.evaluation.summary import summarize_trajectories
            env=dict(os.environ,PYTHONPATH=str(Path(spec['project'])/'src'),OMP_NUM_THREADS='1',B500_DEV_RUN=str(out));results={}
            for suite in SUITES:
                folder=Path(spec['evaluation_root'])/suite;folder.mkdir(parents=True,exist_ok=False);print('START_SUITE='+suite,flush=True)
                with (folder/'driver.log').open('x')as f:subprocess.run(spec['commands'][suite],cwd=spec['project'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
                b=base.validate_suite(spec,suite,summarize_trajectories);print(f'PASSED {suite} {b["strict_successes"]}/{b["expected_tasks"]}',flush=True)
                if suite!='smoke3':
                    a=read(post/f'A800_{suite}_summary.json');aa=set(a['strict_success_task_ids']);bb=set(b['strict_success_task_ids'])
                    results[suite]={'passed':True,'A800':a['strict_successes'],'B800':b['strict_successes'],'net_gain':len(bb)-len(aa),
                        'B_only':sorted(bb-aa),'A_only':sorted(aa-bb),'mean_reward':b['mean_final_reward']}
                    write(post/('dev50_complete.json'if suite=='dev50'else'final200_complete.json'),results[suite])
            write(post/'evaluation_complete.json',{'passed':True,'fixed_step':800,'results':results,'auto_continue':False})
            code=0;print('B800_DEV_AND_FINAL_COMPLETE',flush=True)
        finally:
            legacy.RUN=old_run
            for k,v in old_env.items():
                if v is None:os.environ.pop(k,None)
                else:os.environ[k]=v
            try:write(post/'pipeline_exit_status.json',{'exit_code':code,'at':time.time()})
            finally:base.cleanup(out)


def export_evidence(out):
    import fcntl
    from run_b500_preflight_return import git,configure_author,publish
    with(out/POST/'pipeline.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dest=Path(tempfile.mkdtemp(prefix='b800-eval-return-',dir=ROOT/'outputs'));origin=git(['remote','get-url','origin'],ROOT,capture=True)
        git(['clone','--depth','1','--single-branch','--branch','codex/b800-evaluation',origin,str(dest)],ROOT);configure_author(dest,ROOT)
        stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());branch='codex/b800-eval-results-'+stamp+'-'+dest.name.rsplit('-',1)[-1]
        git(['switch','-c',branch],dest);target=dest/'analysis/b800_evaluation_return'/stamp;target.mkdir(parents=True)
        with zipfile.ZipFile(target/'evidence.zip','w',zipfile.ZIP_DEFLATED)as z:
            for folder in (out/POST,out/'evaluation'):
                for p in folder.rglob('*'):
                    if p.is_file()and p.suffix in ('.json','.jsonl','.log'):z.write(p,p.relative_to(out).as_posix())
            for p in [out/'posttrain.json',out/'supervisor.log']:
                if p.exists():z.write(p,p.name)
            training=Path(read(out/'posttrain.json')['training'])
            for p in training.iterdir():
                if p.is_file()and p.suffix in ('.json','.jsonl','.log'):z.write(p,'training_evidence/'+p.name)
            for folder in ('configs','validation'):
                for p in(training/folder).glob('*'):
                    if p.is_file()and p.suffix in ('.yaml','.jsonl'):z.write(p,'training_evidence/'+folder+'/'+p.name)
            if(training/'train/checkpoint_retention.jsonl').exists():z.write(training/'train/checkpoint_retention.jsonl','training_evidence/checkpoint_retention.jsonl')
            refroot=Path(read(out/'posttrain.json')['original_A800_evaluation_root'])
            for suite in ('dev50','final200_clean'):
                p=refroot/suite/'trajectories.jsonl'
                if p.is_file():z.write(p,'reference_A800/'+suite+'/trajectories.jsonl')
        write(target/'manifest.json',{'archive_sha256':digest(target/'evidence.zip'),'weights_included':False,'fixed_step':800})
        git(['add','--',target.relative_to(dest).as_posix()],dest);git(['commit','-m','Return fixed B800 independent Dev and Final'],dest)
        commit=git(['rev-parse','HEAD'],dest,capture=True);publish(dest,branch,commit);print('RETURN_BRANCH='+branch+'\nRETURN_COMMIT='+commit)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','run','status','return']);p.add_argument('--training',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
    if a.phase=='prepare':prepare(a.training.resolve(),out)
    elif a.phase=='run':pipeline(out)
    elif a.phase=='return':export_evidence(out)
    else:
        for name in ['preparation_receipt.json','export_receipt.json','services_ready.json','dev50_complete.json','final200_complete.json','evaluation_complete.json','pipeline_exit_status.json']:
            p=out/POST/name
            if p.exists():print(name,p.read_text())
