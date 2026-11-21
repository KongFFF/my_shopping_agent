"""Fixed B650 export, original smoke3/Dev50/Final200. Explicitly user authorized."""
import argparse
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import zipfile

import b500_dev50 as base
from b500_run import ROOT,PACKAGE,read,write,digest,verify_runtime_snapshot
legacy=base.legacy
POST='preparation/posttrain_step650_v1'
SUITES=('smoke3','dev50','final200_clean')


def check_code():
    base.check_code()
    for n,h in read(PACKAGE/'b650_eval_code_manifest.json')['files_sha256'].items():
        if digest(ROOT/n)!=h:raise ValueError('B650 evaluation code drift: '+n)


def derive(reference,asset,runtime,out,source):
    spec={k:copy.deepcopy(reference[k])for k in ('protocols','benchmarks')}
    spec['protocols']={s:spec['protocols'][s]for s in SUITES};spec['benchmarks']={s:spec['benchmarks'][s]for s in SUITES}
    sources={}
    for n,h in reference['source_sha256'].items():
        p=Path(n)
        if p.is_relative_to(asset)and p.relative_to(asset).parts[0]in ('src','scripts','configs'):p=runtime/p.relative_to(asset)
        if digest(p)!=h:raise ValueError('Original A650 source differs: '+str(p))
        sources[str(p)]=h
    tag='shopping-agent-b650-group-weight-v1';commands={}
    for s in SUITES:
        a=list(reference['commands'][s]);a[0]=sys.executable;a[1]=str(runtime/'scripts/evaluate_shop_benchmark.py')
        for k,v in [('--model',tag),('--output',out/'evaluation'/s/'trajectories.jsonl'),('--summary',out/'evaluation'/s/'summary.json')]:a=legacy.replace_option(a,k,v)
        if a[a.index('--benchmark')+1]!=spec['benchmarks'][s]['path']:raise ValueError('Command/benchmark binding differs')
        for k,v in [('--base-url','http://127.0.0.1:5700'),('--llm-base-url','http://127.0.0.1:8000/v1')]:
            if k in a and a[a.index(k)+1].rstrip('/')!=v:raise ValueError('Service URL differs: '+k)
        commands[s]=a
    spec.update(project=str(runtime),asset_project=str(asset),checkpoint=str(source),actor=str(source/'actor'),
        raw_export=str(out/'export/hf_unmerged'),merged_model=str(out/'export/merged'),evaluation_root=str(out/'evaluation'),
        model_tag=tag,source_sha256=sources,commands=commands,evaluated_step=650)
    return spec


def prepare(training,out):
    check_code()
    if out.exists():raise ValueError('Output exists; preserve it and use status/return')
    (ROOT/'outputs').mkdir(exist_ok=True);log=ROOT/'outputs/b650_eval_cpu_tests.log';print('EVAL_CPU_LOG='+str(log),flush=True)
    with log.open('w')as f:subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tools'),'-p','test_b500_b650_evaluation.py'],check=True,stdout=f,stderr=subprocess.STDOUT)
    run=read(training/'run.json');done=read(training/'training_complete.json')
    if done['completed_updates']!=650 or done['new_updates']!=150 or(training/'stopped.json').exists():raise ValueError('B650 training is not complete')
    if run['protocol_sha256']!=digest(PACKAGE/'protocol.json'):raise ValueError('Parent training recipe changed')
    if run['target_step']!=650 or run['resume_step']!=500 or run['implementation_manifest_sha256']!=digest(PACKAGE/'implementation_manifest.json'):raise ValueError('Expected fixed B500->650 method identity')
    if digest(training/'configs/phase650.yaml')!=run['config_hashes']['650']:raise ValueError('Parent650 configuration changed')
    asset=Path(run['project']);runtime=Path(run['runtime']);verify_runtime_snapshot(asset,runtime)
    sys.path.insert(0,str(PACKAGE/'overlay'))
    from b500_run_control import checkpoint,checked_folder
    source=Path(checkpoint(training/'train',650));base.require_export_space(source/'actor')
    print('CHECK650_CRC: reading checkpoint, no tensor loading',flush=True);checked_folder(training/'train',650,crc=True)
    meta=read(source/'actor/lora_train_meta.json');fsdp=read(source/'actor/fsdp_config.json')
    if int(meta['r'])!=16 or int(meta['lora_alpha'])!=32 or int(fsdp['world_size'])!=1:raise ValueError('B650 LoRA/FSDP metadata differs')
    reference_path=asset/legacy.RUN/POST/'evaluation_spec.json';ref=read(reference_path)
    if ref.get('evaluated_step',650)!=650 or not ref['checkpoint'].endswith('/global_step_650')or ref['model_tag']!='shopping-agent-grpo-step650':raise ValueError('Expected original A650 evaluation specification')
    spec=derive(ref,asset,runtime,out,source);frozen=read(PACKAGE/'protocol.json')['evaluation']['protocol'];refs={}
    for suite,n,score in [('smoke3',3,None),('dev50',50,26),('final200_clean',200,141)]:
        b=spec['benchmarks'][suite];rs=[json.loads(l)for l in Path(b['path']).read_text().splitlines()if l.strip()];ids=[int(r['task_id'])for r in rs]
        if digest(b['path'])!=b['sha256']or ids!=b['task_ids']or len(ids)!=len(set(ids))or len(ids)!=n:raise ValueError('Original benchmark identity differs: '+suite)
        if suite=='final200_clean'and b['sha256']!=legacy.FINAL_HASH:raise ValueError('Final Clean fingerprint differs')
        if score is not None:
            a=read(Path(ref['evaluation_root'])/suite/'summary.json')
            if a['expected_tasks']!=n or a['strict_successes']!=score or len(set(a['strict_success_task_ids']))!=score or not set(a['strict_success_task_ids']).issubset(ids):raise ValueError('Historical A650 result differs: '+suite)
            if a['protocol'].get('model')!=ref['model_tag']or a['protocol'].get('benchmark')!=b['path']:raise ValueError('Historical A650 summary identity differs: '+suite)
            for k,v in frozen.items():
                if k!='benchmark'and a['protocol'].get(k)!=v:raise ValueError('Original A650 protocol differs: '+k)
            refs[suite]=a
    post=out/POST;post.mkdir(parents=True);write(post/'evaluation_spec.json',spec)
    for suite,a in refs.items():write(post/f'A650_{suite}_summary.json',a)
    write(out/'posttrain.json',{'repo':str(ROOT),'out':str(out),'training':str(training),'source_spec_sha256':digest(reference_path),
        'checkpoint_model_sha256':digest(source/'actor/model_world_size_1_rank_0.pt'),'asset_project':str(asset),
        'original_A650_evaluation_root':ref['evaluation_root']})
    write(post/'preparation_receipt.json',{'passed':True,'fixed_step':650,'source_crc_verified':True,'A650_dev':26,'A650_Final':141,
        'user_authorized_suites':list(SUITES),'Final_runs_regardless_of_Dev_score':True,'Final_is_unseen':False,
        'checkpoint_selection':'User-fixed650 after observing B500 Dev/Final; no best-checkpoint substitution',
        'training_started':False,'export_started':False,'evaluation_started':False})
    print('B650_EVAL_PREPARED=true\nOUT='+str(out)+'\nNo export or evaluation started.')


def pipeline(out):
    import fcntl
    check_code();post=out/POST
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
            legacy.export_model(Path(spec['asset_project']),650)
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
            try:legacy.serve(Path(spec['asset_project']),650)
            finally:legacy.subprocess.Popen=original
            from shopping_grpo.evaluation.summary import summarize_trajectories
            env=dict(os.environ,PYTHONPATH=str(Path(spec['project'])/'src'),OMP_NUM_THREADS='1',B500_DEV_RUN=str(out));results={}
            for suite in SUITES:
                folder=Path(spec['evaluation_root'])/suite;folder.mkdir(parents=True,exist_ok=False);print('START_SUITE='+suite,flush=True)
                with (folder/'driver.log').open('x')as f:subprocess.run(spec['commands'][suite],cwd=spec['project'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
                b=base.validate_suite(spec,suite,summarize_trajectories);print(f'PASSED {suite} {b["strict_successes"]}/{b["expected_tasks"]}',flush=True)
                if suite!='smoke3':
                    a=read(post/f'A650_{suite}_summary.json');aa=set(a['strict_success_task_ids']);bb=set(b['strict_success_task_ids'])
                    results[suite]={'passed':True,'A650':a['strict_successes'],'B650':b['strict_successes'],'net_gain':len(bb)-len(aa),
                        'B_only':sorted(bb-aa),'A_only':sorted(aa-bb),'mean_reward':b['mean_final_reward']}
                    write(post/('dev50_complete.json'if suite=='dev50'else'final200_complete.json'),results[suite])
            write(post/'evaluation_complete.json',{'passed':True,'fixed_step':650,'results':results,'auto_continue':False})
            code=0;print('B650_DEV_AND_FINAL_COMPLETE',flush=True)
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
        dest=Path(tempfile.mkdtemp(prefix='b650-eval-return-',dir=ROOT/'outputs'));origin=git(['remote','get-url','origin'],ROOT,capture=True)
        git(['clone','--depth','1','--single-branch','--branch','codex/b650-evaluation',origin,str(dest)],ROOT);configure_author(dest,ROOT)
        stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());branch='codex/b650-eval-results-'+stamp+'-'+dest.name.rsplit('-',1)[-1]
        git(['switch','-c',branch],dest);target=dest/'analysis/b650_evaluation_return'/stamp;target.mkdir(parents=True)
        with zipfile.ZipFile(target/'evidence.zip','w',zipfile.ZIP_DEFLATED)as z:
            for folder in (out/POST,out/'evaluation'):
                for p in folder.rglob('*'):
                    if p.is_file()and p.suffix in ('.json','.jsonl','.log'):z.write(p,p.relative_to(out).as_posix())
            for p in [out/'posttrain.json',out/'supervisor.log']:
                if p.exists():z.write(p,p.name)
            refroot=Path(read(out/'posttrain.json')['original_A650_evaluation_root'])
            for suite in ('dev50','final200_clean'):
                p=refroot/suite/'trajectories.jsonl'
                if p.is_file():z.write(p,'reference_A650/'+suite+'/trajectories.jsonl')
        write(target/'manifest.json',{'archive_sha256':digest(target/'evidence.zip'),'weights_included':False,'fixed_step':650})
        git(['add','--',target.relative_to(dest).as_posix()],dest);git(['commit','-m','Return fixed B650 independent Dev and Final'],dest)
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
