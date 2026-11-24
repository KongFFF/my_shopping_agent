"""Fixed C500 export, original smoke3/Dev50/Final200. Explicitly user authorized."""
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
from c500_run import ROOT,PACKAGE,read,write,digest,verify_runtime_snapshot
legacy=base.legacy
POST='preparation/posttrain_step500_v1'
SUITES=('smoke3','dev50','final200_clean')


def check_code():
    for n,h in read(PACKAGE/'c500_eval_code_manifest.json')['files_sha256'].items():
        if digest(ROOT/n)!=h:raise ValueError('C500 evaluation code drift: '+n)


def derive(reference,asset,runtime,out,source):
    spec={k:copy.deepcopy(reference[k])for k in ('protocols','benchmarks')}
    spec['protocols']={s:spec['protocols'][s]for s in SUITES};spec['benchmarks']={s:spec['benchmarks'][s]for s in SUITES}
    sources={}
    for n,h in reference['source_sha256'].items():
        p=Path(n)
        if p.is_relative_to(asset)and p.relative_to(asset).parts[0]in ('src','scripts','configs'):p=runtime/p.relative_to(asset)
        if digest(p)!=h:raise ValueError('Original A500 source differs: '+str(p))
        sources[str(p)]=h
    tag='shopping-agent-c500-gamma4-v1';commands={}
    for s in SUITES:
        a=list(reference['commands'][s]);a[0]=sys.executable;a[1]=str(runtime/'scripts/evaluate_shop_benchmark.py')
        for k,v in [('--model',tag),('--output',out/'evaluation'/s/'trajectories.jsonl'),('--summary',out/'evaluation'/s/'summary.json')]:a=legacy.replace_option(a,k,v)
        if a[a.index('--benchmark')+1]!=spec['benchmarks'][s]['path']:raise ValueError('Command/benchmark binding differs')
        for k,v in [('--base-url','http://127.0.0.1:5700'),('--llm-base-url','http://127.0.0.1:8000/v1')]:
            if k in a and a[a.index(k)+1].rstrip('/')!=v:raise ValueError('Service URL differs: '+k)
        commands[s]=a
    spec.update(project=str(runtime),asset_project=str(asset),checkpoint=str(source),actor=str(source/'actor'),
        raw_export=str(out/'export/hf_unmerged'),merged_model=str(out/'export/merged'),evaluation_root=str(out/'evaluation'),
        model_tag=tag,source_sha256=sources,commands=commands,evaluated_step=500)
    return spec


def collect_supplement(training,destination):
    """Read original evidence only; do not rerun CPU acceptance or training."""
    destination.mkdir(parents=True,exist_ok=False)
    run=read(training/'run.json');done=read(training/'training_complete.json')
    if run.get('experiment')!='C500' or run.get('gamma')!=4 or done.get('completed_updates')!=500 or done.get('new_updates')!=500:
        raise ValueError('Expected fixed C500 gamma4 with500 fresh updates')
    if (training/'stopped.json').exists():raise ValueError('Current C training stopped unexpectedly')
    if run['protocol_sha256']!=done['protocol_sha256']or run['protocol_sha256']!=digest(PACKAGE/'protocol.json'):
        raise ValueError('C protocol identity differs')
    if run['implementation_manifest_sha256']!=digest(PACKAGE/'implementation_manifest.json'):
        raise ValueError('C implementation identity differs')
    cpu=read(Path(run['cpu_receipt']))
    if (not cpu.get('passed')or cpu.get('version')!='c500-cpu-acceptance-v1' or cpu.get('tests_run',0)<36
        or cpu.get('gpu_initialized')is not False or cpu.get('model_weights_loaded')is not False or cpu.get('training_started')is not False
        or cpu.get('protocol_sha256')!=run['protocol_sha256']or cpu.get('implementation_manifest_sha256')!=run['implementation_manifest_sha256']or cpu.get('errors')):
        raise ValueError('Original native CPU receipt missing, failed or unbound')
    import shutil
    shutil.copyfile(run['cpu_receipt'],destination/'native_cpu_acceptance.json')
    for n in ['protocol.json','implementation_manifest.json','launch_manifest.json']:
        source=Path(run['repo'])/'experiments/group_weight_c500_v1'/n
        if digest(source)!=digest(PACKAGE/n):raise ValueError('Original C source inventory differs: '+n)
    for n,h in read(PACKAGE/'launch_manifest.json')['files_sha256'].items():
        if digest(Path(run['repo'])/n)!=h:raise ValueError('Original launch source drift: '+n)
    for n,h in read(PACKAGE/'implementation_manifest.json')['files_sha256'].items():
        if digest(Path(run['repo'])/n)!=h:raise ValueError('Original method source drift: '+n)
    previous=None
    recovery=training/'zero_update_recovery.json'
    if recovery.exists():
        info=read(recovery);previous=Path(info['previous_run']).resolve()
        if previous.parent!=training.parent or previous==training:raise ValueError('Unexpected prior run path')
        old=read(previous/'run.json');stopped=read(previous/'stopped.json')
        if old.get('training_started')is not False or 'disk'not in stopped.get('error','').lower():raise ValueError('Prior failure not before training disk preflight')
        if old['protocol_sha256']!=run['protocol_sha256']or old['implementation_manifest_sha256']!=run['implementation_manifest_sha256']:
            raise ValueError('Prior run protocol differs')
        if list((previous/'train').glob('global_step_*'))or list(previous.glob('accepted_step_*.json')):raise ValueError('Prior run has checkpoint/update evidence')
        for name in ['weights.jsonl','training_diagnostics.jsonl']:
            path=previous/name
            if path.exists():
                rs=[json.loads(line)for line in path.read_text().splitlines()if line.strip()]
                if any(e.get('event')=='optimizer_step'or e.get('event','').startswith('actor_update_')for e in rs):raise ValueError('Prior run may have updated')
        if (previous/'generation_budget.json').exists()and any(read(previous/'generation_budget.json').values()):raise ValueError('Prior preflight run has sampling cost')
        shutil.copyfile(recovery,destination/'zero_update_recovery.json')
        (destination/'prior').mkdir()
        for name in ['run.json','stopped.json','supervisor.log','generation_budget.json','weights.jsonl','training_diagnostics.jsonl']:
            path=previous/name
            if path.exists():shutil.copyfile(path,destination/'prior'/name)
    write(destination/'supplement_acceptance.json',{'passed':True,'native_CPU_original_received':True,'prior_disk_zero_update_verified':previous is not None,
        'prior_run':str(previous)if previous else None,'training_started':False,'CPU_tests_rerun':False})
    return run


def verify_training(training,out):
    import fcntl
    from audit_c500_training_return import audit
    with (training/'supervisor.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        evidence=out/'training_evidence';run=collect_supplement(training,evidence)
        archive=evidence/'training_snapshot.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED)as z:
            for p in training.iterdir():
                if p.is_file()and p.suffix in ('.json','.jsonl','.log')and p.name!='environment.log':z.write(p,p.name)
            for p in (training/'configs').glob('*.yaml'):z.write(p,'configs/'+p.name)
            for p in (training/'validation').glob('*.jsonl'):z.write(p,'validation/'+p.name)
            z.write(training/'train/checkpoint_retention.jsonl','checkpoint_retention.jsonl')
            z.write(evidence/'native_cpu_acceptance.json','native_cpu_acceptance.json')
        write(evidence/'manifest.json',{'archive_sha256':digest(archive),'protocol_sha256':run['protocol_sha256']})
        result=audit(archive)
        write(evidence/'training_audit.json',result)
        if not result['training_core_passed']:raise ValueError('C500 training audit failed: '+str(result['errors']))
        # The old-return audit labels the historical prior-scene gap; this
        # prepare has separately verified and copied that original scene.
        write(evidence/'combined_acceptance.json',{'passed':True,'training_core_passed':True,'supplement_verified':True,
            'source_run':str(training),'independent_Dev_started':False,'Final_started':False})
        return run


def prepare(training,out):
    check_code()
    if out.exists():raise ValueError('Output exists; preserve it and use status/return')
    (ROOT/'outputs').mkdir(exist_ok=True);log=ROOT/'outputs/c500_eval_cpu_tests.log';print('EVAL_CPU_LOG='+str(log),flush=True)
    with log.open('w')as f:
        for pattern in ['test_c500_evaluation.py','test_b500_dev50.py']:
            subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tools'),'-p',pattern],check=True,stdout=f,stderr=subprocess.STDOUT)
    run=verify_training(training,out)
    asset=Path(run['project']);runtime=Path(run['runtime']);verify_runtime_snapshot(asset,runtime)
    sys.path.insert(0,str(PACKAGE/'overlay'))
    from c500_run_control import checkpoint,checked_folder
    source=Path(checkpoint(training/'train',500));base.require_export_space(source/'actor')
    print('CHECK500_CRC: reading checkpoint, no tensor loading',flush=True);checked_folder(training/'train',500,crc=True)
    meta=read(source/'actor/lora_train_meta.json');fsdp=read(source/'actor/fsdp_config.json')
    if int(meta['r'])!=16 or int(meta['lora_alpha'])!=32 or int(fsdp['world_size'])!=1:raise ValueError('C500 LoRA/FSDP metadata differs')
    reference_path=asset/legacy.RUN/POST/'evaluation_spec.json';ref=read(reference_path)
    if ref.get('evaluated_step',500)!=500 or not ref['checkpoint'].endswith('/global_step_500'):raise ValueError('Expected original A500 evaluation specification')
    spec=derive(ref,asset,runtime,out,source);frozen=read(PACKAGE/'protocol.json')['evaluation']['protocol'];refs={}
    for suite,n,score in [('smoke3',3,None),('dev50',50,27),('final200_clean',200,137)]:
        b=spec['benchmarks'][suite];rs=[json.loads(l)for l in Path(b['path']).read_text().splitlines()if l.strip()];ids=[int(r['task_id'])for r in rs]
        if digest(b['path'])!=b['sha256']or ids!=b['task_ids']or len(ids)!=len(set(ids))or len(ids)!=n:raise ValueError('Original benchmark identity differs: '+suite)
        if suite=='final200_clean'and b['sha256']!=legacy.FINAL_HASH:raise ValueError('Final Clean fingerprint differs')
        if score is not None:
            a=read(Path(ref['evaluation_root'])/suite/'summary.json')
            if a['expected_tasks']!=n or a['strict_successes']!=score or len(set(a['strict_success_task_ids']))!=score or not set(a['strict_success_task_ids']).issubset(ids):raise ValueError('Historical A500 result differs: '+suite)
            if a['protocol'].get('model')!=ref['model_tag']or a['protocol'].get('benchmark')!=b['path']:raise ValueError('Historical A500 summary identity differs: '+suite)
            for k,v in frozen.items():
                if k!='benchmark'and a['protocol'].get(k)!=v:raise ValueError('Original A500 protocol differs: '+k)
            refs[suite]=a
    post=out/POST;post.mkdir(parents=True);write(post/'evaluation_spec.json',spec)
    for suite,a in refs.items():write(post/f'A500_{suite}_summary.json',a)
    write(out/'posttrain.json',{'repo':str(ROOT),'out':str(out),'training':str(training),'source_spec_sha256':digest(reference_path),
        'checkpoint_model_sha256':digest(source/'actor/model_world_size_1_rank_0.pt'),'asset_project':str(asset),
        'original_A500_evaluation_root':ref['evaluation_root']})
    write(post/'preparation_receipt.json',{'passed':True,'fixed_step':500,'source_crc_verified':True,'A500_dev':27,'A500_Final':137,'B500_dev':28,'B500_Final':141,'gamma':4.0,
        'user_authorized_suites':list(SUITES),'Final_runs_regardless_of_Dev_score':True,'Final_is_unseen':False,
        'checkpoint_selection':'User-fixed C500 gamma4; original independent protocols; no best-checkpoint substitution',
        'training_started':False,'export_started':False,'evaluation_started':False,'supplement_verified':True})
    print('C500_EVAL_PREPARED=true\nOUT='+str(out)+'\nNo export or evaluation started.')


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
            legacy.export_model(Path(spec['asset_project']),500)
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
            try:legacy.serve(Path(spec['asset_project']),500)
            finally:legacy.subprocess.Popen=original
            from shopping_grpo.evaluation.summary import summarize_trajectories
            env=dict(os.environ,PYTHONPATH=str(Path(spec['project'])/'src'),OMP_NUM_THREADS='1',B500_DEV_RUN=str(out));results={}
            for suite in SUITES:
                folder=Path(spec['evaluation_root'])/suite;folder.mkdir(parents=True,exist_ok=False);print('START_SUITE='+suite,flush=True)
                with (folder/'driver.log').open('x')as f:subprocess.run(spec['commands'][suite],cwd=spec['project'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
                b=base.validate_suite(spec,suite,summarize_trajectories);print(f'PASSED {suite} {b["strict_successes"]}/{b["expected_tasks"]}',flush=True)
                if suite!='smoke3':
                    a=read(post/f'A500_{suite}_summary.json');aa=set(a['strict_success_task_ids']);bb=set(b['strict_success_task_ids'])
                    results[suite]={'passed':True,'A500':a['strict_successes'],'C500':b['strict_successes'],'net_gain':len(bb)-len(aa),
                        'B_only':sorted(bb-aa),'A_only':sorted(aa-bb),'mean_reward':b['mean_final_reward']}
                    write(post/('dev50_complete.json'if suite=='dev50'else'final200_complete.json'),results[suite])
            write(post/'evaluation_complete.json',{'passed':True,'fixed_step':500,'results':results,'auto_continue':False})
            code=0;print('C500_DEV_AND_FINAL_COMPLETE',flush=True)
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
        dest=Path(tempfile.mkdtemp(prefix='c500-eval-return-',dir=ROOT/'outputs'));origin=git(['remote','get-url','origin'],ROOT,capture=True)
        git(['clone','--depth','1','--single-branch','--filter=blob:none','--sparse','--branch','codex/c500-evaluation',origin,str(dest)],ROOT);configure_author(dest,ROOT)
        stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());branch='codex/c500-eval-results-'+stamp+'-'+dest.name.rsplit('-',1)[-1]
        git(['sparse-checkout','set','analysis/c500_evaluation_return'],dest)
        git(['switch','-c',branch],dest);target=dest/'analysis/c500_evaluation_return'/stamp;target.mkdir(parents=True)
        with zipfile.ZipFile(target/'evidence.zip','w',zipfile.ZIP_DEFLATED)as z:
            for folder in (out/POST,out/'evaluation'):
                for p in folder.rglob('*'):
                    if p.is_file()and p.suffix in ('.json','.jsonl','.log'):z.write(p,p.relative_to(out).as_posix())
            for p in [out/'posttrain.json',out/'supervisor.log']:
                if p.exists():z.write(p,p.name)
            for p in (out/'training_evidence').rglob('*'):
                if p.is_file():z.write(p,'training_evidence/'+p.relative_to(out/'training_evidence').as_posix())
            refroot=Path(read(out/'posttrain.json')['original_A500_evaluation_root'])
            for suite in ('dev50','final200_clean'):
                p=refroot/suite/'trajectories.jsonl'
                if p.is_file():z.write(p,'reference_A500/'+suite+'/trajectories.jsonl')
        write(target/'manifest.json',{'archive_sha256':digest(target/'evidence.zip'),'weights_included':False,'fixed_step':500,'experiment':'C500','gamma':4.0})
        git(['add','--',target.relative_to(dest).as_posix()],dest);git(['commit','-m','Return fixed C500 independent Dev and Final'],dest)
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
