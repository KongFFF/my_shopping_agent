"""Fixed C800 export, original smoke3/Dev50/Final200. Explicitly user authorized."""
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
POST='preparation/posttrain_step800_v1'
SUITES=('smoke3','dev50','final200_clean')


def check_code():
    for n,h in read(ROOT/'experiments/group_weight_c800_v1/evaluation_manifest.json')['files_sha256'].items():
        if digest(ROOT/n)!=h:raise ValueError('C800 evaluation code drift: '+n)


def derive(reference,asset,runtime,out,source):
    spec={k:copy.deepcopy(reference[k])for k in ('protocols','benchmarks')}
    spec['protocols']={s:spec['protocols'][s]for s in SUITES};spec['benchmarks']={s:spec['benchmarks'][s]for s in SUITES}
    sources={}
    for n,h in reference['source_sha256'].items():
        p=Path(n)
        if p.is_relative_to(asset)and p.relative_to(asset).parts[0]in ('src','scripts','configs'):p=runtime/p.relative_to(asset)
        if digest(p)!=h:raise ValueError('Original A800 source differs: '+str(p))
        sources[str(p)]=h
    tag='shopping-agent-c800-gamma4-v1';commands={}
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


def audit_training_packet(archive,directory):
    """Reconstruct the preserved parent log view; no large historical Git blob needed."""
    import hashlib
    from audit_c800_training_return import audit
    ref=read(ROOT/'experiments/group_weight_c800_v1/evaluation_training_reference.json')
    directory.mkdir(parents=True,exist_ok=True)
    parent_archive=directory/'parent650_reference.zip'
    with zipfile.ZipFile(archive)as z:
        events=[json.loads(line)for line in z.read('training_diagnostics.jsonl').splitlines()if line.strip()]
        prefix=events[:ref['parent_diagnostic_events']]
        canonical=json.dumps(prefix,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
        if hashlib.sha256(canonical).hexdigest()!=ref['parent_canonical_diagnostic_sha256']:
            raise ValueError('Preserved C650 diagnostic prefix differs from archived C650')
        weights=z.read('parent_C650__weights.jsonl')
        if hashlib.sha256(weights).hexdigest()!=ref['parent_weights_sha256']:
            raise ValueError('Preserved C650 weight log differs')
        budget=z.read('parent_C650__generation_budget.json')
        if json.loads(budget)!=ref['parent_budget']:raise ValueError('C650 initial budget differs')
        with zipfile.ZipFile(parent_archive,'w',zipfile.ZIP_DEFLATED)as parent:
            parent.writestr('training_diagnostics.jsonl',''.join(json.dumps(e,ensure_ascii=False)+'\n'for e in prefix))
            parent.writestr('weights.jsonl',weights);parent.writestr('generation_budget.json',budget)
    del events,prefix,canonical,weights
    result=audit(archive,previous_archive=parent_archive)
    if not result['audit_passed']:raise ValueError('C800 training audit failed: '+str(result['errors']))
    write(directory/'training_audit.json',result)
    return result


def verify_training(training,out):
    import fcntl
    with (training/'supervisor.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run=read(training/'run.json');done=read(training/'training_complete.json')
        if run.get('experiment')!='C800'or run.get('gamma')!=4 or done.get('completed_updates')!=800 or done.get('new_updates')!=150:
            raise ValueError('Expected completed C800 gamma4')
        if run['protocol_sha256']!=digest(PACKAGE/'protocol.json')or run['implementation_manifest_sha256']!=digest(PACKAGE/'implementation_manifest.json'):
            raise ValueError('C weighting identity differs')
        if run['continuation_protocol_sha256']!=digest(ROOT/'experiments/group_weight_c800_v1/protocol.json'):
            raise ValueError('Continuation supplement differs')
        evidence=out/'training_evidence';evidence.mkdir(parents=True)
        archive=evidence/'training_snapshot.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED)as z:
            for p in training.iterdir():
                if p.is_file()and p.suffix in ('.json','.jsonl','.log')and p.name!='environment.log':z.write(p,p.name)
            for p in (training/'configs').glob('*.yaml'):z.write(p,'configs/'+p.name)
            for p in (training/'validation').glob('*.jsonl'):z.write(p,'validation/'+p.name)
            z.write(training/'train/checkpoint_retention.jsonl','checkpoint_retention.jsonl')
        write(evidence/'manifest.json',{'archive_sha256':digest(archive),'protocol_sha256':run['protocol_sha256']})
        audit_training_packet(archive,evidence)
        write(evidence/'training_identity.json',{'passed':True,'fixed_step':800,'gamma':4,'training':str(training),'evaluation_started':False})
        return run


def prepare(training,out):
    check_code()
    if out.exists():raise ValueError('Output exists; preserve it and use status/return')
    (ROOT/'outputs').mkdir(exist_ok=True);log=ROOT/'outputs/c800_eval_cpu_tests.log';print('EVAL_CPU_LOG='+str(log),flush=True)
    with log.open('w')as f:
        for pattern in ['test_c800_evaluation.py','test_c800_protocol_guards.py']:
            try:
                subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tools'),'-p',pattern],check=True,stdout=f,stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                f.flush();print('CPU_CHECK_FAILED: '+pattern+' LOG='+str(log),flush=True)
                print(log.read_text(errors='replace')[-12000:],flush=True)
                raise
    run=verify_training(training,out)
    asset=Path(run['project']);runtime=Path(run['runtime']);verify_runtime_snapshot(asset,runtime)
    sys.path.insert(0,str(PACKAGE/'overlay'))
    from c500_run_control import checkpoint,checked_folder
    source=Path(checkpoint(training/'train',800));base.require_export_space(source/'actor')
    print('CHECK800_CRC: reading checkpoint, no tensor loading',flush=True);checked_folder(training/'train',800,crc=True)
    meta=read(source/'actor/lora_train_meta.json');fsdp=read(source/'actor/fsdp_config.json')
    if int(meta['r'])!=16 or int(meta['lora_alpha'])!=32 or int(fsdp['world_size'])!=1:raise ValueError('C800 LoRA/FSDP metadata differs')
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
    import shutil
    shutil.copy2(log,post/'preparation_cpu_tests.log')
    write(out/'posttrain.json',{'repo':str(ROOT),'out':str(out),'training':str(training),'source_spec_sha256':digest(reference_path),
        'checkpoint_model_sha256':digest(source/'actor/model_world_size_1_rank_0.pt'),'asset_project':str(asset),
        'original_A800_evaluation_root':ref['evaluation_root']})
    write(post/'preparation_receipt.json',{'passed':True,'fixed_step':800,'source_crc_verified':True,'A800_dev':29,'A800_Final':143,'B800_dev':28,'B800_Final':142,'gamma':4.0,
        'user_authorized_suites':list(SUITES),'Final_runs_regardless_of_Dev_score':True,'Final_is_unseen':False,
        'checkpoint_selection':'User-fixed C800 after C650 and800 training; no best-checkpoint substitution',
        'training_started':False,'export_started':False,'evaluation_started':False})
    print('C800_EVAL_PREPARED=true\nOUT='+str(out)+'\nNo export or evaluation started.')


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
                    results[suite]={'passed':True,'A800':a['strict_successes'],'C800':b['strict_successes'],'net_gain':len(bb)-len(aa),
                        'C_only':sorted(bb-aa),'A_only':sorted(aa-bb),'mean_reward':b['mean_final_reward']}
                    write(post/('dev50_complete.json'if suite=='dev50'else'final200_complete.json'),results[suite])
            write(post/'evaluation_complete.json',{'passed':True,'fixed_step':800,'results':results,'auto_continue':False})
            code=0;print('C800_DEV_AND_FINAL_COMPLETE',flush=True)
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
        dest=Path(tempfile.mkdtemp(prefix='c800-eval-return-',dir=ROOT/'outputs'));origin=git(['remote','get-url','origin'],ROOT,capture=True)
        git(['clone','--depth','1','--single-branch','--filter=blob:none','--sparse','--branch','codex/c800-evaluation',origin,str(dest)],ROOT);configure_author(dest,ROOT)
        stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());branch='codex/c800-eval-results-'+stamp+'-'+dest.name.rsplit('-',1)[-1]
        git(['sparse-checkout','set','analysis/c800_evaluation_return'],dest)
        git(['switch','-c',branch],dest);target=dest/'analysis/c800_evaluation_return'/stamp;target.mkdir(parents=True)
        with zipfile.ZipFile(target/'evidence.zip','w',zipfile.ZIP_DEFLATED)as z:
            for folder in (out/POST,out/'evaluation'):
                for p in folder.rglob('*'):
                    if p.is_file()and p.suffix in ('.json','.jsonl','.log'):z.write(p,p.relative_to(out).as_posix())
            for p in [out/'posttrain.json',out/'supervisor.log']:
                if p.exists():z.write(p,p.name)
            for p in (out/'training_evidence').rglob('*'):
                if p.is_file():z.write(p,'training_evidence/'+p.relative_to(out/'training_evidence').as_posix())
            refroot=Path(read(out/'posttrain.json')['original_A800_evaluation_root'])
            for suite in ('dev50','final200_clean'):
                p=refroot/suite/'trajectories.jsonl'
                if p.is_file():z.write(p,'reference_A800/'+suite+'/trajectories.jsonl')
        write(target/'manifest.json',{'archive_sha256':digest(target/'evidence.zip'),'weights_included':False,'fixed_step':800,'experiment':'C800','gamma':4.0})
        git(['add','--',target.relative_to(dest).as_posix()],dest);git(['commit','-m','Return fixed C800 independent Dev and Final'],dest)
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
