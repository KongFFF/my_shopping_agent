"""Fixed B500 export + original smoke3/Dev50. Never train, run Final or shut down."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import zipfile

from b500_run import ROOT,PACKAGE,read,write,digest,verify_runtime_snapshot
import step650_evaluation as legacy

SUITES=('smoke3','dev50')
POST_REL='preparation/posttrain_step500_v1'


def require_export_space(actor):
    required=2*(actor/'model_world_size_1_rank_0.pt').stat().st_size+5*2**30
    free=shutil.disk_usage(actor).free
    print(f'EXPORT_DISK_FREE_GiB={free/2**30:.2f} REQUIRED_GiB={required/2**30:.2f}',flush=True)
    if free<required:raise ValueError('Insufficient disk for FSDP export and merged model')


def check_code():
    m=read(PACKAGE/'dev50_code_manifest.json')
    for n,h in m['files_sha256'].items():
        if digest(ROOT/n)!=h:raise ValueError('Posttrain code drift: '+n)
    for n,h in m['reference_text_sha256'].items():
        if hashlib.sha256((ROOT/n).read_text(encoding='utf-8').encode()).hexdigest()!=h:raise ValueError('A reference drift: '+n)


def derive(reference,asset,runtime,out,checkpoint):
    """Preserve every CLI argument except interpreter/script/model/output identity."""
    spec={k:reference[k]for k in ('protocols','benchmarks')}
    spec={k:{s:spec[k][s]for s in SUITES}for k in spec}
    sources={}
    for n,h in reference['source_sha256'].items():
        p=Path(n)
        if p.is_relative_to(asset)and p.relative_to(asset).parts[0]in ('src','scripts','configs'):
            p=runtime/p.relative_to(asset)
        if digest(p)!=h:raise ValueError('Frozen A evaluation/export source differs: '+str(p))
        sources[str(p)]=h
    commands={};tag='shopping-agent-b500-group-weight-v1'
    for suite in SUITES:
        a=list(reference['commands'][suite]);a[0]=sys.executable;a[1]=str(runtime/'scripts/evaluate_shop_benchmark.py')
        for key,value in [('--model',tag),('--output',out/'evaluation'/suite/'trajectories.jsonl'),('--summary',out/'evaluation'/suite/'summary.json')]:
            a=legacy.replace_option(a,key,value)
        commands[suite]=a
        for option,expected in [('--base-url','http://127.0.0.1:5700'),('--llm-base-url','http://127.0.0.1:8000/v1')]:
            if option in a and a[a.index(option)+1].rstrip('/')!=expected.rstrip('/'):
                raise ValueError('Frozen service URL differs: '+option)
    spec.update(project=str(runtime),asset_project=str(asset),checkpoint=str(checkpoint),actor=str(checkpoint/'actor'),
        raw_export=str(out/'export/hf_unmerged'),merged_model=str(out/'export/merged'),model_tag=tag,
        evaluation_root=str(out/'evaluation'),commands=commands,source_sha256=sources)
    return spec


def prepare(training,out):
    check_code()
    if out.exists():raise ValueError('Posttrain output exists; inspect, do not prepare/restart over it')
    (ROOT/'outputs').mkdir(exist_ok=True)
    cpu_log=ROOT/'outputs/dev50_prepare_cpu_tests.log';print('DEV50_CPU_LOG='+str(cpu_log),flush=True)
    with cpu_log.open('w')as log:
        subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tools'),'-p','test_b500_dev50.py'],check=True,stdout=log,stderr=subprocess.STDOUT)
    print('DEV50_CPU_TESTS_PASSED=true',flush=True)
    run=read(training/'run.json');done=read(training/'training_complete.json')
    if done['completed_updates']!=500 or done['independent_Dev_done']or done['Final_done']:raise ValueError('Unexpected completed training/evaluation state')
    if run['protocol_sha256']!=digest(PACKAGE/'protocol.json'):raise ValueError('Training protocol differs')
    asset=Path(run['project']);runtime=Path(run['runtime']);verify_runtime_snapshot(asset,runtime)
    sys.path.insert(0,str(PACKAGE/'overlay'))
    from b500_run_control import checked_folder,checkpoint
    source=Path(checkpoint(training/'train',500));require_export_space(source/'actor')
    print('CHECK500_CRC: reading checkpoint; no tensor loading',flush=True)
    checked_folder(training/'train',500,crc=True)
    meta=read(source/'actor/lora_train_meta.json');fsdp=read(source/'actor/fsdp_config.json')
    if int(meta['r'])!=16 or int(meta['lora_alpha'])!=32 or int(fsdp['world_size'])!=1:raise ValueError('LoRA/FSDP identity differs')
    reference_path=asset/legacy.RUN/POST_REL/'evaluation_spec.json';reference=read(reference_path)
    spec=derive(reference,asset,runtime,out,source)
    Aroot=ROOT/'analysis/grpo_baseline_20261022_202816/grpo_dev50'
    A=read(Aroot/'summary.json');Arows=[json.loads(l)for l in (Aroot/'trajectories.jsonl').read_text(encoding='utf-8').splitlines()if l.strip()]
    if A['strict_successes']!=27 or A['expected_tasks']!=50:raise ValueError('Historical A independent reference differs')
    frozen=read(PACKAGE/'protocol.json')['evaluation']['protocol']
    if any(A['protocol'].get(k)!=v for k,v in frozen.items()):raise ValueError('Historical A protocol differs from B freeze')
    for suite,n in [('smoke3',3),('dev50',50)]:
        b=spec['benchmarks'][suite];rows=[json.loads(l)for l in Path(b['path']).read_text().splitlines()if l.strip()]
        ids=[int(r['task_id'])for r in rows]
        if digest(b['path'])!=b['sha256']or ids!=b['task_ids']or len(ids)!=n or len(set(ids))!=n:raise ValueError('Benchmark identity/order differs: '+suite)
        if suite=='dev50'and ids!=[int(r['task_id'])for r in Arows]:raise ValueError('Independent A/B Dev task order differs')
        for k,v in frozen.items():
            if k=='reward_contract':continue
            option='--'+k.replace('_','-')
            if k=='context_compaction':
                if option in spec['commands'][suite]:raise ValueError('Unexpected context compaction')
            elif k=='benchmark':
                if suite=='dev50'and spec['commands'][suite][spec['commands'][suite].index(option)+1]!=v:raise ValueError('Dev path differs')
            else:
                # Preserve omitted original CLI options too: the same hashed
                # harness supplies their original defaults; verify actual summary.
                if option not in spec['commands'][suite]:continue
                actual=spec['commands'][suite][spec['commands'][suite].index(option)+1]
                if str(actual)!=str(v)and not(isinstance(v,(int,float))and float(actual)==v):raise ValueError('Frozen CLI differs: '+k)
    post=out/POST_REL;post.mkdir(parents=True)
    write(post/'evaluation_spec.json',spec);write(out/'posttrain.json',{'repo':str(ROOT),'training':str(training),'out':str(out),
        'asset_project':str(asset),'source_spec_sha256':digest(reference_path),'source_checkpoint_size':(source/'actor/model_world_size_1_rank_0.pt').stat().st_size})
    write(post/'preparation_receipt.json',{'passed':True,'checkpoint':str(source),'source_crc_verified':True,'suites':list(SUITES),
        'Final_enabled':False,'A500_successes':27,'cpu_tests_passed':True,'cpu_test_log_sha256':digest(cpu_log),'export_started':False,'evaluation_started':False})
    print('DEV50_PREPARED=true\nOUT='+str(out)+'\nNo models loaded or services/evaluations started.')


def validate_suite(spec,suite,summarize):
    folder=Path(spec['evaluation_root'])/suite
    rows=[json.loads(l)for l in (folder/'trajectories.jsonl').read_text().splitlines()if l.strip()]
    ids=[int(r['task_id'])for r in rows];expected=spec['benchmarks'][suite]['task_ids']
    if ids!=expected or len(set(ids))!=len(expected):raise ValueError('Missing/duplicate/out-of-order task results')
    failures=[r['task_id']for r in rows if (r.get('error')and r['error'].get('type')!='ContextBudgetError')or r.get('release_error')]
    if failures:raise ValueError('Non-context execution failures: '+str(failures))
    calc=summarize(expected,rows);reported=read(folder/'summary.json')
    if any(reported.get(k)!=v for k,v in calc.items()):raise ValueError('Native summary recomputation differs')
    frozen=read(PACKAGE/'protocol.json')['evaluation']['protocol']
    for k,v in frozen.items():
        if k=='benchmark':continue
        if reported['protocol'].get(k)!=v:raise ValueError('Actual summary protocol differs: '+k)
    write(folder/'passed.json',{'passed':True,'tasks':len(rows),'strict_successes':calc['strict_successes'],
        'trajectory_sha256':digest(folder/'trajectories.jsonl'),'summary_sha256':digest(folder/'summary.json')})
    return reported


def cleanup(out):
    tag=b'B500_DEV_RUN='+str(out).encode()
    for sig in (signal.SIGTERM,signal.SIGKILL):
        for p in Path('/proc').glob('[0-9]*/environ'):
            try:
                if tag in p.read_bytes().split(b'\0'):os.kill(int(p.parent.name),sig)
            except (OSError,PermissionError):pass
        time.sleep(2)


def pipeline(out):
    import fcntl
    check_code();post=out/POST_REL;lock=(post/'pipeline.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:lock.close();raise
    if (post/'pipeline_started.json').exists():
        lock.close();raise ValueError('Already attempted; preserve run and return evidence')
    write(post/'pipeline_started.json',{'at':time.time()});spec=read(post/'evaluation_spec.json');old_run=legacy.RUN;legacy.RUN=out
    code=1
    try:
        verify_runtime_snapshot(Path(spec['asset_project']),Path(spec['project']))
        for port in (5700,8000):
            with socket.socket()as sock:sock.bind(('127.0.0.1',port))
        legacy.export_model(Path(spec['asset_project']),500);write(post/'export_exit_status.json',{'exit_code':0})
        original=subprocess.Popen
        def tagged(*args,**kw):
            kw['env']=dict(kw.get('env',os.environ),B500_DEV_RUN=str(out),
                SHOP_ENV_CONFIG=str(Path(spec['asset_project'])/'environments/ShopSimulator/shop_env/configs/environment.json'),
                SHOP_SEARCH_INDEX=str(Path(spec['asset_project'])/'environments/ShopSimulator/shop_env/search_engine/products.sqlite3'))
            return original(*args,**kw)
        legacy.subprocess.Popen=tagged
        try:legacy.serve(Path(spec['asset_project']),500)
        finally:legacy.subprocess.Popen=original
        from shopping_grpo.evaluation.summary import summarize_trajectories
        results={};env=dict(os.environ,PYTHONPATH=str(Path(spec['project'])/'src'),OMP_NUM_THREADS='1',B500_DEV_RUN=str(out))
        for suite in SUITES:
            folder=Path(spec['evaluation_root'])/suite;folder.mkdir(parents=True,exist_ok=False)
            print('START_SUITE='+suite,flush=True)
            with (folder/'driver.log').open('x')as log:subprocess.run(spec['commands'][suite],cwd=spec['project'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            results[suite]=validate_suite(spec,suite,summarize_trajectories)
            print('PASSED '+suite+' '+str(results[suite]['strict_successes'])+'/'+str(results[suite]['expected_tasks']),flush=True)
        A=read(ROOT/'analysis/grpo_baseline_20261022_202816/grpo_dev50/summary.json');B=results['dev50']
        a=set(A['strict_success_task_ids']);b=set(B['strict_success_task_ids'])
        write(post/'dev50_complete.json',{'passed':True,'A500':27,'B500':B['strict_successes'],'net_gain':len(b)-len(a),
            'B_only':sorted(b-a),'A_only':sorted(a-b),'continuation_screen_passed':len(b)>=30 and len(b)-len(a)>=3,
            'mean_reward':B['mean_final_reward'],'Final_done':False,'auto_continue':False})
        code=0;print('B500_INDEPENDENT_DEV50_COMPLETE',flush=True)
    finally:
        legacy.RUN=old_run
        try:write(post/'pipeline_exit_status.json',{'exit_code':code,'at':time.time()})
        finally:
            try:cleanup(out)
            finally:lock.close()


def export_evidence(out):
    import fcntl
    post=out/POST_REL;lock=(post/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from run_b500_preflight_return import git,configure_author,publish
    dest=Path(tempfile.mkdtemp(prefix='b500-dev-return-',dir=ROOT/'outputs'));origin=git(['remote','get-url','origin'],ROOT,capture=True)
    git(['clone','--depth','1','--single-branch','--branch','codex/b500-independent-dev50',origin,str(dest)],ROOT);configure_author(dest,ROOT)
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());branch='codex/b500-dev50-results-'+stamp+'-'+dest.name.rsplit('-',1)[-1]
    git(['switch','-c',branch],dest);target=dest/'analysis/b500_independent_dev50_return'/stamp;target.mkdir(parents=True)
    with zipfile.ZipFile(target/'evidence.zip','w',zipfile.ZIP_DEFLATED)as z:
        for folder in (post,out/'evaluation'):
            for p in folder.rglob('*'):
                if p.is_file()and p.suffix in ('.json','.jsonl','.log'):z.write(p,p.relative_to(out).as_posix())
        for p in [out/'posttrain.json',out/'supervisor.log']:
            if p.exists():z.write(p,p.name)
        training=Path(read(out/'posttrain.json')['training'])
        for p in (training/'validation').glob('*.jsonl'):z.write(p,'training_validation/'+p.name)
    write(target/'manifest.json',{'archive_sha256':digest(target/'evidence.zip'),'weights_included':False,'Final_done':False})
    git(['add','--',target.relative_to(dest).as_posix()],dest);git(['commit','-m','Return fixed B500 independent Dev50 evidence'],dest)
    commit=git(['rev-parse','HEAD'],dest,capture=True);publish(dest,branch,commit);print('RETURN_BRANCH='+branch+'\nRETURN_COMMIT='+commit)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','run','status','return']);p.add_argument('--training',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
    if a.phase=='prepare':prepare(a.training.resolve(),out)
    elif a.phase=='run':pipeline(out)
    elif a.phase=='return':export_evidence(out)
    else:
        for name in ['preparation_receipt.json','export_receipt.json','services_ready.json','dev50_complete.json','pipeline_exit_status.json']:
            q=out/POST_REL/name
            if q.exists():print(name,q.read_text())
