"""User-authorized supplemental Final200 on fixed, already exported B500."""
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
POST=base.POST_REL
SUITE='final200_clean'


def check_code():
    base.check_code()
    for n,h in read(PACKAGE/'final200_code_manifest.json')['files_sha256'].items():
        if digest(ROOT/n)!=h:raise ValueError('Final code drift: '+n)


def verify_model(spec,receipt):
    if receipt['checkpoint']!=spec['checkpoint']or receipt['merged_model']!=spec['merged_model']:raise ValueError('Model export identity differs')
    for n,h in receipt['merged_weights_sha256'].items():
        if digest(Path(spec['merged_model'])/n)!=h:raise ValueError('Previously evaluated B500 weights changed: '+n)


def derive(reference,old,out):
    spec=copy.deepcopy(old)
    spec['protocols']={SUITE:reference['protocols'][SUITE]}
    spec['benchmarks']={SUITE:reference['benchmarks'][SUITE],'smoke3':old['benchmarks']['smoke3']}
    command=list(reference['commands'][SUITE]);command[0]=sys.executable
    command[1]=str(Path(spec['project'])/'scripts/evaluate_shop_benchmark.py')
    for k,v in [('--model',spec['model_tag']),('--output',out/'evaluation'/SUITE/'trajectories.jsonl'),('--summary',out/'evaluation'/SUITE/'summary.json')]:
        command=legacy.replace_option(command,k,v)
    spec['commands']={SUITE:command};spec['evaluation_root']=str(out/'evaluation')
    return spec


def prepare(dev,out):
    check_code()
    if out.exists():raise ValueError('Final output already exists; do not overwrite')
    (ROOT/'outputs').mkdir(exist_ok=True);cpu_log=ROOT/'outputs/final200_cpu_tests.log'
    print('FINAL_CPU_LOG='+str(cpu_log),flush=True)
    with cpu_log.open('w')as log:subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tools'),'-p','test_b500_final200.py'],check=True,stdout=log,stderr=subprocess.STDOUT)
    old=read(dev/POST/'evaluation_spec.json');dev_done=read(dev/POST/'dev50_complete.json')
    if not dev_done['passed']or dev_done['B500']!=28 or read(dev/POST/'pipeline_exit_status.json')['exit_code']!=0:raise ValueError('Expected completed B500 Dev28/50')
    asset=Path(old['asset_project']);runtime=Path(old['project']);verify_runtime_snapshot(asset,runtime)
    reference_path=asset/legacy.RUN/POST/'evaluation_spec.json'
    if digest(reference_path)!=read(dev/'posttrain.json')['source_spec_sha256']:raise ValueError('Original A evaluation specification changed since Dev')
    reference=read(reference_path);spec=derive(reference,old,out)
    b=spec['benchmarks'][SUITE];tasks=[json.loads(l)for l in Path(b['path']).read_text().splitlines()if l.strip()];ids=[int(t['task_id'])for t in tasks]
    if b['sha256']!=legacy.FINAL_HASH or digest(b['path'])!=legacy.FINAL_HASH or ids!=b['task_ids']or len(ids)!=200 or len(set(ids))!=200:raise ValueError('Original Final200 Clean identity differs')
    cmd=spec['commands'][SUITE]
    if cmd[cmd.index('--benchmark')+1]!=b['path']:raise ValueError('Final command/benchmark binding differs')
    A=read(Path(reference['evaluation_root'])/SUITE/'summary.json')
    if A['strict_successes']!=137 or A['expected_tasks']!=200 or len(set(A['strict_success_task_ids']))!=137 or not set(A['strict_success_task_ids']).issubset(ids):raise ValueError('Expected historical A500 Final137/200')
    export=read(dev/POST/'export_receipt.json');print('VERIFY_EXISTING_B500_WEIGHTS (no re-export)',flush=True);verify_model(spec,export)
    for n,h in spec['source_sha256'].items():
        if digest(n)!=h:raise ValueError('Frozen evaluation source changed: '+n)
    post=out/POST;post.mkdir(parents=True)
    write(post/'evaluation_spec.json',spec);write(post/'export_receipt.json',export);write(post/'export_exit_status.json',{'exit_code':0,'reused_from':str(dev)})
    write(post/'A500_final_summary.json',A)
    write(post/'preparation_receipt.json',{'passed':True,'purpose':'User-authorized supplemental fixed B500 Final after Dev missed screen',
        'prior_Dev':dev_done,'original_Dev_decision_unchanged':True,'Final_is_unseen':False,'export_reused':True,
        'benchmark_sha256':legacy.FINAL_HASH,'evaluated_step':500,'evaluation_started':False})
    write(out/'posttrain.json',{'repo':str(ROOT),'out':str(out),'dev_run':str(dev),'asset_project':str(asset),'export_reused':True})
    print('FINAL200_PREPARED=true\nOUT='+str(out)+'\nNo model serving or evaluation started.')


def pipeline(out):
    import fcntl
    check_code();post=out/POST
    with (post/'pipeline.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if(post/'pipeline_started.json').exists():raise ValueError('Final already attempted; preserve it and return evidence')
        write(post/'pipeline_started.json',{'at':time.time()});spec=read(post/'evaluation_spec.json');code=1;old_run=legacy.RUN
        try:
            verify_model(spec,read(post/'export_receipt.json'));verify_runtime_snapshot(Path(spec['asset_project']),Path(spec['project']))
            for port in (5700,8000):
                with socket.socket()as sock:sock.bind(('127.0.0.1',port))
            legacy.RUN=out;original=subprocess.Popen
            def tagged(*args,**kw):
                kw['env']=dict(kw.get('env',os.environ),B500_DEV_RUN=str(out),OMP_NUM_THREADS='1',
                    SHOP_ENV_CONFIG=str(Path(spec['asset_project'])/'environments/ShopSimulator/shop_env/configs/environment.json'),
                    SHOP_SEARCH_INDEX=str(Path(spec['asset_project'])/'environments/ShopSimulator/shop_env/search_engine/products.sqlite3'))
                return original(*args,**kw)
            legacy.subprocess.Popen=tagged
            try:legacy.serve(Path(spec['asset_project']),500)
            finally:legacy.subprocess.Popen=original
            from shopping_grpo.evaluation.summary import summarize_trajectories
            folder=Path(spec['evaluation_root'])/SUITE;folder.mkdir(parents=True,exist_ok=False)
            env=dict(os.environ,PYTHONPATH=str(Path(spec['project'])/'src'),OMP_NUM_THREADS='1',B500_DEV_RUN=str(out))
            print('START_SUITE='+SUITE,flush=True)
            with (folder/'driver.log').open('x')as log:subprocess.run(spec['commands'][SUITE],cwd=spec['project'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            summary=base.validate_suite(spec,SUITE,summarize_trajectories)
            A=read(post/'A500_final_summary.json');a=set(A['strict_success_task_ids']);b=set(summary['strict_success_task_ids'])
            write(post/'final200_complete.json',{'passed':True,'B500':summary['strict_successes'],'A500':137,'expected_tasks':200,
                'net_gain':len(b)-len(a),'B_only':sorted(b-a),'A_only':sorted(a-b),'mean_reward':summary['mean_final_reward'],
                'supplemental_evaluation':True,'Dev_screen_still_failed':True,'auto_continue':False})
            code=0;print('B500_FINAL200_COMPLETE',flush=True)
        finally:
            legacy.RUN=old_run
            try:write(post/'pipeline_exit_status.json',{'exit_code':code,'at':time.time()})
            finally:base.cleanup(out)


def export_evidence(out):
    import fcntl
    from run_b500_preflight_return import git,configure_author,publish
    with (out/POST/'pipeline.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dest=Path(tempfile.mkdtemp(prefix='b500-final-return-',dir=ROOT/'outputs'));origin=git(['remote','get-url','origin'],ROOT,capture=True)
        git(['clone','--depth','1','--single-branch','--branch','codex/b500-final200',origin,str(dest)],ROOT);configure_author(dest,ROOT)
        stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());branch='codex/b500-final200-results-'+stamp+'-'+dest.name.rsplit('-',1)[-1]
        git(['switch','-c',branch],dest);target=dest/'analysis/b500_final200_return'/stamp;target.mkdir(parents=True)
        with zipfile.ZipFile(target/'evidence.zip','w',zipfile.ZIP_DEFLATED)as z:
            for folder in (out/POST,out/'evaluation'):
                for p in folder.rglob('*'):
                    if p.is_file()and p.suffix in ('.json','.jsonl','.log'):z.write(p,p.relative_to(out).as_posix())
            for p in [out/'posttrain.json',out/'supervisor.log']:
                if p.exists():z.write(p,p.name)
        write(target/'manifest.json',{'archive_sha256':digest(target/'evidence.zip'),'weights_included':False,'supplemental_Final':True})
        git(['add','--',target.relative_to(dest).as_posix()],dest);git(['commit','-m','Return user-authorized fixed B500 Final200'],dest)
        commit=git(['rev-parse','HEAD'],dest,capture=True);publish(dest,branch,commit);print('RETURN_BRANCH='+branch+'\nRETURN_COMMIT='+commit)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','run','status','return']);p.add_argument('--dev',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
    if a.phase=='prepare':prepare(a.dev.resolve(),out)
    elif a.phase=='run':pipeline(out)
    elif a.phase=='return':export_evidence(out)
    else:
        for name in ['preparation_receipt.json','services_ready.json','final200_complete.json','pipeline_exit_status.json']:
            p=out/POST/name
            if p.exists():print(name,p.read_text())
