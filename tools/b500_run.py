"""Prepare, run0->350->500, inspect and return B500; Linux AutoDL only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1];PACKAGE=ROOT/'experiments/group_weight_b500_v1'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def verify_launch_code():
    m=read(PACKAGE/'launch_manifest.json')
    for n,h in m['files_sha256'].items():
        if digest(ROOT/n)!=h:raise ValueError('Launch code hash drift: '+n)
    if digest(PACKAGE/'protocol.json')!=m['protocol_sha256']:raise ValueError('Protocol drift')
    if digest(PACKAGE/'implementation_manifest.json')!=m['accepted_implementation_sha256']:raise ValueError('CPU-accepted method code manifest changed')
    for n,h in read(PACKAGE/'implementation_manifest.json')['files_sha256'].items():
        if digest(ROOT/n)!=h:raise ValueError('CPU-accepted method source changed: '+n)
def call(cmd,**kw):subprocess.run(cmd,check=True,**kw)
def remap(x,asset,runtime):
    if isinstance(x,dict):return {k:remap(v,asset,runtime)for k,v in x.items()}
    if isinstance(x,list):return [remap(v,asset,runtime)for v in x]
    if isinstance(x,str):
        for part in ('configs','src','scripts'):
            prefix=asset.as_posix()+'/'+part+'/'
            if x.startswith(prefix):return runtime.as_posix()+'/'+part+'/'+x[len(prefix):]
    return x


def verify_runtime_snapshot(asset,runtime):
    snapshot=read(runtime/'runtime_receipt.json')
    if snapshot.get('base')!='a583dba4c149be09622ce58ad6f7a160db77bb41':raise ValueError('Isolated baseline snapshot changed')
    for n,h in snapshot['files'].items():
        if digest(runtime/n)!=h:raise ValueError('Isolated baseline file changed: '+n)
    for name in ('.venv','data','environments'):
        if (runtime/name).resolve()!=(asset/name).resolve():raise ValueError('Shared baseline asset link differs: '+name)


def accepted_cpu_receipt(receipt):
    r=read(receipt)
    if (not r.get('passed')or r.get('implementation_manifest_sha256')!=digest(PACKAGE/'implementation_manifest.json')
        or r.get('protocol_sha256')!=digest(PACKAGE/'protocol.json')
        or r.get('gpu_initialized')is not False or r.get('model_weights_loaded')is not False
        or r.get('training_started')is not False):
        raise ValueError('Native CPU receipt is not bound to this implementation/protocol or initialized GPU/models')


def prepare(asset,out,cpu_receipt=None):
    verify_launch_code()
    if out.exists():raise ValueError('Run directory already exists; use status/return, never prepare over it: '+str(out))
    runtime=asset/'outputs/bounded_resume500_to650/runtime'
    verify_runtime_snapshot(asset,runtime)
    # Revalidate frozen identity; no GPU or tensor loading.
    receipt=cpu_receipt or ROOT/'analysis/b500_implementation_20261118/cloud_cpu_acceptance/20261118T034604Z.json'
    accepted_cpu_receipt(receipt)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.json')as tmp:
        call([sys.executable,str(ROOT/'tools/check_b500_history.py'),'--asset-project',str(asset),'--runtime-project',str(runtime),'--output',tmp.name])
    out.mkdir(parents=True);(out/'configs').mkdir();(out/'train').mkdir()
    import yaml
    cfg=remap(read(PACKAGE/'historical_a_composed_config.json'),asset,runtime)
    train=out/'train';env={'PYTHONHASHSEED':'42','OMP_NUM_THREADS':'1','VLLM_USE_FLASHINFER_SAMPLER':'0',
        'SHOPPING_LENGTH_SHAPING_ENABLE':'false','SHOPSIM_BASE_URL':'http://127.0.0.1:5727',
        'SHOPPING_ENVIRONMENT_VERSION':'shopsimulator-environment-v2.1','SHOPPING_ENV_MANIFEST':str(asset/'data/environment.json'),
        'GRPO_MODEL_PATH':cfg['actor_rollout_ref']['model']['path'],'GRPO_TRAIN_FILE':str(asset/'data/grpo/train.parquet'),
        'GRPO_VAL_FILE':str(asset/'data/grpo/validation.parquet'),'GRPO_OUTPUT_DIR':str(train),
        'SHOPPING_AGENT_LOOP_CONFIG':str(runtime/'configs/agent_loop.yaml'),'SHOPPING_TOOL_CONFIG':str(runtime/'configs/tools.json'),
        'SHOPPING_GRPO_DIAGNOSTICS_PATH':str(out/'training_diagnostics.jsonl'),'B500_WEIGHT_LOG':str(out/'weights.jsonl'),
        'B500_RUN_DIR':str(out),'B500_PROCESS_ROLE':'trainer','B500_CONTROL_FILE':str(out/'control.json'),
        'GRPO_CONFIG_NAME':'grpo',
        'PYTHONPATH':os.pathsep.join([str(PACKAGE/'overlay'),str(runtime/'src'),str(ROOT/'tools')])}
    cfg['trainer'].update(default_local_dir=str(train),validation_data_dir=str(out/'validation'),
        experiment_name='b500-group-weight-v1',resume_mode='disable',resume_from_path=None)
    cfg['actor_rollout_ref']['rollout']['trace']['experiment_name']='b500-group-weight-v1'
    cfg['actor_rollout_ref']['actor']['checkpoint']['async_save']=False
    cfg['ray_kwargs']['ray_init'].update(address='local')
    cfg['ray_kwargs']['ray_init']['runtime_env']={'worker_process_setup_hook':'b500_run_control.setup_worker','env_vars':env}
    for stop in (350,500):
        c=json.loads(json.dumps(cfg));c['ray_kwargs']['ray_init']['_temp_dir']='/tmp/b5-'+digest(PACKAGE/'protocol.json')[:6]+'-'+str(stop)
        if stop==500:c['trainer'].update(resume_mode='resume_path',resume_from_path=str(train/'global_step_350'),val_before_train=False)
        (out/'configs'/f'phase{stop}.yaml').write_text(yaml.safe_dump(c,allow_unicode=True,sort_keys=False))
    write(out/'run.json',{'version':'b500-live-v1','project':str(asset),'runtime':str(runtime),'output':str(out),'repo':str(ROOT),
        'protocol_sha256':digest(PACKAGE/'protocol.json'),'implementation_manifest_sha256':digest(PACKAGE/'implementation_manifest.json'),
        'cpu_receipt':str(receipt),'env':env,'config_hashes':{str(stop):digest(out/'configs'/f'phase{stop}.yaml')for stop in (350,500)},'training_started':False,'launch_attempted':False})
    print('PREPARED='+str(out)+'; no training started')


def belongs_to_run(raw,out):
    entries=raw.split(b'\0')
    return all(n in entries for n in (b'B500_RUN_DIR='+out.as_posix().encode(),b'B500_PROCESS_ROLE=trainer'))


def owned_cleanup(out):
    # Kill only children explicitly tagged with this exact run, never ray stop/pkill.
    def owned():
        found=[]
        for p in Path('/proc').glob('[0-9]*/environ'):
            try:
                if belongs_to_run(p.read_bytes(),out):found.append(int(p.parent.name))
            except (OSError,PermissionError):pass
        return found
    for sig in (signal.SIGTERM,signal.SIGKILL):
        for pid in owned():
            if pid!=os.getpid():
                try:os.kill(pid,sig)
                except ProcessLookupError:pass
        time.sleep(2)


def run(out):
    import fcntl
    verify_launch_code()
    spec=read(out/'run.json')
    if spec['training_started']or spec.get('launch_attempted'):raise ValueError('This run was attempted already; inspect and return it, do not restart automatically')
    lock=(out/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if digest(PACKAGE/'protocol.json')!=spec['protocol_sha256']:raise ValueError('Protocol drift')
    asset=Path(spec['project']);runtime=Path(spec['runtime']);env=dict(os.environ,**spec['env']);env.pop('RAY_ADDRESS',None)
    verify_runtime_snapshot(asset,runtime)
    # All allocated time from this GPU command counts, including preflight and validation.
    deadline=spec.get('inherited_deadline',time.time()+20*3600)
    if time.time()>=deadline-120:raise ValueError('Original B500 wall budget expired; preserve evidence, do not reset budget')
    accepted_cpu_receipt(Path(spec['cpu_receipt']))
    spec.update(launch_attempted=True,started_at=time.time(),deadline=deadline);write(out/'run.json',spec)
    environment=None;trainer=None
    try:
        call([sys.executable,'-c',"import torch; assert torch.cuda.is_available(); f,t=torch.cuda.mem_get_info(); print('GPU free GiB',f/2**30); assert f>=90*2**30"],env=env)
        limit=Path('/sys/fs/cgroup/memory.max')
        if limit.exists()and limit.read_text().strip()!='max'and int(limit.read_text())<110*2**30:raise ValueError('CPU memory quota below110GiB')
        parent=asset/'outputs/runs/grpo_baseline_20261022_202816/formal_resume350_v1/global_step_500'
        checkpoint_size=sum(p.stat().st_size for p in parent.rglob('*')if p.is_file())
        sft_size=sum(p.stat().st_size for p in Path(env['GRPO_MODEL_PATH']).glob('*.safetensors'))
        checkpoint_size=checkpoint_size or 2*sft_size
        required=13*checkpoint_size+2*sft_size+16*2**30
        if not sft_size or shutil.disk_usage(out).free<required:raise ValueError('Insufficient disk for retained checkpoints across resume and export margin')
        with socket.socket()as sock:sock.bind(('127.0.0.1',5727))
        env['SHOPPING_GRPO_ROOT']=str(runtime)
        with (out/'baseline_preflight.log').open('w')as log:
            call([sys.executable,str(runtime/'scripts/check_grpo_runtime.py')],cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT)
        ee=dict(env,SHOPSIM_PORT='5727',SHOPSIM_ENV_SLOTS='8',SHOP_MAX_STEPS='35',B500_PROCESS_ROLE='environment',
            SHOP_ENV_CONFIG=str(asset/'environments/ShopSimulator/shop_env/configs/environment.json'),
            SHOP_SEARCH_INDEX=str(asset/'environments/ShopSimulator/shop_env/search_engine/products.sqlite3'))
        with (out/'environment.log').open('w')as log:
            environment=subprocess.Popen(['bash','scripts/start_environment.sh'],cwd=runtime,env=ee,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for attempt in range(90):
            if environment.poll()is not None:raise RuntimeError('Environment startup failed; inspect environment.log')
            try:
                with socket.create_connection(('127.0.0.1',5727),timeout=1):break
            except OSError:time.sleep(1)
        else:raise TimeoutError('Environment startup timeout')
        sys.path.insert(0,str(runtime/'src'))
        from contextlib import ExitStack
        from shopping_grpo.environment.client import ShopAgentEnv
        task=json.loads((asset/'data/grpo/train.jsonl').read_text().splitlines()[0])['task_id']
        with ExitStack()as stack:
            clients=[stack.enter_context(ShopAgentEnv(env['SHOPSIM_BASE_URL']))for _ in range(8)]
            for client in clients:
                response=client.reset(task)
                if response['environment_version']!='shopsimulator-environment-v2.1':raise ValueError('Environment changed')
            if len({c.env_idx for c in clients})!=8:raise ValueError('Eight distinct slots required')
        for stop in (350,500):
            config=out/'configs'/f'phase{stop}.yaml'
            if digest(config)!=spec['config_hashes'][str(stop)]:raise ValueError('Config drift')
            write(out/'control.json',{'output':str(out),'train_dir':str(out/'train'),'stop_at':stop,'deadline':spec['deadline']})
            command=[sys.executable,'-u','-m','verl.trainer.main_ppo',f'--config-path={out/"configs"}',f'--config-name=phase{stop}']
            with (out/f'train_to_{stop}.log').open('w')as log:
                print(f'START_PHASE={stop} LOG={log.name}',flush=True)
                spec['training_started']=True;write(out/'run.json',spec)
                trainer=subprocess.Popen(command,cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write(out/'active_process.json',{'pid':trainer.pid,'stop_at':stop})
                while trainer.poll()is None:
                    if time.time()>=spec['deadline']-30:
                        os.killpg(trainer.pid,signal.SIGTERM);raise TimeoutError('20h budget exhausted')
                    time.sleep(2)
            owned_cleanup(out)
            if trainer.returncode or (out/'budget_truncated.json').exists():raise RuntimeError(f'Phase{stop} failed/truncated; do not restart blindly')
            needed='phase350_complete.json'if stop==350 else 'accepted_step_500.json'
            if not(out/needed).is_file():raise RuntimeError('Expected complete phase receipt missing')
        events=[json.loads(l)for l in (out/'training_diagnostics.jsonl').read_text().splitlines()]
        steps=[e['global_step']for e in events if e['event']=='optimizer_step']
        if steps!=list(range(1,501)):raise ValueError('Effective update chain is not exactly1..500')
        write(out/'training_complete.json',{'completed_updates':500,'protocol_sha256':spec['protocol_sha256'],
            'generation_budget':read(out/'generation_budget.json'),'elapsed_seconds':time.time()-spec.get('original_started_at',spec['started_at']),
            'independent_Dev_done':False,'Final_done':False})
        print('B500_TRAINING_COMPLETE; retain all checkpoints; return evidence before posttraining export/evaluation.')
    except Exception as e:
        write(out/'stopped.json',{'error':str(e),'at':time.time()});raise
    finally:
        if trainer and trainer.poll()is None:
            try:os.killpg(trainer.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        owned_cleanup(out)
        if environment and environment.poll()is None:
            try:os.killpg(environment.pid,signal.SIGTERM)
            except ProcessLookupError:pass


def export(out):
    import fcntl
    export_lock=(out/'supervisor.lock').open('a');fcntl.flock(export_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from run_b500_preflight_return import git,configure_author,publish
    import tempfile
    spec=read(out/'run.json');repo=Path(spec['repo']);(repo/'outputs').mkdir(exist_ok=True)
    destination=Path(tempfile.mkdtemp(prefix='b500-training-return-',dir=repo/'outputs'))
    origin=git(['remote','get-url','origin'],repo,capture=True)
    git(['clone','--single-branch','--branch','codex/b500-bypass-guard-fix',origin,str(destination)],repo)
    configure_author(destination,repo);stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());branch='codex/b500-training-results-'+stamp+'-'+destination.name.rsplit('-',1)[-1]
    git(['switch','-c',branch],destination)
    evidence=destination/'analysis/b500_training_return_20261118'/stamp;evidence.mkdir(parents=True)
    selected=[p for p in out.iterdir()if p.is_file()and p.suffix in ('.json','.jsonl','.log')]
    with zipfile.ZipFile(evidence/'evidence.zip','w',zipfile.ZIP_DEFLATED)as z:
        for p in selected:z.write(p,p.name)
        for p in (out/'configs').glob('*.yaml'):z.write(p,'configs/'+p.name)
    write(evidence/'manifest.json',{'run':str(out),'archive_sha256':digest(evidence/'evidence.zip'),'training_complete':(out/'training_complete.json').exists(),
        'protocol_sha256':spec['protocol_sha256'],'weights_included':False})
    git(['add','--',evidence.relative_to(destination).as_posix()],destination);git(['commit','-m','Return B500 training evidence'],destination)
    commit=git(['rev-parse','HEAD'],destination,capture=True);publish(destination,branch,commit)
    print('RETURN_BRANCH='+branch+'\nRETURN_COMMIT='+commit)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','run','status','return'])
    p.add_argument('--asset-project',type=Path,default=Path('/root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon'))
    p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();out=a.run_dir.resolve()
    if a.phase=='prepare':prepare(a.asset_project.resolve(),out)
    elif a.phase=='run':run(out)
    elif a.phase=='return':export(out)
    else:
        for n in ['run.json','generation_budget.json','accepted_step_1.json','phase350_complete.json','accepted_step_500.json','training_complete.json','stopped.json']:
            if(out/n).exists():print(n,(out/n).read_text())


if __name__=='__main__':main()
