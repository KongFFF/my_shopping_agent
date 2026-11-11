"""Manage checkpoint-pair CPU preflight and resumable inference; never train."""
import argparse
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from urllib.request import Request,urlopen
from checkpoint_pair_common import *


def http(url,payload=None,timeout=60):
    req=Request(url,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urlopen(req,timeout=timeout) as f:return json.loads(f.read())


def assert_free(port):
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',port))==0:raise ValueError(f'Port {port} is occupied; existing processes are untouched')


def checked(run):
    check_package();cfg=load(run/'run.json')
    if cfg['protocol_sha256']!=file_hash(PACKAGE/'manifest.json'):raise ValueError('Run protocol changed')
    return cfg


def model_hashes(path):
    path=Path(path).resolve()
    config=load(path/'config.json')
    if config.get('model_type')!='qwen3_5':raise ValueError('Expected Qwen3.5 checkpoint')
    tensors=sorted(path.glob('*.safetensors'))
    if not tensors:raise ValueError('Missing merged model weights: '+str(path))
    files=tensors+sorted(path.glob('*.json'))+sorted(path.glob('*.jinja'))
    return {p.name:file_hash(p) for p in files}


def init(run,bundle,cpu_check):
    check_package()
    if run.exists():raise ValueError('Use a new run directory; existing records will not be overwritten')
    if not load(cpu_check/'cpu_check.json')['passed']:raise ValueError('Actual CPU preflight did not pass')
    receipt=runtime_receipt(bundle)
    if receipt!=load(PACKAGE/'runtime_receipt.json'):raise ValueError('Wrong baseline runtime')
    models={arm:{'path':str((UP/p).resolve()),'files':model_hashes(UP/p)} for arm,p in MODEL_PATHS.items()}
    for name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja'):
        if name not in models['sft']['files'] or models['sft']['files'][name]!=models['grpo']['files'].get(name):
            raise ValueError('Tokenization/chat template differs: '+name)
    version=importlib.metadata.version('vllm')
    if version!='0.25.1':raise ValueError('Expected original vLLM 0.25.1, found '+version)
    run.mkdir(parents=True)
    save(run/'run.json',{'version':VERSION,'models':models,'bundle':str(bundle.resolve()),
        'protocol_sha256':file_hash(PACKAGE/'manifest.json'),'runtime_sha256':digest(dump(receipt)),
        'index_sha256':file_hash(bundle/'environment/search_engine/products.sqlite3'),
        'cpu_check':load(cpu_check/'cpu_check.json'),'vllm_version':version,
        'settings':SETTINGS,'prepared_at':time.time()})
    print('PASS: 262 tasks, both existing checkpoints, tokenizer and baseline runtime prepared')


@contextmanager
def services(run,bundle,model=None,phase='cpu'):
    """Own only these child process groups. No billing clock or budget command."""
    assert_free(5721)
    if model is not None:assert_free(8021)
    children=[];handles=[];label=model or 'cpu'
    env=dict(os.environ,PYTHONPATH='',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',VLLM_USE_FLASHINFER_SAMPLER='0')
    commands=[('environment',[str(UP/'environments/ShopSimulator/.venv-shopsim/bin/python'),
        str(ROOT/'tools/checkpoint_pair_environment.py'),'--bundle',str(bundle)])]
    if model:
        cfg=checked(run)
        if model_hashes(cfg['models'][model]['path'])!=cfg['models'][model]['files']:
            raise ValueError('Model files changed since init')
        commands.append(('model',[str(UP/'.venv/bin/vllm'),'serve',cfg['models'][model]['path'],
            '--served-model-name','checkpoint-pair-'+model,'--port','8021','--max-model-len','24576',
            '--enable-auto-tool-choice','--tool-call-parser','qwen3_coder']))
    try:
        for name,cmd in commands:
            log=(run/f'{label}_{name}.log').open('ab');handles.append(log)
            children.append(subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        started={'event':'started','model':label,'phase':phase,'at':time.time(),'pids':[p.pid for p in children]}
        save(run/f'{label}_{phase}_services_started.json',started);append(run/'services_history.jsonl',started)
        deadline=time.monotonic()+600
        while True:
            if any(p.poll() is not None for p in children):raise RuntimeError('Service exited; inspect service logs')
            try:
                identity=http(ENV_URL+'/api/checkpoint_pair_identity',timeout=3)
                if identity['runtime_sha256']!=digest(dump(load(PACKAGE/'runtime_receipt.json'))) or identity['code_sha256']!=code_receipt():
                    raise ValueError('Wrong environment identity')
                if model:
                    cfg=checked(run)
                    if identity['index_sha256']!=cfg['index_sha256']:raise ValueError('Search index changed')
                    data=http(MODEL_URL+'/models',timeout=3)['data']
                    matches=[m for m in data if m['id']=='checkpoint-pair-'+model]
                    if not matches:raise OSError('Model not ready')
                    if Path(matches[0]['root']).resolve()!=Path(cfg['models'][model]['path']).resolve() or matches[0]['max_model_len']!=24576:
                        raise ValueError('Wrong model loaded')
                    save(run/f'{model}_{phase}_ready.json',{'identity':identity,'model':matches[0],'at':time.time()})
                break
            except (OSError,TimeoutError):
                if time.monotonic()>=deadline:raise RuntimeError('Services not ready within startup timeout')
                time.sleep(2)
        yield
    finally:
        for child in children:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait()
            # Parent may exit before its engine workers; clean this owned group.
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        for f in handles:f.close()
        stopped={'event':'stopped','model':label,'phase':phase,'at':time.time(),'children_exit_codes':[p.returncode for p in children]}
        save(run/f'{label}_{phase}_services_stopped.json',stopped);append(run/'services_history.jsonl',stopped)


def canonical_result(z):return {k:v for k,v in z.items() if k not in ('idx','env_idx')}


def cpu_preflight(output,bundle):
    check_package()
    if output.exists():raise ValueError('CPU preflight output must be new')
    output.mkdir(parents=True);results=[];r=load_agent(bundle)
    try:
        with services(output,bundle):
            for task in rows(PACKAGE/'tasks.jsonl')[:3]:
                a=r.ShopAgentEnv(base_url=ENV_URL);b=r.ShopAgentEnv(base_url=ENV_URL)
                try:
                    x=a.reset(task['task_id']);y=b.reset(task['task_id'])
                    if a.env_idx==b.env_idx or canonical_result(x)!=canonical_result(y):raise ValueError('Independent reset mismatch')
                    query=x['instruction'].removeprefix('Instruction: ')
                    if query!=task['query']:raise ValueError('Frozen query mismatch')
                    action='search['+task['query']+']'
                    if canonical_result(a.step(action))!=canonical_result(b.step(action)):raise ValueError('Public action replay differs')
                    results.append({'task_id':task['task_id'],'distinct_slots':True,'public_search_matched':True})
                finally:
                    try:a.release()
                    finally:b.release()
        save(output/'cpu_check.json',{'passed':len(results)==3,'tasks':results,'model_used':False})
    except BaseException as exc:
        save(output/'cpu_check.json',{'passed':False,'tasks':results,'error':str(exc)});raise
    print('PASS: actual original-environment CPU preflight')


def slot_dir(run,arm,task_id,attempt):return run/arm/'attempts'/f'{task_id}_{attempt}'


def latest_try(slot):
    tries=sorted(slot.glob('try_*')) if slot.exists() else []
    return tries[-1] if tries else None


def selected(slot):
    last=latest_try(slot)
    if last is None:return None
    return load(last/'result.json') if (last/'result.json').exists() else {'classification':'interrupted','try_dir':str(last)}


def require_repair(slot):
    previous=selected(slot)
    if previous is None:raise ValueError('Repair requires a previously attempted slot')
    if previous['classification']=='completed':raise ValueError('Completed policy outcomes must never be rerolled')
    return previous


def evaluate_slot(run,arm,task,attempt,repair=False,client_factory=None,env_factory=None):
    cfg=checked(run);r=load_agent(cfg['bundle']);slot=slot_dir(run,arm,task['task_id'],attempt)
    previous=selected(slot)
    if repair:require_repair(slot)
    elif previous:
        if previous['classification']=='completed':return previous
        raise ValueError(f'Unresolved infrastructure/interruption in {task["task_id"]}:{attempt}; use explicit repair')
    old=latest_try(slot);number=int(old.name.split('_')[1])+1 if old else 0
    folder=slot/f'try_{number:03d}';folder.mkdir(parents=True)
    save(folder/'start.json',{'model':arm,'task_id':task['task_id'],'attempt':attempt,'at':time.time(),
                            'repair_of':str(old) if old else None,'query_sha256':task['query_sha256']})
    turn=0
    def transport(url,payload,headers,timeout):
        nonlocal turn
        payload['seed']=seed_for(task['task_id'],attempt,turn);turn+=1
        record={'turn':turn-1,'seed':payload['seed'],'at':time.time()}
        try:
            response=http(url,payload,timeout)
            record.update(status='ok',usage=response.get('usage'));append(folder/'usage.jsonl',record)
            return response
        except Exception as exc:
            record.update(status='error',error=type(exc).__name__);append(folder/'usage.jsonl',record);raise
    settings={k:v for k,v in SETTINGS.items() if k not in ('attempts','max_steps')}
    client=client_factory() if client_factory else r.OpenAIChatClient(model='checkpoint-pair-'+arm,
        base_url=MODEL_URL,api_key='EMPTY',timeout=60,transport=transport,**settings)
    # Disable implicit transport retries: repair must preserve the attempt record
    # and start from the same deterministic turn seed, rather than silently retry.
    r.MODEL_COMPLETION_RETRIES=0
    factory=env_factory or r.ShopAgentEnv
    class CheckedEnv:
        def __init__(self,**kwargs):self.env=factory(**kwargs)
        def reset(self,tid):
            initial=self.env.reset(tid)
            if initial.get('environment_version')!='shopsimulator-environment-v2.1' or initial['instruction'].removeprefix('Instruction: ')!=task['query']:
                raise ValueError('Environment/query identity mismatch')
            return initial
        def step(self,action):return self.env.step(action)
        def release(self):return self.env.release()
    trace=r.collect_for_task({'task_id':task['task_id']},client,env_factory=CheckedEnv,base_url=ENV_URL,max_steps=35,attempt_index=attempt)
    save(folder/'trajectory.json',trace)
    result={'model':arm,'task_id':task['task_id'],'attempt':attempt,'try_index':number,
            'trajectory_sha256':file_hash(folder/'trajectory.json'),**outcome(trace)}
    save(folder/'result.json',result)
    if result['classification']!='completed':raise RuntimeError(f'Infrastructure error at {task["task_id"]}:{attempt}; saved {folder}')
    return result


def planned(phase):
    tasks=rows(PACKAGE/'tasks.jsonl')
    return [(t,a) for t in (tasks[:3] if phase=='smoke' else tasks) for a in (range(1) if phase=='smoke' else range(8))]


def evaluate(run,arm,phase,repair_slot=None):
    cfg=checked(run);lock=run/'driver.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    old_handler=signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    status={'model':arm,'phase':phase,'passed':False,'at':time.time()}
    try:
        if phase=='full':
            for model in ('sft','grpo'):
                proof=run/f'smoke_{model}.json'
                if not proof.exists() or not load(proof)['passed']:raise ValueError('Both smoke phases must pass before full evaluation')
        jobs=planned(phase)
        if repair_slot:
            tid,attempt=map(int,repair_slot.split(':'))
            jobs=[(t,a) for t,a in jobs if t['task_id']==tid and a==attempt]
            if len(jobs)!=1:raise ValueError('Repair slot not in this phase')
            require_repair(slot_dir(run,arm,tid,attempt))
        else:
            for t,a in jobs:
                previous=selected(slot_dir(run,arm,t['task_id'],a))
                if previous and previous['classification']!='completed':raise ValueError(f'Unresolved slot {t["task_id"]}:{a}; explicit repair required')
            jobs=[(t,a) for t,a in jobs if selected(slot_dir(run,arm,t['task_id'],a)) is None]
        if jobs:
            with services(run,Path(cfg['bundle']),arm,phase):
                for t,a in jobs:
                    evaluate_slot(run,arm,t,a,repair=bool(repair_slot))
                    print(f'completed {arm} {t["task_id"]}:{a}',flush=True)
        status['completed_slots']=sum((selected(slot_dir(run,arm,t['task_id'],a)) or {}).get('classification')=='completed' for t,a in planned(phase))
        status['expected_slots']=len(planned(phase));status['passed']=status['completed_slots']==status['expected_slots']
        if not repair_slot and not status['passed']:raise RuntimeError('Phase is incomplete')
    except BaseException as exc:
        status.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:
        save(run/f'{phase}_{arm}.json',status)
        signal.signal(signal.SIGTERM,old_handler);lock.unlink()


def status(run):
    checked(run)
    counts={a:Counter((selected(slot_dir(run,a,t['task_id'],i)) or {'classification':'not_started'})['classification']
                      for t in rows(PACKAGE/'tasks.jsonl') for i in range(8)) for a in ('sft','grpo')}
    print(dump({a:dict(v) for a,v in counts.items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['cpu-check','init','evaluate','status'])
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--bundle',type=Path,default=ROOT/'outputs/checkpoint_pair_v1/runtime')
    p.add_argument('--cpu-check',type=Path,default=ROOT/'outputs/checkpoint_pair_v1/cpu_check_01')
    p.add_argument('--model',choices=['sft','grpo']);p.add_argument('--phase',choices=['smoke','full']);p.add_argument('--repair-slot')
    a=p.parse_args();run=a.run_dir.resolve()
    if a.command=='cpu-check':cpu_preflight(run,a.bundle)
    elif a.command=='init':init(run,a.bundle,a.cpu_check)
    elif a.command=='status':status(run)
    else:
        if not a.model or not a.phase:p.error('evaluate requires --model and --phase')
        evaluate(run,a.model,a.phase,a.repair_slot)
