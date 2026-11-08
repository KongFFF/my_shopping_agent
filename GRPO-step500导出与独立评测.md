# GRPO step 500：导出、独立 Dev-50 与 Final-200 Clean

固定候选为 `formal_resume350_v1/global_step_500`，训练内 Dev-50 为 29/50，平均奖励 0.508667。后续不根据 Final 成绩换 checkpoint。本流程只提供 AutoDL 命令，没有在本地或云端代执行模型导出或评测。

按 1–8 节顺序执行，每段通过再继续。第 1 节以及最后的结果汇总可无卡完成。第 2 节权重导出主要用 CPU，但无卡实例只有约 2 GiB 内存，所以本次第 2–7 节放在有卡模式，沿用至少 110 GiB CPU 内存的实例。独立推理需要 GPU。所有新产物写入新目录，不覆盖 SFT、训练 checkpoint 或已有评测。

## 1. 固定候选、复用 SFT 评测命令（无卡可做）

复用 SFT 的 `eval_sft_step_300_smoke3.sh` 和 `eval_sft_step_300_dev50.sh` 中的实际评测参数。Final 任务路径和协议从 SFT Final 的 `summary.json` 读取，同时校验 Clean 任务文件哈希。两者不会被当作 GRPO 选模依据。

下面脚本生成恢复会话、导出记录目录及评测辅助脚本；不加载权重。若重复执行已完成的准备，会恢复现有会话，不覆盖它。

```bash
set +e
set +u
if /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/.venv/bin/python - <<'PY'
import ast, hashlib, importlib.util, json, os, shlex, sys
from pathlib import Path

root = Path('/root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon')
run = root/'outputs/runs/grpo_baseline_20261022_202816'
post = run/'preparation/posttrain_step500_v1'
export = run/'export/step500_v1'
raw, merged = export/'hf_unmerged', export/'merged'
eval_root = run/'evaluation/step500_independent_v1'
model_tag = 'shopping-agent-grpo-step500'
if (post/'evaluation_spec.json').exists():
    assert (post/'session.env').is_file() and (post/'run_eval.py').is_file()
    print('已有准备配置；恢复本次 session.env 后继续，不重复覆盖。')
    raise SystemExit(0)
selection = json.loads((run/'preparation/dev50_selection_v1/dev50_selection.json').read_text())
assert selection['selected']['step']==500
checkpoint = Path(selection['selected']['checkpoint'])
assert checkpoint==run/'formal_resume350_v1/global_step_500'
actor = checkpoint/'actor'
assert json.loads((run/'preparation/formal_resume350_v1/exit_status.json').read_text())['exit_code']==0
meta = json.loads((actor/'lora_train_meta.json').read_text())
assert int(meta['r'])==16 and int(meta['lora_alpha'])==32, meta
fsdp = json.loads((actor/'fsdp_config.json').read_text())
assert int(fsdp['world_size'])==1
train_spec = json.loads((run/'preparation/formal_resume350_v1/launch_spec.json').read_text())
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''): h.update(chunk)
    return h.hexdigest()
for name,expected in train_spec['source_sha256'].items():
    assert digest(name)==expected, name
for p in (run/'preparation/formal_memfix_v1',run/'preparation/formal_resume350_v1'):
    # 旧监控已结束或随实例关机消失；取消标记防止旧状态触发后处理阶段关机。
    (p/'auto_shutdown.cancel').touch()

sft = root/'outputs/runs/sft_20261021_165243'
cpu = sft/'cpu_prep'
sys.path.insert(0,str(root/'src'))
loader=importlib.util.spec_from_file_location('post_eval_cli',root/'scripts/evaluate_shop_benchmark.py')
module=importlib.util.module_from_spec(loader); loader.loader.exec_module(module)
def extract(path):
    found=[]
    for line in path.read_text().splitlines():
        if 'scripts/evaluate_shop_benchmark.py' not in line: continue
        tokens=shlex.split(line)
        if len(tokens)>1 and Path(tokens[1]).name=='evaluate_shop_benchmark.py': found.append(tokens)
    assert len(found)==1, f'未找到唯一评测命令：{path}'
    return found[0]
def replace(argv,key,value):
    assert argv.count(key)==1,(key,argv)
    argv[argv.index(key)+1]=str(value)
def parsed(argv):
    saved=sys.argv
    try:
        sys.argv=[argv[1],*argv[2:]]
        return module.parse_args()
    finally: sys.argv=saved
def tasks(path,n):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids=[int(r['task_id']) for r in rows]
    assert len(ids)==len(set(ids))==n,(path,len(ids))
    return ids

templates={'smoke3':extract(cpu/'eval_sft_step_300_smoke3.sh'),
           'dev50':extract(cpu/'eval_sft_step_300_dev50.sh')}
final_sft=sft/'evaluation/final200_clean/sft_step_300'
final_summary=json.loads((final_sft/'summary.json').read_text())
dev_summary=json.loads((cpu/'evaluation/sft_step_300/dev50/summary.json').read_text())
assert final_summary['strict_successes']==134 and final_summary['expected_tasks']==200
final_benchmark=Path(final_summary['protocol']['benchmark'])
if not final_benchmark.is_absolute(): final_benchmark=root/final_benchmark
assert digest(final_benchmark)=='d99112a20ef47534c27a32e4b38229bf048dcc6b06fef2e3e919aac3093662f5'
commands={}; benchmarks={}; protocols={}
for suite,n in (('smoke3',3),('dev50',50),('final200_clean',200)):
    argv=list(templates['smoke3' if suite=='smoke3' else 'dev50'])
    argv[0]=str(root/'.venv/bin/python')
    if suite=='final200_clean': replace(argv,'--benchmark',final_benchmark)
    replace(argv,'--model',model_tag)
    replace(argv,'--output',eval_root/suite/'trajectories.jsonl')
    replace(argv,'--summary',eval_root/suite/'summary.json')
    args=parsed(argv)
    benchmark=args.benchmark if args.benchmark.is_absolute() else root/args.benchmark
    ids=tasks(benchmark,n)
    if suite=='smoke3': assert set(ids)=={302,2718,18534}
    if suite=='dev50': assert set(ids)==set(tasks(root/'data/grpo/validation.jsonl',50))
    assert args.base_url=='http://127.0.0.1:5700'
    assert args.llm_base_url.rstrip('/')=='http://127.0.0.1:8000/v1'
    reference=(final_summary if suite=='final200_clean' else dev_summary)['protocol']
    for key,value in reference.items():
        if key in ('benchmark','model','reward_contract'): continue
        assert hasattr(args,key),f'未知的已封存协议字段：{key}'
        assert getattr(args,key)==value,(suite,key,getattr(args,key),value)
    commands[suite]=argv
    benchmarks[suite]={'path':str(benchmark),'sha256':digest(benchmark),'task_ids':ids}
    protocols[suite]={k:v for k,v in vars(args).items()
        if k not in ('benchmark','output','summary','api_key','model')}

sources=[root/'scripts/evaluate_shop_benchmark.py',root/'scripts/merge_lora_adapter.py',
    root/'scripts/serve_model.sh',root/'scripts/build_glm_report.py',
    *sorted((root/'src/shopping_grpo/evaluation').glob('*.py')),
    *sorted((root/'src/shopping_grpo/environment').glob('*.py'))]
verl=root/'.venv/lib/python3.12/site-packages/verl/model_merger'
for name in ('base_model_merger.py','fsdp_model_merger.py'):
    assert (verl/name).is_file(); sources.append(verl/name)
assert 'lora_train_meta.json' in (verl/'base_model_merger.py').read_text()
assert 'lora_adapter' in (verl/'base_model_merger.py').read_text()
assert '--use_cpu_initialization' in (verl/'base_model_merger.py').read_text(), '本机导出器缺少预期的 CPU 初始化参数，先核对源码。'
post.mkdir(parents=True,exist_ok=True)
export.mkdir(parents=True,exist_ok=True)
record={'project':str(root),'post':str(post),'checkpoint':str(checkpoint),'actor':str(actor),
    'raw_export':str(raw),'merged_model':str(merged),'model_tag':model_tag,'evaluation_root':str(eval_root),
    'commands':commands,'benchmarks':benchmarks,'protocols':protocols,
    'sft_final_dir':str(final_sft),'sft_dev_dir':str(cpu/'evaluation/sft_step_300/dev50'),
    'source_sha256':{str(p):digest(p) for p in sources}}

runner=r'''import argparse, collections, fcntl, hashlib, json, os, subprocess, sys, time
from pathlib import Path
from urllib.request import urlopen
prep=Path(__file__).resolve().parent
spec=json.loads((prep/'evaluation_spec.json').read_text())
parser=argparse.ArgumentParser()
parser.add_argument('suite',choices=['smoke3','dev50','final200_clean'])
parser.add_argument('--launch',action='store_true')
parser.add_argument('--shutdown-after-exit',action='store_true')
args=parser.parse_args()
suite=args.suite
out=Path(spec['evaluation_root'])/suite
previous={'dev50':'smoke3','final200_clean':'dev50'}.get(suite)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''): h.update(chunk)
    return h.hexdigest()
def preflight():
    assert (prep/'export_receipt.json').is_file(),'导出验收尚未通过。'
    if previous:
        prior=Path(spec['evaluation_root'])/previous/'passed.json'
        assert prior.is_file() and json.loads(prior.read_text())['passed'],prior
    for name,expected in spec['source_sha256'].items():
        assert digest(name)==expected,name
    benchmark=spec['benchmarks'][suite]
    assert digest(benchmark['path'])==benchmark['sha256']
    with urlopen('http://127.0.0.1:8000/v1/models',timeout=10) as response:
        models=json.load(response)['data']
    found=[m for m in models if m['id']==spec['model_tag']]
    assert len(found)==1 and Path(found[0]['root']).resolve()==Path(spec['merged_model']).resolve(),found
    assert int(found[0]['max_model_len'])==24576,found

if args.launch:
    preflight()
    out.mkdir(parents=True,exist_ok=True)
    # 独占创建日志；重复提交不会覆盖已有结果或启动第二次。
    with (out/'driver.log').open('x') as log:
        command=[sys.executable,str(Path(__file__).resolve()),suite]
        if args.shutdown_after_exit: command.append('--shutdown-after-exit')
        env=dict(os.environ,PYTHONPATH=str(Path(spec['project'])/'src'),OMP_NUM_THREADS='1')
        proc=subprocess.Popen(command,cwd=spec['project'],env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (out/'launcher.pid').write_text(str(proc.pid))
    print('后台评测 PID=',proc.pid,'日志：',out/'driver.log',flush=True)
    raise SystemExit(0)

started=time.time(); code=1
with (out/'evaluation.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        preflight()
        assert not (out/'trajectories.jsonl').exists(),'已有轨迹，不能自动重跑或覆盖。'
        with (out/'eval.log').open('x') as log:
            proc=subprocess.run(spec['commands'][suite],cwd=spec['project'],
                stdout=log,stderr=subprocess.STDOUT)
        assert proc.returncode==0,f'评测退出码 {proc.returncode}，检查 eval.log。'
        rows=[json.loads(line) for line in (out/'trajectories.jsonl').read_text().splitlines() if line.strip()]
        expected=spec['benchmarks'][suite]['task_ids']
        ids=[int(r['task_id']) for r in rows]
        assert len(ids)==len(set(ids))==len(expected) and set(ids)==set(expected),ids
        abnormal=[{'task_id':r['task_id'],'error':r.get('error'),'release_error':r.get('release_error')}
            for r in rows if r.get('release_error') or
            (r.get('error') and r['error'].get('type')!='ContextBudgetError')]
        print('非上下文预算类执行异常：',json.dumps(abnormal,ensure_ascii=False),flush=True)
        assert not abnormal,'先处理执行异常，不启动下一套任务。'
        if suite=='smoke3': assert sum(len(r.get('steps') or []) for r in rows)>0
        from shopping_grpo.evaluation.summary import summarize_trajectories
        calculated=summarize_trajectories(expected,rows)
        summary=json.loads((out/'summary.json').read_text())
        for key,value in calculated.items(): assert summary[key]==value,key
        assert not summary['missing_tasks']
        protocol=summary['protocol']
        for key,value in spec['protocols'][suite].items():
            if key in ('base_url','llm_base_url','timeout'): continue
            assert protocol[key]==value,(key,protocol.get(key),value)
        with (out/'report.log').open('w') as log:
            subprocess.run([sys.executable,'scripts/build_eval_report.py','--run-dir',str(out)],
                cwd=spec['project'],stdout=log,stderr=subprocess.STDOUT,check=True)
        assert (out/'report.html').is_file()
        print('严格成功：',summary['strict_successes'],'/',summary['expected_tasks'],flush=True)
        for key in ('strict_success_rate','purchase_success_rate','done_rate','reward_valid_rate',
            'mean_final_reward','average_steps','reward_type_counts','status_counts','guard_reason_counts'):
            print(key,':',summary[key],flush=True)
        (out/'passed.json').write_text(json.dumps({'passed':True,'suite':suite,
            'trajectories_sha256':digest(out/'trajectories.jsonl'),'summary_sha256':digest(out/'summary.json')},indent=2))
        code=0
    except BaseException:
        import traceback
        traceback.print_exc()
    finally:
        ended=time.time()
        (out/'exit_status.json').write_text(json.dumps({'exit_code':code,'started_unix':started,
            'ended_unix':ended,'elapsed_seconds':ended-started},indent=2))
        print('EVAL_EXIT_CODE=',code,flush=True)
        if args.shutdown_after_exit:
            print('本轮已退出，60 秒后请求关机；可创建本轮 auto_shutdown.cancel 取消。',flush=True)
            for _ in range(12):
                if (out/'auto_shutdown.cancel').exists(): break
                time.sleep(5)
            else:
                if not (out/'auto_shutdown.cancel').exists():
                    (out/'shutdown_request.json').write_text(json.dumps({'requested_unix':time.time(),'eval_exit_code':code}))
                    os.sync()
                    subprocess.run(['/usr/bin/shutdown'],check=True)
sys.exit(code)
'''
ast.parse(runner)
(post/'run_eval.py').write_text(runner)
session=''.join(f'export {k}={shlex.quote(v)}\n' for k,v in {
    'PROJECT':str(root),'GRPO_RUN':str(run),'POST':str(post),'ACTOR':str(actor),
    'RAW_EXPORT':str(raw),'GRPO_MERGED':str(merged),'EVAL_ROOT':str(eval_root),
    'MODEL_TAG':model_tag,'PYTHONPATH':str(root/'src'),'OMP_NUM_THREADS':'1',
    'VLLM_USE_FLASHINFER_SAMPLER':'0','SERVED_MODEL_NAME':model_tag,'LLM_PORT':'8000',
    'SHOPSIM_BASE_URL':'http://127.0.0.1:5700'}.items())
(post/'session.env').write_text(session)
(post/'evaluation_spec.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
print('候选与独立评测协议已固定。准备目录：',post)
PY
then
  source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/posttrain_step500_v1/session.env
  cd "$PROJECT"
else
  echo '准备失败，先处理上方错误；不要继续。'
fi
```

## 2. 导出 FSDP，随后合并 GRPO LoRA（本次使用有卡模式的内存配额）

veRL 的导出结果含一份底座和 `lora_adapter/`。必须继续合并，才能得到完整的 GRPO 推理模型；直接服务第一阶段的底座会漏掉 GRPO adapter。本段不更新模型、不再训练。为防多次复制覆盖，已完成的阶段会校验并复用；不完整的非空目录则停止，保留错误现场。

```bash
set +e
set +u
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/posttrain_step500_v1/session.env
cd "$PROJECT"
if .venv/bin/python -u - <<'PY'
import hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
post=Path(os.environ['POST'])
spec=json.loads((post/'evaluation_spec.json').read_text())
actor=Path(spec['actor']); raw=Path(spec['raw_export']); merged=Path(spec['merged_model'])
limit=Path('/sys/fs/cgroup/memory.max')
if limit.exists() and limit.read_text().strip()!='max':
    assert int(limit.read_text())>=110*2**30,'本次请使用至少 110 GiB 内存配额；2 GiB 无卡实例不能导出。'
if (post/'export_receipt.json').exists():
    receipt=json.loads((post/'export_receipt.json').read_text())
    assert receipt['merged_model']==str(merged) and (merged/'config.json').is_file()
    print('本轮导出已验收，复用现有模型：',merged)
    raise SystemExit(0)
bytes_in=(actor/'model_world_size_1_rank_0.pt').stat().st_size
required=2*bytes_in+5*2**30
print(f'导出磁盘可用 {shutil.disk_usage(actor).free/2**30:.2f} GiB；本次保守要求 {required/2**30:.2f} GiB')
assert shutil.disk_usage(actor).free>=required,'空间不足，不删除 checkpoint。'

if not (post/'fsdp_export_done.json').exists():
    assert not raw.exists() or not any(raw.iterdir()),'第一阶段目录非空且无完成记录，先查 fsdp_export.log。'
    code="""import runpy,sys,tempfile
from pathlib import Path
import torch.distributed as dist
with tempfile.TemporaryDirectory() as td:
    dist.init_process_group('gloo',init_method=Path(td,'init').as_uri(),rank=0,world_size=1)
    try:
        sys.argv=['verl.model_merger','merge','--backend','fsdp','--local_dir',sys.argv[1],
            '--target_dir',sys.argv[2],'--trust-remote-code','--use_cpu_initialization']
        runpy.run_module('verl.model_merger',run_name='__main__')
    finally:
        dist.destroy_process_group()
"""
    with (post/'fsdp_export.log').open('w') as log:
        subprocess.run([sys.executable,'-u','-c',code,str(actor),str(raw)],
            cwd=spec['project'],stdout=log,stderr=subprocess.STDOUT,check=True)
    adapter=raw/'lora_adapter'
    assert (adapter/'adapter_model.safetensors').is_file() and (raw/'config.json').is_file()
    config=json.loads((adapter/'adapter_config.json').read_text())
    assert int(config['r'])==16 and int(config['lora_alpha'])==32,config
    (post/'fsdp_export_done.json').write_text(json.dumps({'raw_export':str(raw),'rank':16,'alpha':32},indent=2))

adapter=raw/'lora_adapter'
import torch
from safetensors import safe_open
count=0; nonzero_b=0
with safe_open(str(adapter/'adapter_model.safetensors'),framework='pt',device='cpu') as handle:
    for key in handle.keys():
        tensor=handle.get_tensor(key)
        assert torch.isfinite(tensor).all(),key
        count+=1
        if '.lora_B.' in key: nonzero_b+=bool(torch.count_nonzero(tensor))
assert count>0 and nonzero_b>0,'导出的 adapter 没有非零 LoRA B 参数。'
assert not merged.exists() or not any(merged.iterdir()),'完整模型目录非空且未验收；先查 adapter_merge.log。'
with (post/'adapter_merge.log').open('w') as log:
    subprocess.run([sys.executable,'-u','scripts/merge_lora_adapter.py','--base-model',str(raw),
        '--adapter',str(adapter),'--output',str(merged),'--bf16'],cwd=spec['project'],
        stdout=log,stderr=subprocess.STDOUT,check=True)
assert (merged/'config.json').is_file()
files=sorted(merged.glob('*.safetensors'))
assert files,'没有完整模型权重。'
for path in files:
    with safe_open(str(path),framework='pt',device='cpu') as handle:
        assert not any('lora_' in key for key in handle.keys()),'仍有未合并 LoRA 参数。'
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
receipt={'operation':'GRPO FSDP export followed by PEFT merge_and_unload',
    'checkpoint':spec['checkpoint'],'raw_export':str(raw),'adapter':str(adapter),
    'merged_model':str(merged),'lora_rank':16,'lora_alpha':32,'adapter_tensor_count':count,
    'actor_model_sha256':digest(actor/'model_world_size_1_rank_0.pt'),
    'merged_weights_sha256':{p.name:digest(p) for p in files}}
(post/'export_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print('GRPO step 500 完整模型导出并合并完成：',merged)
print('adapter 参数张量数：',count,'；LoRA rank/alpha：16/32')
PY
then
  echo '导出完成，可以启动环境和模型服务。'
else
  echo '导出失败，保留现场，查看下方日志，不继续评测。'
fi
tail -n 25 "$POST/fsdp_export.log"
tail -n 25 "$POST/adapter_merge.log"
```

这里复用了项目的 `merge_lora_adapter.py` 作为 PEFT 通用合并工具；该工具生成的旧 `merge_manifest.json` 文案仍提到 SFT/GRPO 起点，本轮准确的 GRPO 导出来源与用途记录在 `POST/export_receipt.json`。

## 3. 启动 ShopSimulator 并验证环境（有卡模式，环境自身不使用 GPU）

```bash
set +e
set +u
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/posttrain_step500_v1/session.env
cd "$PROJECT"
if .venv/bin/python - <<'PY'
import socket
with socket.create_connection(('127.0.0.1',5700),timeout=2):pass
PY
then
  echo '5700 已监听，下面核对接口。'
else
  nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 SHOPSIM_PORT=5700 \
    SHOPSIM_ENV_SLOTS=8 SHOP_MAX_STEPS=35 bash scripts/start_environment.sh \
    > "$POST/environment.log" 2>&1 < /dev/null &
fi
if .venv/bin/python - <<'PY'
import json,os,socket,time
from pathlib import Path
from contextlib import ExitStack
from shopping_grpo.environment.client import ShopAgentEnv
deadline=time.monotonic()+120
while True:
    try:
        with socket.create_connection(('127.0.0.1',5700),timeout=2):break
    except OSError:
        if time.monotonic()>=deadline:raise SystemExit('环境未就绪，检查 environment.log。')
        time.sleep(2)
with ExitStack() as stack:
    indices=[]
    for _ in range(8):
        env=stack.enter_context(ShopAgentEnv('http://127.0.0.1:5700'))
        result=env.reset(302)
        assert result['environment_version']=='shopsimulator-environment-v2.1'
        indices.append(env.env_idx)
    assert len(set(indices))==8
(Path(os.environ['POST'])/'environment_ready.json').write_text(json.dumps({'passed':True}))
print('8 个槽位已实测可用且全部释放：',indices)
PY
then
  echo '环境检查通过。'
else
  echo '环境检查未通过，先排查，不启动评测。'
fi
```

## 4. 启动合并后的 GRPO 模型服务（需要 GPU）

沿用项目 `serve_model.sh`：上下文 24,576、`qwen3_coder` 工具解析器；保留已验证的 `VLLM_USE_FLASHINFER_SAMPLER=0`。不要再启动训练或另一个 SFT 模型服务。本段不会终止未知的端口占用；8000 已占用时先核对已有服务。

```bash
set +e
set +u
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/posttrain_step500_v1/session.env
cd "$PROJECT"
nvidia-smi
if .venv/bin/python - <<'PY'
import json,os,socket
from pathlib import Path
assert (Path(os.environ['POST'])/'export_receipt.json').is_file()
assert json.loads((Path(os.environ['POST'])/'environment_ready.json').read_text())['passed']
with socket.socket() as sock:
    assert sock.connect_ex(('127.0.0.1',8000))!=0,'8000 已占用，先核对已有模型服务。'
import torch
assert torch.cuda.is_available()
free,_=torch.cuda.mem_get_info()
assert free>=90*2**30,f'显存未空闲：仅余 {free/2**30:.1f} GiB'
PY
then
  nohup env OMP_NUM_THREADS=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
    SERVED_MODEL_NAME="$MODEL_TAG" LLM_PORT=8000 \
    bash scripts/serve_model.sh "$GRPO_MERGED" \
    > "$POST/vllm.log" 2>&1 < /dev/null &
  echo "vLLM 启动 PID=$!"
else
  echo '本段未启动服务；检查上方原因。'
fi
tail -n 35 "$POST/vllm.log"
```

启动可能需要几分钟。之后执行下面核对命令；若尚未就绪，查看 `vllm.log`，不要重复启动。只有输出“模型服务身份检查通过”才继续。

```bash
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/posttrain_step500_v1/session.env
cd "$PROJECT"
.venv/bin/python - <<'PY'
import json,os
from pathlib import Path
from urllib.request import urlopen
with urlopen('http://127.0.0.1:8000/v1/models',timeout=10) as response:
    result=json.load(response)
print(json.dumps(result,ensure_ascii=False,indent=2))
models=[m for m in result['data'] if m['id']==os.environ['MODEL_TAG']]
assert len(models)==1
assert Path(models[0]['root']).resolve()==Path(os.environ['GRPO_MERGED']).resolve()
assert models[0]['max_model_len']==24576
print('模型服务身份检查通过。')
PY
```

## 5. 后台跑 smoke3，并检查验收（需要 GPU 服务）

```bash
set +e
set +u
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/posttrain_step500_v1/session.env
cd "$PROJECT"
.venv/bin/python "$POST/run_eval.py" smoke3 --launch
tail -n 60 -F "$EVAL_ROOT/smoke3/driver.log"
```

看到 `EVAL_EXIT_CODE=0` 后按 Ctrl+C 退出日志查看器，核对：

```bash
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/posttrain_step500_v1/session.env
cat "$EVAL_ROOT/smoke3/exit_status.json"
cat "$EVAL_ROOT/smoke3/passed.json"
tail -n 30 "$EVAL_ROOT/smoke3/eval.log"
```

smoke3 用于链路检查，不要求三题全成功。出现退出码非 0、缺少通过记录或非上下文预算类执行异常时先处理；不要直接启动 Dev-50。后台评测不依赖本地电脑一直在线。

## 6. 后台跑独立 Dev-50（需要 GPU 服务）

这是使用此前 SFT 的独立评测脚本核对导出模型，结果不要求与训练内 29/50 完全相同。候选仍锁定 step 500；若存在执行异常或异常大的偏差，先排查导出与协议，不根据 Final 改选。

```bash
set +e
set +u
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/posttrain_step500_v1/session.env
cd "$PROJECT"
.venv/bin/python "$POST/run_eval.py" dev50 --launch
tail -n 60 -F "$EVAL_ROOT/dev50/driver.log"
```

看到 `EVAL_EXIT_CODE=0` 后按 Ctrl+C，再核对：

```bash
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/posttrain_step500_v1/session.env
cat "$EVAL_ROOT/dev50/exit_status.json"
cat "$EVAL_ROOT/dev50/passed.json"
tail -n 35 "$EVAL_ROOT/dev50/driver.log"
```

HTML 报告位于 `EVAL_ROOT/dev50/report.html`。固定脚本会拦截缺题、重复任务、环境释放异常和非 ContextBudgetError 执行异常。上下文超限仍计失败，不从 50 题中剔除。

## 7. 后台跑 Final-200 Clean（需要 GPU 服务）

只有 Dev-50 验收通过后才启动。同一 Clean 任务文件、同一套 SFT 评测参数，完整分母为 200。该命令**默认在本轮评测退出、写完验收和退出记录后，等待约 60 秒请求关机**；成功或报错退出均会触发，避免空转。若希望保留服务器在线，启动时去掉 `--shutdown-after-exit`。不使用旧训练关机脚本。

```bash
set +e
set +u
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/posttrain_step500_v1/session.env
cd "$PROJECT"
.venv/bin/python "$POST/run_eval.py" final200_clean --launch --shutdown-after-exit
tail -n 60 -F "$EVAL_ROOT/final200_clean/driver.log"
```

如需取消本轮自动关机，在倒计时结束前另一个终端执行（不影响评测）：

```bash
touch /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/evaluation/step500_independent_v1/final200_clean/auto_shutdown.cancel
```

驱动日志主要在评测结束后打印汇总；需要看当前任务输出，可另开终端 `tail -n 30 -F "$EVAL_ROOT/final200_clean/eval.log"`，先恢复本次 session.env。若最终失败，保留原始轨迹和日志排查；启动器不会自动覆盖、重试或挑选性重跑失败题。

## 8. 验收 Final，并与 SFT 做逐任务比较（无卡可做）

若第 7 节已自动关机，可无卡开机执行。以下重新从两份原始轨迹计算成功任务，检查一致的 200 题分母；报告两模型都成功、仅 SFT 成功、仅 GRPO 成功和都失败。不读取其他 GRPO checkpoint 的 Final 结果。

```bash
set +e
set +u
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/posttrain_step500_v1/session.env
cd "$PROJECT"
.venv/bin/python - <<'PY'
import json,os
from pathlib import Path
from shopping_grpo.evaluation.summary import summarize_trajectories
post=Path(os.environ['POST'])
spec=json.loads((post/'evaluation_spec.json').read_text())
grpo=Path(spec['evaluation_root'])/'final200_clean'
status_path=grpo/'exit_status.json'
assert status_path.is_file(),'尚无退出状态，查看本轮 driver.log。'
status=json.loads(status_path.read_text())
print('Final 评测退出状态：',status)
assert status['exit_code']==0 and json.loads((grpo/'passed.json').read_text())['passed']
sft=Path(spec['sft_final_dir'])
expected=spec['benchmarks']['final200_clean']['task_ids']
def load(folder):
    rows=[json.loads(line) for line in (folder/'trajectories.jsonl').read_text().splitlines() if line.strip()]
    ids=[int(r['task_id']) for r in rows]
    assert len(ids)==len(set(ids))==200 and set(ids)==set(expected),folder
    summary=summarize_trajectories(expected,rows)
    stored=json.loads((folder/'summary.json').read_text())
    for key,value in summary.items():assert stored[key]==value,(folder,key)
    return summary
a,b=load(sft),load(grpo)
assert a['strict_successes']==134
print('\n指标 | SFT step 300 | GRPO step 500')
for key in ('strict_successes','strict_success_rate','purchase_success_rate','done_rate',
    'reward_valid_rate','mean_final_reward','average_steps'):
    print(key,':',a[key],'|',b[key])
sa,sb=set(a['strict_success_task_ids']),set(b['strict_success_task_ids'])
all_ids=set(expected)
paired={'both_success':sorted(sa&sb),'only_sft_success':sorted(sa-sb),
    'only_grpo_success':sorted(sb-sa),'both_fail':sorted(all_ids-sa-sb)}
print('\n逐任务配对：')
for key,ids in paired.items():print(key,'数量：',len(ids),'任务：',ids)
delta=100*(b['strict_success_rate']-a['strict_success_rate'])
print(f'GRPO 相对 SFT 严格成功率变化：{delta:+.2f} 个百分点')
print('GRPO 结局分布：',b['reward_type_counts'])
print('GRPO Guard 拒绝：',b['guard_reason_counts'])
result={'sft':a,'grpo':b,'paired':paired,'strict_success_delta_percentage_points':delta,
    'grpo_checkpoint':spec['checkpoint'],'independent_protocol':'same as frozen SFT commands'}
(post/'sft_vs_grpo_final200_clean.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print('GRPO HTML：',grpo/'report.html')
print('逐任务比较：',post/'sft_vs_grpo_final200_clean.json')
print('Final 完整性与比较完成；结果好坏均保留，不据此更换 checkpoint。')
PY
```

恢复终端时统一使用第 1 节生成的 `posttrain_step500_v1/session.env`；不能 source 回旧训练会话替代它。所有评测均后台运行，`tail -F` 的 Ctrl+C 只结束查看器。

导出分成底座与 adapter 的行为参考 [veRL 0.8 model merger](https://github.com/verl-project/verl/blob/v0.8.0/verl/model_merger/base_model_merger.py)。本次记录两阶段导出与 rank/alpha，服务目标仅为最终 `merged/` 目录。
