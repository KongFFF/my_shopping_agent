# 正式 GRPO：从 step 350 恢复至 step 500

2026-10-25。依据本次诊断，修复版运行记录了连续的 1–358 次更新，没有非有限标量指标；最新保存点是 350，0/50/…/350 的验证均有 50 条。没有最终退出状态或本会话自动关机请求记录，结合用户账单，按平台欠费停机后的断点恢复处理。

恢复的是正式 GRPO 的模型、优化器、调度器和框架保存的随机状态，并恢复 `data.pt` 中的数据迭代状态。**从 350 补做 150 次更新，总上限保持 500**。351–358 的未保存更新需要重算；旧日志保留，验收时不能将它们重复累计。异步采样和进程重启可能改变后续具体轨迹，不承诺与未中断运行逐位一致。

下面命令在 AutoDL 的 Bash 终端执行。本文件未在云端代执行训练；已完成本地语法和关键逻辑检查。

## 1. 准备独立恢复会话（无卡可做）

不修改训练源码，不删除旧 checkpoint。新日志和新保存点写入 `formal_resume350_v1`。本轮预计新增 400、450、500 三份保存点；空间门槛按实际已有单份体积 × 3 + 15 GiB（日志和余量）计算，约 46 GiB **可用空间**。不再套用从头训练的空间门槛；后续模型导出的空间另外检查。由于新旧目录分开，不能假定新会话的保留上限会清理旧目录；本流程允许新旧合计保留 13 份。

若空间不足，本段会在创建新配置前停止并打印缺口。不要绕过检查；无需先盲目扩容到某个总容量。

```bash
set +e
set +u
if /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/.venv/bin/python - <<'PY'
import argparse, ast, hashlib, importlib.util, json, os, shlex, shutil, sys, zipfile
from pathlib import Path

root = Path('/root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon')
run = root / 'outputs/runs/grpo_baseline_20261022_202816'
old_prep = run / 'preparation/formal_memfix_v1'
prep = run / 'preparation/formal_resume350_v1'
out = run / 'formal_resume350_v1'
old = json.loads((old_prep / 'launch_spec.json').read_text())
checkpoint = Path(old['output']) / 'global_step_350'
assert checkpoint == run / 'formal_memfix_v1/global_step_350'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

for name, expected in old['source_sha256'].items():
    assert digest(name) == expected, f'源码与昨晚封存版本不一致：{name}'

if (prep / 'launch_spec.json').exists():
    saved = json.loads((prep / 'launch_spec.json').read_text())
    assert saved['resume_from'] == str(checkpoint) and saved['output'] == str(out)
    for name in ('session.env', 'launch_formal.py', 'auto_shutdown.py', 'math_probe.json'):
        assert (prep / name).is_file(), name
    print('恢复配置已经生成，不重复覆盖。恢复下方 session.env 后继续；已有启动日志时不要重复启动。')
    raise SystemExit(0)

assert not out.exists() or (out.is_dir() and not any(out.iterdir())), '新输出已有内容，先检查。'
# 旧 launcher / 关机监控不能与新会话并行。
targets = {str(old_prep / 'launch_formal.py').encode(), str(old_prep / 'auto_shutdown.py').encode()}
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        args = (proc / 'cmdline').read_bytes().split(b'\0')
    except OSError:
        continue
    assert not targets.intersection(args), f'旧会话进程仍在运行：PID={proc.name}'

required_files = ['actor/model_world_size_1_rank_0.pt', 'actor/optim_world_size_1_rank_0.pt',
    'actor/extra_state_world_size_1_rank_0.pt', 'actor/fsdp_config.json', 'data.pt']
for name in required_files:
    path = checkpoint / name
    assert path.is_file() and path.stat().st_size > 0, path
    if path.suffix == '.pt':
        with zipfile.ZipFile(path) as archive:
            assert archive.namelist(), path

observed = sum(p.stat().st_size for p in checkpoint.rglob('*') if p.is_file())
required = observed * 3 + 15 * 2**30
available = shutil.disk_usage(root).free
print(f'当前磁盘可用：{available/2**30:.2f} GiB；本次续跑要求：{required/2**30:.2f} GiB')
assert available >= required, f'尚缺 {(required-available)/2**30:.2f} GiB；未创建恢复配置，不删除旧文件。'

baseline = json.loads((run / 'preparation/baseline_plan.json').read_text())
for rel, expected in baseline['data_sha256'].items():
    assert digest(root / rel) == expected, rel
for name, expected in baseline['model_sha256'].items():
    assert digest(Path(baseline['model']) / name) == expected, name

# 从昨晚的真实命令派生，基础模型仍为同一 SFT，GRPO 状态由 resume_path 恢复。
argv = list(old['command'])
split = argv.index('--')
prefix, overrides = argv[:split], argv[split+1:]
prefix[prefix.index('--output')+1] = str(out)
prefix[prefix.index('--experiment-name')+1] = 'shopping-grpo-formal-resume350-v1'
changes = {
    'trainer.resume_mode': 'resume_path',
    'trainer.resume_from_path': json.dumps(str(checkpoint)),
    'trainer.del_local_ckpt_after_load': 'false',
    'trainer.val_before_train': 'false',
    'trainer.validation_data_dir': json.dumps(str(out / 'validation')),
}
overrides = [v for v in overrides if v.lstrip('+').split('=',1)[0] not in changes]
overrides += ['++' + k + '=' + v for k, v in changes.items()]
argv = prefix + ['--'] + overrides
env = dict(old['env'])
env.update(GRPO_OUTPUT_DIR=str(out), SHOPPING_GRPO_DIAGNOSTICS_PATH=str(out / 'training_diagnostics.jsonl'))
os.environ.update(env)

loader = importlib.util.spec_from_file_location('resume350_launcher', root / 'scripts/train_grpo.py')
module = importlib.util.module_from_spec(loader)
loader.loader.exec_module(module)
def option(name):
    return prefix[prefix.index(name)+1]
args = argparse.Namespace(model=Path(option('--model')), train_data=Path(option('--train-data')),
    val_data=Path(option('--val-data')), config=Path(option('--config')), output=out,
    env_url=option('--env-url'), logger='console', experiment_name=option('--experiment-name'),
    hydra_overrides=overrides)
command, _ = module.build_command(args)

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
with initialize_config_dir(version_base=None, config_dir=str(root / 'configs')):
    cfg = compose(config_name='grpo', overrides=command[5:])
assert cfg.trainer.total_training_steps == 500
assert cfg.trainer.save_freq == cfg.trainer.test_freq == 50
assert cfg.trainer.resume_mode == 'resume_path' and cfg.trainer.resume_from_path == str(checkpoint)
assert not cfg.trainer.del_local_ckpt_after_load and not cfg.trainer.val_before_train
contents = OmegaConf.select(cfg, 'actor_rollout_ref.actor.checkpoint.load_contents')
assert contents is None or set(contents) == {'model', 'optimizer', 'extra'}, contents

def differences(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in a.keys() | b.keys():
            name = f'{path}.{key}' if path else key
            if key not in a or key not in b:
                result.append(name)
            else:
                result.extend(differences(a[key], b[key], name))
        return result
    return [] if a == b else [path]

before = OmegaConf.to_container(OmegaConf.load(old_prep / 'composed_config.yaml'), resolve=True)
after = OmegaConf.to_container(cfg, resolve=True)
allowed = {'trainer.resume_mode', 'trainer.resume_from_path', 'trainer.del_local_ckpt_after_load',
    'trainer.val_before_train', 'trainer.validation_data_dir', 'trainer.default_local_dir',
    'trainer.experiment_name', 'actor_rollout_ref.rollout.trace.experiment_name'}
# trace 名称由 trainer.experiment_name 插值而来；仅允许这一关联的记录字段变化。
for resolved in (before, after):
    assert (resolved['actor_rollout_ref']['rollout']['trace']['experiment_name']
            == resolved['trainer']['experiment_name']), 'trace 名称未与实验名称保持一致，需核对。'
changed = differences(before, after)
assert set(changed) <= allowed, f'恢复配置中存在预期外差异：{changed}'

verl = root / '.venv/lib/python3.12/site-packages/verl'
trainer_tree = ast.parse((verl / 'trainer/ppo/ray_trainer.py').read_text())
load_fn = next(n for n in ast.walk(trainer_tree) if isinstance(n, ast.FunctionDef) and n.name == '_load_checkpoint')
load_source = ast.unparse(load_fn)
assert all(s in load_source for s in ('resume_path', 'resume_from_path',
    'actor_rollout_wg.load_checkpoint', 'train_dataloader.load_state_dict'))
manager = (verl / 'utils/checkpoint/checkpoint_manager.py').read_text()
assert 'checkpoint_load_contents = ["model", "optimizer", "extra"]' in manager

probe = json.loads((old_prep / 'math_probe.json').read_text())
assert probe['passed'] and probe['engine_sha256'] == old['source_sha256'][probe['engine_file']]
driver = (old_prep / 'launch_formal.py').read_text()
ast.parse(driver)
assert not any(isinstance(n, ast.Import) and any(a.name == 'torch' for a in n.names)
               for n in ast.parse(driver).body), '不能恢复为长期占用 CUDA 的旧 launcher。'
assert (old_prep / 'auto_shutdown.py').is_file(), '缺少自动关机脚本，先检查旧准备目录。'
ast.parse((old_prep / 'auto_shutdown.py').read_text())

prep.mkdir(parents=True, exist_ok=True)
# 只取消旧会话监控；不会给新会话创建取消标记。
(old_prep / 'auto_shutdown.cancel').touch()
for name in ('launch_formal.py', 'auto_shutdown.py', 'math_probe.json'):
    shutil.copy2(old_prep / name, prep / name)
record = dict(old)
record.update(command=argv, preflight=[sys.executable, str(root / 'scripts/check_grpo_runtime.py'), *command[5:]],
    env=env, output=str(out), required_free_bytes=required, resume_from=str(checkpoint),
    parent_preparation=str(old_prep), checkpoint_steps=[400,450,500],
    resume_step=350, remaining_updates=150, total_updates=500)
(prep / 'composed_config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
(prep / 'command.txt').write_text(shlex.join(argv) + '\n')
(prep / 'resume_receipt.json').write_text(json.dumps({
    'checkpoint': str(checkpoint), 'resume_step':350, 'target_step':500,
    'rolled_back_unsaved_steps':list(range(351,359)), 'config_differences':sorted(changed),
    'new_checkpoint_slots':3, 'required_free_bytes':required,
    'checkpoint_structure_only':True, 'weight_load_test_pending':True,
}, ensure_ascii=False, indent=2))
session = ''.join(f'export {k}={shlex.quote(v)}\n' for k,v in {
    'PROJECT':str(root), 'GRPO_RUN':str(run), 'FORMAL_PREP':str(prep), 'FORMAL_OUT':str(out),
    'PYTHONPATH':str(root/'src'), 'OMP_NUM_THREADS':'1', 'VLLM_USE_FLASHINFER_SAMPLER':'0',
    'PYTHONHASHSEED':'42', 'SHOPPING_LENGTH_SHAPING_ENABLE':'false',
    'SHOPSIM_BASE_URL':'http://127.0.0.1:5700'}.items())
(prep / 'session.env').write_text(session)
# launch_spec 最后写入，作为配置已准备好的标志。
(prep / 'launch_spec.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
print('恢复配置准备完成；尚未启动训练。配置差异：', sorted(changed))
print('恢复点：', checkpoint)
print('新输出：', out)
PY
then
  source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/formal_resume350_v1/session.env
  cd "$PROJECT"
  echo '第 1 步通过。'
else
  echo '准备失败，先处理上方具体错误，不继续启动。'
fi
```

第 350 步已做过完整 Dev-50，因此本次关闭“开始前再验证”，避免重复评测；400/450/500 的周期验证保留。`rollout.trace.experiment_name` 会随 `trainer.experiment_name` 插值变化，属于实验记录名称；本段同时核对修改前后两者相等，再允许这个字段变化。配置逐项比较仍拦截学习率、总步数、LoRA、batch、奖励或长度等预期外改动。若此前仅在这个差异检查处失败，尚未写入新会话配置，重新复制本节更新后的完整命令即可。checkpoint 的真正加载由后面的有卡训练执行；本步 ZIP 目录检查不能证明所有张量可加载。

## 2. 有卡开机，恢复会话并检查资源

仍使用约 95 GiB 显存的同档 GPU，CPU 配额至少沿用 110 GiB。GPU 检查在短命进程中执行，不会留下占显存的长期 launcher。每次换终端可重复本段的 `source`。

```bash
set +e
set +u
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/formal_resume350_v1/session.env
cd "$PROJECT"
nvidia-smi
if .venv/bin/python - <<'PY'
import json, os, shutil
from pathlib import Path
prep = Path(os.environ['FORMAL_PREP'])
(prep / 'resource_check.json').unlink(missing_ok=True)
spec = json.loads((prep / 'launch_spec.json').read_text())
free = shutil.disk_usage(spec['project']).free
print(f'磁盘可用 {free/2**30:.2f} GiB；要求 {spec["required_free_bytes"]/2**30:.2f} GiB')
assert free >= spec['required_free_bytes'], '可用磁盘不足。'
limit = Path('/sys/fs/cgroup/memory.max')
if limit.exists() and limit.read_text().strip() != 'max':
    print('CPU 内存配额 GiB：', int(limit.read_text())/2**30)
    assert int(limit.read_text()) >= 110 * 2**30, 'CPU 配额低于采用的 110 GiB。'
import torch
assert torch.cuda.is_available()
gpu_free, total = torch.cuda.mem_get_info()
print(f'GPU 可用/总量：{gpu_free/2**30:.2f}/{total/2**30:.2f} GiB')
assert gpu_free >= 90 * 2**30, 'GPU 有其他占用或规格不符。'
(prep / 'resource_check.json').write_text(json.dumps({'passed':True}))
print('资源检查通过。')
PY
then
  echo '可以执行第 3 步。'
else
  echo '检查失败，不启动训练。'
fi
```

## 3. 启动 ShopSimulator，验证 Ray 连接并运行预检（有卡模式）

不要另启 SFT 评测用的独立 vLLM 服务；GRPO 自己管理模型采样服务。环境端口是 5700。

```bash
set +e
set +u
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/formal_resume350_v1/session.env
cd "$PROJECT"
if .venv/bin/python - <<'PY'
import socket
with socket.create_connection(('127.0.0.1',5700),timeout=2): pass
PY
then
  echo '5700 已监听，下面实测环境。'
else
  nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 SHOPSIM_PORT=5700 \
    SHOPSIM_ENV_SLOTS=8 SHOP_MAX_STEPS=35 bash scripts/start_environment.sh \
    > "$FORMAL_PREP/environment.log" 2>&1 < /dev/null &
fi

if .venv/bin/python -u - <<'PY'
import json, os, socket, subprocess, time
from pathlib import Path
from contextlib import ExitStack
root = Path(os.environ['PROJECT'])
prep = Path(os.environ['FORMAL_PREP'])
(prep / 'environment_preflight.json').unlink(missing_ok=True)
assert json.loads((prep / 'resource_check.json').read_text())['passed']
deadline = time.monotonic()+120
while True:
    try:
        with socket.create_connection(('127.0.0.1',5700),timeout=2): break
    except OSError:
        if time.monotonic() >= deadline: raise SystemExit('环境启动超时，检查 environment.log。')
        time.sleep(2)
task = int(json.loads((root/'data/grpo/train.jsonl').read_text().splitlines()[0])['task_id'])
from shopping_grpo.environment.client import ShopAgentEnv
with ExitStack() as stack:
    indices = []
    for _ in range(8):
        env = stack.enter_context(ShopAgentEnv(os.environ['SHOPSIM_BASE_URL']))
        result = env.reset(task)
        assert result['environment_version'] == 'shopsimulator-environment-v2.1'
        indices.append(env.env_idx)
    assert len(set(indices)) == 8
print('8 个槽位均可申请并已释放：', indices)
import ray
ray.init(address='local',num_cpus=1,num_gpus=0,object_store_memory=128*1024**2,
    include_dashboard=False,runtime_env={'env_vars':{
        'PYTHONPATH':str(root/'src'),'SHOPSIM_BASE_URL':os.environ['SHOPSIM_BASE_URL'],
        'OMP_NUM_THREADS':'1','VLLM_USE_FLASHINFER_SAMPLER':'0'}})
try:
    @ray.remote(num_cpus=0.1)
    def probe(task):
        import os
        from shopping_grpo.environment.client import ShopAgentEnv
        with ShopAgentEnv(os.environ['SHOPSIM_BASE_URL']) as env:
            result = env.reset(task)
            assert result['environment_version'] == 'shopsimulator-environment-v2.1'
        assert env.env_idx is None
        assert os.environ['OMP_NUM_THREADS']=='1'
        assert os.environ['VLLM_USE_FLASHINFER_SAMPLER']=='0'
        return 'reset_and_release passed'
    print('Ray worker：',ray.get(probe.remote(task)))
finally:
    ray.shutdown()
spec = json.loads((prep/'launch_spec.json').read_text())
with (prep/'preflight.log').open('w') as log:
    status = subprocess.call(spec['preflight'],cwd=root,env=dict(os.environ,**spec['env']),
        stdout=log,stderr=subprocess.STDOUT)
print((prep/'preflight.log').read_text())
assert status == 0, f'运行时预检退出码：{status}'
(prep/'environment_preflight.json').write_text(json.dumps({'passed':True}))
print('续跑环境和运行时预检通过。')
PY
then
  echo '可以执行第 4 步。'
else
  echo '预检失败，不启动训练。'
fi
```

## 4. 后台续跑（有卡模式）

```bash
set +e
set +u
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/formal_resume350_v1/session.env
cd "$PROJECT"
if .venv/bin/python - <<'PY'
import json, os
from pathlib import Path
prep = Path(os.environ['FORMAL_PREP'])
for name in ('resource_check.json','environment_preflight.json'):
    assert json.loads((prep/name).read_text())['passed'], name
for name in ('driver.log','train.log','train.pid','exit_status.json'):
    assert not (prep/name).exists(), f'已有启动记录：{name}；先检查，不能重复启动。'
print('启动保护检查通过。')
PY
then
  nohup .venv/bin/python -u "$FORMAL_PREP/launch_formal.py" \
    > "$FORMAL_PREP/driver.log" 2>&1 < /dev/null &
  echo "后台 launcher PID=$!"
  sleep 3
  tail -n 30 "$FORMAL_PREP/driver.log"
else
  echo '未启动；查看上方错误。'
fi
```

随后查看训练日志：

```bash
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/formal_resume350_v1/session.env
tail -n 60 -F "$FORMAL_PREP/train.log"
```

预期加载路径包含 `formal_memfix_v1/global_step_350`，出现 `Setting global step to 350` / `Resuming from ...global_step_350`，并报告模型、优化器、随机状态和调度器的恢复。随后首个真正完成的优化器更新为 **351**。进度条可能从 `350/500` 开始显示，这是正确的。若出现从 0 开始、数据状态缺失等提示，先保存日志核查。

`Ctrl+C` 退出的是日志查看器，不影响后台训练。先退出日志查看器或另开终端，执行下一段挂上本次关机监控。

## 5. 为本次续跑挂自动关机（另一个终端，训练已进入启动流程后）

使用新目录的监控，不能运行旧 `formal_launch` 或 `formal_memfix_v1` 的监控。本段检查 launcher 已建立记录，且没有取消标记。仍按此前行为：训练正常或报错退出后，等待约 60 秒请求关机；外部强杀或欠费停机不能依赖脚本收尾。

```bash
set +e
set +u
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/formal_resume350_v1/session.env
cd "$PROJECT"
if .venv/bin/python - <<'PY'
import os
from pathlib import Path
prep = Path(os.environ['FORMAL_PREP'])
assert (prep/'train.pid').is_file(), '启动器尚未进入训练流程，先检查 driver.log。'
assert not (prep/'auto_shutdown.cancel').exists(), '存在取消标记，先检查。'
assert (prep/'auto_shutdown.py').is_file()
PY
then
  nohup .venv/bin/python -u "$FORMAL_PREP/auto_shutdown.py" \
    >> "$FORMAL_PREP/auto_shutdown.log" 2>&1 < /dev/null &
  sleep 2
  tail -n 25 "$FORMAL_PREP/auto_shutdown.log"
else
  echo '监控未启动，查看上方错误。'
fi
```

看到“监控已启动”且没有后续异常，才算挂上。新会话结束前可用下列命令取消自动关机，不影响训练：

```bash
touch /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/formal_resume350_v1/auto_shutdown.cancel
```

## 6. 进度检查与完成后的合并验收（无卡可做；训练进行时勿切换模式）

这里合并的是实验记录，不合并模型。旧运行仅采用 1–350；新运行采用 351–500。旧记录中的 351–358 保留为回滚历史，不计入有效训练链。验证记录应为旧 0/50/…/350 加新 400/450/500，共 11 次、每次 50 条。运行中打印“尚未完成”是正常的；完成后的基础验收不替代导出和模型评测。

```bash
set +e
set +u
/root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/.venv/bin/python - <<'PY'
import collections, json, math, numbers, zipfile
from pathlib import Path
root = Path('/root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon')
run = root/'outputs/runs/grpo_baseline_20261022_202816'
prep = run/'preparation/formal_resume350_v1'
old_out, new_out = run/'formal_memfix_v1', run/'formal_resume350_v1'

def finite(v):
    if isinstance(v,dict): return all(finite(x) for x in v.values())
    if isinstance(v,list): return all(finite(x) for x in v)
    return math.isfinite(v) if isinstance(v,numbers.Real) else True

chains = []
bad_lines, bad_metrics = [], []
last = None
for folder, low, high in ((old_out,1,350),(new_out,351,500)):
    steps = []
    path = folder/'training_diagnostics.jsonl'
    if path.exists():
        with path.open() as f:
            for line in f:
                if not line.strip(): continue
                try: event = json.loads(line)
                except json.JSONDecodeError:
                    bad_lines.append(str(path)); continue
                if event.get('event') != 'optimizer_step': continue
                step = int(event['global_step'])
                if not low <= step <= high:
                    if folder == new_out: bad_metrics.append(('unexpected_step',step))
                    continue
                steps.append(step)
                m = event['metrics']
                if not finite(m) or m.get('training/optimizer_updated') != 1:
                    bad_metrics.append(('metrics',step))
                last = {'step':step,'lr':m.get('actor/lr'),'grad_norm':m.get('actor/grad_norm')}
    print(folder.name,'有效更新数：',len(steps),'范围：',(steps[0],steps[-1]) if steps else None)
    chains.extend(steps)
print('有效训练链是否连续：', chains == list(range(1,len(chains)+1)))
print('最近更新：', last, '诊断坏行：',len(bad_lines),'异常指标：',bad_metrics[:10])
status_path = prep/'exit_status.json'
status = json.loads(status_path.read_text()) if status_path.exists() else None
print('续跑退出状态：',status)
if status is None:
    print('尚未记录最终退出状态；结合进程和日志确认。')
    raise SystemExit(0)
assert status['exit_code']==0, '续跑异常退出，先检查新会话 train.log。'
assert chains==list(range(1,501)) and not bad_lines and not bad_metrics

validation_counts = collections.Counter()
for folder, allowed in ((old_out,set(range(0,351,50))),(new_out,{400,450,500})):
    for path in (folder/'validation').glob('*.jsonl'):
        with path.open() as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                step = int(row['step'])
                assert step in allowed, (path,step)
                assert math.isfinite(float(row['score']))
                validation_counts[step] += 1
print('各步验证样本数：',dict(sorted(validation_counts.items())))
assert dict(validation_counts)=={s:50 for s in range(0,501,50)}
for step in range(50,501,50):
    folder = (old_out if step <=350 else new_out)/f'global_step_{step}'
    for name in ('actor/model_world_size_1_rank_0.pt','actor/optim_world_size_1_rank_0.pt',
                 'actor/extra_state_world_size_1_rank_0.pt','actor/fsdp_config.json','data.pt'):
        path = folder/name
        assert path.is_file() and path.stat().st_size>0, path
        if path.suffix=='.pt':
            with zipfile.ZipFile(path) as archive: assert archive.namelist(), path
print('基础验收通过：有效更新 1–500、10 个候选保存点、11 次各 50 条验证。')
print('下一步按 Dev-50 选择候选，导出后独立评测；不默认 step 500 最好。')
PY
```

恢复语义参考 [veRL 0.8 trainer 的 checkpoint 加载入口](https://github.com/verl-project/verl/blob/v0.8.0/verl/trainer/ppo/ray_trainer.py) 和 [FSDP 状态恢复实现](https://github.com/verl-project/verl/blob/v0.8.0/verl/utils/checkpoint/fsdp_checkpoint_manager.py)。实际运行以昨晚封存的本机源码及配置为准，本流程会校验源码哈希，并保留配置差异回执。
