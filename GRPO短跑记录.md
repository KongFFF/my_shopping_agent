# GRPO 短跑：修复后的执行流程与已完成结果

本节的目的：从选定的 SFT step 300 出发，新挂 LoRA，执行 3 次 GRPO 优化器更新，检查采样、奖励、反向传播、保存、参数变化和环境释放。

下面是把已验证修复提前后的**重新执行流程**，不是失败命令的合集。它使用新的输出目录，不覆盖已有实验。当前短跑已经通过，记录这些命令即可，**无需为了补记录再付费重跑**。下列新流程尚未在云端重新执行；本次实际完成的结果单独列在末尾。

## 前提与固定配方

- 使用有卡模式：本次为单张 RTX PRO 6000 Blackwell，约 95 GiB 可用总显存，CPU 内存限额 120 GiB。环境服务与验收不依赖 GPU，真实 GRPO 更新需要 GPU。
- 数据盘已由 50G 扩为 100G。新跑之前要求至少 **40 GiB 可用磁盘空间**：单份 FSDP checkpoint 约 11G，保留 2 份时，保存新文件与清理旧文件之间可能同时存在 3 份。40 GiB 是本记录的保守门槛，不是精确最低需求。本次结束后仅余约 33G，不能再无检查地追加新跑。
- 已完成依赖安装、模型下载、动态采样补丁和“固定起点与配方”；原有 `preparation/baseline_plan.json` 保留。
- 不另开 SFT 评测用的独立 vLLM 模型服务。GRPO 自己管理训练模型与 rollout 引擎；启动前 GPU 应空闲。

| 项目 | 设置 |
| --- | --- |
| 上游提交 | `a583dba4c149be09622ce58ad6f7a160db77bb41` |
| 起点 | `outputs/runs/sft_20261021_165243/merged`，即选定的 SFT step 300 |
| 数据 | GRPO train 1000 任务 / validation 50 任务 |
| 算法 / 奖励 | 原版 GRPO / Reward v3，关闭额外长度 shaping |
| LoRA | 新 adapter，rank 16、alpha 32 |
| 学习率 | `1e-6` |
| 采样 | 每任务 4 条，temperature/top-p 为 0.7/0.9 |
| 长度上限 | prompt 4096 + response 20480 = 24576 |
| batch | train batch 2；actor mini batch 2、micro batch 1 |
| 短跑覆盖项 | 总更新 3 步、每步保存、最多保留 2 份、不在训练前或期间运行验证 |
| 运行修复 | 修正 launcher 参数传递与环境清单；零系数 entropy 日志分支 detach；主进程与 Ray worker 固定线程和采样器设置 |

以下 7 步按顺序在 **AutoDL 的 Bash 终端**执行。命令仅写入云端代码修复及忽略的 `outputs/` 实验目录；本地 Windows 不执行训练。

## 1. 建立独立会话，检查资源

作用：集中定义路径，避免后续 Python 找不到 `PROJECT`，并在训练前拦截磁盘不足、GPU 被占用等问题。

```bash
cd /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon
set -euo pipefail
export PROJECT="$PWD"
export GRPO_RUN="$PROJECT/outputs/runs/grpo_baseline_20261022_202816"
export GRPO_PREP="$GRPO_RUN/preparation"
export SFT_MODEL="$PROJECT/outputs/runs/sft_20261021_165243/merged"
export SMOKE_TAG="$(date +%Y%m%d_%H%M%S)_$$"
export SMOKE_PREP="$GRPO_PREP/smoke_clean_$SMOKE_TAG"
export SMOKE_OUT="$GRPO_RUN/smoke_clean_$SMOKE_TAG"
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONHASHSEED=42
export SHOPPING_LENGTH_SHAPING_ENABLE=false
export SHOPSIM_BASE_URL=http://127.0.0.1:5700

.venv/bin/python - <<'PY'
import os, shutil
from pathlib import Path
import torch

root = Path(os.environ['PROJECT'])
assert (Path(os.environ['GRPO_PREP']) / 'baseline_plan.json').is_file()
assert (Path(os.environ['SFT_MODEL']) / 'merge_manifest.json').is_file()
assert not Path(os.environ['SMOKE_OUT']).exists()
free_disk = shutil.disk_usage(root).free / 2**30
assert free_disk >= 40, f'新跑磁盘余量不足：{free_disk:.1f} GiB；先归档不再需要的权重或扩容。'
assert torch.cuda.is_available()
free_gpu, total_gpu = torch.cuda.mem_get_info()
assert free_gpu / 2**30 >= 90, f'GPU 尚有其他占用：剩余 {free_gpu / 2**30:.1f} GiB'
limit = Path('/sys/fs/cgroup/memory.max')
if limit.exists() and limit.read_text().strip() != 'max':
    assert int(limit.read_text()) >= 120 * 2**30, 'CPU 内存配额低于本次已验证配置。'
print(f'磁盘可用 {free_disk:.1f} GiB；GPU 可用/总量 {free_gpu/2**30:.1f}/{total_gpu/2**30:.1f} GiB')
PY

mkdir -p "$SMOKE_PREP"
.venv/bin/python - <<'PY'
import os, shlex
from pathlib import Path
keys = ['PROJECT', 'GRPO_RUN', 'GRPO_PREP', 'SFT_MODEL', 'SMOKE_TAG',
        'SMOKE_PREP', 'SMOKE_OUT', 'PYTHONPATH', 'OMP_NUM_THREADS',
        'VLLM_USE_FLASHINFER_SAMPLER', 'PYTHONHASHSEED',
        'SHOPPING_LENGTH_SHAPING_ENABLE', 'SHOPSIM_BASE_URL']
text = ''.join(f'export {k}={shlex.quote(os.environ[k])}\n' for k in keys)
prep = Path(os.environ['SMOKE_PREP'])
(prep / 'session.env').write_text(text)
# 仅更新便于找回当前记录会话的指针；每次会话的原文件仍在独立目录中。
(Path(os.environ['GRPO_PREP']) / 'clean_smoke_session.env').write_text(text)
print('本次准备目录：', prep)
print('本次训练目录：', os.environ['SMOKE_OUT'])
PY
```

若中途换终端，先恢复会话再继续；不要再次执行第 1 步生成另一个新跑：

```bash
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/clean_smoke_session.env
cd "$PROJECT"
set -euo pipefail
```

## 2. 提前应用并记录三项代码修复

作用：把已查明的问题一次处理完。已修复的文件只校验，不重复插入补丁；代码与固定版本不符时停止，不盲目覆盖。

```bash
.venv/bin/python - <<'PY'
import ast, hashlib, importlib.metadata as md, importlib.util
import json, os, subprocess
from pathlib import Path

root = Path(os.environ['PROJECT'])
audit = Path(os.environ['SMOKE_PREP']) / 'fixes'
audit.mkdir(exist_ok=True)
pin = 'a583dba4c149be09622ce58ad6f7a160db77bb41'
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
assert head == pin, (head, pin)
sha = lambda b: hashlib.sha256(b).hexdigest()
receipts = []

def git_bytes(rel):
    return subprocess.check_output(['git', 'show', f'{pin}:{rel}'], cwd=root)

def install(path, data, label):
    before = path.read_bytes()
    (audit / f'{label}.before').write_bytes(before)
    (audit / f'{label}.after').write_bytes(data)
    if before != data:
        # 原子替换，避免原地修改可能由 uv 创建的硬链接文件。
        temp = path.with_name(path.name + f'.smoke-{os.getpid()}.tmp')
        temp.write_bytes(data)
        temp.chmod(path.stat().st_mode & 0o777)
        os.replace(temp, path)
    receipts.append({'file': str(path), 'before_sha256': sha(before),
                     'after_sha256': sha(data), 'changed_now': before != data})

# A. launcher：将实际构造出的 Hydra 参数传给预检。
rel = 'scripts/train_grpo.py'
original = git_bytes(rel)
old = (b'        str(ROOT / "scripts/check_grpo_runtime.py"),\n'
       b'        *overrides,\n        *extra,\n')
assert original.count(old) == 1
fixed = original.replace(old, b'        str(ROOT / "scripts/check_grpo_runtime.py"),\n        *command[5:],\n')
path = root / rel
assert path.read_bytes() in (original, fixed), 'launcher 有未识别修改，请先核对。'
ast.parse(fixed.decode())
install(path, fixed, 'launcher')

# B. 环境清单：确认五个运行文件都等于固定提交，再修正漏更新的一项。
rel = 'data/environment.json'
manifest = json.loads((root / rel).read_text())
original_manifest = json.loads(git_bytes(rel))
paths = {
    'observation.py': 'environments/ShopSimulator/shop_env/web_agent_site/engine/observation.py',
    'pack_api.py': 'environments/ShopSimulator/shop_env/shop_env/pack_api.py',
    'reward.py': 'environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py',
    'slot_lease_pool.py': 'environments/ShopSimulator/shop_env/shop_env/slot_lease_pool.py',
    'web_agent_text_env.py': 'environments/ShopSimulator/shop_env/web_agent_site/envs/web_agent_text_env.py',
}
fixed_manifest = json.loads(json.dumps(original_manifest))
for name, runtime_rel in paths.items():
    actual = (root / runtime_rel).read_bytes()
    assert actual == git_bytes(runtime_rel), f'运行文件偏离固定提交：{name}'
    if name != 'web_agent_text_env.py':
        assert sha(actual) == original_manifest['runtime_files_sha256'][name]
    else:
        assert sha(actual) == 'd6db15f282849847a364d8d36af0eed113c1c97184a1ded98eb1fd29dab577d1'
        fixed_manifest['runtime_files_sha256'][name] = sha(actual)
assert manifest in (original_manifest, fixed_manifest), '环境清单还有其他未识别修改。'
install(root / rel, (json.dumps(fixed_manifest, ensure_ascii=False, indent=2) + '\n').encode(), 'manifest')

# C. entropy：系数为零时只记录数值，不保留这一分支的梯度图。
assert md.version('verl') == '0.8.0'
verl_root = Path(importlib.util.find_spec('verl').origin).resolve().parent
path = verl_root / 'workers/utils/losses.py'
assert path.is_relative_to(root / '.venv'), path
source = path.read_text()
anchor = '    entropy = model_output.get("entropy", None)\n'
addition = ('    # SHOPPING_GRPO_ZERO_COEFF_ENTROPY_DETACH_V1\n'
            '    if entropy is not None and float(config.entropy_coeff) == 0.0:\n'
            '        entropy = entropy.detach()\n'
            '        model_output["entropy"] = entropy\n')
marker = 'SHOPPING_GRPO_ZERO_COEFF_ENTROPY_DETACH_V1'
if marker not in source:
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'ppo_loss')
    fn_text = '\n'.join(source.splitlines()[fn.lineno-1:fn.end_lineno]) + '\n'
    assert fn_text.count(anchor) == 1 and source.count(anchor) == 1
    source = source.replace(anchor, anchor + addition)
else:
    assert source.count(marker) == 1 and anchor + addition in source, 'entropy 补丁内容不符。'
ast.parse(source)
install(path, source.encode(), 'entropy_loss')
(audit / 'receipt.json').write_text(json.dumps(receipts, ensure_ascii=False, indent=2))
print('launcher、环境清单、零系数 entropy 修复均已确认。')
PY
```

三项修复不修改奖励、学习率、采样数或长度上限。entropy 补丁会改动安装包代码，因此必须保留源文件和哈希；不能称为完全未经修改的运行环境。

## 3. 固定实际命令、配置和文件哈希

作用：沿用之前封存的起点，记录修复后的代码，并显式把线程数和采样器设置传到 Ray worker。只保存必要环境变量，不把整个进程环境中的密钥写入记录。

```bash
.venv/bin/python - <<'PY'
import argparse, hashlib, importlib.metadata as md, importlib.util, json, os, shlex, subprocess, sys
from pathlib import Path

root = Path(os.environ['PROJECT'])
prep = Path(os.environ['SMOKE_PREP'])
out = Path(os.environ['SMOKE_OUT'])
plan = json.loads((Path(os.environ['GRPO_PREP']) / 'baseline_plan.json').read_text())
assert Path(plan['model']).resolve() == Path(os.environ['SFT_MODEL']).resolve()
assert Path(plan['config']).resolve() == root / 'configs/grpo.yaml'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

# 与前一小节的模型及数据封存结果核对。
for name, expected in plan['model_sha256'].items():
    assert digest(Path(plan['model']) / name) == expected, name
for rel, expected in plan['data_sha256'].items():
    assert digest(root / rel) == expected, rel
changed_files = {'scripts/train_grpo.py', 'data/environment.json',
                 '.venv/lib/python3.12/site-packages/verl/workers/utils/losses.py'}
for rel, expected in plan['source_sha256'].items():
    if rel not in changed_files:
        assert digest(root / rel) == expected, rel
verl_root = Path(importlib.util.find_spec('verl').origin).resolve().parent
ray_source = verl_root / 'trainer/ppo/ray_trainer.py'
assert digest(ray_source) == plan['verl_ray_trainer_sha256']
assert 'SHOPPING_GRPO_DYNAMIC_SAMPLING_PATCH_V3' in ray_source.read_text()

runtime = dict(plan['runtime_env'])
runtime.update(OMP_NUM_THREADS='1', VLLM_USE_FLASHINFER_SAMPLER='0',
               PYTHONHASHSEED='42', SHOPPING_LENGTH_SHAPING_ENABLE='false')
overrides = [x for x in plan['hydra_overrides']
             if not x.startswith('++ray_kwargs.ray_init.runtime_env.env_vars.')]
overrides += ['++ray_kwargs.ray_init.runtime_env.env_vars.' + k + '=' + json.dumps(v)
              for k, v in runtime.items()]
overrides += ['trainer.total_training_steps=3', 'trainer.save_freq=1',
              'trainer.test_freq=-1', 'trainer.val_before_train=false',
              'trainer.max_actor_ckpt_to_keep=2', 'trainer.resume_mode=disable']

spec = importlib.util.spec_from_file_location('smoke_launcher', root / 'scripts/train_grpo.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
args = argparse.Namespace(model=Path(plan['model']), train_data=Path(plan['train_data']),
    val_data=Path(plan['val_data']), config=Path(plan['config']), output=out,
    env_url='http://127.0.0.1:5700', logger='console',
    experiment_name='shopping-grpo-smoke-clean-' + os.environ['SMOKE_TAG'],
    hydra_overrides=overrides)
command, env = module.build_command(args)
env.update(runtime)
keys = ['PYTHONPATH', 'SHOPPING_GRPO_ROOT', 'SHOPPING_ENVIRONMENT_VERSION',
        'SHOPPING_ENV_MANIFEST', 'GRPO_MODEL_PATH', 'GRPO_TRAIN_FILE', 'GRPO_VAL_FILE',
        'GRPO_OUTPUT_DIR', 'SHOPPING_GRPO_DIAGNOSTICS_PATH',
        'SHOPSIM_BASE_URL', 'SHOPPING_AGENT_LOOP_CONFIG', 'SHOPPING_TOOL_CONFIG', 'GRPO_CONFIG_NAME']
keys += list(runtime)
selected_env = {k: env[k] for k in set(keys)}

argv = [sys.executable, str(root / 'scripts/train_grpo.py'),
        '--model', str(args.model), '--train-data', str(args.train_data),
        '--val-data', str(args.val_data), '--config', str(args.config),
        '--output', str(out), '--env-url', args.env_url,
        '--logger', 'console', '--experiment-name', args.experiment_name,
        '--', *overrides]
snapshot = prep / 'snapshot'
snapshot.mkdir()
rels = set(plan['source_sha256']) | changed_files | {'configs/agent_loop.yaml', 'configs/tools.json'}
hashes = {}
for rel in sorted(rels):
    path = root / rel
    assert path.is_file(), path
    hashes[str(path)] = digest(path)
    dest = snapshot / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(path.read_bytes())
hashes[str(ray_source)] = digest(ray_source)
record = {'project': str(root), 'output': str(out), 'command': argv,
          'preflight': [sys.executable, str(root / 'scripts/check_grpo_runtime.py'), *command[5:]],
          'env': selected_env, 'source_sha256': hashes}
(prep / 'launch_spec.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
(prep / 'command.txt').write_text(shlex.join(argv) + '\n')
packages = sorted((d.metadata['Name'], d.version) for d in md.distributions() if d.metadata['Name'])
(prep / 'packages.json').write_text(json.dumps(packages, indent=2))
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
os.environ.update(selected_env)
with initialize_config_dir(version_base=None, config_dir=str(root / 'configs')):
    cfg = compose(config_name='grpo', overrides=command[5:])
    (prep / 'composed_config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
with (prep / 'dry_run.log').open('w') as log:
    pos = argv.index('--')
    subprocess.run([*argv[:pos], '--dry-run', *argv[pos:]], env=env, cwd=root,
                   stdout=log, stderr=subprocess.STDOUT, check=True)
print('实际命令、完整解析配置、源码快照和 dry-run 已保存：', prep)
PY
```

`dry-run` 通过只说明参数与路径可构造。后面还必须进行运行时预检和真实训练。

## 4. 启动环境，并从终端与 Ray worker 实测连接

作用：环境服务独立于训练启动；不用不存在的 `/health` 接口，以真实 `reset → release` 验证。

```bash
if .venv/bin/python - <<'PY'
import socket
with socket.create_connection(('127.0.0.1', 5700), timeout=2):
    pass
PY
then
  echo '5700 已监听，下面验证服务是否确实可用。'
else
  nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 \
    SHOPSIM_PORT=5700 SHOPSIM_ENV_SLOTS=8 SHOP_MAX_STEPS=35 \
    bash scripts/start_environment.sh > "$SMOKE_PREP/environment.log" 2>&1 < /dev/null &
  echo "$!" > "$SMOKE_PREP/environment.pid"
fi

.venv/bin/python - <<'PY'
import json, os, socket, time
from contextlib import ExitStack
from pathlib import Path
from shopping_grpo.environment.client import ShopAgentEnv

deadline = time.monotonic() + 120
while True:
    try:
        with socket.create_connection(('127.0.0.1', 5700), timeout=2):
            break
    except OSError:
        if time.monotonic() >= deadline:
            raise SystemExit('环境未开始监听；检查本次 environment.log。')
        time.sleep(2)
root = Path(os.environ['PROJECT'])
task = int(json.loads((root / 'data/grpo/train.jsonl').read_text().splitlines()[0])['task_id'])
with ExitStack() as stack:
    indices = []
    for _ in range(8):
        env = stack.enter_context(ShopAgentEnv(os.environ['SHOPSIM_BASE_URL']))
        env.reset(task)
        indices.append(env.env_idx)
    assert len(set(indices)) == 8
print('终端可申请 8 个槽位，且已全部释放：', indices)

import ray
runtime = {'env_vars': {'PYTHONPATH': str(root / 'src'),
                      'SHOPSIM_BASE_URL': os.environ['SHOPSIM_BASE_URL'],
                      'OMP_NUM_THREADS': '1', 'VLLM_USE_FLASHINFER_SAMPLER': '0'}}
ray.init(address='local', num_cpus=1, num_gpus=0,
         object_store_memory=128 * 1024**2, include_dashboard=False, runtime_env=runtime)
try:
    @ray.remote(num_cpus=0.1)
    def probe(task):
        import os, torch
        from shopping_grpo.environment.client import ShopAgentEnv
        with ShopAgentEnv(os.environ['SHOPSIM_BASE_URL']) as env:
            env.reset(task)
            idx = env.env_idx
        assert os.environ['OMP_NUM_THREADS'] == '1'
        assert torch.get_num_threads() > 0
        return {'reset_and_release': 'passed', 'env_idx': idx,
                'OMP_NUM_THREADS': os.environ['OMP_NUM_THREADS'],
                'torch_num_threads': torch.get_num_threads()}
    result = ray.get(probe.remote(task))
    (Path(os.environ['SMOKE_PREP']) / 'environment_probe.json').write_text(json.dumps(result, indent=2))
    print('Ray worker 检查：', result)
finally:
    ray.shutdown()
PY
```

复用已经监听的服务时，本步证明接口和槽位可用；还需确保该服务就是此前固定的 Environment v2.1 配置。本记录不自动终止未知的已有服务。

## 5. 运行时预检

作用：检查版本、环境清单、动态采样补丁及内存相关配置。检查失败就停止，不跳过它直接训练。

```bash
.venv/bin/python - <<'PY'
import json, os, subprocess
from pathlib import Path
prep = Path(os.environ['SMOKE_PREP'])
spec = json.loads((prep / 'launch_spec.json').read_text())
env = dict(os.environ, **spec['env'])
with (prep / 'preflight.log').open('w') as log:
    status = subprocess.call(spec['preflight'], cwd=spec['project'], env=env,
                             stdout=log, stderr=subprocess.STDOUT)
print((prep / 'preflight.log').read_text())
assert status == 0, f'预检失败：{status}'
print('运行时预检通过。')
PY
```

## 6. 后台运行 3 步真实训练

作用：使用刚刚封存的命令启动训练，训练前重查源码；保存退出码和进程耗时。关闭本地电脑不会结束 `nohup` 进程；关闭 AutoDL 实例会停止训练。

```bash
cat > "$SMOKE_PREP/launch.py" <<'PY'
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

prep = Path(__file__).resolve().parent
spec = json.loads((prep / 'launch_spec.json').read_text())
for name, expected in spec['source_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
out = Path(spec['output'])
assert not out.exists() or (out.is_dir() and not any(out.iterdir())), '拒绝覆盖已有训练输出。'
env = dict(os.environ, **spec['env'])
start = time.time()
status = subprocess.call(spec['command'], cwd=spec['project'], env=env)
receipt = {'exit_code': status, 'elapsed_seconds': time.time() - start,
           'started_unix': start, 'ended_unix': time.time(), 'output': str(out)}
(prep / 'exit_status.json').write_text(json.dumps(receipt, indent=2))
print(f'\nTRAIN_EXIT_CODE={status}', flush=True)
sys.exit(status)
PY

if [[ -e "$SMOKE_PREP/train.log" || -e "$SMOKE_PREP/train.pid" ]]; then
  echo '本会话已经启动过训练，请先检查原进程和日志，不重复启动。'
else
  nohup .venv/bin/python -u "$SMOKE_PREP/launch.py" \
    > "$SMOKE_PREP/train.log" 2>&1 < /dev/null &
  echo "$!" > "$SMOKE_PREP/train.pid"
  echo "训练 PID：$(cat "$SMOKE_PREP/train.pid")"
fi

tail -n 60 -f "$SMOKE_PREP/train.log"
```

看到 `TRAIN_EXIT_CODE=0` 后，用 `Ctrl+C` 退出日志查看器，再执行验收。`tail -f` 不会因为训练结束而自动退出；此处 `Ctrl+C` 停止的是日志查看器。

## 7. 验收更新、checkpoint、LoRA 变化与环境释放

作用：区分“程序没报错”和“确实完成了训练”。以下针对本记录的**新三步运行**，预期诊断中有 step 1/2/3；比较最终保留的 step 2 与 step 3 参数。

```bash
.venv/bin/python - <<'PY'
import collections, json, math, numbers, os
from pathlib import Path

prep = Path(os.environ['SMOKE_PREP'])
out = Path(os.environ['SMOKE_OUT'])
status = json.loads((prep / 'exit_status.json').read_text())
assert status['exit_code'] == 0, status
events = [json.loads(x) for x in (out / 'training_diagnostics.jsonl').read_text().splitlines() if x.strip()]
updates = [x for x in events if x['event'] == 'optimizer_step']
assert [int(x['global_step']) for x in updates] == [1, 2, 3]

def finite(value, path='metrics'):
    if isinstance(value, dict):
        for key, item in value.items(): finite(item, path + '.' + str(key))
    elif isinstance(value, list):
        for i, item in enumerate(value): finite(item, path + f'[{i}]')
    elif isinstance(value, numbers.Real):
        assert math.isfinite(value), (path, value)

for update in updates:
    m = update['metrics']
    finite(m)
    assert int(m['training/optimizer_updated']) == 1
assert any(float(x['metrics'].get('actor/lr', 0)) > 0 for x in updates)
assert any(float(x['metrics'].get('actor/grad_norm', 0)) > 0 for x in updates)
# 调度器可能在最后一步结束后将日志中的 lr 记为 0，不能要求每条日志都大于 0。
generations = [x for x in events if x['event'] == 'generation_batch']
assert generations
groups = [g for event in generations for g in event['groups']]
assert groups
for group in groups:
    assert len(group['rewards']) == len(group['terminal_utilities']) == 4
    finite(group['rewards'], 'group.rewards')
    finite(group['terminal_utilities'], 'group.terminal_utilities')
    if group['accepted']:
        assert group['kept'] and not group['sampling_invalid']
        assert max(group['terminal_utilities']) - min(group['terminal_utilities']) > 1e-8
reasons = collections.Counter(reason for g in groups for reason in g['sampling_invalid_reasons'])
drop_reasons = collections.Counter(g['drop_reason'] for g in groups if g['drop_reason'])
print('采样无效原因：', dict(reasons))
print('组过滤原因：', dict(drop_reasons))
for step in (2, 3):
    checkpoint = out / f'global_step_{step}'
    for name in ('model_world_size_1_rank_0.pt', 'optim_world_size_1_rank_0.pt',
                 'extra_state_world_size_1_rank_0.pt', 'fsdp_config.json'):
        path = checkpoint / 'actor' / name
        assert path.is_file() and path.stat().st_size > 0, path
    assert (checkpoint / 'data.pt').is_file()

for event in updates:
    m = event['metrics']
    selected = {k: v for k, v in m.items() if k in ('actor/lr', 'actor/grad_norm')
                or k.startswith(('group/', 'timing_s/'))
                or 'memory' in k.lower() or 'terminal_utility' in k}
    print('step', event['global_step'], json.dumps(selected, ensure_ascii=False))
summary = {'exit_code': 0, 'optimizer_steps': [1, 2, 3],
           'events': dict(collections.Counter(x['event'] for x in events)),
           'sampling_invalid_reasons': dict(reasons), 'group_drop_reasons': dict(drop_reasons),
           'process_elapsed_seconds': status['elapsed_seconds'],
           'checkpoint_basic_checks': 'passed'}
(prep / 'acceptance.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2))
print('以上耗时包含进程中的模型初始化和预检，不等于纯训练吞吐。')
PY

.venv/bin/python - <<'PY'
import json, os, tempfile
from pathlib import Path
import torch
import torch.distributed as dist

out = Path(os.environ['SMOKE_OUT'])
# 仅加载本次自己生成的可信 checkpoint；不加载来源未知的 pickle 文件。
with tempfile.TemporaryDirectory() as td:
    dist.init_process_group('gloo', init_method=Path(td, 'init').as_uri(), rank=0, world_size=1)
    try:
        def load(step):
            return torch.load(out / f'global_step_{step}' / 'actor/model_world_size_1_rank_0.pt',
                              map_location='cpu', mmap=True, weights_only=False)
        before, after = load(2), load(3)
        keys = sorted(k for k in before if 'lora_' in k.lower())
        assert keys and set(keys) == {k for k in after if 'lora_' in k.lower()}
        def local(value):
            if hasattr(value, 'local_shards'):
                shards = value.local_shards()
                assert len(shards) == 1
                return shards[0].tensor
            if hasattr(value, 'to_local'): return value.to_local()
            return value
        changes = []
        for key in keys:
            a, b = local(before[key]), local(after[key])
            assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all(), key
            changes.append((float((a.float() - b.float()).abs().max()), key))
        changes.sort(reverse=True)
        changed = sum(delta > 0 for delta, _ in changes)
        assert changed > 0, 'LoRA 参数没有实际变化。'
        result = {'compared_steps': [2, 3], 'lora_tensor_count': len(keys),
                  'changed_tensor_count': changed, 'max_abs_delta': changes[0][0],
                  'top_changes': changes[:5]}
        (Path(os.environ['SMOKE_PREP']) / 'lora_delta.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        dist.destroy_process_group()
PY

.venv/bin/python - <<'PY'
import json, os
from contextlib import ExitStack
from pathlib import Path
from shopping_grpo.environment.client import ShopAgentEnv
root = Path(os.environ['PROJECT'])
task = int(json.loads((root / 'data/grpo/train.jsonl').read_text().splitlines()[0])['task_id'])
with ExitStack() as stack:
    indices = []
    for _ in range(8):
        env = stack.enter_context(ShopAgentEnv(os.environ['SHOPSIM_BASE_URL']))
        env.reset(task)
        indices.append(env.env_idx)
    assert len(set(indices)) == 8
result = {'slot_count': 8, 'indices': indices, 'all_released': True}
(Path(os.environ['SMOKE_PREP']) / 'lease_check.json').write_text(json.dumps(result, indent=2))
print('8 个环境槽位均可申请，且已全部释放：', indices)
PY

df -h "$PROJECT"
nvidia-smi
```

数值与文件检查是基础验收，不替代完整 checkpoint 导出和模型评测。`generation_batch`、`skipped_update`、`optimizer_step` 是不同事件：生成多个 batch 不代表更新了多次；奖励全相同等情况会触发有上限的重采样或跳过。诊断中还应保留并查看采样无效原因、奖励和有效组比例，而不能只看最终退出码。

## 本次实际完成的结果（2026-10-24）

实际运行经历中断：第一次保留了 step 1，之后从该 checkpoint 恢复，成功完成 step 2、step 3。上面新整理的流程把已知修复放在训练前，从 SFT 一次执行三步，因此**新流程的目录、耗时和参数变化数字不能冒用下面的历史结果**。

| 检查 | 本次实际结果 |
| --- | --- |
| 续跑退出、step 2/3 更新、数值与 checkpoint 基础检查 | 通过 |
| LoRA 参数比较 | 比较原 step 1 与续跑 step 3，共 568 个张量 |
| 实际变化的张量 | 372 个 |
| 最大绝对变化 | `2.004897851293208e-06` |
| 环境租约 | 8 个槽位均可申请，且全部释放 |
| 数据盘 | 100G，总计已用约 68G，可用约 33G |

实际产物位置，相对于云端项目根目录：

```text
outputs/runs/grpo_baseline_20261022_202816/
├── preparation/
│   ├── baseline_plan.json
│   ├── resume_session.env                 # 定位实际续跑日志及命令
│   └── entropy_logging_fix/               # 已应用补丁的前后源码及回执
├── smoke/global_step_1/                   # 实际续跑起点
└── smoke_resume_entropy_20261024_103342/   # 成功续跑输出
    ├── training_diagnostics.jsonl
    ├── global_step_2/
    └── global_step_3/
```

实测总耗时、逐步显存峰值、有效组比例和奖励分布尚未在本记录回填，不写估算数字。历史比较中 372/568 个张量变化不是未来每次运行的固定预期。

结论：**GRPO 真实短跑、断点恢复与基础验收完成；正式 GRPO 尚未开始。** 短跑没有证明模型成绩超过 SFT。下一步整理吞吐和资源数据、固定正式训练预算及 checkpoint 保留策略；正式实验仍从选定的 SFT step 300 新挂 LoRA 开始，不把短跑模型作为正式起点。
