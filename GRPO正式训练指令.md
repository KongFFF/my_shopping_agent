# 正式 GRPO：顺序命令

> 2026-10-24 状态更新：正式训练已启动，随后在 actor 前向计算 `logprobs_from_logits_v2 → F.log_softmax` 时发生 CUDA OOM。报错显示 PyTorch 已分配约 89.33 GiB、保留未分配约 2.78 GiB。中断步数及可恢复 checkpoint 待核对，不能将本节命令视为已通过完整 500 步验证。不要直接重跑第 6 步覆盖或复用已有输出；先完成下方第 9 节只读诊断。

> 后续诊断确认：完成 35 次更新，在第 36 步尝试中 OOM；进程共约 56 分 34 秒，尚未产生正式 checkpoint，无法恢复第 35 步。新的显存修复和早期保存方案单列于 [GRPO正式训练OOM修复与重跑.md](GRPO正式训练OOM修复与重跑.md)，不要混用两份文档的运行目录及后台监控。

适用状态：SFT step 300 已选定，GRPO step 1 → 续跑 step 2/3 的真实短跑和基础验收已经通过，云端修复仍然保留。本节从同一个 SFT 合并模型**新挂 LoRA 开始正式训练**，不继续优化短跑权重。

本节命令面向 AutoDL Linux 的 Bash 终端。仅整理命令，未从本地启动训练。第 1—3 步可以在无卡模式完成；第 4—7 步切换到有卡模式后执行。换终端、关机重开后，要先恢复本节会话变量。

## 配方与空间安排

采用固定提交的原配方：**上限 500 次更新**，每 50 步保存并验证，训练前验证一次，保留最多 10 份 checkpoint。训练 1000 题、验证仍为 Dev-50，Final-200 Clean 不进入训练或模型选择。核心学习率、LoRA、采样数、上下文与 Reward 均沿用短跑所验证的设置。

500 是本次明确采用的上游默认上限，不是从三步短跑证明出的最优步数或费用预算。正式调度器按 500 步建立，不能把“先将总步数设为 100、之后改成 500”当作等价的连续训练。训练内验证用于初筛 checkpoint，之后仍需用已固定的独立评测协议核验候选。

目前数据盘总计 100G、剩余约 33G。每份完整 checkpoint 约 10.4 GiB，10 份合计约 104 GiB，还需预留导出模型、日志和余量。**建议把数据盘扩到总计 200G（再增加约 100G）**，第 2 步会按实际 checkpoint 大小计算门槛，第 4 步会阻止空间不足时开跑。若实际余量仍未达门槛，再补扩；不是只看“总容量 200G”就认为必然足够。

本方案保留既有 SFT、短跑产物和全部计划内正式候选，不删除原始证据。仅滚动保留最后两份虽能省空间，但无法保证较早的最佳 Dev checkpoint 仍可导出，因此这里不采用该方式。

2026-10-24 资源校准：正式实例 CPU 内存配额为 **110 GiB**，短跑实例为 120 GiB。120 GiB 是此前的实例配额，不是测得的最低内存需求；本节将启动门槛调整为 110 GiB。该变化不修改训练配方，110 GiB 下的实际训练峰值仍需观察。已经生成的云端 `launch_formal.py` 也需要同步修改这一门槛，单改本文不会改变云端脚本。

veRL checkpoint 保留与验证输出机制参考 [veRL 0.8 checkpoint manager](https://github.com/verl-project/verl/blob/v0.8.0/verl/utils/checkpoint/checkpoint_manager.py) 与 [RayPPOTrainer](https://github.com/verl-project/verl/blob/v0.8.0/verl/trainer/ppo/ray_trainer.py)；实际实验仍以已安装并封存哈希的源文件为准。

## 1. 建立正式会话，回收短跑诊断（无卡模式可做）

本步不加载权重、不导出模型，也不初始化 CUDA。只读取配置、日志与文件大小，适合无卡实例的 2 GiB 内存配额。

```bash
cd /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon
set -euo pipefail
export PROJECT="$PWD"
export GRPO_RUN="$PROJECT/outputs/runs/grpo_baseline_20261022_202816"
export GRPO_PREP="$GRPO_RUN/preparation"
export FORMAL_PREP="$GRPO_PREP/formal_launch"
export FORMAL_OUT="$GRPO_RUN/formal"
export SFT_MODEL="$PROJECT/outputs/runs/sft_20261021_165243/merged"
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONHASHSEED=42
export SHOPPING_LENGTH_SHAPING_ENABLE=false
export SHOPSIM_BASE_URL=http://127.0.0.1:5700

mkdir -p "$FORMAL_PREP"
.venv/bin/python - <<'PY'
import collections, json, math, os, shlex, statistics
from pathlib import Path

root = Path(os.environ['PROJECT'])
run = Path(os.environ['GRPO_RUN'])
prep = Path(os.environ['FORMAL_PREP'])
assert (Path(os.environ['GRPO_PREP']) / 'baseline_plan.json').is_file()
keys = ['PROJECT', 'GRPO_RUN', 'GRPO_PREP', 'FORMAL_PREP', 'FORMAL_OUT',
        'SFT_MODEL', 'PYTHONPATH', 'OMP_NUM_THREADS', 'VLLM_USE_FLASHINFER_SAMPLER',
        'PYTHONHASHSEED', 'SHOPPING_LENGTH_SHAPING_ENABLE', 'SHOPSIM_BASE_URL']
session = ''.join(f'export {key}={shlex.quote(os.environ[key])}\n' for key in keys)
(prep / 'session.env').write_text(session)
(Path(os.environ['GRPO_PREP']) / 'formal_session.env').write_text(session)

# 只保留已提交更新的标量指标，不把整份轨迹日志一次性载入内存。
updates = {}
counts = collections.Counter()
for path, allowed in [(run / 'smoke/training_diagnostics.jsonl', {1}),
    (run / 'smoke_resume_entropy_20261024_103342/training_diagnostics.jsonl', {2, 3})]:
    assert path.is_file(), path
    with path.open() as f:
        for line in f:
            if not line.strip(): continue
            event = json.loads(line)
            counts[event['event']] += 1
            step = int(event['global_step'])
            if event['event'] == 'optimizer_step' and step in allowed:
                assert step not in updates, f'重复更新记录：{step}'
                updates[step] = event['metrics']
assert sorted(updates) == [1, 2, 3], sorted(updates)
for step, metrics in sorted(updates.items()):
    assert int(metrics['training/optimizer_updated']) == 1
    selected = {k: v for k, v in metrics.items() if k in ('actor/lr', 'actor/grad_norm')
        or k.startswith(('group/', 'timing_s/')) or 'memory' in k.lower()}
    print('短跑 step', step, json.dumps(selected, ensure_ascii=False))
assert any(float(m.get('actor/lr', 0)) > 0 for m in updates.values())
assert any(float(m.get('actor/grad_norm', 0)) > 0 for m in updates.values())
summary = {'events': dict(counts), 'steps': updates}
(prep / 'smoke_metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
times = []
for m in updates.values():
    step_time = m.get('timing_s/step')
    if isinstance(step_time, (int, float)) and math.isfinite(step_time):
        times.append(max(0, step_time - float(m.get('timing_s/save_checkpoint', 0))))
if times:
    estimate = statistics.median(times) * 500 / 3600
    print(f'仅作量级参考：500 × 短跑去保存耗时的中位数 ≈ {estimate:.1f} 小时。')
    print('这不是总耗时承诺：未计入 11 次验证、初始化、正式保存及分布变化；跳过更新也可能未完整计入。')
else:
    print('没有可直接使用的 step 计时字段，不生成耗时估算。')
print('短跑指标已保存：', prep / 'smoke_metrics.json')
PY

df -h "$PROJECT"
```

500 步训练可能持续较长时间；先看本步打印的实测量级，再执行启动段。已有结果不足以给出可信的精确总耗时。

## 2. 核对修复，冻结正式命令并计算磁盘预算（无卡模式可做）

本步不会再次打补丁，只确认成功短跑所需的修复仍在。若校验失败，应恢复已验证环境；不要通过重新安装依赖把安装包内的补丁覆盖掉。

```bash
.venv/bin/python - <<'PY'
import argparse, ast, hashlib, importlib.metadata as md, importlib.util
import json, math, os, shlex, shutil, subprocess, sys
from pathlib import Path

root = Path(os.environ['PROJECT'])
run = Path(os.environ['GRPO_RUN'])
prep = Path(os.environ['FORMAL_PREP'])
out = Path(os.environ['FORMAL_OUT'])
assert not (prep / 'launch_spec.json').exists(), '正式配置已生成；直接恢复会话继续，不重复生成。'
assert not out.exists() or (out.is_dir() and not any(out.iterdir())), '正式输出非空，不能重新从 SFT 覆盖开跑。'
plan = json.loads((Path(os.environ['GRPO_PREP']) / 'baseline_plan.json').read_text())
pin = 'a583dba4c149be09622ce58ad6f7a160db77bb41'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() == pin
assert Path(plan['model']).resolve() == Path(os.environ['SFT_MODEL']).resolve()
assert Path(plan['config']).resolve() == root / 'configs/grpo.yaml'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()

repaired = {'scripts/train_grpo.py', 'data/environment.json',
            '.venv/lib/python3.12/site-packages/verl/workers/utils/losses.py'}
for rel, expected in plan['source_sha256'].items():
    if rel not in repaired: assert digest(root / rel) == expected, rel
for rel, expected in plan['data_sha256'].items():
    assert digest(root / rel) == expected, rel
for name, expected in plan['model_sha256'].items():
    assert digest(Path(plan['model']) / name) == expected, name

def task_ids(path, count):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = {int(row['task_id']) for row in rows}
    assert len(rows) == len(ids) == count, path
    return ids
train_ids = task_ids(root / 'data/grpo/train.jsonl', 1000)
val_ids = task_ids(root / 'data/grpo/validation.jsonl', 50)
final_ids = task_ids(root / 'data/evaluation/tasks.jsonl', 200)
assert not (train_ids & val_ids or train_ids & final_ids or val_ids & final_ids)
dev_path = root / 'outputs/runs/sft_20261021_165243/cpu_prep/dev50.jsonl'
assert val_ids == task_ids(dev_path, 50)

# launcher 必须只包含此前已经确认的参数传递修复。
original = subprocess.check_output(['git', 'show', f'{pin}:scripts/train_grpo.py'], cwd=root)
anchor = (b'        str(ROOT / "scripts/check_grpo_runtime.py"),\n'
          b'        *overrides,\n        *extra,\n')
assert original.count(anchor) == 1
fixed = original.replace(anchor, b'        str(ROOT / "scripts/check_grpo_runtime.py"),\n        *command[5:],\n')
assert (root / 'scripts/train_grpo.py').read_bytes() == fixed, 'launcher 修复未保留或有其他修改。'
manifest = json.loads((root / 'data/environment.json').read_text())
expected_manifest = json.loads(subprocess.check_output(['git', 'show', f'{pin}:data/environment.json'], cwd=root))
expected_manifest['runtime_files_sha256']['web_agent_text_env.py'] = 'd6db15f282849847a364d8d36af0eed113c1c97184a1ded98eb1fd29dab577d1'
assert manifest == expected_manifest, '环境清单偏离已经确认的修复。'

assert md.version('verl') == '0.8.0'
verl_root = Path(importlib.util.find_spec('verl').origin).resolve().parent
loss_file = verl_root / 'workers/utils/losses.py'
ray_source = verl_root / 'trainer/ppo/ray_trainer.py'
assert digest(ray_source) == plan['verl_ray_trainer_sha256']
assert 'validation_data_dir' in ray_source.read_text(), '安装版本缺少本节所需的验证输出功能。'
fix = plan['zero_coeff_entropy_fix']
assert digest(loss_file) == fix['after_sha256'], 'entropy 补丁偏离短跑时封存的版本。'
assert 'SHOPPING_GRPO_ZERO_COEFF_ENTROPY_DETACH_V1' in loss_file.read_text()

runtime = dict(plan['runtime_env'])
runtime.update(OMP_NUM_THREADS='1', VLLM_USE_FLASHINFER_SAMPLER='0',
               PYTHONHASHSEED='42', SHOPPING_LENGTH_SHAPING_ENABLE='false')
assert all(isinstance(v, str) for v in runtime.values())
overrides = [v for v in plan['hydra_overrides'] if not v.startswith('trainer.')
             and not v.startswith('++ray_kwargs.ray_init.runtime_env.env_vars.')]
overrides += ['++ray_kwargs.ray_init.runtime_env.env_vars.' + k + '=' + json.dumps(v)
              for k, v in runtime.items()]
overrides += ['trainer.total_training_steps=500', 'trainer.total_epochs=50',
              'trainer.save_freq=50', 'trainer.test_freq=50',
              'trainer.val_before_train=true', 'trainer.max_actor_ckpt_to_keep=10',
              'trainer.resume_mode=disable',
              '++trainer.validation_data_dir=' + json.dumps(str(out / 'validation'))]

loader = importlib.util.spec_from_file_location('formal_launcher', root / 'scripts/train_grpo.py')
module = importlib.util.module_from_spec(loader)
loader.loader.exec_module(module)
args = argparse.Namespace(model=Path(plan['model']), train_data=Path(plan['train_data']),
    val_data=Path(plan['val_data']), config=Path(plan['config']), output=out,
    env_url='http://127.0.0.1:5700', logger='console',
    experiment_name='shopping-grpo-baseline-formal', hydra_overrides=overrides)
command, env = module.build_command(args)
env.update(runtime)
keys = ['PYTHONPATH', 'SHOPPING_GRPO_ROOT', 'SHOPPING_ENVIRONMENT_VERSION',
        'SHOPPING_ENV_MANIFEST', 'GRPO_MODEL_PATH', 'GRPO_TRAIN_FILE', 'GRPO_VAL_FILE',
        'GRPO_OUTPUT_DIR', 'SHOPPING_GRPO_DIAGNOSTICS_PATH', 'SHOPSIM_BASE_URL',
        'SHOPPING_AGENT_LOOP_CONFIG', 'SHOPPING_TOOL_CONFIG', 'GRPO_CONFIG_NAME', *runtime]
selected_env = {k: env[k] for k in set(keys)}
os.environ.update(selected_env)

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
with initialize_config_dir(version_base=None, config_dir=str(root / 'configs')):
    cfg = compose(config_name='grpo', overrides=command[5:])
    assert cfg.trainer.total_training_steps == 500 and cfg.trainer.save_freq == cfg.trainer.test_freq == 50
    assert cfg.trainer.val_before_train and cfg.trainer.max_actor_ckpt_to_keep == 10
    assert cfg.trainer.resume_mode == 'disable'
    assert cfg.actor_rollout_ref.model.get('lora_adapter_path') in (None, '')
    assert cfg.actor_rollout_ref.model.lora_rank == 16 and cfg.actor_rollout_ref.model.lora_alpha == 32
    assert cfg.actor_rollout_ref.actor.optim.lr == 1e-6
    assert cfg.actor_rollout_ref.actor.entropy_coeff == 0 and cfg.actor_rollout_ref.actor.calculate_entropy
    assert cfg.actor_rollout_ref.rollout.n == 4 and cfg.actor_rollout_ref.rollout.max_model_len == 24576
    assert cfg.data.seed == 42
    assert cfg.actor_rollout_ref.rollout.val_kwargs.n == 1
    assert cfg.actor_rollout_ref.rollout.val_kwargs.do_sample is False
    (prep / 'composed_config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
    # 直接调用纯文件合同检查，不调用需要 CUDA 的 main。
    checker_spec = importlib.util.spec_from_file_location('formal_checker', root / 'scripts/check_grpo_runtime.py')
    checker = importlib.util.module_from_spec(checker_spec)
    checker_spec.loader.exec_module(checker)
    checker.validate_environment_contract()
    for name, expected in checker.EXPECTED_VERSIONS.items():
        assert md.version(name).split('+', 1)[0] == expected, (name, md.version(name), expected)
    checker.validate_transformers_revision()

snap = prep / 'snapshot'
snap.mkdir(exist_ok=True)
hashes = {}
rels = set(plan['source_sha256']) | repaired | {'configs/agent_loop.yaml', 'configs/tools.json'}
for rel in sorted(rels):
    path = root / rel
    hashes[str(path)] = digest(path)
    dest = snap / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(path.read_bytes())
hashes[str(ray_source)] = digest(ray_source)
(snap / 'installed_ray_trainer.py').write_bytes(ray_source.read_bytes())
(prep / 'packages.json').write_text(json.dumps(sorted((d.metadata['Name'], d.version)
    for d in md.distributions() if d.metadata['Name']), indent=2))

# 用完整的短跑 checkpoint 实测大小，避免按 LoRA 文件很小误判空间。
folders = [run / 'smoke/global_step_1',
           run / 'smoke_resume_entropy_20261024_103342/global_step_2',
           run / 'smoke_resume_entropy_20261024_103342/global_step_3']
sizes = []
for folder in folders:
    for name in ('model_world_size_1_rank_0.pt', 'optim_world_size_1_rank_0.pt',
                 'extra_state_world_size_1_rank_0.pt'):
        path = folder / 'actor' / name
        assert path.is_file() and path.stat().st_size > 0, path
    sizes.append(sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()))
largest = max(sizes)
required = largest * 10 + (12 + 5 + 10) * 2**30
available = shutil.disk_usage(root).free
budget = {'checkpoint_observed_bytes': largest, 'candidate_count': 10,
          'export_reserve_gib': 12, 'log_reserve_gib': 5, 'headroom_gib': 10,
          'required_free_bytes': required, 'available_bytes_when_prepared': available}
(prep / 'disk_budget.json').write_text(json.dumps(budget, indent=2))

argv = [sys.executable, str(root / 'scripts/train_grpo.py'), '--model', str(args.model),
        '--train-data', str(args.train_data), '--val-data', str(args.val_data),
        '--config', str(args.config), '--output', str(out), '--env-url', args.env_url,
        '--logger', 'console', '--experiment-name', args.experiment_name, '--', *overrides]
record = {'project': str(root), 'output': str(out), 'command': argv,
          'preflight': [sys.executable, str(root / 'scripts/check_grpo_runtime.py'), *command[5:]],
          'env': selected_env, 'source_sha256': hashes, 'total_updates': 500,
          'validation_every': 50, 'checkpoint_every': 50,
          'checkpoint_selection': 'Dev-50 only; preserve scheduled candidates; Final excluded',
          'required_free_bytes': required}
with (prep / 'dry_run.log').open('w') as log:
    pos = argv.index('--')
    subprocess.run([*argv[:pos], '--dry-run', *argv[pos:]], cwd=root, env=env,
                   stdout=log, stderr=subprocess.STDOUT, check=True)
(prep / 'command.txt').write_text(shlex.join(argv) + '\n')
(prep / 'launch_spec.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
print(f'单份 checkpoint 观测大小：{largest / 2**30:.2f} GiB')
print(f'正式训练所需可用空间门槛：{required / 2**30:.1f} GiB；当前 {available / 2**30:.1f} GiB')
print(f'至少还需增加的可用空间：{max(0, required-available)/2**30:.1f} GiB')
print('正式配置已固定，尚未训练：', prep)
PY
```

预算按 10 个计划保存点计算，额外手工复制、另跑实验和额外强制保存会另占空间。保留源目录时不要同时批量导出所有 10 个候选；导出一个确认后的候选即可。

## 3. 写入后台启动脚本（无卡模式可做）

脚本在 GPU 模式实际运行时才检查 CUDA、内存配额、磁盘余量和源码。日志用独占方式创建，重复执行不会覆盖已有训练日志。

```bash
cat > "$FORMAL_PREP/launch_formal.py" <<'PY'
import hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

prep = Path(__file__).resolve().parent
spec = json.loads((prep / 'launch_spec.json').read_text())
for name, expected in spec['source_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
out = Path(spec['output'])
assert not out.exists() or (out.is_dir() and not any(out.iterdir())), '输出已非空，不能当成新训练重启。'
assert shutil.disk_usage(spec['project']).free >= spec['required_free_bytes'], '磁盘未达到封存预算。'
import torch
assert torch.cuda.is_available()
free, _ = torch.cuda.mem_get_info()
assert free >= 90 * 2**30, f'GPU 可用显存不足：{free/2**30:.1f} GiB'
limit = Path('/sys/fs/cgroup/memory.max')
if limit.exists() and limit.read_text().strip() != 'max':
    assert int(limit.read_text()) >= 110 * 2**30, 'CPU 内存配额低于本次采用的 110 GiB 配置。'
env = dict(os.environ, **spec['env'])
start = time.time()
with (prep / 'train.log').open('x') as log:
    (prep / 'train.pid').write_text(str(os.getpid()) + '\n')
    status = subprocess.call(spec['command'], cwd=spec['project'], env=env,
                             stdout=log, stderr=subprocess.STDOUT)
    log.write(f'\nTRAIN_EXIT_CODE={status}\n')
(prep / 'exit_status.json').write_text(json.dumps({'exit_code': status,
    'started_unix': start, 'ended_unix': time.time(), 'elapsed_seconds': time.time()-start}, indent=2))
print(f'TRAIN_EXIT_CODE={status}', flush=True)
sys.exit(status)
PY

.venv/bin/python - <<'PY'
import ast, os
from pathlib import Path
path = Path(os.environ['FORMAL_PREP']) / 'launch_formal.py'
ast.parse(path.read_text())
print('后台脚本语法通过，尚未启动训练。')
PY
```

完成这里后，可以先结束无卡会话，在 AutoDL 控制台扩容到总计约 200G，再切换有卡模式。

## 4. 恢复会话，核对 GPU 与磁盘（需要有卡模式）

重新开机后先执行本段；不要重新生成第 2 步配置。磁盘不够时按打印的缺口补扩，训练不会启动。本段在子 shell 中执行严格检查，失败只结束子 shell，保留当前终端和诊断日志。

```bash
set +e
set +u
set +o pipefail
RESOURCE_LOG="/tmp/grpo_resource_check_$(date +%Y%m%d_%H%M%S).log"
(
set -euo pipefail
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/formal_session.env
cd "$PROJECT"

df -h "$PROJECT"
nvidia-smi
.venv/bin/python - <<'PY'
import json, os, shutil
from pathlib import Path
import torch
spec = json.loads((Path(os.environ['FORMAL_PREP']) / 'launch_spec.json').read_text())
available = shutil.disk_usage(spec['project']).free
required = spec['required_free_bytes']
print(f'磁盘可用 {available/2**30:.1f} GiB；要求至少 {required/2**30:.1f} GiB')
assert available >= required, '请继续扩容或先归档另行确认不再需要的权重；本指令不删除历史产物。'
assert torch.cuda.is_available()
free, total = torch.cuda.mem_get_info()
print(f'GPU 可用/总量：{free/2**30:.1f}/{total/2**30:.1f} GiB')
assert free >= 90 * 2**30, '请先停止本实验旧的独立模型服务或其他 GPU 任务。'
limit = Path('/sys/fs/cgroup/memory.max')
if limit.exists() and limit.read_text().strip() != 'max':
    print('CPU 内存限额 GiB：', int(limit.read_text()) / 2**30)
    assert int(limit.read_text()) >= 110 * 2**30, 'CPU 内存配额低于本次采用的 110 GiB 配置。'
print('资源检查通过。')
PY
) 2>&1 | tee "$RESOURCE_LOG"
RESOURCE_STATUS=${PIPESTATUS[0]}
echo "资源检查退出码：$RESOURCE_STATUS"
echo "完整日志：$RESOURCE_LOG"
echo '仅退出码为 0 时继续第 5 步；否则先处理日志中的错误。'
if [[ "$RESOURCE_STATUS" -eq 0 ]]; then
  # 子 shell 的 export 不会返回父终端；成功后在当前终端恢复后续步骤所需的变量。
  source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/formal_session.env
  cd "$PROJECT"
fi
```

## 5. 启动 ShopSimulator，验证终端和 Ray worker，并运行预检（有卡模式执行）

环境服务本身不使用 GPU；放在这里执行是因为模式切换或关机会结束旧服务。GRPO 训练需要的独立服务是 ShopSimulator，**不要另启 SFT 评测用的 vLLM 服务**。

```bash
if .venv/bin/python - <<'PY'
import socket
with socket.create_connection(('127.0.0.1', 5700), timeout=2): pass
PY
then
  echo '5700 已监听，下面核对真实环境接口。'
else
  nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 \
    SHOPSIM_PORT=5700 SHOPSIM_ENV_SLOTS=8 SHOP_MAX_STEPS=35 \
    bash scripts/start_environment.sh > "$FORMAL_PREP/environment_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
  echo "$!" > "$FORMAL_PREP/environment.pid"
fi

.venv/bin/python - <<'PY'
import json, os, socket, time
from contextlib import ExitStack
from pathlib import Path
from shopping_grpo.environment.client import ShopAgentEnv

deadline = time.monotonic() + 120
while True:
    try:
        with socket.create_connection(('127.0.0.1', 5700), timeout=2): break
    except OSError:
        if time.monotonic() >= deadline: raise SystemExit('环境启动超时，请查看本次 environment 日志。')
        time.sleep(2)
root = Path(os.environ['PROJECT'])
task = int(json.loads((root / 'data/grpo/train.jsonl').read_text().splitlines()[0])['task_id'])
with ExitStack() as stack:
    indices = []
    for _ in range(8):
        env = stack.enter_context(ShopAgentEnv(os.environ['SHOPSIM_BASE_URL']))
        result = env.reset(task)
        assert result['environment_version'] == 'shopsimulator-environment-v2.1'
        indices.append(env.env_idx)
    assert len(set(indices)) == 8
print('8 个槽位已实测可用并释放：', indices)

import ray
ray.init(address='local', num_cpus=1, num_gpus=0, object_store_memory=128*1024**2,
    include_dashboard=False, runtime_env={'env_vars': {
        'PYTHONPATH': str(root / 'src'), 'SHOPSIM_BASE_URL': os.environ['SHOPSIM_BASE_URL'],
        'OMP_NUM_THREADS': '1', 'VLLM_USE_FLASHINFER_SAMPLER': '0'}})
try:
    @ray.remote(num_cpus=0.1)
    def probe(task):
        import os, torch
        from shopping_grpo.environment.client import ShopAgentEnv
        with ShopAgentEnv(os.environ['SHOPSIM_BASE_URL']) as env:
            result = env.reset(task)
            assert result['environment_version'] == 'shopsimulator-environment-v2.1'
        return {'reset_and_release': 'passed',
                'environment_version': result['environment_version'],
                'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
                'torch_num_threads': torch.get_num_threads(),
                'VLLM_USE_FLASHINFER_SAMPLER': os.environ.get('VLLM_USE_FLASHINFER_SAMPLER')}
    result = ray.get(probe.remote(task))
    print('Ray worker：', result)
    (Path(os.environ['FORMAL_PREP']) / 'environment_probe.json').write_text(json.dumps(result, indent=2))
    assert result['environment_version'] == 'shopsimulator-environment-v2.1', result
    assert result['OMP_NUM_THREADS'] == '1', result
    assert result['VLLM_USE_FLASHINFER_SAMPLER'] == '0', result
    assert result['torch_num_threads'] > 0, result
    # OMP 环境变量与 PyTorch 已初始化的线程池分别检查，不要求二者数值必然相等。
finally:
    ray.shutdown()
PY

.venv/bin/python - <<'PY'
import json, os, subprocess
from pathlib import Path
prep = Path(os.environ['FORMAL_PREP'])
spec = json.loads((prep / 'launch_spec.json').read_text())
with (prep / 'preflight.log').open('w') as log:
    status = subprocess.call(spec['preflight'], cwd=spec['project'],
        env=dict(os.environ, **spec['env']), stdout=log, stderr=subprocess.STDOUT)
print((prep / 'preflight.log').read_text())
assert status == 0, f'正式运行时预检失败：{status}'
print('正式运行时预检通过。')
PY
```

## 6. 启动正式 GRPO（需要 GPU；从此开始实际训练）

```bash
if [[ -e "$FORMAL_PREP/train.log" || -e "$FORMAL_PREP/driver.log" || -e "$FORMAL_PREP/train.pid" ]]; then
  echo '本次正式会话已有启动记录，请先查看进程与日志，不重复启动。'
else
  nohup .venv/bin/python -u "$FORMAL_PREP/launch_formal.py" \
    > "$FORMAL_PREP/driver.log" 2>&1 < /dev/null &
  echo "后台启动器 PID=$!"
fi

tail -n 40 "$FORMAL_PREP/driver.log"
tail -n 60 -F "$FORMAL_PREP/train.log"
```

`tail -F` 在日志尚未创建时可能暂时提示文件不存在，会继续重试。若启动器已经失败，错误在 `driver.log` 中；不要无视它反复启动。进入训练前先跑一次完整 Dev-50 验证，此时暂时没有 GRPO 更新记录是正常的。之后在 step 50/100/…/500 保存并验证；达到 500 或训练因错误提前结束后，退出码写入 `exit_status.json`。

`Ctrl+C` 退出本段的日志查看器，不会终止 `nohup` 的训练。可以关闭本地电脑；训练未结束时不要关闭 AutoDL 实例。正式第一个保存点是 step 50：在它之前中断没有本次正式 checkpoint，之后中断通常只能从最近保存点恢复，丢失其后的未保存更新。

## 7. 查看进度或完成后验收（标量日志检查可无卡；训练未结束时保持有卡）

新终端先恢复会话：

```bash
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/formal_session.env
cd "$PROJECT"
set -euo pipefail

.venv/bin/python - <<'PY'
import collections, json, math, numbers, os, shutil
from pathlib import Path
prep = Path(os.environ['FORMAL_PREP'])
out = Path(os.environ['FORMAL_OUT'])
diagnostics = out / 'training_diagnostics.jsonl'
counts = collections.Counter()
steps, last, validations = [], None, []

def check_finite(value):
    if isinstance(value, dict):
        for item in value.values(): check_finite(item)
    elif isinstance(value, list):
        for item in value: check_finite(item)
    elif isinstance(value, numbers.Real):
        assert math.isfinite(value), f'非有限指标：{value}'

if diagnostics.exists():
    with diagnostics.open() as f:
        for line in f:
            if not line.strip(): continue
            try: event = json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith('\n') and not (prep / 'exit_status.json').exists():
                    print('最后一行正在写入，本次忽略。')
                    continue
                raise
            counts[event['event']] += 1
            if event['event'] != 'optimizer_step': continue
            m = event['metrics']
            check_finite(m)
            assert int(m['training/optimizer_updated']) == 1
            step = int(event['global_step'])
            steps.append(step)
            last = {'step': step, 'lr': m.get('actor/lr'), 'grad_norm': m.get('actor/grad_norm'),
                    'group_effective_ratio': m.get('group/effective_ratio')}
            val = {k: v for k, v in m.items() if k.startswith(('val-core/', 'val-aux/'))}
            if val: validations.append({'step': step, 'metrics': val})
assert steps == list(range(1, len(steps)+1)), '更新步数不连续或重复。'
print('诊断事件数量：', dict(counts))
print('已完成更新：', len(steps), '/ 500；最近一次：', last)
available = shutil.disk_usage(Path(os.environ['PROJECT'])).free / 2**30
print(f'当前磁盘可用：{available:.1f} GiB')
print('验证输出：', sorted(p.name for p in (out / 'validation').glob('*.jsonl')))
status_file = prep / 'exit_status.json'
if status_file.exists():
    status = json.loads(status_file.read_text())
    print('进程退出状态：', status)
    assert status['exit_code'] == 0, '训练失败；先检查 train.log / driver.log，不直接重新启动。'
    assert steps == list(range(1, 501)), '正常退出但未完成预期 500 步，请核查。'
    for step in range(50, 501, 50):
        checkpoint = out / f'global_step_{step}'
        for name in ('model_world_size_1_rank_0.pt', 'optim_world_size_1_rank_0.pt',
                     'extra_state_world_size_1_rank_0.pt', 'fsdp_config.json'):
            path = checkpoint / 'actor' / name
            assert path.is_file() and path.stat().st_size > 0, path
        assert (checkpoint / 'data.pt').is_file(), checkpoint
    # 按内容中的 step 汇总，不依赖验证文件的具体命名方式。
    validation_counts = collections.Counter()
    for path in (out / 'validation').glob('*.jsonl'):
        with path.open() as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                assert math.isfinite(float(row['score']))
                validation_counts[int(row['step'])] += 1
    assert dict(validation_counts) == {s: 50 for s in [0, *range(50, 501, 50)]}, dict(validation_counts)
    report = {'exit_status': status, 'optimizer_updates': len(steps),
              'events': dict(counts), 'periodic_validation_metrics': validations,
              'validation_sample_counts': dict(validation_counts),
              'checkpoint_and_validation_basic_checks': 'passed'}
    (prep / 'formal_acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('500 步更新、10 个保存点和 11 次验证输出基础检查通过。')
else:
    print('尚无最终退出状态：可能仍在初始化、验证或训练；结合进程和日志确认。')
PY

df -h "$PROJECT"
tail -n 40 "$FORMAL_PREP/driver.log"
tail -n 60 "$FORMAL_PREP/train.log"
```

训练完成后再检查 8 个槽位释放情况（不使用 GPU；无卡模式只要环境服务仍在，也可执行）。**不要在训练进行中一次占满 8 个槽位**：

```bash
.venv/bin/python - <<'PY'
import json, os
from contextlib import ExitStack
from pathlib import Path
from shopping_grpo.environment.client import ShopAgentEnv
prep = Path(os.environ['FORMAL_PREP'])
assert json.loads((prep / 'exit_status.json').read_text())['exit_code'] == 0
root = Path(os.environ['PROJECT'])
task = int(json.loads((root / 'data/grpo/train.jsonl').read_text().splitlines()[0])['task_id'])
with ExitStack() as stack:
    indices = []
    for _ in range(8):
        env = stack.enter_context(ShopAgentEnv(os.environ['SHOPSIM_BASE_URL']))
        env.reset(task)
        indices.append(env.env_idx)
    assert len(set(indices)) == 8
(prep / 'lease_check.json').write_text(json.dumps({'indices': indices, 'all_released': True}, indent=2))
print('8 个槽位均可申请，且已全部释放：', indices)
PY
```

到这里完成的是正式训练与基础验收。下一步读取 Dev-50 验证表现选候选、按固定独立协议核验并导出，再进行一次正式 Final-200 Clean 对比。不要直接默认 step 500 最好，也不要将训练 batch 的奖励当成 Dev 成绩。

若进程中断，保留原输出目录、日志和最近完整 checkpoint；本文件的启动器专门用于首次从 SFT 开跑，**不能原样重跑它来续训**。恢复时需保持总训练上限和调度器设置，从正式 checkpoint 恢复到新的输出目录，并连接已有记录；不能回到短跑 checkpoint 或更改总步数冒充同一次实验。

## 8. 训练已经开始后，加挂结束自动关机（可选，现在可执行）

按第 6 步的 `nohup` 启动后，本地电脑断网、关机或关闭网页不影响云端训练。自动关机监控也使用 `nohup`，不依赖本地电脑保持在线。

另开一个 AutoDL 终端执行下面整段；不要重启训练，也不要修改正在运行的 launcher。监控等待本次 launcher 写出 `exit_status.json`，这发生在训练子进程退出之后，包含最终保存与验证。**成功或报错退出都会等待约 60 秒后关机**，以避免失败后的 GPU 空转。若进程被强杀、没有写出状态，监控停止并留日志，不把它误判成训练结束而关机。

AutoDL 官方支持使用 [`/usr/bin/shutdown`](https://api.autodl.com/docs/save_money/) 结束实例运行。本段不删除模型或释放实例；关机后付费扩容盘仍按平台规则收费，见[数据盘说明](https://www.autodl.com/docs/local_disk/)。

```bash
set +e
set +u
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/formal_session.env
cd "$PROJECT"

cat > "$FORMAL_PREP/auto_shutdown.py" <<'PY'
import fcntl, json, os, subprocess, time
from pathlib import Path

def monitor(prep, do_shutdown, proc_root=Path('/proc'), pause=time.sleep):
    pid = int((prep / 'train.pid').read_text())
    status_file = prep / 'exit_status.json'
    cancel = prep / 'auto_shutdown.cancel'
    launcher = str((prep / 'launch_formal.py').resolve())

    def token():
        try:
            fields = (proc_root / str(pid) / 'stat').read_text().rpartition(')')[2].split()
            return None if fields[0] == 'Z' else fields[19]
        except FileNotFoundError:
            return None

    identity = token()
    if identity is not None:
        args = (proc_root / str(pid) / 'cmdline').read_bytes().split(b'\0')
        assert launcher.encode() in args, 'PID 不属于本次 launcher，拒绝监控。'
    elif not status_file.exists():
        raise RuntimeError('训练 launcher 不在，且没有退出状态；不自动关机。')
    print(f'监控已启动：launcher PID={pid}；等待 {status_file}', flush=True)

    while True:
        if cancel.exists():
            print('已取消自动关机。', flush=True)
            return
        result = None
        if status_file.exists():
            try:
                result = json.loads(status_file.read_text())
            except json.JSONDecodeError:
                pass  # 状态文件可能正处于写入过程中。
        if result is not None:
            assert type(result.get('exit_code')) is int, result
            assert result['ended_unix'] >= result['started_unix'], result
            break
        if identity is None or token() != identity:
            raise RuntimeError('launcher 已消失或 PID 被复用，但没有完整退出状态；不自动关机。')
        pause(15)

    print('训练子进程已退出：' + json.dumps(result), flush=True)
    print('约 60 秒后关机；期间可创建 auto_shutdown.cancel 取消。', flush=True)
    for _ in range(12):
        if cancel.exists():
            print('已取消自动关机。', flush=True)
            return
        pause(5)
    if cancel.exists(): return
    (prep / 'auto_shutdown_result.json').write_text(json.dumps({
        'training_exit_status': result, 'shutdown_requested_unix': time.time()
    }, indent=2))
    os.sync()
    if cancel.exists(): return
    print('日志已同步，调用 AutoDL 关机命令。', flush=True)
    do_shutdown()

if __name__ == '__main__':
    prep = Path(__file__).resolve().parent
    shutdown = '/usr/bin/shutdown'
    assert os.access(shutdown, os.X_OK), f'关机命令不可执行：{shutdown}'
    with (prep / 'auto_shutdown.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('已有自动关机监控在运行，本次不重复启动。')
        (prep / 'auto_shutdown.pid').write_text(str(os.getpid()))
        monitor(prep, lambda: subprocess.run([shutdown], check=True))
PY

nohup .venv/bin/python -u "$FORMAL_PREP/auto_shutdown.py" \
  >> "$FORMAL_PREP/auto_shutdown.log" 2>&1 < /dev/null &
echo "监控启动 PID=$!"
sleep 2
tail -n 20 "$FORMAL_PREP/auto_shutdown.log"
```

看到“监控已启动”且无后续报错，才表示监控正常挂上。它不会重新启动训练。成功关机后，下次可无卡开机读取训练日志与执行文件基础验收；需要模型推理时再恢复有卡模式。

如需取消自动关机，在调用关机命令之前执行（不影响训练）：

```bash
touch /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/formal_launch/auto_shutdown.cancel
```

## 9. 本次正式运行 CUDA OOM 的只读诊断（无卡模式可做）

本段不导入 Torch、Ray，不加载 checkpoint 权重，也不修改数据或训练参数。若实例已由自动关机监控关闭，可无卡开机执行。若仍在运行且训练已经退出，可先关机再无卡开机，避免 GPU 空转；不要释放实例。

输出包含：本次退出状态、最后几次已提交更新、最近生成批次的上下文记录、checkpoint 基础完整性，以及实际安装版本的 log-probability 计算源码。只有文件存在及大小合理，不代表 checkpoint 已经通过可加载性验证。

```bash
set +e
set +u
/root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/.venv/bin/python - <<'PY'
import ast, collections, hashlib, json
from pathlib import Path

root = Path('/root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon')
run = root / 'outputs/runs/grpo_baseline_20261022_202816'
prep = run / 'preparation/formal_launch'
spec = json.loads((prep / 'launch_spec.json').read_text())
out = Path(spec['output'])
print('训练输出：', out)
launcher_pid = prep / 'train.pid'
print('本次 launcher 的记录 PID：', launcher_pid.read_text().strip() if launcher_pid.exists() else '未记录')
status = prep / 'exit_status.json'
print('退出状态：', status.read_text() if status.exists() else '未发现退出状态文件')
for key in ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF', 'OMP_NUM_THREADS'):
    print('封存环境变量', key, '=', spec['env'].get(key, '未在显式环境中设置'))

counts = collections.Counter()
updates = collections.deque(maxlen=3)
last_generation = None
diagnostics = out / 'training_diagnostics.jsonl'
if diagnostics.exists():
    with diagnostics.open() as f:
        for line in f:
            if not line.strip(): continue
            try: event = json.loads(line)
            except json.JSONDecodeError:
                print('忽略一行不完整诊断记录。')
                continue
            counts[event['event']] += 1
            if event['event'] == 'optimizer_step':
                m = event['metrics']
                selected = {k: v for k, v in m.items() if 'memory' in k.lower()
                    or k in ('actor/lr', 'actor/grad_norm', 'response_length/max',
                             'response_length/mean', 'prompt_length/max')}
                updates.append({'step': event['global_step'], 'metrics': selected})
            elif event['event'] == 'generation_batch':
                last_generation = {'step': event['global_step'], 'groups': event.get('groups'),
                    'rollouts': [{k: r.get(k) for k in ('task_id', 'steps',
                        'context_max_input_tokens', 'error', 'infrastructure_invalid')}
                        for r in event.get('rollouts', [])]}
print('诊断事件数量：', dict(counts))
print('最近已提交更新：', json.dumps(list(updates), ensure_ascii=False, indent=2))
print('最近采样批次：', json.dumps(last_generation, ensure_ascii=False, indent=2))

reference = run / 'smoke/global_step_1/actor/model_world_size_1_rank_0.pt'
reference_size = reference.stat().st_size if reference.exists() else 0
folders = sorted(out.glob('global_step_*'), key=lambda p: int(p.name.rsplit('_', 1)[1]))
print('checkpoint 数量：', len(folders))
for folder in folders:
    names = ['actor/model_world_size_1_rank_0.pt', 'actor/optim_world_size_1_rank_0.pt',
             'actor/extra_state_world_size_1_rank_0.pt', 'actor/fsdp_config.json', 'data.pt']
    missing = [n for n in names if not (folder / n).is_file() or (folder / n).stat().st_size == 0]
    model = folder / names[0]
    size = model.stat().st_size if model.exists() else 0
    plausible = not missing and size >= reference_size * 0.99
    print(folder.name, '模型 MiB=', round(size/2**20, 2),
          '基础文件检查=', plausible, '缺失或空文件=', missing)

verl = root / '.venv/lib/python3.12/site-packages/verl'
loss = verl / 'workers/utils/losses.py'
plan = json.loads((run / 'preparation/baseline_plan.json').read_text())
print('原 entropy 补丁哈希是否保留：', hashlib.sha256(loss.read_bytes()).hexdigest()
      == plan['zero_coeff_entropy_fix']['after_sha256'])
for path, functions in [(verl / 'utils/torch_functional.py',
                        {'logprobs_from_logits', 'logprobs_from_logits_v2'})]:
    text = path.read_text()
    print('\n源码文件：', path, 'SHA256=', hashlib.sha256(path.read_bytes()).hexdigest())
    lines = text.splitlines()
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            print('\n' + '\n'.join(f'{i+1}: {lines[i]}' for i in range(node.lineno-1, node.end_lineno)))

engine = verl / 'workers/engine/fsdp/transformer_impl.py'
lines = engine.read_text().splitlines()
print('\nFSDP 引擎 SHA256：', hashlib.sha256(engine.read_bytes()).hexdigest())
# 围绕这次调用栈所指的真实位置读取，不拿网上版本行号替代本机实现。
for i in range(max(0, 1214-24), min(len(lines), 1214+14)):
    print(f'{i+1}: {lines[i]}')
hits = [i for i, line in enumerate(lines) if 'enable_activation_offload' in line][:3]
for i in hits:
    print('\nactivation offload 相关实现：')
    for j in range(max(0, i-3), min(len(lines), i+5)):
        print(f'{j+1}: {lines[j]}')
PY
```
