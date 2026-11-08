# 正式 GRPO：第 36 步 CUDA OOM 的修复与重跑

## 已确认的事实

- 本次进程耗时 `3394.31` 秒，约 **56 分 34 秒**，包含初始化和训练前验证；完成 **35 次优化器更新**，在第 36 次更新的前向 `log_softmax` 处 OOM。
- `generation_batch=151`、`skipped_update=23`、`optimizer_step=35`。生成批次及跳过更新不等于已训练步数。
- 第 36 步最近采样中，有轨迹的最大输入上下文达到 **21,774 tokens**；不是环境执行异常，两个组均有奖励差异且通过过滤。完整失败 micro batch 的张量形状未记录，不能仅凭上下文记录计算精确显存需求。
- **正式 checkpoint 数量为 0**。首个保存点原设为 step 50，因此不能恢复第 35 步，必须从选定的 SFT step 300 重新开始；旧短跑 step 3 也不能代替丢失的正式状态。
- entropy 的原 `detach` 补丁仍在。但当前引擎先计算 entropy、再计算 log-probability，最后才进入 loss。下游 detach 无法降低到达 loss 之前的前向峰值。
- actor micro batch 已为 1，当前引擎仍用 `torch.cat` 将单个样本的 logits 再复制一次。新方案在单样本时直接使用 view，多样本保留原拼接。
- 报错中占 550 MiB 的 PID 3179 正好是本次 launcher；它为了检查 GPU 导入 Torch 并建立了长期存在的 CUDA 上下文。新 launcher 将这项检查放在短命子进程，检查完就释放。

本次先修复上述明确的张量生命周期与复制问题。暂不启用 CPU activation offload，也不单凭缓存统计认定分配器开关能解决问题。

## 本轮调整

| 项目 | 调整 |
| --- | --- |
| entropy 前向 | 设置 `actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=true`，显式传入 `use_reentrant=True`；保留 entropy 日志与原零系数 detach 补丁 |
| 单样本 logits | 使用现有张量 view，避免单元素 `torch.cat` 的完整复制 |
| launcher | GPU 资源检查改为子进程，移除长期驻留的检查用 CUDA 上下文 |
| 保存时机 | 除 50/100/…/500 外，增加 **1、10、25** 三个早期保存点；滚动保留最多 10 份，正常计划下最终保留 50/100/…/500 |
| 优化目标 | 不改学习率、500 步调度、LoRA、奖励、4 条采样、batch 或 24,576 长度预算 |

entropy 系数为 0，因此该分支不贡献策略梯度；checkpointing 降低其中间结果的保留。view 与单元素 cat 的数值相同。局部 CPU 测试覆盖 FP64/FP32/BF16，CUDA autocast BF16 测试覆盖单样本、含 padding 及双样本，log-probability、loss 与梯度均逐元素一致。**本地测试环境是 Torch 2.6.0/CUDA 11.8，不是云端 Torch 2.11；所以仍需下方云端检验。** 局部算子检验不替代完整模型训练，不能承诺所有后续批次都不会 OOM。

原理参考 [PyTorch checkpoint](https://docs.pytorch.org/docs/2.11/checkpoint.html)；具体补丁以用户贴出的云端文件哈希为准。本节会修改安装包源码并保留前后快照，属于固定训练配方的显存修复版本，不能称为完全未经修改的运行实现。

## 1. 应用修复并准备独立重跑（无卡模式可完成）

**只执行一次创建配置。** 若中途脚本报错，先保留输出再排查；不要启动旧 `formal` 目录。代码不加载权重、不导入 Torch。旧退出状态、日志、诊断和配置保持原样，旧关机监控会被取消，避免对新跑误触发。

```bash
set +e
set +u
/root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/.venv/bin/python - <<'PY'
import argparse, ast, hashlib, importlib.util, json, os, shlex, shutil, subprocess, sys
from pathlib import Path

root = Path('/root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon')
run = root / 'outputs/runs/grpo_baseline_20261022_202816'
old_prep = run / 'preparation/formal_launch'
prep = run / 'preparation/formal_memfix_v1'
out = run / 'formal_memfix_v1'
old = json.loads((old_prep / 'launch_spec.json').read_text())
assert json.loads((old_prep / 'exit_status.json').read_text())['exit_code'] == 1
assert not (prep / 'launch_spec.json').exists(), '修复版配置已生成，不重复覆盖；恢复新会话后继续。'
assert not out.exists() or (out.is_dir() and not any(out.iterdir())), '新输出已经非空。'
prep.mkdir(parents=True, exist_ok=True)
fixes = prep / 'fixes'
fixes.mkdir(exist_ok=True)
(old_prep / 'auto_shutdown.cancel').touch()
verl = root / '.venv/lib/python3.12/site-packages/verl'
engine = verl / 'workers/engine/fsdp/transformer_impl.py'
trainer = verl / 'trainer/ppo/ray_trainer.py'
sha = lambda data: hashlib.sha256(data).hexdigest()
baseline = json.loads((run / 'preparation/baseline_plan.json').read_text())
def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()
for rel, expected in baseline['data_sha256'].items():
    assert digest(root / rel) == expected, rel
for name, expected in baseline['model_sha256'].items():
    assert digest(Path(baseline['model']) / name) == expected, name
torch_functional = verl / 'utils/torch_functional.py'
assert digest(torch_functional) == 'cb37d045dec2161d637e2633032cc97c97334e5547cea62dace42630c9683259'
for name, expected in old['source_sha256'].items():
    if Path(name) not in (engine, trainer):
        assert sha(Path(name).read_bytes()) == expected, name

receipts = []
def patch(path, expected_sha, transform, label):
    backup = fixes / (label + '.before.py')
    original = backup.read_bytes() if backup.exists() else path.read_bytes()
    assert sha(original) == expected_sha, f'未知版本：{path}'
    fixed = transform(original.decode()).encode()
    ast.parse(fixed.decode())
    assert path.read_bytes() in (original, fixed), f'存在其他未识别修改：{path}'
    if not backup.exists(): backup.write_bytes(original)
    (fixes / (label + '.after.py')).write_bytes(fixed)
    if path.read_bytes() != fixed:
        temp = path.with_name(path.name + f'.memoryfix-{os.getpid()}.tmp')
        temp.write_bytes(fixed)
        temp.chmod(path.stat().st_mode & 0o777)
        os.replace(temp, path)
    receipts.append({'file': str(path), 'before_sha256': sha(original), 'after_sha256': sha(fixed)})

def fix_engine(text):
    lines = text.splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines) if line.strip() == 'logits_rmpad = torch.cat([t for t in logits.unbind()])']
    assert len(matches) == 1
    i = matches[0]
    indent = lines[i][:len(lines[i])-len(lines[i].lstrip())]
    lines[i] = (indent + '# SHOPPING_GRPO_SINGLE_ITEM_LOGITS_VIEW_V1\n'
                + indent + 'logits_parts = logits.unbind()\n'
                + indent + 'logits_rmpad = logits_parts[0] if len(logits_parts) == 1 else torch.cat(logits_parts)\n')
    text = ''.join(lines)
    call = 'torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)'
    assert text.count(call) == 1
    return text.replace(call, call[:-1] + ', use_reentrant=True)')

def fix_saves(text):
    lines = text.splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines) if line.strip() == 'or current_step % self.config.trainer.save_freq == 0']
    assert len(matches) == 1
    i = matches[0]
    indent = lines[i][:len(lines[i])-len(lines[i].lstrip())]
    lines[i] += (indent + '# SHOPPING_GRPO_EARLY_SAVE_STEPS_V1\n'
                 + indent + 'or current_step in self.config.trainer.get("early_save_steps", [])\n')
    return ''.join(lines)

patch(engine, 'b4f6243471f22c08dbfa472d13f708e8de609705ecff13f23713258160cf7902', fix_engine, 'fsdp_engine')
patch(trainer, old['source_sha256'][str(trainer)], fix_saves, 'ray_trainer')
(fixes / 'receipt.json').write_text(json.dumps(receipts, indent=2))

# 沿用失败运行的实际参数；只覆盖本节明确列出的工程选项。
argv = list(old['command'])
split = argv.index('--')
prefix, overrides = argv[:split], argv[split+1:]
def option(name): return prefix[prefix.index(name)+1]
prefix[prefix.index('--output')+1] = str(out)
prefix[prefix.index('--experiment-name')+1] = 'shopping-grpo-formal-memfix-v1'
changes = {
    'trainer.total_training_steps': '500', 'trainer.save_freq': '50',
    'trainer.test_freq': '50', 'trainer.val_before_train': 'true',
    'trainer.max_actor_ckpt_to_keep': '10', 'trainer.resume_mode': 'disable',
    'trainer.validation_data_dir': json.dumps(str(out / 'validation')),
    'trainer.early_save_steps': '[1,10,25]',
    'actor_rollout_ref.actor.entropy_checkpointing': 'true',
    'actor_rollout_ref.actor.fsdp_config.entropy_checkpointing': 'true',
}
def key(value): return value.lstrip('+').split('=', 1)[0]
overrides = [v for v in overrides if key(v) not in changes and key(v) != 'trainer.resume_from_path']
overrides += [('++' if k in ('trainer.early_save_steps', 'trainer.validation_data_dir') else '')
              + k + '=' + v for k, v in changes.items()]
argv = prefix + ['--'] + overrides
env = dict(old['env'])
env.update(GRPO_OUTPUT_DIR=str(out), SHOPPING_GRPO_DIAGNOSTICS_PATH=str(out / 'training_diagnostics.jsonl'))
os.environ.update(env)

loader = importlib.util.spec_from_file_location('memoryfix_launcher', root / 'scripts/train_grpo.py')
module = importlib.util.module_from_spec(loader)
loader.loader.exec_module(module)
args = argparse.Namespace(model=Path(option('--model')), train_data=Path(option('--train-data')),
    val_data=Path(option('--val-data')), config=Path(option('--config')), output=out,
    env_url=option('--env-url'), logger='console', experiment_name=option('--experiment-name'),
    hydra_overrides=overrides)
command, _ = module.build_command(args)
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
with initialize_config_dir(version_base=None, config_dir=str(root / 'configs')):
    cfg = compose(config_name='grpo', overrides=command[5:])
    assert cfg.trainer.total_training_steps == 500 and cfg.trainer.resume_mode == 'disable'
    assert list(cfg.trainer.early_save_steps) == [1, 10, 25]
    assert cfg.trainer.save_freq == cfg.trainer.test_freq == 50
    assert cfg.actor_rollout_ref.actor.fsdp_config.entropy_checkpointing
    assert cfg.actor_rollout_ref.actor.entropy_checkpointing
    assert cfg.actor_rollout_ref.actor.entropy_coeff == 0 and cfg.actor_rollout_ref.actor.calculate_entropy
    assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 1
    assert cfg.actor_rollout_ref.actor.optim.lr == 1e-6
    assert cfg.actor_rollout_ref.model.lora_rank == 16 and cfg.actor_rollout_ref.rollout.n == 4
    assert cfg.actor_rollout_ref.rollout.max_model_len == 24576
    (prep / 'composed_config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
    actor_source = (verl / 'workers/config/actor.py').read_text()
    assert 'self.engine = self.fsdp_config' in actor_source, '需要核对本机 actor → engine 配置映射。'

old_budget = json.loads((old_prep / 'disk_budget.json').read_text())
required = old_budget['checkpoint_observed_bytes'] * 11 + 27 * 2**30
available = shutil.disk_usage(root).free
print(f'最多保留 10 份、按 11 份短时峰值及预留计算：{required/2**30:.1f} GiB；当前可用 {available/2**30:.1f} GiB')

hashes = dict(old['source_sha256'])
hashes[str(engine)] = sha(engine.read_bytes())
hashes[str(trainer)] = sha(trainer.read_bytes())
hashes[str(torch_functional)] = digest(torch_functional)
snapshot = prep / 'snapshot'
for name, expected in hashes.items():
    path = Path(name)
    assert sha(path.read_bytes()) == expected, name
    dest = snapshot / path.relative_to(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(path.read_bytes())

# 从原 launcher 派生，保留全部资源检查与日志行为，只将 GPU 检查放入短命子进程。
driver = (old_prep / 'launch_formal.py').read_text()
tree = ast.parse(driver)
idx = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Import)
           and any(alias.name == 'torch' for alias in n.names))
nodes = tree.body[idx:idx+4]
assert [type(n).__name__ for n in nodes] == ['Import', 'Assert', 'Assign', 'Assert']
block = '\n'.join(driver.splitlines()[nodes[0].lineno-1:nodes[-1].end_lineno])
assert 'torch.cuda.is_available()' in block and 'torch.cuda.mem_get_info()' in block
gpu_check = ('subprocess.run([sys.executable, "-c", '
    '"import torch; assert torch.cuda.is_available(); '
    'free, _ = torch.cuda.mem_get_info(); assert free >= 90 * 2**30, free"], check=True)')
lines = driver.splitlines(keepends=True)
lines[nodes[0].lineno-1:nodes[-1].end_lineno] = [gpu_check + '\n']
driver = ''.join(lines)
driver = driver.replace("env = dict(os.environ, **spec['env'])",
    "probe = json.loads((prep / 'math_probe.json').read_text())\n"
    "assert probe['passed'] and probe['engine_sha256'] == spec['source_sha256'][probe['engine_file']]\n"
    "env = dict(os.environ, **spec['env'])")
assert "probe['passed']" in driver and '>= 110 * 2**30' in driver
ast.parse(driver)
(prep / 'launch_formal.py').write_text(driver)
(prep / 'launch_formal.before_gpu_probe_fix.py').write_text((old_prep / 'launch_formal.py').read_text())
if (old_prep / 'auto_shutdown.py').is_file():
    (prep / 'auto_shutdown.py').write_bytes((old_prep / 'auto_shutdown.py').read_bytes())

record = dict(old)
record.update(command=argv, output=str(out), env=env, source_sha256=hashes,
    preflight=[sys.executable, str(root / 'scripts/check_grpo_runtime.py'), *command[5:]],
    required_free_bytes=required, checkpoint_steps=[1,10,25,*range(50,501,50)],
    checkpoint_retention_limit=10, checkpoint_peak_slots_budgeted=11,
    restart_reason='CUDA OOM at attempted update 36; 35 completed updates had no checkpoint',
    failed_run=str(old['output']), initial_policy='same SFT step 300 with new LoRA')
with (prep / 'dry_run.log').open('w') as log:
    pos = argv.index('--')
    subprocess.run([*argv[:pos], '--dry-run', *argv[pos:]], cwd=root,
        env=dict(os.environ, **env), stdout=log, stderr=subprocess.STDOUT, check=True)
(prep / 'command.txt').write_text(shlex.join(argv) + '\n')
(prep / 'launch_spec.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
session = ''.join(f'export {k}={shlex.quote(v)}\n' for k, v in {
    'PROJECT': str(root), 'GRPO_RUN': str(run), 'FORMAL_PREP': str(prep), 'FORMAL_OUT': str(out),
    'PYTHONPATH': str(root / 'src'), 'OMP_NUM_THREADS': '1', 'VLLM_USE_FLASHINFER_SAMPLER': '0',
    'PYTHONHASHSEED': '42', 'SHOPPING_LENGTH_SHAPING_ENABLE': 'false',
    'SHOPSIM_BASE_URL': 'http://127.0.0.1:5700'}.items())
(prep / 'session.env').write_text(session)
(run / 'preparation/formal_memfix_session.env').write_text(session)
print('CPU 修复和配置冻结完成；尚未开始训练。')
print('新准备目录：', prep)
print('新训练输出：', out)
PY
```

当前版本最多保留 10 份，先保守预留保存瞬间多出一份的空间，门槛约 141.6 GiB 可用空间。若本机实现能确认在写入前清理旧文件，第 5 节可以进一步校准为约 131.2 GiB。此前已经生成 13 份保留配置的用户，不要重做第一步，使用第 5 节同步修正已有配置。所有门槛均指可用空间，不是数据盘总容量。

## 2. 云端数值与最长 logits 前后向检验（需要有卡模式）

这一步只检查关键计算组件，**不加载完整购物模型、不做 500 步训练**。先在小张量上比较修复前后数值和梯度，再按实际词表大小、24,576 tokens 检查修复路径的前向和反向。完整模型仍有额外参数与激活占用，本检验通过不代表完整训练的显存峰值已经测得。

```bash
set +e
set +u
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/formal_memfix_session.env
cd "$PROJECT"

if .venv/bin/python -u - <<'PY'
import gc, hashlib, json, os, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

assert torch.cuda.is_available()
free, _ = torch.cuda.mem_get_info()
assert free >= 90 * 2**30, f'先释放旧 GPU 进程，当前仅余 {free/2**30:.1f} GiB'
from verl.utils.torch_functional import logprobs_from_logits_v2, entropy_from_logits
prep = Path(os.environ['FORMAL_PREP'])
spec = json.loads((prep / 'launch_spec.json').read_text())
(prep / 'math_probe.json').unlink(missing_ok=True)

def calculate(x, lengths, labels, fixed):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        entropy = checkpoint(entropy_from_logits, x, use_reentrant=True) if fixed else entropy_from_logits(x)
        nested = torch.nested.narrow(x, 1, torch.zeros_like(lengths), lengths, layout=torch.jagged)
        parts = nested.unbind()
        packed = parts[0] if fixed and len(parts) == 1 else torch.cat(parts)
        lp = logprobs_from_logits_v2(packed, labels)
        loss = lp.mean() - 0.0 * entropy.detach().mean()
    loss.backward()
    return lp.detach(), loss.detach(), x.grad.detach(), entropy.detach()

for lengths_list in ([17], [11], [11,17]):
    torch.manual_seed(42)
    source = torch.randn(len(lengths_list), 17, 257, device='cuda', dtype=torch.bfloat16)
    lengths = torch.tensor(lengths_list, device='cuda')
    labels = torch.randint(257, (sum(lengths_list),), device='cuda')
    results = [calculate(source.clone().requires_grad_(), lengths, labels, fixed) for fixed in (False,True)]
    for a, b in zip(results[0][:3], results[1][:3]):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    assert torch.isfinite(results[0][3]).all() and torch.isfinite(results[1][3]).all()
    entropy_delta = float((results[0][3].float()-results[1][3].float()).abs().max())
    print('小张量 logprob/loss/gradient 一致：', lengths_list, 'entropy 最大差异：', entropy_delta, flush=True)
del source, lengths, labels, results
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

model = Path(spec['env']['GRPO_MODEL_PATH'])
cfg = json.loads((model / 'config.json').read_text())
vocab = int((cfg.get('text_config') or cfg)['vocab_size'])
tokens = 24576
print(f'开始最长 logits 检验：shape=(1,{tokens},{vocab})，只运行修复路径。', flush=True)
start = time.monotonic()
x = torch.randn(1, tokens, vocab, device='cuda', dtype=torch.bfloat16, requires_grad=True)
lengths = torch.tensor([tokens], device='cuda')
labels = torch.randint(vocab, (tokens,), device='cuda')
lp, loss, grad, entropy = calculate(x, lengths, labels, True)
assert torch.isfinite(lp).all() and torch.isfinite(loss)
for chunk in grad.reshape(-1, vocab).split(128):
    assert torch.isfinite(chunk).all()
torch.cuda.synchronize()
engine = Path(spec['project']) / '.venv/lib/python3.12/site-packages/verl/workers/engine/fsdp/transformer_impl.py'
result = {'passed': True, 'torch_version': torch.__version__, 'tokens': tokens, 'vocab': vocab,
          'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
          'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
          'elapsed_seconds': time.monotonic()-start, 'engine_file': str(engine),
          'engine_sha256': hashlib.sha256(engine.read_bytes()).hexdigest()}
(prep / 'math_probe.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
PY
then
  echo '关键组件检验通过。先核对打印的峰值，再执行后面的环境预检和正式启动。'
else
  echo '检验失败：暂不启动正式训练，保留完整输出继续排查。'
fi
```

## 3. 环境检查与运行时预检（有卡模式执行）

```bash
set +e
set +u
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/formal_memfix_session.env
cd "$PROJECT"
if .venv/bin/python - <<'PY'
import socket
with socket.create_connection(('127.0.0.1', 5700), timeout=2): pass
PY
then
  echo '环境端口已监听，下面实际验证。'
else
  nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 SHOPSIM_PORT=5700 \
    SHOPSIM_ENV_SLOTS=8 SHOP_MAX_STEPS=35 bash scripts/start_environment.sh \
    > "$FORMAL_PREP/environment.log" 2>&1 < /dev/null &
fi

if .venv/bin/python -u - <<'PY'
import json, os, socket, subprocess, time
from pathlib import Path
root = Path(os.environ['PROJECT'])
prep = Path(os.environ['FORMAL_PREP'])
(prep / 'environment_preflight.json').unlink(missing_ok=True)
deadline = time.monotonic() + 120
while True:
    try:
        with socket.create_connection(('127.0.0.1',5700),timeout=2): break
    except OSError:
        if time.monotonic() >= deadline: raise SystemExit('环境未就绪，请查 environment.log。')
        time.sleep(2)
task = int(json.loads((root / 'data/grpo/train.jsonl').read_text().splitlines()[0])['task_id'])
import ray
ray.init(address='local', num_cpus=1, num_gpus=0, object_store_memory=128*1024**2,
    include_dashboard=False, runtime_env={'env_vars': {
        'PYTHONPATH': str(root/'src'), 'SHOPSIM_BASE_URL': os.environ['SHOPSIM_BASE_URL'],
        'OMP_NUM_THREADS':'1', 'VLLM_USE_FLASHINFER_SAMPLER':'0'}})
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
        return {'reset_and_release':'passed'}
    print('Ray 环境检查：', ray.get(probe.remote(task)))
finally:
    ray.shutdown()
spec = json.loads((prep / 'launch_spec.json').read_text())
with (prep / 'preflight.log').open('w') as log:
    status = subprocess.call(spec['preflight'], cwd=root, env=dict(os.environ,**spec['env']),
        stdout=log, stderr=subprocess.STDOUT)
print((prep / 'preflight.log').read_text())
assert status == 0, status
(prep / 'environment_preflight.json').write_text(json.dumps({'passed': True}))
print('修复版运行时预检通过。')
PY
then
  echo '可以执行第 4 步。'
else
  echo '环境或预检失败，不启动正式训练。'
fi
```

## 4. 启动独立正式重跑（需要 GPU）

```bash
set +e
set +u
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/formal_memfix_session.env
cd "$PROJECT"
if [[ ! -f "$FORMAL_PREP/math_probe.json" || ! -f "$FORMAL_PREP/environment_preflight.json" ]]; then
  echo '组件检验或环境预检尚未通过，不启动训练。'
elif [[ -e "$FORMAL_PREP/train.log" || -e "$FORMAL_PREP/driver.log" || -e "$FORMAL_PREP/train.pid" ]]; then
  echo '本会话已有启动记录，请先检查，不重复启动。'
else
  nohup .venv/bin/python -u "$FORMAL_PREP/launch_formal.py" \
    > "$FORMAL_PREP/driver.log" 2>&1 < /dev/null &
  echo "后台启动 PID=$!"
fi
tail -n 40 "$FORMAL_PREP/driver.log"
tail -n 60 -F "$FORMAL_PREP/train.log"
```

这次重新从 0 计数，计划上限仍为 500；旧失败运行的 35 步不能加进新模型的累计更新数。保存点为 **1、10、25、50、100、…、500**，滚动保留最多 10 份，因此早期 1/10/25 在后期陆续被淘汰，正常计划结束后保留 50/100/…/500 十个验证候选。额外保存不改变学习率调度或 Dev 验证频率。平台强制保存或额外手工保存需另行计入保留数量。完整任务训练和实际最长轨迹仍需观察，不能把组件检验当作已通过完整训练。

如果要训练退出后自动关机，在**另一个终端**执行新会话的监控（不能再使用旧 `formal_launch` 的监控）：

```bash
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/formal_memfix_session.env
cd "$PROJECT"
if [[ -f "$FORMAL_PREP/auto_shutdown.py" ]]; then
  nohup .venv/bin/python -u "$FORMAL_PREP/auto_shutdown.py" \
    >> "$FORMAL_PREP/auto_shutdown.log" 2>&1 < /dev/null &
  sleep 2
  tail -n 20 "$FORMAL_PREP/auto_shutdown.log"
else
  echo '旧运行没有自动关机脚本；需要按原文档第 8 节创建脚本，但使用当前新会话变量。'
fi
```

若重新开机或换终端，先 `source` 上面的 `formal_memfix_session.env` 并 `cd "$PROJECT"`。原始正式训练文档第 7 节的日志/周期验证检查可以在恢复新会话后使用；它检查 50/100/…/500 这 10 个候选。早期保存点用于中断恢复，不要求在整个实验结束后仍保留。最终模型仍只通过 Dev-50 选择，Final-200 Clean 不参与调参。

## 5. 已生成配置被磁盘预算拦截时：同步修正保留策略（无卡可做）

适用于只有 `driver.log`、报错为“磁盘未达到封存预算”、尚未启动修复版训练的情况。不会删除已有 checkpoint，也不会跳过空间检查。先检查本机 checkpoint 管理器：只有明确识别出保存前清理旧文件的实现，才按 10 份峰值预算；否则保守按 11 份。

下面命令同时更新实际训练参数、预检参数、解析后的配置和空间预算，并保留修改回执。仅在空间足够时移动这一次失败的 `driver.log` 到备份目录，解除启动段的重复启动保护；不覆盖失败证据。

```bash
set +e
set +u
if /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/.venv/bin/python - <<'PY'
import ast, hashlib, json, shutil, time
from pathlib import Path
from omegaconf import OmegaConf

root = Path('/root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon')
run = root / 'outputs/runs/grpo_baseline_20261022_202816'
prep = run / 'preparation/formal_memfix_v1'
spec_path = prep / 'launch_spec.json'
spec = json.loads(spec_path.read_text())
out = Path(spec['output'])
assert out == run / 'formal_memfix_v1'
assert not out.exists() or (out.is_dir() and not any(out.iterdir())), '输出已非空，不能当作未启动运行修改。'
assert not (prep / 'train.log').exists() and not (prep / 'train.pid').exists(), '已有训练启动记录。'
target = str(prep / 'launch_formal.py').encode()
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try: args = (proc / 'cmdline').read_bytes().split(b'\0')
    except OSError: continue
    assert target not in args, f'启动器 PID={proc.name} 仍在运行。'
for name, expected in spec['source_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name

cfg_path = prep / 'composed_config.yaml'
cfg = OmegaConf.load(cfg_path)
assert cfg.trainer.save_freq == cfg.trainer.test_freq == 50
assert cfg.trainer.total_training_steps == 500
assert list(cfg.trainer.early_save_steps) == [1,10,25]
assert cfg.trainer.max_actor_ckpt_to_keep in (10,13)
planned = [1,10,25,*range(50,501,50)]
assert planned[-10:] == list(range(50,501,50))

# 对本机保存前清理逻辑做严格结构识别；识别不了则使用更保守的 11 份峰值。
package = root / '.venv/lib/python3.12/site-packages/verl/utils/checkpoint'
base_tree = ast.parse((package / 'checkpoint_manager.py').read_text())
fsdp_tree = ast.parse((package / 'fsdp_checkpoint_manager.py').read_text())
ensure = next((n for n in ast.walk(base_tree) if isinstance(n, ast.FunctionDef)
               and n.name == 'ensure_checkpoint_capacity'), None)
expected = ast.parse('''
def expected(self, max_ckpt_to_keep):
    if not (max_ckpt_to_keep and isinstance(max_ckpt_to_keep, int) and max_ckpt_to_keep > 1):
        return
    if len(self.previous_saved_paths) >= max_ckpt_to_keep:
        keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1
        self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])
        self.previous_saved_paths = self.previous_saved_paths[keep_start:]
''').body[0]
def body(fn):
    nodes = list(fn.body)
    if nodes and isinstance(nodes[0], ast.Expr) and isinstance(nodes[0].value, ast.Constant) and isinstance(nodes[0].value.value, str):
        nodes = nodes[1:]
    return ast.dump(ast.Module(body=nodes, type_ignores=[]))
prune_before = False
save_fns = [n for n in ast.walk(fsdp_tree) if isinstance(n, ast.FunctionDef) and n.name == 'save_checkpoint']
if ensure is not None and body(ensure) == body(expected) and len(save_fns) == 1:
    calls = [n for n in ast.walk(save_fns[0]) if isinstance(n, ast.Call)]
    pruning = [n.lineno for n in calls if isinstance(n.func, ast.Attribute)
               and n.func.attr == 'ensure_checkpoint_capacity'
               and ((n.args and isinstance(n.args[0], ast.Name) and n.args[0].id == 'max_ckpt_to_keep')
                    or any(k.arg == 'max_ckpt_to_keep' and isinstance(k.value, ast.Name)
                           and k.value.id == 'max_ckpt_to_keep' for k in n.keywords))]
    saving = [n.lineno for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'save'
              and isinstance(n.func.value, ast.Name) and n.func.value.id == 'torch']
    prune_before = bool(pruning and saving and min(pruning) < min(saving))
peak_slots = 10 if prune_before else 11
observed = json.loads((run / 'preparation/formal_launch/disk_budget.json').read_text())['checkpoint_observed_bytes']
required = observed * peak_slots + 27 * 2**30
available = shutil.disk_usage(root).free
print(f'当前可用：{available/2**30:.2f} GiB')
print(f'原预算门槛：{spec["required_free_bytes"]/2**30:.2f} GiB')
print('是否明确识别保存前清理：', prune_before)
print(f'保留上限 10，短时峰值按 {peak_slots} 份计算，新门槛：{required/2**30:.2f} GiB')
if available < required:
    raise SystemExit(f'仍差 {(required-available)/2**30:.2f} GiB；本次未修改配置，不删除数据，也不解除启动保护。')

def change_limit(argv):
    values = list(argv)
    hits = [i for i, v in enumerate(values) if v.lstrip('+').startswith('trainer.max_actor_ckpt_to_keep=')]
    assert len(hits) == 1, hits
    i = hits[0]
    assert values[i].rsplit('=',1)[1] in ('10','13')
    values[i] = values[i].rsplit('=',1)[0] + '=10'
    return values
spec['command'] = change_limit(spec['command'])
spec['preflight'] = change_limit(spec['preflight'])
spec['required_free_bytes'] = required
spec['checkpoint_retention_limit'] = 10
spec['checkpoint_peak_slots_budgeted'] = peak_slots
cfg.trainer.max_actor_ckpt_to_keep = 10
driver_log = prep / 'driver.log'
if driver_log.exists():
    assert '磁盘未达到封存预算' in driver_log.read_text(), 'driver.log 包含其他错误，先核对，不移动。'
archive = prep / f'disk_retention_fix_{time.time_ns()}'
archive.mkdir()
for path in (spec_path, cfg_path, prep / 'command.txt'):
    if path.exists(): shutil.copy2(path, archive / path.name)
updates = {
    cfg_path: OmegaConf.to_yaml(cfg, resolve=True),
    spec_path: json.dumps(spec, ensure_ascii=False, indent=2),
}
import shlex
updates[prep / 'command.txt'] = shlex.join(spec['command']) + '\n'
for path, text in updates.items():
    temp = path.with_name(path.name + '.retention.tmp')
    temp.write_text(text)
    temp.replace(path)
(archive / 'receipt.json').write_text(json.dumps({
    'retention_limit':10, 'peak_slots_budgeted':peak_slots, 'required_free_bytes':required,
    'prune_before_save_recognized':prune_before, 'training_objective_changed':False,
    'scheduled_validation_candidates':list(range(50,501,50))
}, indent=2))
if driver_log.exists(): driver_log.rename(archive / 'driver.failed_disk.log')
print('保留策略与预算已同步修正，原配置和失败日志已备份：', archive)
print('不必重做第一步。源码未变，已通过的第二步组件检验仍有效；再运行第三步预检后启动第四步。')
PY
then
  source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/formal_memfix_session.env
  cd "$PROJECT"
else
  echo '暂不启动，请先查看上方实际空间差额或具体错误。'
fi
```
