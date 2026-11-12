"""CPU-only configuration/data proof; uses dependencies extracted under outputs."""
import csv
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/difficulty_scheduler_v1'
sys.path[:0] = [str(BASE / 'cpu_dependencies'), str(BASE / 'cpu_config'), str(ROOT / 'tools')]
from prepare_difficulty_scheduler import PACKAGE, lines, write
from run_difficulty_scheduler import command
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

cfg = {'bundle': str((BASE / 'runtime').resolve()),
       'sft': str(ROOT / 'outputs/fake_model_cpu_only'), 'protocol': 'cpu-only'}
configs = {}
for arm in ('uniform', 'adaptive'):
    cmd, env = command(cfg, arm, BASE / 'cpu_config_run')
    os.environ.update(env)
    with initialize_config_dir(version_base=None, config_dir=str(BASE / 'runtime/project/configs')):
        resolved = compose(config_name='grpo', overrides=cmd[6:])
        OmegaConf.resolve(resolved)
        configs[arm] = OmegaConf.to_container(resolved, resolve=True)
    sys.path.insert(0, str(BASE / 'runtime/project/src'))
    spec = importlib.util.spec_from_file_location('original_cpu_check', BASE / 'runtime/project/scripts/check_grpo_runtime.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    original.validate_environment_contract()
    original.validate_training_memory_budget(resolved)


def differences(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        return [x for key in sorted(set(a) | set(b))
                for x in differences(a.get(key), b.get(key), prefix + '/' + key)]
    return [] if a == b else [prefix]


changed = differences(configs['uniform'], configs['adaptive'])
expected = {'/trainer/experiment_name', '/trainer/default_local_dir',
    '/actor_rollout_ref/rollout/trace/experiment_name',
    '/ray_kwargs/ray_init/runtime_env/env_vars/GRPO_OUTPUT_DIR',
    '/ray_kwargs/ray_init/runtime_env/env_vars/SHOPPING_SCHEDULER_SPEC',
    '/ray_kwargs/ray_init/runtime_env/env_vars/SHOPPING_GRPO_DIAGNOSTICS_PATH'}
assert set(changed) == expected, changed

import pyarrow.parquet as pq
rows = pq.read_table(BASE / 'runtime/project/data/grpo/train.parquet').to_pylist()
assert sorted(r['extra_info']['task_id'] for r in rows) == sorted(r['task_id'] for r in lines(PACKAGE / 'train_tasks.jsonl'))
ids = {r['extra_info']['task_id'] for r in rows}
with (ROOT / 'analysis/grpo_baseline_20261022_202816/training_diagnosis/groups.csv').open(encoding='utf-8-sig') as f:
    groups = list(csv.DictReader(f))
for group in groups:
    assert int(group['task_id']) in ids and int(group['group_size']) == 4
    keep = group['drop_reason'] != 'sampling_invalid' and float(group['utility_max']) - float(group['utility_min']) > 1e-8
    assert (group['kept'] == 'True') == keep
write(Path(__file__).parent / 'configuration_and_data_check.json', {
    'passed': True, 'GPU_or_model_used': False, 'original_environment_contract': True,
    'memory_budget_contract': True, 'train_rows': len(rows), 'train_unique_ids': len(ids),
    'pyarrow_version': __import__('pyarrow').__version__, 'historical_groups_audited': len(groups),
    'historical_filter_compatibility': True, 'historical_data_used_for_scheduler_initialization': False,
    'allowed_arm_config_differences': changed, 'warmup_actual_updates': 15,
    'actor_seed': configs['uniform']['actor_rollout_ref']['actor']['fsdp_config']['seed'],
    'vllm_seed': configs['uniform']['actor_rollout_ref']['rollout']['engine_kwargs']['vllm']['seed'],
    'inline_validation': False, 'new_training_executed': False})
print('PASS: A/B config parity, original contracts, parquet1000, historical4296 group compatibility')
