"""Prepare, preflight, run and resume the frozen A/B pilot. Linux execution only."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from difficulty_scheduler import VERSION, append, canonical, read, sha, write
from prepare_difficulty_scheduler import (ROOT, UP, PACKAGE, SETTINGS, check_package,
    check_runtime, file_hash, lines, receipt, text_hash)

ENV_URL = 'http://127.0.0.1:5723'
MODEL_URL = 'http://127.0.0.1:8023/v1'
MODEL_RELATIVE = 'outputs/runs/sft_20261021_165243/merged'


def http(url, payload=None, timeout=15):
    request = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def assert_free(port):
    with socket.socket() as connection:
        if connection.connect_ex(('127.0.0.1', port)) == 0:
            raise RuntimeError(f'Port {port} already occupied; existing services left untouched')


@contextmanager
def lock(path):
    path = Path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, canonical({'pid': os.getpid()}).encode())
    os.close(fd)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def model_hashes(path):
    path = Path(path)
    weights = list(path.glob('*.safetensors')) + list(path.glob('pytorch_model*.bin'))
    if not weights or not (path / 'config.json').exists():
        raise ValueError('Missing merged SFT model: ' + str(path))
    files = sorted(set(weights + [p for p in path.iterdir() if p.suffix in ('.json', '.jinja')]))
    return {p.name: file_hash(p) for p in files}


def checked(run, overlay=True, models=False):
    cfg = read(Path(run) / 'run.json')
    check_runtime(cfg['bundle'], overlay)
    if cfg['protocol'] != text_hash(PACKAGE / 'manifest.json'):
        raise ValueError('Run protocol changed')
    index = Path(cfg['bundle']) / 'project/environments/ShopSimulator/shop_env/search_engine/products.sqlite3'
    if cfg['index'] != file_hash(index):
        raise ValueError('Run search index changed')
    if models and model_hashes(cfg['sft']) != cfg['sft_files']:
        raise ValueError('SFT model changed after init')
    return cfg


def init(run, bundle):
    check_runtime(bundle, overlay=True)
    if run.exists():
        checked(run)
        print('PASS: existing run verified; not overwritten')
        return
    sft = (UP / MODEL_RELATIVE).resolve()
    files = model_hashes(sft)
    run.mkdir(parents=True)
    index = Path(bundle) / 'project/environments/ShopSimulator/shop_env/search_engine/products.sqlite3'
    write(run / 'run.json', {'version': VERSION, 'bundle': str(Path(bundle).resolve()),
          'protocol': text_hash(PACKAGE / 'manifest.json'), 'sft': str(sft), 'sft_files': files,
          'index': file_hash(index), 'settings': SETTINGS})
    print('PASS: frozen A/B pilot initialized, original SFT model bound')


def environment(cfg, spec=None):
    bundle = Path(cfg['bundle'])
    project = bundle / 'project'
    env = dict(os.environ)
    # Remove inherited experimental/reward and Ray settings from prior jobs.
    for key in list(env):
        if key.startswith(('SHOPPING_', 'GRPO_')) or key in ('RAY_ADDRESS', 'RAY_RUNTIME_ENV_HOOK'):
            env.pop(key)
    env.update(PYTHONPATH=os.pathsep.join((str(bundle / 'overlay'), str(project / 'src'), str(ROOT / 'tools'))),
        SHOPPING_GRPO_ROOT=str(project), SHOPPING_ENVIRONMENT_VERSION='shopsimulator-environment-v2.1',
        SHOPPING_ENV_MANIFEST=str(project / 'data/environment.json'),
        SHOPPING_AGENT_LOOP_CONFIG=str(project / 'configs/agent_loop.yaml'),
        SHOPPING_TOOL_CONFIG=str(project / 'configs/tools.json'),
        SHOPPING_EVIDENCE_MANIFEST='', SHOPPING_REWARD_TOOLS_DIR='', SHOPPING_LENGTH_SHAPING_ENABLE='false',
        SHOPSIM_BASE_URL=ENV_URL, GRPO_MODEL_PATH=cfg['sft'],
        GRPO_TRAIN_FILE=str(project / 'data/grpo/train.parquet'),
        GRPO_VAL_FILE=str(project / 'data/grpo/validation.parquet'),
        PYTHONHASHSEED=str(SETTINGS['seed']), OMP_NUM_THREADS='1',
        VLLM_USE_FLASHINFER_SAMPLER='0', PYTHONUNBUFFERED='1',
        RAY_USAGE_STATS_ENABLED='0', PYTHONUTF8='1')
    if spec:
        env.update(SHOPPING_SCHEDULER_SPEC=str(spec),
                   GRPO_OUTPUT_DIR=str(spec.parent),
                   SHOPPING_GRPO_DIAGNOSTICS_PATH=str(spec.parent / 'training_diagnostics.jsonl'))
    return env


def command(cfg, arm, run):
    output = Path(run) / arm
    spec = output / 'spec.json'
    value = {'version': VERSION, 'arm': arm, 'seed': SETTINGS['seed'], 'budget': SETTINGS['budget'],
             'task_ids': sorted(r['task_id'] for r in lines(PACKAGE / 'train_tasks.jsonl')),
             'output': str(output.resolve()), 'protocol': cfg['protocol']}
    if spec.exists() and read(spec) != value:
        raise ValueError('Existing arm spec changed')
    write(spec, value)
    env = environment(cfg, spec)
    project = Path(cfg['bundle']) / 'project'
    overrides = [
        'data.seed=' + str(SETTINGS['seed']), 'data.dataloader_num_workers=0',
        'actor_rollout_ref.actor.data_loader_seed=' + str(SETTINGS['seed']),
        'actor_rollout_ref.actor.fsdp_config.seed=' + str(SETTINGS['seed']),
        '++actor_rollout_ref.rollout.engine_kwargs.vllm.seed=' + str(SETTINGS['seed']),
        'actor_rollout_ref.actor.optim.lr_warmup_steps=15',
        'actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0',
        'actor_rollout_ref.actor.optim.lr_scheduler_type=constant',
        'actor_rollout_ref.actor.checkpoint.async_save=false',
        'trainer.total_epochs=1', 'trainer.total_training_steps=500',
        'trainer.save_freq=50', 'trainer.max_actor_ckpt_to_keep=2',
        'trainer.val_before_train=false', 'trainer.test_freq=0',
        'trainer.logger=[console]', 'trainer.project_name=difficulty-scheduler-v1',
        'trainer.experiment_name=' + arm, 'trainer.resume_mode=auto',
        '++ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=difficulty_scheduler_hook.setup_worker',
    ]
    for name in ('PYTHONPATH', 'SHOPPING_GRPO_ROOT', 'SHOPPING_ENVIRONMENT_VERSION', 'SHOPPING_ENV_MANIFEST',
        'SHOPPING_AGENT_LOOP_CONFIG', 'SHOPPING_TOOL_CONFIG', 'SHOPPING_EVIDENCE_MANIFEST',
        'SHOPPING_REWARD_TOOLS_DIR', 'SHOPPING_LENGTH_SHAPING_ENABLE', 'SHOPSIM_BASE_URL',
        'SHOPPING_SCHEDULER_SPEC', 'GRPO_MODEL_PATH', 'GRPO_TRAIN_FILE', 'GRPO_VAL_FILE',
        'GRPO_OUTPUT_DIR', 'SHOPPING_GRPO_DIAGNOSTICS_PATH', 'PYTHONHASHSEED', 'OMP_NUM_THREADS',
        'VLLM_USE_FLASHINFER_SAMPLER'):
        overrides.append('++ray_kwargs.ray_init.runtime_env.env_vars.' + name + '=' + json.dumps(env[name]))
    cmd = [sys.executable, '-u', '-m', 'verl.trainer.main_ppo',
           '--config-path=' + str(project / 'configs'), '--config-name=grpo', *overrides]
    return cmd, env


def preflight(run, arm='uniform'):
    """No model loading/CUDA calls: validate resolved config, APIs, packages and rows."""
    cfg = checked(run, models=True)
    cmd, env = command(cfg, arm, run)
    script = r'''
import importlib.util,importlib.metadata,json,os,pathlib,sys
project=pathlib.Path(os.environ['SHOPPING_GRPO_ROOT'])
spec=importlib.util.spec_from_file_location('baseline_check',project/'scripts/check_grpo_runtime.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
config=m.compose_runtime_config(sys.argv[1:])
m.validate_environment_contract();m.validate_training_memory_budget(config)
if sys.version_info[:2]!=(3,12):raise ValueError('Requires original Python3.12 environment')
installed={k:importlib.metadata.version(k) for k in m.EXPECTED_VERSIONS}
for k,v in installed.items():
 if v.split('+',1)[0]!=m.EXPECTED_VERSIONS[k]:raise ValueError('Dependency mismatch: '+k+' '+v)
m.validate_transformers_revision()
from transformers import AutoModelForMultimodalLM
from peft import PeftModel
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
if not hasattr(FSDPCheckpointManager,'_get_lora_train_meta'):raise ValueError('Missing LoRA rank/alpha checkpoint metadata support')
import verl
if not pathlib.Path(verl.__file__).resolve().is_relative_to(pathlib.Path(os.environ['PYTHONPATH'].split(os.pathsep)[0])):raise ValueError('Wrong veRL overlay')
m.validate_dynamic_sampling(config,pathlib.Path(verl.__file__),installed)
import pyarrow.parquet as pq
if int(importlib.metadata.version('pyarrow').split('.')[0])<25:raise ValueError('Original parquet requires pyarrow>=25; do not regenerate it')
rows=pq.read_table(os.environ['GRPO_TRAIN_FILE']).to_pylist()
ids=[int(r['extra_info']['task_id']) for r in rows]
frozen=json.load(open(os.environ['SHOPPING_SCHEDULER_SPEC']))
if sorted(ids)!=frozen['task_ids']:raise ValueError('Parquet identity mismatch')
if config.actor_rollout_ref.actor.optim.lr_warmup_steps!=15:raise ValueError('Warmup differs')
if config.trainer.val_before_train or config.trainer.test_freq!=0:raise ValueError('Unexpected inline validation')
bindings=config.ray_kwargs.ray_init.runtime_env.env_vars
for key in ('PYTHONPATH','SHOPPING_SCHEDULER_SPEC','SHOPSIM_BASE_URL'):
 if bindings[key]!=os.environ[key] or '${' in bindings[key]:raise ValueError('Lazy Ray binding: '+key)
print('SCHEDULER_CPU_PREFLIGHT_PASS')
'''
    result = subprocess.run([sys.executable, '-c', script, *cmd[6:]], env=env, cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (Path(run) / (arm + '_preflight.log')).write_text(result.stdout, encoding='utf-8')
    if result.returncode or 'SCHEDULER_CPU_PREFLIGHT_PASS' not in result.stdout:
        raise RuntimeError('CPU config preflight failed; inspect ' + arm + '_preflight.log')
    write(Path(run) / (arm + '_preflight.json'), {'passed': True, 'command': cmd,
                                               'protocol': cfg['protocol']})
    print('PASS: CPU dependency, config, Ray binding and train1000 preflight:', arm)


@contextmanager
def services(run, cfg, model_path=None, label='environment'):
    assert_free(5723)
    if model_path:
        assert_free(8023)
    children, logs = [], []
    env = environment(cfg)
    commands = [('environment', [str(UP / 'environments/ShopSimulator/.venv-shopsim/bin/python'),
        str(ROOT / 'tools/difficulty_scheduler_environment.py'), '--bundle', cfg['bundle']])]
    if model_path:
        commands.append(('model', [str(UP / '.venv/bin/vllm'), 'serve', str(model_path),
            '--served-model-name', 'scheduler-' + label, '--port', '8023', '--max-model-len', '24576',
            '--seed', str(SETTINGS['seed']), '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder']))
    try:
        for name, cmd in commands:
            handle = (Path(run) / (label + '_' + name + '.log')).open('ab')
            logs.append(handle)
            children.append(subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle,
                                            stderr=subprocess.STDOUT, start_new_session=True))
        deadline = time.monotonic() + 600
        while True:
            if any(child.poll() is not None for child in children):
                raise RuntimeError('Child service exited; inspect service logs')
            try:
                identity = http(ENV_URL + '/api/difficulty_scheduler_identity', timeout=3)
                if identity['runtime'] != sha(canonical(receipt(cfg['bundle']))) or identity['index'] != cfg['index']:
                    raise ValueError('Environment service identity mismatch')
                if model_path:
                    models = http(MODEL_URL + '/models', timeout=3)['data']
                    match = next((m for m in models if m['id'] == 'scheduler-' + label), None)
                    if not match:
                        raise OSError('Model not ready')
                    if Path(match['root']).resolve() != Path(model_path).resolve() or match['max_model_len'] != 24576:
                        raise ValueError('Loaded model identity mismatch')
                break
            except (OSError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise RuntimeError('Service startup timeout')
                time.sleep(2)
        yield
    finally:
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for handle in logs:
            handle.close()


def cpu_check(run):
    cfg = checked(run)
    output = Path(run) / 'cpu_check.json'
    if output.exists():
        if read(output)['passed']:
            print('PASS: existing CPU environment preflight')
            return
        raise ValueError('Prior CPU preflight failed; inspect artifacts')
    sys.path.insert(0, str(Path(cfg['bundle']) / 'project/src'))
    from shopping_grpo.evaluation.rollout import ShopAgentEnv
    tasks = lines(PACKAGE / 'train_tasks.jsonl')[:3]
    checks = []
    with services(run, cfg, label='cpu'):
        for task in tasks:
            a, b = ShopAgentEnv(base_url=ENV_URL), ShopAgentEnv(base_url=ENV_URL)
            try:
                x, y = a.reset(task['task_id']), b.reset(task['task_id'])
                clean = lambda z: {k: v for k, v in z.items() if k not in ('idx', 'env_idx')}
                if a.env_idx == b.env_idx or clean(x) != clean(y):
                    raise ValueError('Independent session reset differs')
                query = x['instruction'].removeprefix('Instruction: ')
                if clean(a.step('search[' + query + ']')) != clean(b.step('search[' + query + ']')):
                    raise ValueError('Public search replay differs')
                checks.append(task['task_id'])
            finally:
                try:
                    a.release()
                finally:
                    b.release()
    write(output, {'passed': len(checks) == 3, 'task_ids': checks, 'model_used': False})
    print('PASS: actual original-environment CPU preflight')


def train(run, arm):
    cfg = checked(run, models=True)
    if (Path(run) / 'STOP').exists():
        raise RuntimeError('Pipeline safely paused between stages')
    if not read(Path(run) / 'cpu_check.json')['passed']:
        raise ValueError('CPU environment preflight required')
    if not read(Path(run) / (arm + '_preflight.json'))['passed']:
        raise ValueError('CPU config preflight required')
    end = Path(run) / arm / 'training_complete.json'
    if end.exists() and read(end)['complete']:
        print('PASS: existing completed training:', arm)
        return
    cmd, env = command(cfg, arm, run)
    with services(run, cfg, label=arm):
        with (Path(run) / (arm + '_training.log')).open('ab') as handle:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Training failed; keep all artifacts: ' + arm)
    if not end.exists():
        raise RuntimeError('Training exited without budget/safe-stop receipt')
    if not read(end)['complete']:
        raise RuntimeError('Training safely paused; rerun pipeline to resume')


def export_model(run, arm):
    cfg = checked(run)
    end = read(Path(run) / arm / 'training_complete.json')
    if not end['complete'] or end['generated_rollouts'] != 4000:
        raise ValueError('Export requires completed fixed budget')
    destination = Path(run) / (arm + '_merged')
    raw_export = Path(run) / (arm + '_fsdp_export')
    marker = Path(run) / (arm + '_export.json')
    if marker.exists():
        if read(marker)['files'] != model_hashes(destination):
            raise ValueError('Export changed')
        return destination
    if destination.exists() or raw_export.exists():
        raise ValueError('Incomplete export exists; retain and investigate')
    source = Path(run) / arm / f"global_step_{end['optimizer_updates']}" / 'actor'
    command_line = [sys.executable, '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
                    '--local_dir', str(source), '--target_dir', str(raw_export), '--trust-remote-code']
    with (Path(run) / (arm + '_export.log')).open('ab') as handle:
        subprocess.run(command_line, env=environment(cfg), cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        adapter = raw_export / 'lora_adapter'
        if not (adapter / 'adapter_model.safetensors').exists():
            raise ValueError('Expected newly trained GRPO LoRA adapter, not a base-only export')
        adapter_config = read(adapter / 'adapter_config.json')
        if adapter_config.get('r') != 16 or adapter_config.get('lora_alpha') != 32:
            raise ValueError('Exported adapter rank/alpha differ from frozen training')
        merger = Path(cfg['bundle']) / 'project/scripts/merge_lora_adapter.py'
        subprocess.run([sys.executable, str(merger), '--base-model', str(raw_export),
                        '--adapter', str(adapter), '--output', str(destination), '--bf16'],
                       env=environment(cfg), cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    # Some veRL mergers leave tokenizer assets under actor/huggingface only.
    import shutil
    tokenizer_rebound = {}
    for path in Path(cfg['sft']).iterdir():
        if path.is_file() and ('token' in path.name or path.suffix == '.jinja' or path.name in ('vocab.json', 'merges.txt', 'preprocessor_config.json')):
            target = destination / path.name
            # GRPO never trains the tokenizer: bind exact original assets rather
            # than serializer-specific metadata from the intermediate export.
            tokenizer_rebound[path.name] = {'exported': file_hash(target) if target.exists() else None,
                                           'original_sft': file_hash(path)}
            shutil.copy2(path, target)
    write(marker, {'files': model_hashes(destination), 'optimizer_updates': end['optimizer_updates'],
                   'adapter_sha256': file_hash(adapter / 'adapter_model.safetensors'),
                   'tokenizer_assets': tokenizer_rebound,
                   'operation': 'fsdp_export_then_peft_merge_and_unload'})
    return destination


def evaluate(run, label):
    cfg = checked(run, models=True)
    if (Path(run) / 'STOP').exists():
        raise RuntimeError('Pipeline safely paused between stages')
    model = Path(cfg['sft']) if label == 'sft' else export_model(run, label)
    folder = Path(run) / ('dev50_' + label)
    folder.mkdir(exist_ok=True)
    result = folder / 'summary.json'
    if result.exists():
        if read(result)['complete']:
            print('PASS: existing Dev50 evaluation:', label)
            return
        raise ValueError('Incomplete evaluation must be investigated, not silently rerolled')
    sys.path.insert(0, str(Path(cfg['bundle']) / 'project/src'))
    from shopping_grpo.evaluation import rollout as r
    from checkpoint_pair_common import outcome
    r.MODEL_COMPLETION_RETRIES = 0
    values = []
    with services(run, cfg, model_path=model, label=label):
        for task in lines(PACKAGE / 'dev50_tasks.jsonl'):
            if (Path(run) / 'STOP').exists():
                raise RuntimeError('Pipeline safely paused between completed Dev50 tasks')
            path = folder / (str(task['task_id']) + '.json')
            if path.exists():
                prior = read(path)
                if prior['outcome']['classification'] != 'completed':
                    raise ValueError('Unresolved evaluation infrastructure error')
                values.append(prior)
                continue
            turn = 0
            def transport(url, payload, headers, timeout):
                nonlocal turn
                payload['seed'] = int(sha(f"scheduler-dev50:{task['task_id']}:{turn}")[:8], 16) % 2147483647
                turn += 1
                response = http(url, payload, timeout)
                append(folder / 'usage.jsonl', {'task_id': task['task_id'], 'turn': turn - 1,
                                               'usage': response.get('usage')})
                return response
            client = r.OpenAIChatClient(model='scheduler-' + label, base_url=MODEL_URL, api_key='EMPTY',
                temperature=0., top_p=1., max_tokens=512, context_window=24576, context_safety_margin=512,
                observation_token_budget=1536, observation_detail_token_budget=4096,
                observation_generic_token_budget=768, observation_search_top_k=20, transport=transport)
            trace = r.collect_for_task(task, client=client, base_url=ENV_URL, max_steps=35, attempt_index=0)
            value = {'task_id': task['task_id'], 'trajectory': trace, 'outcome': outcome(trace)}
            write(path, value)
            if value['outcome']['classification'] != 'completed':
                raise RuntimeError('Evaluation infrastructure error; original attempt retained')
            values.append(value)
    write(result, {'complete': len(values) == 50, 'tasks': len(values),
          'successes': sum(x['outcome']['strict_success'] for x in values),
          'protocol': cfg['protocol'], 'model': str(model), 'model_files': model_hashes(model)})
    print('PASS: fixed Dev50 evaluation:', label)


def pipeline(run):
    checked(run, models=True)
    with lock(Path(run) / 'pipeline.lock'):
        # A new explicit invocation resumes a previous safe-stop request. A
        # second concurrent invocation never gets past the lock to clear it.
        (Path(run) / 'STOP').unlink(missing_ok=True)
        for arm in SETTINGS['arms']:
            (Path(run) / arm / 'STOP').unlink(missing_ok=True)
        status = {'complete': False, 'exit_code': 1}
        try:
            evaluate(run, 'sft')
            for arm in SETTINGS['arms']:
                train(run, arm)
                evaluate(run, arm)
            from report_difficulty_scheduler import report
            report(run)
            status.update(complete=True, exit_code=0)
            print('SCHEDULER_PILOT_COMPLETE', flush=True)
        except BaseException as exc:
            status['error'] = repr(exc)
            raise
        finally:
            write(Path(run) / 'pipeline_exit.json', status)


def pack(run):
    """Return audit material, excluding all model/optimizer tensor payloads."""
    import tarfile
    run = Path(run)
    cfg = checked(run)
    target = run.parent / (run.name + '_return.tar.gz')
    if target.exists():
        raise ValueError('Return archive already exists; retain it or choose a new run directory')
    with tarfile.open(target, 'w:gz') as archive:
        for path in sorted(run.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.log', '.yaml', '.md', '.txt'):
                archive.add(path, arcname='run/' + path.relative_to(run).as_posix())
        for name in ('manifest.json', 'runtime_receipt.json', 'train_tasks.jsonl', 'dev50_tasks.jsonl', 'diagnostic_labels.json'):
            archive.add(PACKAGE / name, arcname='protocol/' + name)
        for name in ('overlay_receipt.json', 'index_manifest.json', 'metadata_correction.json'):
            archive.add(Path(cfg['bundle']) / name, arcname='runtime/' + name)
    print('RETURN_ARCHIVE=' + str(target))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['init', 'preflight', 'cpu-check', 'pipeline', 'status', 'stop', 'pack'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'outputs/difficulty_scheduler_v1/runtime')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.command == 'init':
        init(run, args.bundle)
    elif args.command == 'preflight':
        for arm in SETTINGS['arms']:
            preflight(run, arm)
    elif args.command == 'cpu-check':
        cpu_check(run)
    elif args.command == 'pipeline':
        pipeline(run)
    elif args.command == 'pack':
        pack(run)
    elif args.command == 'stop':
        checked(run)
        (run / 'STOP').touch()
        for arm in SETTINGS['arms']:
            (run / arm).mkdir(exist_ok=True)
            (run / arm / 'STOP').touch()
        print('Requested safe stop at next update/cleared-buffer boundary; keep process running')
    else:
        result = {}
        for arm in SETTINGS['arms']:
            path = run / arm / 'scheduler_state.json'
            state = read(path)['scheduler'] if path.exists() else {}
            result[arm] = {key: state.get(key) for key in ('generated', 'updates', 'batches', 'pending')}
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
