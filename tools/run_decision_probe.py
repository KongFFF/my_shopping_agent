"""CPU preparation, bounded model services, and no-training A/B local rollouts."""
import argparse
import copy
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from urllib.request import Request, urlopen
from decision_probe_common import (ROOT, PACKAGE, VERSION, Budget, BudgetExceeded, append,
                               check_package, code_hash, digest, dump,
                               file_hash, load, rows, runtime_receipt, save)

MODEL_NAME = 'decision-probe-grpo800'
MODEL_URL = 'http://127.0.0.1:8014/v1'
ENV_URL = 'http://127.0.0.1:5714'
from decision_probe_rules import trigger


def http(url, payload=None, timeout=30):
    request = Request(url, data=None if payload is None else json.dumps(payload).encode('utf-8'),
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def init_run(run, bundle, model):
    check_package()
    if runtime_receipt(bundle) != load(PACKAGE/'runtime_receipt.json'):
        raise ValueError('Runtime differs from frozen receipt')
    config = load(model/'config.json')
    if config.get('model_type') != 'qwen3_5':
        raise ValueError('Expected Qwen3.5 GRPO800 model')
    shards = sorted(model.glob('*.safetensors'))
    if not shards:
        raise ValueError('Merged GRPO800 safetensors are missing')
    hashes = {p.name: file_hash(p) for p in shards}
    for p in sorted(model.glob('*.json')):
        hashes[p.name] = file_hash(p)
    for p in sorted(model.glob('*.jinja')):
        hashes[p.name] = file_hash(p)
    # Must name the actual selected GRPO800 checkpoint, not an RL trial checkpoint.
    if model.name != 'merged' or model.parent.name != 'step800_v1':
        raise ValueError('Use the original GRPO step800_v1/merged path')
    receipt_path = model.parents[2]/'preparation/posttrain_step800_v1/export_receipt.json'
    receipt = load(receipt_path)
    if Path(receipt['merged_model']).resolve() != model.resolve() or not str(receipt['checkpoint']).endswith('/global_step_800'):
        raise ValueError('Original step800 export receipt does not identify this model')
    if receipt['merged_weights_sha256'] != {p.name: hashes[p.name] for p in shards}:
        raise ValueError('Weights differ from the original step800 export receipt')
    if hashes.get('tokenizer.json') != '06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523':
        raise ValueError('Original tokenizer differs')
    if run.exists():
        raise ValueError('Use a new run directory; existing results are never overwritten')
    run.mkdir(parents=True)
    save(run/'run.json', {'version': VERSION, 'bundle': str(bundle.resolve()),
                         'model_dir': str(model.resolve()), 'model_hashes': hashes,
                         'protocol_sha256': file_hash(PACKAGE/'manifest.json'),
                         'runtime_sha256': digest(dump(load(PACKAGE/'runtime_receipt.json'))),
                         'index_physical_sha256': file_hash(bundle/'environment/search_engine/products.sqlite3'),
                         'model_url': MODEL_URL, 'env_url': ENV_URL,
                         'model_name': MODEL_NAME, 'prepared_at': time.time()})
    save(run/'original_export_receipt.json', receipt)
    print('Prepared on CPU: ' + str(run))


def checked_run(run):
    check_package()
    cfg = load(run/'run.json')
    if cfg['protocol_sha256'] != file_hash(PACKAGE/'manifest.json'):
        raise ValueError('Protocol changed after run preparation')
    return cfg


def clock(run):
    checked_run(run)
    path = run/'budget.json'
    if path.exists():
        raise ValueError('Budget clock already exists; it cannot be reset')
    save(path, {'started_at': time.time(), 'seconds': 7200,
                'note': 'Start immediately when GPU rental begins; includes service startup and waiting'})
    print('Started immutable two-hour wall-clock budget')


def budget_for(run):
    return Budget(load(run/'budget.json')['started_at'])


def serve(run):
    cfg = checked_run(run)
    budget = budget_for(run)
    children = []
    handles = []
    for port in (5714, 8014):
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                raise ValueError(f'Port {port} is occupied; no existing service will be killed')
    env = dict(os.environ, PYTHONPATH='', OMP_NUM_THREADS='1', VLLM_USE_FLASHINFER_SAMPLER='0',
               SERVED_MODEL_NAME=MODEL_NAME, LLM_PORT='8014', PYTHONUNBUFFERED='1')
    up = ROOT/'upstream/shopping-grpo-longhorizon'
    envpy = up/'environments/ShopSimulator/.venv-shopsim/bin/python'
    commands = [(str(envpy), str(ROOT/'tools/run_decision_probe_environment.py'), '--bundle', cfg['bundle'], '--port', '5714'),
                (str(up/'.venv/bin/vllm'), 'serve', cfg['model_dir'], '--served-model-name', MODEL_NAME,
                 '--port', '8014', '--max-model-len', '24576', '--enable-auto-tool-choice',
                 '--tool-call-parser', 'qwen3_coder')]
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        budget.check()
        for name, expected in cfg['model_hashes'].items():
            if file_hash(Path(cfg['model_dir'])/name) != expected:
                raise ValueError('Model files changed since CPU preparation: ' + name)
        if (run/'STOP_SERVICES').exists():
            raise ValueError('Run has been stopped; do not reset its budget')
        for name, command in zip(('environment', 'model'), commands):
            handle = (run/(name+'.log')).open('ab')
            handles.append(handle)
            children.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                             stderr=subprocess.STDOUT, start_new_session=True))
        save(run/'service_pids.json', {'supervisor': os.getpid(), 'children': [p.pid for p in children]})
        while budget.remaining() > 0 and not (run/'STOP_SERVICES').exists():
            if any(p.poll() is not None for p in children):
                raise RuntimeError('A probe service exited; inspect model.log and environment.log')
            time.sleep(min(2, budget.remaining()))
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try: child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for h in handles: h.close()
        save(run/'services_stopped.json', {'at': time.time(), 'budget_expired': budget.remaining() <= 0,
                                          'note': 'Processes stopped; cloud instance billing requires user shutdown'})


def wait_ready(run):
    cfg = checked_run(run)
    budget = budget_for(run)
    end = min(time.time()+600, budget.deadline)
    last = None
    while time.time() < end:
        try:
            identity = http(ENV_URL+'/api/local_probe_identity', timeout=budget.timeout(3))
            if identity['runtime_sha256'] != cfg['runtime_sha256']:
                raise ValueError('Wrong environment runtime')
            if identity['index_physical_sha256'] != cfg['index_physical_sha256']:
                raise ValueError('Search index changed since CPU preparation')
            if identity['code_sha256'] != check_package()['code_sha256']:
                raise ValueError('Wrong diagnostic server code')
            models = http(MODEL_URL+'/models', timeout=budget.timeout(3))['data']
            model = next(m for m in models if m['id'] == MODEL_NAME)
            if str(Path(model['root']).resolve()) != str(Path(cfg['model_dir']).resolve()):
                raise ValueError('Model service root does not match selected GRPO800 checkpoint')
            save(run/'ready.json', {'at': time.time(), 'identity': identity, 'models': models})
            print('PASS: isolated environment and selected GRPO800 service ready')
            return
        except ValueError:
            raise
        except Exception as exc:
            last = str(exc)
            time.sleep(min(2, max(0, end-time.time())))
    raise RuntimeError('Services not ready within ten minutes: ' + str(last))


def normalize_result(result):
    result = copy.deepcopy(result)
    result.pop('env_idx', None)
    result.pop('idx', None)
    return result


def cpu_check(bundle, output):
    """Check actual two-slot reset/action replay on CPU, without an actor model."""
    check_package()
    if output.exists(): raise ValueError('CPU check output must be a new directory')
    if runtime_receipt(bundle) != load(PACKAGE/'runtime_receipt.json'):
        raise ValueError('Runtime differs')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', 5714)) == 0:
            raise ValueError('5714 occupied; do not replace an existing environment')
    output.mkdir(parents=True)
    envpy = Path(os.environ.get('DECISION_ENV_PYTHON', str(ROOT/'upstream/shopping-grpo-longhorizon/environments/ShopSimulator/.venv-shopsim/bin/python')))
    proc = None
    results = []
    try:
        with (output/'environment.log').open('wb') as log:
            proc = subprocess.Popen([str(envpy), str(ROOT/'tools/run_decision_probe_environment.py'),
                                     '--bundle', str(bundle.resolve()), '--port', '5714'], cwd=ROOT,
                    env=dict(os.environ, PYTHONPATH=os.environ.get('DECISION_CPU_DEPS', ''), PYTHONUTF8='1', OMP_NUM_THREADS='1'), stdout=log, stderr=subprocess.STDOUT)
            end = time.time()+180
            while True:
                try:
                    identity = http(ENV_URL+'/api/local_probe_identity', timeout=2)
                    if identity['runtime_sha256'] != digest(dump(load(PACKAGE/'runtime_receipt.json'))):
                        raise ValueError('Wrong environment identity')
                    break
                except (OSError, TimeoutError):
                    if proc.poll() is not None or time.time() >= end:
                        raise RuntimeError('CPU environment did not start; inspect environment.log')
                    time.sleep(1)
            sys.path.insert(0, str(bundle.resolve()/'agent'))
            from shopping_grpo.evaluation import rollout as r
            # Verify all frozen public queries against actual native reset order.
            for task in rows(PACKAGE/'tasks.jsonl'):
                probe = r.ShopAgentEnv(base_url=ENV_URL)
                try:
                    actual = probe.reset(task['task_id'])['instruction']
                    if actual.removeprefix('Instruction: ') != task['query']:
                        raise ValueError('Frozen query differs at task '+str(task['task_id']))
                finally: probe.release()
            for task in rows(PACKAGE/'tasks.jsonl')[:3]:
                first, second = r.ShopAgentEnv(base_url=ENV_URL), r.ShopAgentEnv(base_url=ENV_URL)
                try:
                    a = first.reset(task['task_id'])
                    query = task['query'][:60]
                    actions = ['search['+query+']']
                    result = first.step(actions[0])
                    expected_results = [result]
                    products = (result.get('observation_state') or {}).get('products') or []
                    if products:
                        actions.append('click['+products[0]['asin']+']')
                        expected_results.append(first.step(actions[-1]))
                        options = (expected_results[-1].get('observation_state') or {}).get('available_options') or {}
                        value = next((str(v[0]) for v in options.values() if isinstance(v,list) and v), None)
                        if value:
                            actions.append('click['+value+']')
                            expected_results.append(first.step(actions[-1]))
                    before = http(ENV_URL+'/api/local_probe_state', {'env_idx': first.env_idx})
                    b = second.reset(task['task_id'])
                    if first.env_idx == second.env_idx or normalize_result(a) != normalize_result(b):
                        save(output/'reset_mismatch.json', {'first': a, 'second': b})
                        raise ValueError('Reset/independent lease check failed')
                    for action, expected in zip(actions, expected_results):
                        actual = second.step(action)
                        if normalize_result(actual) != normalize_result(expected):
                            raise ValueError('CPU action replay result differs')
                    after = http(ENV_URL+'/api/local_probe_state', {'env_idx': second.env_idx})
                    if before != after: raise ValueError('CPU complete state digest differs')
                    first.step('click[Back to Search]')
                    unchanged = http(ENV_URL+'/api/local_probe_state', {'env_idx': second.env_idx})
                    if unchanged != after: raise ValueError('Mutable sessions are shared')
                    results.append({'task_id': task['task_id'], 'replayed_actions': actions,
                                    'distinct_slots': True, 'state': before})
                finally:
                    try: first.release()
                    finally: second.release()
            save(output/'cpu_check.json', {'passed': len(results) == 3, 'tasks': results,
                 'verified_query_count': 48, 'independent_mutation_checked': True,
                 'protocol_sha256': file_hash(PACKAGE/'manifest.json'),
                 'code_sha256': check_package()['code_sha256'],
                 'actor_model_used': False, 'gpu_used': False,
                 'note': 'Synthetic public actions verify environment replay only; natural GRPO800 branch remains untested'})
            print('PASS: actual CPU replay on three tasks; no model inference')
    except BaseException as exc:
        save(output/'cpu_check.json', {'passed': False, 'tasks': results, 'error': str(exc)})
        raise
    finally:
        if proc is not None:
            proc.terminate()
            try: proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


class Engine:
    def __init__(self, run, budget, env_factory=None, client_factory=None):
        self.run, self.budget = run, budget
        cfg = checked_run(run)
        sys.path.insert(0, str(Path(cfg['bundle'])/'agent'))
        from shopping_grpo.evaluation import rollout
        self.r = rollout
        if not Path(rollout.__file__).resolve().is_relative_to(Path(cfg['bundle']).resolve()):
            raise ValueError('Wrong agent import path')
        self.env_factory = env_factory or rollout.ShopAgentEnv
        self.client_factory = client_factory

    def state(self, env):
        return http(ENV_URL+'/api/local_probe_state', {'env_idx': env.env_idx}, self.budget.timeout(20))

    def client(self, key):
        if self.client_factory: return self.client_factory(key)
        counter = 0
        def transport(url, payload, headers, timeout):
            nonlocal counter
            # Derive an independent reproducible request seed per path and turn.
            payload['seed'] = int(digest(key+':'+str(counter))[:8], 16) % 2147483647
            counter += 1
            settings = check_package()['settings']
            used = rows(self.run/'usage.jsonl') if (self.run/'usage.jsonl').exists() else []
            if len(used) >= settings['max_model_calls'] or sum((u.get('usage') or {}).get('completion_tokens', 0) for u in used) >= settings['max_completion_tokens']:
                raise BudgetExceeded('Frozen model-call/token cap reached')
            completion_used = sum((u.get('usage') or {}).get('completion_tokens', 0) for u in used)
            payload['max_tokens'] = min(payload.get('max_tokens', 512), settings['max_completion_tokens']-completion_used)
            begin = time.time()
            try:
                response = http(url, payload, self.budget.timeout(60))
                usage = response.get('usage') or {}
                if not all(isinstance(usage.get(k), int) and usage[k] >= 0
                           for k in ('prompt_tokens', 'completion_tokens')):
                    raise ValueError('Missing model token usage; cannot compare costs')
                append(self.run/'usage.jsonl', {'path': key, 'at': begin, 'elapsed': time.time()-begin,
                                              'seed': payload['seed'], 'usage': usage, 'status': 'ok'})
                return response
            except Exception as exc:
                append(self.run/'usage.jsonl', {'path': key, 'at': begin, 'elapsed': time.time()-begin,
                                              'status': 'error', 'error': type(exc).__name__,
                                              'usage': None})
                raise
        return self.r.OpenAIChatClient(model=MODEL_NAME, base_url=MODEL_URL, api_key='EMPTY',
            temperature=.7, top_p=.9, timeout=30, max_tokens=512, context_window=24576,
            context_safety_margin=512, context_compaction_enable=False,
            observation_token_budget=1536, observation_detail_token_budget=4096,
            observation_generic_token_budget=768, observation_search_top_k=20, transport=transport)

    def execute(self, task, key, prefix=None, stop_at_branch=False):
        r = self.r
        started = time.time()
        trace = {'task_id': task['task_id'], 'path': key, 'status': 'running',
                 'messages': [], 'steps': [], 'blocked_tool_calls': [], 'replay': [],
                 'tool_call_truncations': [], 'error': None, 'release_error': None,
                 'terminal_result': {}, 'done': False, 'branch': False,
                 'prefix_steps': len(prefix['steps']) if prefix else 0}
        env = self.env_factory(base_url=ENV_URL, timeout=self.budget.timeout(30))
        client = self.client(key)
        try:
            self.budget.check()
            initial = env.reset(task['task_id'])
            trace['initial_result'] = initial
            if initial.get('environment_version') != 'shopsimulator-environment-v2.1':
                raise ValueError('Wrong environment version')
            actual_query = initial['instruction']
            if actual_query.startswith('Instruction: '): actual_query = actual_query[len('Instruction: '):]
            if actual_query != task['query']:
                raise ValueError('Reset query differs from frozen task')
            messages = r._initial_messages({'task_id': task['task_id']}, initial)
            latest = r.render_structured_observation(initial['observation_state'])
            blocked = 0
            if prefix:
                if normalize_result(initial) != normalize_result(prefix['initial_result']):
                    raise ValueError('Replay reset mismatch')
                for old in prefix['steps']:
                    self.budget.check()
                    env.timeout = self.budget.timeout(30)
                    begin = time.time()
                    fresh = r._execute_tool_call(env, old['tool_call'], old['step_index'])
                    matched = normalize_result(fresh['result']) == normalize_result(old['result'])
                    trace['replay'].append({'step_index': old['step_index'], 'matched': matched,
                                            'seconds': time.time()-begin})
                    if not matched: raise ValueError('Replay observation/result mismatch')
                fresh_receipt = self.state(env)
                trace['restored_state'] = fresh_receipt
                if fresh_receipt != prefix['branch_state']:
                    raise ValueError('Replay complete environment state mismatch')
                blocked = prefix['consecutive_blocked_calls']
                messages = copy.deepcopy(prefix['messages'])
                trace['steps'] = copy.deepcopy(prefix['steps'])
                trace['blocked_tool_calls'] = copy.deepcopy(prefix['blocked_tool_calls'])
                trace['tool_call_truncations'] = copy.deepcopy(prefix['tool_call_truncations'])
                latest = prefix['steps'][-1]['observation']
                trace['prefix_message_sha256'] = digest(dump(messages))
                if trace['prefix_message_sha256'] != prefix['branch_message_sha256']:
                    raise ValueError('Replay actor history mismatch')
                append(self.run/'restorations.jsonl', {'task_id': task['task_id'], 'path': key,
                       'verified': True, 'state': fresh_receipt,
                       'replayed_steps': len(trace['replay']), 'at': time.time()})
            trace['messages'] = messages
            while len(trace['steps']) < 35:
                self.budget.check()
                for counter in (getattr(client, 'token_counter', None), getattr(client, 'observation_token_counter', None)):
                    if counter is not None: counter.timeout = self.budget.timeout(30)
                assistant, dropped = r._enforce_serial_tool_call(client.complete(messages, r.SHOP_TOOL_SCHEMAS))
                if dropped: trace['tool_call_truncations'].append(dropped)
                calls = assistant.get('tool_calls') or []
                if stop_at_branch and calls and len(trace['steps']) <= 32:
                    try:
                        proposed_name, proposed_args = r._tool_call_name_args(calls[0])
                        allowed = not r.action_reject_reason(proposed_name, proposed_args, latest)
                        decision = trigger(task['query'], latest, proposed_name, proposed_args) if allowed else None
                    except (ValueError, KeyError, TypeError):
                        decision = None
                    if decision:
                        trace.update(status='branch', branch=True, branch_state=self.state(env),
                            branch_message_sha256=digest(dump(messages)), consecutive_blocked_calls=blocked,
                            trigger={**decision, 'proposal': assistant, 'visible': latest, 'executed': False})
                        break
                messages.append(assistant)
                if not calls:
                    trace['status'] = 'assistant_final'
                    break
                call = calls[0]
                try:
                    name, args = r._tool_call_name_args(call)
                    reason = r.action_reject_reason(name, args, latest)
                except Exception as exc:
                    reason = 'invalid_tool_call:' + type(exc).__name__
                if reason:
                    blocked += 1
                    trace['blocked_tool_calls'].append({'step_index': len(trace['steps']),
                        'tool_call': call, 'reason': reason, 'consecutive_count': blocked})
                    messages.append(r.action_guard_tool_message(call, reason, latest))
                    if blocked >= r.MAX_BLOCKED_TOOL_CALLS:
                        trace['status'] = 'invalid_action_limit'
                        break
                    continue
                self.budget.check()
                env.timeout = self.budget.timeout(30)
                step = r._execute_tool_call(env, call, len(trace['steps']))
                trace['steps'].append(step)
                blocked = 0
                self.budget.check()
                visible, projection = client.project_observation(step['tool_name'], step['observation'], step['parameters'])
                step['raw_observation'], step['observation'], step['projection'] = step['observation'], visible, projection
                messages.append(r._tool_message(call, step))
                latest = visible
                if step['done']:
                    trace.update(status='done', done=True, terminal_result=step['result'])
                    break
            else:
                trace['status'] = 'max_steps'
        except BaseException as exc:
            if isinstance(exc, (SystemExit, KeyboardInterrupt)): raise
            if isinstance(exc, r.ToolExecutionError):
                trace['steps'].append(exc.step)
            trace['status'] = 'budget_exhausted' if isinstance(exc, BudgetExceeded) else 'error'
            trace['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        finally:
            try:
                env.timeout = 5
                env.release()
            except Exception as exc:
                trace['release_error'] = str(exc)
            trace['elapsed_seconds'] = time.time()-started
            append(self.run/'trajectories.jsonl', trace)
        # Operational errors stop the run; ordinary policy/context failures remain observations.
        if trace['release_error']:
            raise RuntimeError('Environment lease release failed')
        if trace['status'] == 'error' and trace['error']['type'] != 'ContextBudgetError':
            raise RuntimeError('Probe operation failed: ' + trace['error']['message'])
        if trace['status'] == 'budget_exhausted':
            raise BudgetExceeded(trace['error']['message'])
        self.budget.check()
        return trace


def sample(run, phase):
    cfg = checked_run(run)
    budget = budget_for(run)
    if not (run/'ready.json').exists(): raise ValueError('Run wait before sampling')
    if not load(run/'cpu_check.json')['passed']: raise ValueError('CPU replay must pass before GPU sampling')
    # Atomic process lock prevents concurrent drivers sharing a run.
    lock = run/'driver.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        if phase == 'full' and not load(run/'smoke.json')['passed']:
            raise ValueError('Smoke did not verify restoration; stop and inspect')
        engine = Engine(run, budget)
        tasks = rows(PACKAGE/'tasks.jsonl')
        events = rows(run/'events.jsonl') if (run/'events.jsonl').exists() else []
        started_ids = {e['task_id'] for e in events if e['event'] == 'task_started'}
        completed_ids = {e['task_id'] for e in events if e['event'] == 'task_completed'}
        if started_ids - completed_ids:
            raise ValueError('An interrupted task exists; preserve partial run and inspect, do not reroll')
        if phase == 'smoke' and started_ids:
            raise ValueError('Smoke already attempted; do not silently rerun')
        for index, task in enumerate(tasks[:4] if phase == 'smoke' else tasks):
            if task['task_id'] in completed_ids: continue
            if budget.remaining() < 900: break
            tid = task['task_id']
            append(run/'events.jsonl', {'event': 'task_started', 'task_id': tid, 'at': time.time()})
            for arm in (('A', 'B') if index % 2 == 0 else ('B', 'A')):
                if arm == 'A':
                    for j in range(4): engine.execute(task, f'{tid}/A/{j}')
                else:
                    prefix = engine.execute(task, f'{tid}/B/prefix', stop_at_branch=True)
                    if prefix['branch']:
                        for j in range(4): engine.execute(task, f'{tid}/B/{j}', prefix=prefix)
            append(run/'events.jsonl', {'event': 'task_completed', 'task_id': tid, 'at': time.time()})
        if phase == 'smoke':
            events = rows(run/'events.jsonl')
            completed = [e for e in events if e['event'] == 'task_completed']
            restored = rows(run/'restorations.jsonl') if (run/'restorations.jsonl').exists() else []
            result = {'passed': len(completed) == 4 and (len(restored) == 0 or len(restored) % 4 == 0),
                      'completed_tasks': len(completed), 'verified_restorations': len(restored),
                      'natural_restoration_verified': len(restored) >= 4,
                      'note': 'Zero trigger is a coverage result; CPU replay is required; first natural restoration must match before continuation'}
            save(run/'smoke.json', result)
            if not result['passed']: raise RuntimeError('Smoke incomplete or no natural branch; stop')
        print('Sampling phase completed; remaining seconds: ' + str(round(budget.remaining())))
    except BaseException as exc:
        save(run/'sampling_stop.json', {'at': time.time(), 'phase': phase, 'error': str(exc),
                                       'type': type(exc).__name__})
        (run/'STOP_SERVICES').touch()
        raise
    finally:
        lock.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['init', 'clock', 'serve', 'wait', 'smoke', 'full', 'stop', 'cpu-check', 'pipeline'])
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--bundle', type=Path, default=ROOT/'outputs/decision_probe_v1/runtime')
    p.add_argument('--model-dir', type=Path)
    a = p.parse_args()
    run = a.run_dir.resolve()
    if a.command == 'cpu-check': cpu_check(a.bundle, run)
    elif a.command == 'init':
        if a.model_dir is None: p.error('--model-dir is required for init')
        init_run(run, a.bundle, a.model_dir)
    elif a.command == 'pipeline':
        cfg = checked_run(run)
        if not load(run/'cpu_check.json')['passed']: raise ValueError('CPU check missing')
        clock(run)
        log = (run/'supervisor.log').open('ab')
        proc = subprocess.Popen([sys.executable, __file__, 'serve', '--run-dir', str(run)], stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_ready(run)
            sample(run, 'smoke')
            sample(run, 'full')
        finally:
            (run/'STOP_SERVICES').touch()
            try: proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                # Supervisor owns and cleans its own children on SIGTERM.
                proc.terminate(); proc.wait(timeout=30)
            log.close()
            from report_decision_probe import report
            report(run, run/'report')
    elif a.command == 'clock': clock(run)
    elif a.command == 'serve': serve(run)
    elif a.command == 'wait':
        try: wait_ready(run)
        except BaseException:
            (run/'STOP_SERVICES').touch()
            raise
    elif a.command == 'stop':
        checked_run(run)
        (run/'STOP_SERVICES').touch()
        print('Requested shutdown of this probe\'s own model/environment processes')
    else: sample(run, a.command)


if __name__ == '__main__': main()
