"""Local CPU-only proof, using existing conda Flask and isolated Gym dependency."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from difficulty_scheduler import write
from prepare_difficulty_scheduler import PACKAGE, check_runtime, lines

bundle = ROOT / 'outputs/difficulty_scheduler_v1/runtime'
check_runtime(bundle)
output = Path(__file__).parent
with socket.socket() as connection:
    if connection.connect_ex(('127.0.0.1', 5723)) == 0:
        raise RuntimeError('Port occupied')
log = (output / 'cpu_environment.log').open('wb')
env = dict(os.environ, PYTHONPATH=str(ROOT / 'outputs/difficulty_scheduler_v1/cpu_dependencies'), PYTHONUTF8='1')
child = subprocess.Popen([sys.executable, str(ROOT / 'tools/difficulty_scheduler_environment.py'),
    '--bundle', str(bundle)], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env,
    **({'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {'start_new_session': True}))
checks = []
try:
    for _ in range(90):
        if child.poll() is not None:
            raise RuntimeError('CPU server exited: inspect log')
        try:
            with urllib.request.urlopen('http://127.0.0.1:5723/api/difficulty_scheduler_identity', timeout=2) as response:
                identity = json.load(response)
            break
        except OSError:
            time.sleep(1)
    else:
        raise RuntimeError('CPU startup timeout')
    sys.path.insert(0, str(bundle / 'project/src'))
    from shopping_grpo.evaluation.rollout import ShopAgentEnv
    for task in lines(PACKAGE / 'train_tasks.jsonl')[:3]:
        a, b = (ShopAgentEnv(base_url='http://127.0.0.1:5723') for _ in range(2))
        try:
            x, y = a.reset(task['task_id']), b.reset(task['task_id'])
            clean = lambda z: {k: v for k, v in z.items() if k not in ('idx', 'env_idx')}
            assert a.env_idx != b.env_idx and clean(x) == clean(y)
            query = x['instruction'].removeprefix('Instruction: ')
            assert clean(a.step('search[' + query + ']')) == clean(b.step('search[' + query + ']'))
            checks.append({'task_id': task['task_id'], 'distinct_slots': True, 'public_search_matched': True})
        finally:
            try:
                a.release()
            finally:
                b.release()
    write(output / 'cpu_environment.json', {'passed': len(checks) == 3, 'tasks': checks,
          'identity': identity, 'model_used': False})
    print('PASS: actual CPU original environment, independent reset and search x3')
finally:
    child.terminate()
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
    log.close()
