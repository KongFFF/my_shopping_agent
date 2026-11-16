"""Audit and CPU-replay retained online windows; no model inference or legacy reevaluation data."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import zipfile
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'analysis/online_training_window_return_20261116/evidence.zip'
OUT = ROOT / 'analysis/online_window_audit_20261116'
BUNDLE = ROOT / 'outputs/difficulty_scheduler_v1/runtime/project'
TOKENIZER = ROOT / 'outputs/difficulty_scheduler_v1/return_20261113/run/adaptive_merged/tokenizer.json'


def sha(raw): return hashlib.sha256(raw).hexdigest()
def dump(obj): return json.dumps(obj, ensure_ascii=False, sort_keys=True)
def save(name, obj): (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def worker_init(envmeta):
    global WORKER_REPLAY
    sys.dont_write_bytecode = True
    WORKER_REPLAY, _ = runtime(envmeta)


def worker_job(item):
    key, record = item
    return {'key': key, **WORKER_REPLAY(record)}


def inputs():
    with zipfile.ZipFile(ARCHIVE) as z:
        manifest = json.loads(z.read('manifest.json'))
        for name, info in manifest['sources'].items():
            assert sha(z.read(name)) == info['archive_sha256'], name
        for name, info in manifest['generated_files'].items():
            assert sha(z.read(name)) == info['sha256'], name
        train = [json.loads(l) for l in z.read('inputs/train.jsonl').decode('utf-8-sig').splitlines() if l.strip()]
        assert len(train) == len({t['task_id'] for t in train}) == 1000
        occurrences = []
        for label in manifest['windows']:
            events = [json.loads(l) for l in z.read(label + '/training_diagnostics.jsonl').decode().splitlines() if l.strip()]
            used = {r['uid'] for e in events if e['event'] == 'optimizer_step' for r in e['rollouts']}
            for e in events:
                if e['event'] != 'generation_batch': continue
                for r in e['rollouts']:
                    key = sha(dump({'task_id': r['task_id'], 'actions': r['actions']}).encode())
                    occurrences.append({'window': label, 'step': e['global_step'], 'consumed': r['uid'] in used, 'key': key, 'rollout': r})
        envmeta = json.loads(z.read('inputs/environment.json'))
        # Reference drift is recoverable only when actual archived bytes elsewhere
        # match its declared historical SHA256. Do not use the drifted file.
        available = {sha(z.read(n)): n for n in z.namelist() if '/bound_sources/' in n}
        source_audit = {}
        for label in manifest['windows']:
            source_audit[label] = [{**r, 'historical_bytes_found': available.get(r['expected_sha256'])}
                                   for r in json.loads(z.read(label + '/source_checks.json'))]
        return manifest, occurrences, envmeta, source_audit


def runtime(envmeta):
    envdir = BUNDLE / 'environments/ShopSimulator/shop_env'
    hashes = {}
    for name, expected in envmeta['runtime_files_sha256'].items():
        matches = [p for p in envdir.rglob(name) if '__pycache__' not in p.parts]
        exact = [p for p in matches if sha(p.read_bytes()) == expected]
        assert exact, (name, 'environment binding differs')
        hashes[str(exact[0].relative_to(ROOT))] = expected
    data = envdir / 'data/items_eval_train.json'
    assert sha(data.read_bytes()) == envmeta['product_data_sha256']
    hashes[str(data.relative_to(ROOT))] = sha(data.read_bytes())
    index = envdir / 'search_engine/products.sqlite3'
    local_index_binding = json.loads((BUNDLE.parent / 'index_manifest.json').read_text(encoding='utf-8-sig'))
    assert sha(index.read_bytes()) == local_index_binding['index_sha256']
    import sqlite3
    con = sqlite3.connect(index.resolve().as_uri() + '?mode=ro', uri=True)
    try: assert con.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
    finally: con.close()
    hashes[str(index.relative_to(ROOT))] = sha(index.read_bytes())
    # Agent rendering/projection/tools must equal the pinned upstream Git bytes.
    import subprocess
    for name in ('observation.py', 'projection.py', 'tools.py', 'actions.py', 'product_id.py'):
        rel = 'src/shopping_grpo/environment/' + name
        p = BUNDLE / rel
        pinned = subprocess.check_output(['git', '-C', str(ROOT / 'upstream/shopping-grpo-longhorizon'), 'show', 'a583dba:' + rel])
        assert p.read_bytes().replace(b'\r\n', b'\n') == pinned.replace(b'\r\n', b'\n'), rel
        hashes[str(p.relative_to(ROOT))] = sha(p.read_bytes())
    assert sha(TOKENIZER.read_bytes()) == '06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523'
    os.environ.update(SHOP_ENV_CONFIG=str(envdir / 'configs/environment.json'), SHOP_MAX_STEPS='35',
                      SHOP_SEARCH_INDEX=str(envdir / 'search_engine/products.sqlite3'))
    os.chdir(envdir / 'shop_env')
    sys.path[:0] = [str(ROOT / 'outputs/online_window_cpu_deps'), str(ROOT / 'outputs/difficulty_scheduler_v1/cpu_dependencies'), str(envdir / 'shop_env'), str(envdir), str(BUNDLE / 'src')]
    logging.disable(logging.CRITICAL)
    from tokenizers import Tokenizer
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    from web_agent_site.utils import DEBUG_PROD_SIZE
    from shop_agent import shop_agent
    from shopping_grpo.environment.tools import tool_call_to_action
    from shopping_grpo.environment.observation import render_structured_observation
    from shopping_grpo.environment.projection import project_observation
    tok = Tokenizer.from_file(str(TOKENIZER))
    @lru_cache(maxsize=20000)
    def token_count(text): return len(tok.encode(text, add_special_tokens=False).ids)
    env = WebAgentTextEnv(observation_mode='text', split='train', num_products=DEBUG_PROD_SIZE, session_prefix='online-audit')
    def replay(record):
        tid = record['task_id']
        env.server.user_sessions.pop(f'online-audit-{tid}', None)
        terminal = shop_agent(env, 0, 'reset', tid)
        query = terminal['instruction'].removeprefix('Instruction: ')
        steps, projections = [], []
        try:
            for i, action in enumerate(record['actions']):
                raw = tool_call_to_action(action['tool'], action['parameters'])
                if raw is None: continue
                terminal = shop_agent(env, 0, 'interact', response=raw)
                if terminal.get('error'): raise ValueError(terminal['error'])
                visible = None
                if not terminal.get('done'):
                    text = render_structured_observation(terminal['observation_state'])
                    visible, meta = project_observation(action['tool'], text, parameters=action['parameters'],
                        count_tokens=token_count,
                        token_budget=1536, detail_token_budget=4096, generic_token_budget=768, search_top_k=20)
                    projections.append(meta.to_dict())
                steps.append({'turn': i + 1, 'action': action, 'observation': visible})
            detail = terminal.get('reward_detail') or {}
            return {'query': query, 'steps': steps, 'done': bool(terminal.get('done')),
                    'reward_type': detail.get('reward_type'), 'reward_valid': terminal.get('reward_valid'),
                    'reward': terminal.get('reward', 0), 'error': None,
                    'projection': {'observation_projection_count': len(projections),
                        'observation_raw_tokens': sum(p['raw_tokens'] for p in projections),
                        'observation_visible_tokens': sum(p['visible_tokens'] for p in projections),
                        'observation_truncated_count': sum(p['truncated'] for p in projections)}}
        except Exception as exc:
            return {'query': query, 'steps': steps, 'error': repr(exc)}
    return replay, hashes


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--limit', type=int); parser.add_argument('--workers', type=int, default=1); args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, occurrences, envmeta, source_audit = inputs()
    save('source_audit.json', {'archive_sha256': sha(ARCHIVE.read_bytes()), 'source_checks': source_audit})
    replay, hashes = runtime(envmeta)
    save('runtime.json', {'files': hashes, 'tokenizer_sha256': sha(TOKENIZER.read_bytes()), 'tokenizers_version': '0.22.2', 'model_calls': 0})
    cache_path = OUT / 'replays_parallel.jsonl.gz'
    cache = {}
    recovered = 0
    cache_inputs = (OUT / 'restored_observations.jsonl.gz',) if (OUT / 'restored_observations.jsonl.gz').exists() else (OUT / 'replays.jsonl.gz', cache_path)
    for old_path in cache_inputs:
        if old_path.exists():
            with gzip.open(old_path, 'rt', encoding='utf-8') as f:
                while True:
                    try:
                        line = f.readline()
                        if not line: break
                        r = json.loads(line); cache[r['key']] = r
                    except (EOFError, json.JSONDecodeError):
                        # Interrupted gzip tail: retain complete JSON records only.
                        recovered += 1; break
    eligible = [o for o in occurrences if o['rollout'].get('guard_rejections', 0) == 0 and not o['rollout'].get('error')
                and not o['rollout'].get('infrastructure_invalid') and o['rollout'].get('context_compactions', 0) == 0]
    unique = {o['key']: o['rollout'] for o in eligible}
    work = sorted(unique)
    if args.limit: work = work[:args.limit]
    with gzip.open(cache_path, 'at', encoding='utf-8') as f:
        pending = [(key, unique[key]) for key in work if key not in cache]
        print(f'CPU_RESUME cached={len(cache)} pending={len(pending)} workers={args.workers}', flush=True)
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init, initargs=(envmeta,)) as pool:
                for i, value in enumerate(pool.map(worker_job, pending, chunksize=4)):
                    f.write(dump(value) + '\n'); cache[value['key']] = value
                    if (i + 1) % 100 == 0:
                        f.flush(); print(f'CPU_REPLAY_NEW {i + 1}/{len(pending)}', flush=True)
        else:
            for i, (key, record) in enumerate(pending):
                value = {'key': key, **replay(record)}
                f.write(dump(value) + '\n'); cache[key] = value
                if (i + 1) % 100 == 0:
                    f.flush(); print(f'CPU_REPLAY_NEW {i + 1}/{len(pending)}', flush=True)
    counters = defaultdict(Counter)
    mismatch = []
    for o in occurrences:
        c = counters[o['window']]; c['all_generated'] += 1; c['consumed'] += o['consumed']
        r = o['rollout']
        if r.get('guard_rejections', 0) or r.get('error') or r.get('infrastructure_invalid') or r.get('context_compactions', 0):
            c['excluded_missing_guard_or_error_history'] += 1; continue
        if o['key'] not in cache: c['not_replayed_yet'] += 1; continue
        p = cache[o['key']]
        terminal_match = not p['error'] and p['done'] == r['done'] and (not r['done'] or
            (p['reward_type'] == r['reward_type'] and p['reward_valid'] == r['reward_valid'] and math.isclose(float(p['reward']), float(r['reward']['terminal_utility']), abs_tol=1e-7)))
        projection_match = not p['error'] and all(p['projection'][k] == r.get(k) for k in p['projection'])
        c['terminal_match'] += terminal_match; c['projection_aggregate_match'] += projection_match
        if not terminal_match or not projection_match:
            mismatch.append({'window': o['window'], 'step': o['step'], 'uid': r['uid'], 'rollout_index': r['rollout_index'], 'key': o['key'],
                             'terminal_match': terminal_match, 'projection_match': projection_match, 'error': p['error']})
    save('replay_audit.json', {'counts': {k: dict(v) for k, v in counters.items()}, 'unique_eligible_actions': len(unique),
         'unique_replayed': len(cache), 'mismatches': mismatch, 'partial': bool(args.limit), 'interrupted_gzip_tails_recovered': recovered,
         'limits': 'Executed action and projected observation restoration only; no model text or full original prompt reconstruction.'})
    print(dump({k: dict(v) for k, v in counters.items()}))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8'); sys.dont_write_bytecode = True; main()
