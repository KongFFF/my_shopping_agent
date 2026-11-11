"""Frozen protocol and standard-library utilities for checkpoint-pair v1."""
import hashlib
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'experiments/checkpoint_pair_v1'
UP = ROOT/'upstream/shopping-grpo-longhorizon'
BASE = 'a583dba4c149be09622ce58ad6f7a160db77bb41'
VERSION = 'checkpoint-pair-v1'
ENV_URL = 'http://127.0.0.1:5721'
MODEL_URL = 'http://127.0.0.1:8021/v1'
MODEL_PATHS = {'sft':'outputs/runs/sft_20261021_165243/merged',
              'grpo':'outputs/runs/grpo_baseline_20261022_202816/export/step500_v1/merged'}
SCRIPTS = ['checkpoint_pair_common.py','prepare_checkpoint_pair.py','checkpoint_pair_environment.py',
           'run_checkpoint_pair.py','report_checkpoint_pair.py','run_checkpoint_pair_pipeline.py']
SETTINGS = {'attempts':8,'temperature':.7,'top_p':.9,'max_tokens':512,'max_steps':35,
            'context_window':24576,'context_safety_margin':512,'context_compaction_enable':False,
            'observation_token_budget':1536,'observation_detail_token_budget':4096,
            'observation_generic_token_budget':768,'observation_search_top_k':20}


def dump(v): return json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
def digest(raw): return hashlib.sha256(raw.encode('utf-8') if isinstance(raw,str) else raw).hexdigest()
def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def file_hash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def code_hash(p): return digest(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_bytes(dump(v).encode('utf-8'));temp.replace(p)
def append(p,v):
    with Path(p).open('a',encoding='utf-8') as f:
        f.write(json.dumps(v,ensure_ascii=False,sort_keys=True)+'\n');f.flush()
def seed_for(task_id,attempt,turn):
    return int(digest(f'checkpoint-pair-v1:{task_id}:{attempt}:{turn}')[:8],16)%2147483647
def code_receipt(): return {n:code_hash(ROOT/'tools'/n) for n in SCRIPTS}
def check_package():
    m=load(PACKAGE/'manifest.json')
    if m['version']!=VERSION or m['code_sha256']!=code_receipt() or m['settings']!=SETTINGS:
        raise ValueError('Frozen checkpoint-pair protocol/code changed')
    for n,h in m['files_sha256'].items():
        if file_hash(PACKAGE/n)!=h:raise ValueError('Frozen input changed: '+n)
    tasks=rows(PACKAGE/'tasks.jsonl')
    if len(tasks)!=262 or len({t['task_id'] for t in tasks})!=262:raise ValueError('Invalid frozen task list')
    return m


def index_hash(p):
    con=sqlite3.connect(Path(p).resolve().as_uri()+'?mode=ro',uri=True)
    try:
        if con.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('Index integrity failed')
        m=json.loads(con.execute('SELECT payload FROM manifest').fetchone()[0])
        for k in ('python_version','sqlite_version'):m.pop(k,None)
        h=hashlib.sha256(dump(m).encode())
        for r in con.execute('SELECT * FROM products ORDER BY rowid'):h.update(dump(r).encode())
        return h.hexdigest()
    finally:con.close()


def runtime_receipt(bundle):
    bundle=Path(bundle)
    paths=[p for base in (bundle/'agent',bundle/'environment') for p in base.rglob('*')
           if p.is_file() and p.suffix in ('.py','.json','.html','.jinja','.sql')]
    index=bundle/'environment/search_engine/products.sqlite3'
    if file_hash(index)!=load(bundle/'index_manifest.json')['index_sha256']:
        raise ValueError('Physical index changed since build')
    paths.append(index)
    return {p.relative_to(bundle).as_posix():index_hash(p) if p.suffix=='.sqlite3'
            else code_hash(p) if p.suffix=='.py' else file_hash(p) for p in sorted(set(paths))}


def load_agent(bundle):
    import sys
    bundle=Path(bundle).resolve();sys.path.insert(0,str(bundle/'agent'))
    from shopping_grpo.evaluation import rollout
    if not Path(rollout.__file__).resolve().is_relative_to(bundle):raise ValueError('Wrong imported agent')
    return rollout


def outcome(trace):
    from shopping_grpo.evaluation.trajectory import normalize_trajectory
    from shopping_grpo.evaluation.metrics import compute_deterministic_metrics
    metrics=compute_deterministic_metrics(normalize_trajectory(trace))
    error=trace.get('error') or {}
    infra=bool(trace.get('release_error') or (error and error.get('type')!='ContextBudgetError'))
    infra=infra or not metrics['validity']['trajectory_contract_valid']
    strict=metrics['reward_and_outcome']['strict_gold_success'] and not infra
    result=trace.get('terminal_result') or {};detail=result.get('reward_detail') or {}
    proxy=bool(trace.get('done') and result.get('done') and detail.get('reward_type')=='gold_purchase'
               and result.get('reward_valid') is True and not infra and not detail.get('sampling_invalid'))
    return {'classification':'infrastructure_error' if infra else 'completed',
            'strict_success':bool(strict),'training_gold_proxy':proxy,'status':trace['status'],
            'reward_type':detail.get('reward_type'),'final_reward':trace.get('final_reward'),
            'terminal_reward':result.get('reward') if result.get('done') else None,
            'guard_rejections':len(trace.get('blocked_tool_calls') or []),
            'tool_steps':len(trace.get('steps') or []),'metrics':metrics}
