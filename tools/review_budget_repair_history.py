"""Verify the existing frozen v2 repair against the newly audited35 queries."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    package=ROOT/'repairs/task_contract_v2'
    lock=json.loads((package/'freeze.lock.json').read_text(encoding='utf-8'))
    manifest=json.loads((package/'manifest.json').read_text(encoding='utf-8'))
    rows=json.loads((ROOT/'analysis/grpo_mechanism_audit_20261116/budget_parser_candidates.json').read_text(encoding='utf-8'))
    historical={r['task_id']:r for r in map(json.loads,(package/'all_train_audit.jsonl').read_text(encoding='utf-8-sig').splitlines())}
    sources={};hashes={}
    for name in ('constraints.py','reward_features.py','reward.py'):
        raw=(package/'runtime'/name).read_bytes();h=hashlib.sha256(raw).hexdigest()
        assert h==lock['contract_files_sha256']['runtime/'+name]
        sources[name]=('',raw.decode('utf-8'));hashes[name]=h
    # Use frozen files directly; do not regenerate them from the dirty checkout.
    sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'outputs/decision_probe_v1/runtime/environment')]
    from build_task_contract_v2 import load_runtime
    loaded=load_runtime(sources)
    parser,features,reward=loaded['constraints.py'],loaded['reward_features.py'],loaded['reward.py']
    results=[]
    for r in rows:
        assert r['native_upper'] is None
        assert historical[r['task_id']]['query']==r['query']
        d=parser.budget_details_v2(r['query']);limit=r['requirement']['expected']['value']
        assert d['status']=='parsed'and d['upper']==limit
        assert r['task_id']in manifest['retained_task_ids']
        product={'asin':'synthetic','category':'配件','title':'测试商品','attribute':[],'customization_options':{}}
        goal={'asin':'synthetic',**features.compile_reward_features({'instruction':r['query'],'attributes':[],'instruction_options':[]},product)}
        gates=[]
        for price in (limit-.01,limit,limit+.01):
            result=reward.evaluate_purchase(product,goal,selected_options={},price=price)
            status=result.hard_gates['budget']['status'];gates.append({'price':price,'status':status})
        assert [g['status']for g in gates]==['pass','pass','fail']
        results.append({'task_id':r['task_id'],'query':r['query'],'baseline_upper':None,
            'frozen_v2_upper':d['upper'],'upper_inclusive':d['upper_inclusive'],
            'retained_in_old928':True,'synthetic_gate_boundary_checks':gates})
    strict=parser.budget_details_v2('价格要低于20元')
    assert strict['upper']==20 and not strict['upper_inclusive']
    strict_goal={'asin':'synthetic',**features.compile_reward_features(
        {'instruction':'价格要低于20元','attributes':[],'instruction_options':[]},product)}
    assert reward.evaluate_purchase(product,strict_goal,selected_options={},price=20).hard_gates['budget']['status']=='fail'
    out=ROOT/'analysis/budget_repair_history_20261116';out.mkdir(parents=True,exist_ok=True)
    packet={'freeze_id':lock['freeze_id'],'frozen_on':lock['frozen_on'],
        'runtime_hashes_verified':hashes,'current35_parsed_correctly':len(results),
        'all35_retained_in_old928':True,'synthetic_boundary_checks_passed':3*len(results),
        'strict_less_than20_preserved':True,'rows':results,
        'scope':'Frozen repair capability and local synthetic gates, not deployment or trained-model benefit.'}
    (out/'verification.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS: old frozen repair recognizes35/35;105 synthetic boundary checks pass; strict less-than preserved.')


if __name__=='__main__':main()
