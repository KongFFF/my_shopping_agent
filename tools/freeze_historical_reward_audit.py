"""Freeze actual SFT/GRPO Dev50 histories before adding acceptance facts."""
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'analysis/purchase_guard_v1_1_dev50_20261031'
OUT=ROOT/'analysis/semantic_history_acceptance_20261107'
ARCHIVE=ROOT/'upstream/shopping-grpo-longhorizon/environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz'
SEED='semantic-history-acceptance-20261107-v1'
REGRESSIONS=[5293,8666,12703]

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def lines(p,rs):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs),encoding='utf-8')


def purchase_event(row):
    terminal=row.get('terminal_result') or {};steps=row.get('steps') or []
    last=steps[-1] if steps else {};result=last.get('result') or {}
    guard=((result.get('reward_detail') or {}).get('evidence') or {}).get('purchase_guard')
    if guard and guard.get('status')=='exhausted':return {'status':'purchase_guard_exhausted','purchase_executed':False,'evidence':{'last_step_index':last.get('step_index'),'guard':guard}}
    if row.get('error'):return {'status':'execution_error','purchase_executed':False,'error':str(row['error'])}
    purchase=terminal.get('purchase') or {}
    if not purchase:
        return {'status':'no_purchase','purchase_executed':False,'evidence':{'last_tool':last.get('tool_name'),'termination_reason':terminal.get('termination_reason')}}
    event=result.get('purchase') or {}
    keys=('asin','options')
    if last.get('tool_name')!='buy_now' or not result.get('done') or not terminal.get('done') or any(event.get(k)!=purchase.get(k) for k in keys):
        return {'status':'purchase_event_unproven','purchase_executed':False}
    return {'status':'executed_purchase','purchase_executed':True,'asin':str(purchase['asin']),
            'selected_options':purchase.get('options') or {},'recorded_price':purchase.get('price'),
            'evidence':{'last_step_index':last.get('step_index'),'last_tool':'buy_now','terminal_event_snapshot_equal':True}}


def main():
    OUT.mkdir(exist_ok=True)
    manifest=json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))
    with gzip.open(ARCHIVE,'rt',encoding='utf-8') as f:products=json.load(f)
    tasks=[(p,i) for p in products for i in p.get('instructions',[]) if i.get('attributes')]
    catalog={str(p['asin']):p for p in products}
    histories={};hashes={}
    for run in ('sft_dev50','grpo_dev50'):
        path=SOURCE/run/'trajectories.jsonl';hashes[str(path.relative_to(ROOT))]=digest(path)
        assert digest(path)==manifest['files_sha256'][run+'/trajectories.jsonl']
        histories[run]=rows(path);assert len(histories[run])==50 and len({r['task_id'] for r in histories[run]})==50
    ids=set(manifest['task_ids']);assert all({r['task_id'] for r in rs}==ids for rs in histories.values())
    train_ids={r['task_id'] for r in rows(ROOT/'upstream/shopping-grpo-longhorizon/data/grpo/train.jsonl')}
    assert not ids&train_ids
    ordered=sorted(ids,key=lambda t:hashlib.sha256(f'{SEED}:{t}'.encode()).hexdigest())
    sample=ordered[:12]
    selected=sorted(set(sample)|set(REGRESSIONS))
    queries=[{'task_id':t,'query':tasks[t][1]['instruction']} for t in sorted(ids)]
    querymap={r['task_id']:r['query'] for r in queries}
    records=[];asins=[];variant_keys=set()
    for run,rs in histories.items():
        for r in rs:
            q=querymap[r['task_id']]
            initial=(r.get('initial_result') or {}).get('instruction') or ''
            assert q in initial,(run,r['task_id'],'query not present in initial observation')
            event=purchase_event(r)
            record={'run':run,'trajectory_id':r['trajectory_id'],'task_id':r['task_id'],**event,
                    'query_sha256':hashlib.sha256(q.encode()).hexdigest(),'selected_for_manual_acceptance':r['task_id'] in selected,
                    'old_reward_for_reference_only':r.get('final_reward'),
                    'old_terminal_for_reference_only':(r.get('terminal_result') or {}).get('termination_reason')}
            if event['purchase_executed']:
                assert event['asin'] in catalog
                asins.append(event['asin']);variant_keys.add((event['asin'],json.dumps(event['selected_options'],sort_keys=True,ensure_ascii=False)))
            records.append(record)
    lock={'seed':SEED,'all_tasks':sorted(ids),'all_trajectories':100,'hash_sample12':sample,'known_regressions_separate':REGRESSIONS,
          'manual_acceptance_tasks':selected,'source_sha256':hashes,'product_data_sha256':digest(ARCHIVE),
          'selection':'deterministic hash sample before current fact annotations; known regression cases reported separately, not a blind test',
          'not_a_final_test':True,'training_ids_overlap':[]}
    if (OUT/'sample_lock.json').exists():assert json.loads((OUT/'sample_lock.json').read_text(encoding='utf-8'))==lock
    write(OUT/'sample_lock.json',lock);lines(OUT/'dev50_queries.jsonl',queries)
    lines(OUT/'trajectory_inventory100.jsonl',records)
    lines(OUT/'actual_purchases.jsonl',[r for r in records if r['purchase_executed']])
    write(OUT/'actual_product_asins.json',sorted(set(asins)))
    summary={'trajectories':100,'tasks':50,'events':dict(Counter(r['status'] for r in records)),
             'actual_products':len(set(asins)),'actual_variants':len(variant_keys),'train_dev_contract_overlap':0,
             'fixed_hash_sample12':sample,'known_regressions':REGRESSIONS,'manual_acceptance_tasks':len(selected),'api_calls':0}
    write(OUT/'inventory_summary.json',summary);print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
