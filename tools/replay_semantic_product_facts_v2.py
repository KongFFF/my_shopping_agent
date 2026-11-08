"""Replay the unchanged fixed catalog probes with independent fact cache."""
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from semantic_product_verifier_v2 import verify

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/semantic_product_facts_v2_20261107'
CONTRACTS=ROOT/'analysis/semantic_product_stage_20261107/validated1000.jsonl'
ROSTER=ROOT/'analysis/semantic_product_verifier_20261107/train20_target_probes.jsonl'
OLD=ROOT/'analysis/semantic_product_verifier_20261107/train20_comparisons.jsonl'
ARCHIVE=ROOT/'upstream/shopping-grpo-longhorizon/environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz'

def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def key(d,r):return json.dumps([d,r['subtype'],r['attribute'],r['subject'],r['operator'],r['expected']],ensure_ascii=False,sort_keys=True)


def main():
    with gzip.open(ARCHIVE,'rt',encoding='utf-8') as f:products={str(p['asin']):p for p in json.load(f)}
    new={r['task_id']:r for r in rows(CONTRACTS)}
    prior_contracts={r['task_id']:r for r in rows(ROOT/'analysis/semantic_closeout_20261107/validated1000.jsonl')}
    previous={r['task_id']:r for r in rows(OLD) if 'statuses' in r}
    facts=rows(OUT/'catalog_facts.jsonl')
    records=[];counts=Counter();paired=Counter();changes=[];rejections=[]
    for probe in rows(ROSTER):
        t=probe['task_id'];p=products[probe['asin']]
        relevant=[r for r in facts if r['product_asin']==probe['asin']]
        result=verify(new[t]['normalized'],p,probe['selected_options'],catalog_facts=relevant)
        result['probe_source']='same frozen target-catalog probe; not an agent rollout'
        counts.update(result['statuses'].values())
        oldc=prior_contracts[t]['normalized'];newc=new[t]['normalized']
        oldmap={key(d,r):previous[t]['statuses'][r['id']] for d in ('category','budget','functions','specifications','brand','model') for r in oldc[d]}
        for d in ('category','budget','functions','specifications','brand','model'):
            for r in newc[d]:
                ident=key(d,r)
                if ident in oldmap:
                    before=oldmap[ident];after=result['statuses'][r['id']];paired[(before,after)]+=1
                    if before!=after:changes.append({'task_id':t,'dimension':d,'requirement':r,'before':before,'after':after,
                        'comparison':result['comparisons'][r['id']]})
        if result['rejected_catalog_facts']:rejections.append({'task_id':t,'rejections':result['rejected_catalog_facts']})
        records.append(result)
    assert len(records)==19 and not rejections,rejections
    (OUT/'comparisons.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    (OUT/'changed_comparisons.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in changes),encoding='utf-8')
    summary={'same_probe_tasks':19,'catalog_facts':len(facts),'all_comparison_counts':dict(counts),
             'paired_unchanged_requirements':sum(paired.values()),'paired_status_transitions':{a+'->'+z:n for (a,z),n in paired.items()},
             'train_contracts_approved':0,'reward_ready':sum(r['reward_ready'] for r in records),
             'unknown_is_not_failure':True,'rejected_facts':rejections,'api_calls':0,'training_runs':0,
             'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (CONTRACTS,ROSTER,OUT/'catalog_facts.jsonl',ARCHIVE)},
             'limitations':'Static target-catalog coverage, not model success or full reward acceptance; only 19 pilot products have reviewed facts'}
    (OUT/'replay_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
