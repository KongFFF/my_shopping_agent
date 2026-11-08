"""Fully reviewed train181 static SKU counterfactual, not a model evaluation."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
from semantic_contract_v2_r7_final12 import DIMENSIONS,validate
from semantic_product_verifier_v3 import verify
from semantic_reward_r7_final12 import score_verified

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/semantic_final12_20261107'


def main():
    source=ROOT/'analysis/semantic_final12_20261107/validated1000.jsonl'
    row=next(json.loads(x) for x in source.read_text(encoding='utf-8').splitlines() if json.loads(x)['task_id']==181)
    c=copy.deepcopy(row['normalized'])
    # Original query and all eight requirements were read together: one set,
    # six strings, ion coating, rust resistance, guitar-string category,
    # 10-46, a gift, <=44 under the already confirmed approximate budget.
    assert len([r for d in DIMENSIONS for r in c[d]])==8 and not c['pending_mentions']
    review={'task_id':181,'query':c['query'],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'scope':'whole original query and all eight requirements, independent test copy only',
            'checked':['one set','six strings','ion coating','not easy to rust','electric guitar strings','10-46','gift presence','40 around -> <=44'],
            'main_train1000_modified':False}
    for d in DIMENSIONS:
        for r in c[d]:r['review_status']='approved'
    validate(c,source_query=c['query'],require_reviewed=True)
    archive=ROOT/'upstream/shopping-grpo-longhorizon/environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz'
    with gzip.open(archive,'rt',encoding='utf-8') as handle:p=next(p for p in json.load(handle) if str(p['asin'])=='927581317566')
    facts=[json.loads(x) for x in (ROOT/'analysis/semantic_product_facts_v2_20261107/catalog_facts.jsonl').read_text(encoding='utf-8').splitlines() if json.loads(x)['product_asin']==str(p['asin'])]
    cases=[]
    for label in ('小鱼M1 电吉他覆膜防锈【10-46】+赠品','小鱼M1 电吉他覆膜防锈【09-42】+赠品','小鱼A1 电吉他镍钢【10-46】+赠品'):
        result=verify(c,p,{'颜色分类':label},catalog_facts=facts,purchase_executed=True)
        score=score_verified(c,result,source_query=c['query'])
        cases.append({'selected_options':{'颜色分类':label},'simulated_purchase_execution':True,
                      'verification':result,'score':score,'source':'static counterfactual, not a historical rollout'})
    assert cases[0]['score']['reward']==1
    assert cases[1]['score']['reward']<0.25 and cases[1]['score']['semantic_success'] is False
    assert cases[2]['score']['reward'] is None and cases[2]['score']['reward_valid'] is False
    (OUT/'reviewed_case181_contract.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'reviewed_case181_record.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'real_train181_counterfactuals.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary={'task_id':181,'passed':True,'cases':[{'options':r['selected_options'],'reward':r['score']['reward'],'reward_valid':r['score']['reward_valid']} for r in cases],
             'api_calls':0,'training_runs':0,'history_trajectory_acceptance':False}
    (OUT/'real_reward_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
