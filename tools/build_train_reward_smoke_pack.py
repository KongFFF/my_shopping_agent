"""Seven source-reviewed training contracts for OFFLINE integration tests.

Not a replacement training dataset, not a rollout or a training launch. Roster
comes from the pre-existing catalog probe pack and is disjoint from Dev50.
Original train1000 is read-only. Labels are reviewed before product scoring.
"""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import semantic_contract_v2_r7_final12 as b
from semantic_product_verifier_v4 import verify
from semantic_history_adapter import canonicalize_purchase
from semantic_evidence_reward import score_training_verified

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/semantic_train_smoke7_20261107'
IDS=(181,313,315,366,390,396,460)


def main():
    OUT.mkdir(exist_ok=True)
    source=ROOT/'analysis/semantic_final12_20261107/validated1000.jsonl';before=hashlib.sha256(source.read_bytes()).hexdigest()
    records={r['task_id']:r for r in map(json.loads,source.read_text(encoding='utf-8').splitlines())}
    probes={r['task_id']:r for r in map(json.loads,(ROOT/'analysis/semantic_product_verifier_20261107/train20_target_probes.jsonl').read_text(encoding='utf-8').splitlines())}
    dev=set(json.loads((ROOT/'analysis/semantic_history_acceptance_20261107/sample_lock.json').read_text(encoding='utf-8'))['all_tasks'])
    assert set(IDS)<=set(probes) and not set(IDS)&dev
    reviewed=[];edits=[]
    for task in IDS:
        c=copy.deepcopy(records[task]['normalized']);assert not c['pending_mentions']
        def field(identity):return next(r for d in b.DIMENSIONS for r in c[d] if r['id']==identity)
        def edit(identity,**changes):
            r=field(identity);old=copy.deepcopy(r);r.update(changes)
            edits.append({'task_id':task,'id':identity,'before':old,'after':copy.deepcopy(r)})
        if task==315:
            edit('r003',attribute='function',operator='contains_all',expected={'type':'text_list','values':['太阳能供电','外接电源供电']})
        if task==366:
            edit('r005',operator='contains_all',expected={'type':'text_list','values':['轻','薄']})
        if task==390:
            text='女款';pos=c['query'].index(text)
            r={'id':'review_target_user','subtype':'target_user','attribute':'target_user','subject':{'kind':'primary_product','name':'主商品'},
                'importance':'required','operator':'eq','expected':{'type':'text','value':text},'price_basis':None,
                'evidence':[{'text':text,'start':pos,'end':pos+len(text)}],'review_status':'pending'}
            c['functions'].append(r);edits.append({'task_id':task,'id':r['id'],'before':None,'after':copy.deepcopy(r)})
        if task==396:
            edit('r008',operator='one_of',expected={'type':'text_list','values':['打篮球','骑行']})
        if task==460:
            edit('r006',subject={'kind':'component','name':'灯光'})
            edit('r007',subject={'kind':'component','name':'有效照明'})
            edit('r002',operator='one_of',expected={'type':'text_list','values':['户外庭院','草坪','花园']})
        for d in b.DIMENSIONS:
            for r in c[d]:r['review_status']='approved'
        b.validate(c,source_query=records[task]['query'],require_reviewed=True)
        reviewed.append({'task_id':task,'query':c['query'],'whole_contract_approved':True,'normalized':c,
                         'scope':'all source requirements read and checked; source label approval, not product satisfaction approval'})
    (OUT/'reviewed_contracts7.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reviewed),encoding='utf-8')
    (OUT/'source_edits.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in edits),encoding='utf-8')
    # Static selected-SKU integration only, explicitly not historical purchases.
    archive=ROOT/'upstream/shopping-grpo-longhorizon/environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz'
    with gzip.open(archive,'rt',encoding='utf-8') as h:products={str(p['asin']):p for p in json.load(h)}
    fs=[json.loads(s) for s in (ROOT/'analysis/semantic_product_facts_v2_20261107/catalog_facts.jsonl').read_text(encoding='utf-8').splitlines()]
    results=[]
    for row in reviewed:
        probe=probes[row['task_id']];p=products[str(probe['asin'])]
        mapping=canonicalize_purchase(p,probe['selected_options']);assert not mapping['errors']
        z=verify(row['normalized'],p,mapping['selected_options'],catalog_facts=[f for f in fs if f['product_asin']==str(p['asin'])],purchase_executed=True)
        score=score_training_verified(row['normalized'],z,source_query=row['query'])
        results.append({'task_id':row['task_id'],'synthetic_purchase_for_integration_only':True,'verification':z,'training_score':score})
    (OUT/'static_integration7.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf-8')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    summary={'reviewed_source_contracts':len(reviewed),'all_program_valid':True,'original_train1000_unchanged':True,
             'source_sha256':before,'training_dev_overlap':[],'static_integration_scores':sum(r['training_score']['reward_valid'] for r in results),
             'training_runs':0,'api_calls':0,'not_a_replacement_dataset':True,
             'cases':[{'task_id':r['task_id'],'training_reward':r['training_score']['reward'],
                       'complete_semantic_success':r['training_score']['semantic_success']} for r in results]}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
