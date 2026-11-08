"""Offline real-catalog smoke audit. Target variants are probes, not rollouts."""
import copy
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
import semantic_contract_v2_r7_closeout as b
from semantic_product_verifier import verify, fingerprint, snapshot, automatic_facts, reviewed_facts, compare_requirement
from semantic_reward_r7_closeout import score_verified

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/semantic_product_verifier_20261107'
ARCHIVE=ROOT/'upstream/shopping-grpo-longhorizon/environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz'
CONTRACTS=ROOT/'analysis/semantic_closeout_20261107/validated1000.jsonl'


def write(path,v):path.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def lines(path,rs):path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs),encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True)
    with gzip.open(ARCHIVE,'rt',encoding='utf-8') as handle:products=json.load(handle)
    tasks=[(p,i) for p in products for i in p.get('instructions',[]) if i.get('attributes')]
    contracts=[json.loads(x) for x in CONTRACTS.read_text(encoding='utf-8').splitlines()]
    # Fixed first 20 original train IDs, chosen before reading outcomes. These
    # are static target-catalog probes, not agent success measurements.
    sample=contracts[:20]; results=[]; purchases=[]; statuses=Counter(); reasons=Counter()
    for row in sample:
        task=row['task_id'];product,instruction=tasks[task]; options={};errors=[]
        for target in instruction.get('instruction_options') or []:
            hits=[(axis,e['value']) for axis,entries in product.get('customization_options',{}).items()
                  for e in entries if e.get('value')==target]
            if len(hits)!=1 or hits[0][0] in options:errors.append({'value':target,'reason':'ambiguous_target_option'})
            else:options[hits[0][0]]=hits[0][1]
        if errors:result={'task_id':task,'probe_errors':errors}
        else:
            purchase={'task_id':task,'asin':str(product['asin']),'selected_options':options,'probe_source':'original_target_annotation_not_agent'}
            purchases.append(purchase)
            result=verify(row['normalized'],product,options)
            result['probe_source']='original_target_annotation_not_agent'
            statuses.update(result['statuses'].values());reasons.update(r['reason'] for r in result['comparisons'].values())
        results.append(result)
    lines(OUT/'train20_target_probes.jsonl',purchases);lines(OUT/'train20_comparisons.jsonl',results)
    # A previously known Dev regression only. This does not select training
    # tasks or tune the formula, and does not inspect Final.
    product,instruction=tasks[8666];query=instruction['instruction']
    raw={'schema_version':b.VERSION,'task_id':8666,'query':query,**{d:[] for d in b.DIMENSIONS},'pending_mentions':[]}
    def add(d,st,attr,val,quote,op='eq'):
        raw[d].append({'subtype':st,'attribute':attr,'subject':{'kind':'order','name':'本次订单'} if d=='budget' else {'kind':'primary_product','name':'主商品'},
                       'importance':'required','operator':op,'expected':val,'price_basis':None,'evidence':[{'text':quote}],'review_status':'pending'})
    add('category','product_category','product_category',{'type':'text','value':'压脚'},'压脚')
    add('specifications','material','material',{'type':'text','value':'全钢'},'全钢材质')
    add('functions','function','function',{'type':'text','value':'工业型电脑平车用'},'工业型电脑平车用')
    add('functions','function','function',{'type':'text','value':'单边隐形拉链'},'单边隐形拉链')
    add('specifications','bundle','promotion',{'type':'bundle','paid_quantity':10,'gift_quantity':1,'received_quantity':11,'unit':'piece','same_item':True},'买10个要能送1个','matches_bundle')
    add('budget','total_price','total_price',{'type':'text','value':'预算100元以内'},'预算100元以内')
    # This is an atomic-predicate regression, not approval of the entire Dev
    # query. Its initial 需要一个 and later promotion quantity still trigger
    # full-contract coverage checks; retain that limitation visibly.
    try:
        c,audit=b.normalize(raw,{'task_id':8666,'query':query})
        full_contract_error=None
    except ValueError as error:
        full_contract_error=str(error);c=copy.deepcopy(raw);index=0
        for d in b.DIMENSIONS:
            for r in c[d]:
                index+=1;r['id']=f'probe{index}'
                for e in r['evidence']:
                    e['start']=query.index(e['text']);e['end']=e['start']+len(e['text'])
                if d=='budget':r['operator'],r['expected'],_=b.budget_amount(r['expected']['value'])
    facts=[]
    def reviewed(d,st,attr,val,path,actual,*,options=None):
        return {'product_asin':str(product['asin']),'product_sha256':fingerprint(product),'review_status':'approved',
                'scope':'product' if options is None else 'sku','options':options,'dimension':d,'subtype':st,'attribute':attr,
                'subject':{'kind':'primary_product','name':'主商品'},'value':val,'complete':True,
                'evidence':[{'pointer':path,'value':actual}]}
    title=product['title']
    facts.append(reviewed('category','product_category','product_category',{'type':'text','value':'压脚'},'/title',title))
    facts.append(reviewed('functions','function','function',{'type':'text','value':'工业型电脑平车用'},'/title',title))
    facts.append(reviewed('functions','function','function',{'type':'text','value':'单边隐形拉链'},'/title',title))
    # Reviews assert only facts supported by the public label plus the
    # product identity (presser-foot promotion), never instruction_options.
    for marker in (False,True):
        options={'T168':'单边拉链压脚买10送1'}
        if marker:options['颜色分类']='规格'
        facts.append(reviewed('specifications','bundle','promotion',{'type':'bundle','paid_quantity':10,'gift_quantity':1,'received_quantity':11,'unit':'piece','same_item':True},
                              '/customization_options/T168/1/value','单边拉链压脚买10送1',options=options))
    cases=[]
    for package in ('隐形单边拉链压脚1个','单边拉链压脚买10送1'):
        for marker in (False,True):
            options={'T168':package}
            if marker:options['颜色分类']='规格'
            relevant=[f for f in facts if f['scope']=='product' or f['options']==options]
            state=snapshot(product,options);extra,rejected=reviewed_facts(product,state,relevant)
            assert not rejected,rejected
            all_facts=automatic_facts(product,state)+extra
            comparisons={r['id']:compare_requirement(r,d,product,state,all_facts) for d in b.DIMENSIONS for r in c[d]}
            result={'statuses':{k:v['status'] for k,v in comparisons.items()},'comparisons':comparisons,
                    'selection_complete':not state['missing_axes'],'execution_complete':False,'reward_ready':False,'reward':None,
                    'full_contract_error':full_contract_error,'scope':'atomic comparison regression only'}
            cases.append({'options':options,'comparisons':result})
    assert cases[0]['comparisons']['statuses']==cases[1]['comparisons']['statuses']
    assert cases[2]['comparisons']['statuses']==cases[3]['comparisons']['statuses']
    bundle_id=next(r['id'] for r in c['specifications'] if r['subtype']=='bundle')
    assert cases[1]['comparisons']['statuses'][bundle_id]=='fail'
    assert cases[3]['comparisons']['statuses'][bundle_id]=='pass'
    assert not cases[0]['comparisons']['selection_complete'] and not cases[2]['comparisons']['selection_complete']
    write(OUT/'task8666_partial_probe_contract.json',c);lines(OUT/'task8666_reviewed_catalog_facts.jsonl',facts)
    write(OUT/'task8666_counterfactuals.json',cases)
    summary={'train_probe_tasks':len(sample),'train_probe_ids':[r['task_id'] for r in sample],
             'train_probe_source':'original target variants, not model trajectories or accuracy',
             'comparison_counts':dict(statuses),'comparison_reasons':dict(reasons),
             'train_reward_ready':sum(r.get('reward_ready',False) for r in results),
             'known_dev_regression8666':{'atomic_comparison_passed':True,'wrong_bundle_status':'fail',
                'correct_bundle_status':'pass','full_contract_ready':False,'full_contract_error':full_contract_error,
                'missing_placeholder':'semantic statuses unchanged; selection incomplete; static probes are not executed purchases, no reward emitted'},
             'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ARCHIVE,CONTRACTS)},
             'api_calls':0,'training_runs':0,'final_evaluations_run':0,'final_task_ids_selected':0}
    write(OUT/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
