"""Public-catalog probes for confirmed closeout semantics; no model calls."""
import gzip
import json
import re
from pathlib import Path
from semantic_product_verifier_v3 import verify

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/semantic_final12_20261107'


def main():
    contracts={r['task_id']:r for r in (json.loads(x) for x in (OUT/'validated1000.jsonl').read_text(encoding='utf-8').splitlines())}
    archive=ROOT/'upstream/shopping-grpo-longhorizon/environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz'
    with gzip.open(archive,'rt',encoding='utf-8') as f:products=json.load(f)
    tasks=[p for p in products for i in p.get('instructions',[]) if i.get('attributes')]
    probes=[
        (938,{'颜色分类':'φ0.9mm*1米'},'unit_price','pass'),
        (9984,{'颜色分类':'(20个)贴片AO3415 丝印3415'},'unit_price','pass'),
        (10272,{'颜色分类':'15mm宽 1米'},'unit_price','pass'),
        (5598,{'刀尖角度':'60°以上','颜色分类':'军绿色','刃长':'33cm'},'blade','pass'),
        (11018,{'刀尖角度':'60°以上','分类':'白色','刀身长度':'65cm'},'blade','pass'),
        (7483,{'颜色分类':'零跑C10【方向盘+摄像头TPU膜】3片装'},'purchase_count','pass'),
        (7483,{'颜色分类':'零跑C10【方向盘+摄像头TPU膜】3片装'},'soft','unknown'),
        (8188,{'颜色分类':'金色两件套(1正羽球+1倒羽球)'},'metal','unknown'),
        (12308,{'颜色分类':'超值彩虹全套【24+36+48+72】送教程书','魔方种类':'其他'},'members','pass'),
        (12308,{'颜色分类':'超值套餐【彩虹24+36+72】送教程书','魔方种类':'其他'},'members','fail'),
    ]
    results=[]
    for task,options,focus,expected in probes:
        product=tasks[task];c=contracts[task]['normalized']
        result=verify(c,product,options)
        assert not result['snapshot']['selection_errors'],result['snapshot']['selection_errors']
        candidates=[(d,r) for d in ('category','budget','functions','specifications','brand','model') for r in c[d]]
        if focus=='unit_price':matched=[r for d,r in candidates if r['subtype']=='unit_price']
        elif focus=='blade':matched=[r for d,r in candidates if r['subtype']=='condition' and r['expected'].get('value')=='未开刃']
        elif focus=='purchase_count':matched=[r for d,r in candidates if r['subtype']=='quantity']
        elif focus=='members':matched=[r for d,r in candidates if r['expected']['type']=='member_ranges']
        elif focus=='soft':matched=[r for d,r in candidates if r['expected'].get('value')=='质感柔软']
        else:matched=[r for d,r in candidates if r['expected'].get('value')=='金属质感']
        assert len(matched)==1,(task,focus,matched)
        comparison=result['comparisons'][matched[0]['id']]
        assert comparison['status']==expected,(task,focus,comparison)
        results.append({'task_id':task,'product_asin':str(product['asin']),'selected_options':options,'focus':focus,'expected_status':expected,
                        'actual_comparison':comparison,'execution_complete':result['execution_complete'],'reward':None,
                        'scope':'static public-catalog comparison, not agent accuracy or full-task reward acceptance'})
    # Check the shared lower bound on the actual labelled USB/Type-C variant.
    p=tasks[8477];value=next(e['value'] for e in p['customization_options']['颜色分类'] if re.sub(r'\s+','',e['value'])=='黑色美标三脚插头3usb+2C1.5米')
    result=verify(contracts[8477]['normalized'],p,{'颜色分类':value})
    ports=[r for r in contracts[8477]['normalized']['specifications'] if r['subtype']=='attribute_count']
    assert len(ports)==2 and all(result['statuses'][r['id']]=='pass' for r in ports)
    results.append({'task_id':8477,'selected_options':{'颜色分类':value},'focus':'two_port_lower_bounds',
                    'actual_comparisons':[result['comparisons'][r['id']] for r in ports],'reward':None})
    (OUT/'real_catalog_checks.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary={'cases_passed':len(results),'unit_price_cases':3,'blade_cases':2,'member_matching_positive_and_negative':True,
             'TPU_not_assumed_soft':True,'iron_or_gold_not_assumed_metallic_appearance':True,
             'history_trajectory_acceptance':False,'api_calls':0,'training_runs':0}
    (OUT/'real_catalog_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
