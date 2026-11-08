"""Apply confirmed closeout choices, retain unresolved items, replay train1000."""
import copy
import hashlib
import json
from pathlib import Path
from collections import Counter
import semantic_contract_v2_r7_product as b

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'analysis/semantic_closeout_20261107'
OUT=ROOT/'analysis/semantic_product_stage_20261107'
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def lines(p,v):p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in v),encoding='utf-8')
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True)
    old={r['task_id']:r for r in rows(SOURCE/'validated1000.jsonl')}
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE/'validated1000.jsonl',SOURCE/'candidate_drafts1000.jsonl')}
    drafts=[];results=[];edits=[];derived=[]
    for row in rows(SOURCE/'candidate_drafts1000.jsonl'):
        c=copy.deepcopy(row['raw_draft']);before=copy.deepcopy(c);t=row['task_id'];c['schema_version']=b.VERSION
        if t in (18823,19437):
            quote='两袋五百克' if t==18823 else '需要120g，共10包'
            r=next(r for r in c['specifications'] if r['subtype']=='net_content')
            r.update(subtype='quantity',attribute='purchase_mass',subject={'kind':'order','name':'本次订单'},importance='required',evidence=[{'text':quote}])
            c['pending_mentions']=[p for p in c['pending_mentions'] if p['evidence']['text']!=quote]
            derived.append({'task_id':t,'confirmed_by_user':True,'order_total_g':500 if t==18823 else 120,
                            'bags':2 if t==18823 else 10,'derived_per_bag_g':250 if t==18823 else 12,
                            'added_as_extra_reward_requirement':False,'source':quote})
        if t==7483:
            # These are targets for applying the film, not goods included in
            # the order. This source-supported role repair needs no guessing.
            c['specifications']=[r for r in c['specifications'] if not (r['subtype']=='components' and
                any(i['name'] in ('方向盘','车载摄像头') for i in r['expected'].get('items',[])))]
            for name in ('方向盘','车载摄像头'):
                c['specifications'].append({'subtype':'compatibility','attribute':'compatibility',
                    'subject':{'kind':'compatible_target','name':name},'importance':'required','operator':'eq',
                    'expected':{'type':'text','value':'适用于'+name},'price_basis':None,
                    'evidence':[{'text':'给方向盘和车载摄像头贴一层'}],'review_status':'pending'})
        if t==8188:
            r=next(r for r in c['specifications'] if r['subtype']=='components' and
                   any(i['name']=='羽毛球造型' for i in r['expected'].get('items',[])))
            r.update(subtype='form_factor',attribute='form_factor',operator='eq',expected={'type':'text','value':'羽毛球造型'})
            r=next(r for r in c['specifications'] if r['subtype']=='quantity' and r['expected'].get('value')==2)
            r.update(subtype='packaging',attribute='pack_count')
        # Two concrete misclassifications found in the same fixed product
        # probes are corrected, rather than making catalog facts fit them.
        if t==187:
            c['category']=[]
            c['pending_mentions'].append({'evidence':{'text':'妆效'},'reason':'ambiguous_meaning','importance':'required'})
        if t==366:
            r=next(r for r in c['specifications'] if r['subtype']=='material' and r['expected'].get('value')=='轻薄')
            r.update(subtype='function',attribute='function')
        changed={k:v for k,v in c.items() if k!='schema_version'}!={k:v for k,v in before.items() if k!='schema_version'}
        if changed:edits.append({'task_id':t,'query':row['query'],'before':before,'after':copy.deepcopy(c),'reward_approved':False})
        identity={k:row[k] for k in ('task_id','query','source_attempt_line')}
        drafts.append({**identity,'raw_draft':c,'reward_approved':False})
        out={**identity,'program_valid':False,'whole_contract_approved':False,'reward_approved':False}
        try:n,a=b.normalize(c,{'task_id':t,'query':row['query']})
        except Exception as error:out.update(error=getattr(error,'message',str(error)),queue='program_blocked')
        else:
            fs=[f for x in a if x['kind']=='semantic_review_findings' for f in x['findings']]
            pending=n['pending_mentions'] or any(r['importance']=='uncertain' for d in b.DIMENSIONS for r in n[d])
            out.update(program_valid=True,normalized=n,audit=a,review_findings=fs,queue='semantic_review_pending' if pending or fs else 'no_static_signals')
            for _,e in b._evidences(n):assert row['query'][e['start']:e['end']]==e['text']
        results.append(out)
        if len(results)%250==0:print(f'{len(results)}/1000',flush=True)
    newly=[r['task_id'] for r in results if old[r['task_id']]['program_valid'] and not r['program_valid']]
    assert not newly,newly
    assert len(results)==1000
    for path,h in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
    lines(OUT/'candidate_drafts1000.jsonl',drafts);lines(OUT/'validated1000.jsonl',results)
    lines(OUT/'draft_edits.jsonl',edits);lines(OUT/'derived_weight_notes.jsonl',derived)
    summary={'total_tasks':1000,'program_valid':sum(r['program_valid'] for r in results),'newly_blocked':newly,
        'raw_edited_ids':[r['task_id'] for r in edits],'known12_weight_scopes_resolved':[18823,19437],
        'other_known12_still_open':[938,9984,10272,5598,11018,8477,7131,7483,8188,12308],
        'extra_probe_label_repairs':[187,366],'source_sha256':hashes,'queues':dict(Counter(r['queue'] for r in results)),
        'api_calls':0,'training_runs':0,'whole_contracts_approved':0}
    write(OUT/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False));b.write_schema()

if __name__=='__main__':main()
