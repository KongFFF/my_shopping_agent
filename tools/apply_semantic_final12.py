"""Apply every user-confirmed decision for the fixed 12-task closeout."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
import semantic_contract_v2_r7_final12 as b

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'analysis/semantic_product_stage_20261107'
OUT=ROOT/'analysis/semantic_final12_20261107'
KNOWN=[18823,19437,938,9984,10272,8477,7483,12308,5598,11018,7131,8188]
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def lines(p,rs):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs),encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True)
    inputs=[SOURCE/'candidate_drafts1000.jsonl',SOURCE/'validated1000.jsonl']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    old={r['task_id']:r for r in rows(inputs[1])}
    drafts=[];results=[];edits=[];notes=[];decisions=[]
    for row in rows(inputs[0]):
        c=copy.deepcopy(row['raw_draft']);before=copy.deepcopy(c);t=c['task_id'];c['schema_version']=b.VERSION
        s=c['specifications']
        if t in (938,9984,10272):
            for r in c['budget']:
                if r['subtype']=='unit_price':r['price_basis']={'amount':1.0,'unit':'sale_unit'}
        if t==8477:
            quote='至少三个usb接口和两个type-c接口'
            for r in s:
                if r['subtype']=='attribute_count' and r['attribute']=='other_feature_count':
                    name='usb接口' if r['expected']['value']==3 else 'type-c接口'
                    r.update(subject={'kind':'component','name':name},operator='at_least',evidence=[{'text':quote}])
        if t==7483:
            r=next(r for r in s if r['subtype']=='quantity')
            r.update(importance='required',operator='between',expected={'type':'range','lower':3,'upper':4,'unit':'piece','lower_inclusive':True,'upper_inclusive':True})
            c['pending_mentions']=[p for p in c['pending_mentions'] if p['evidence']['text'] not in ('需要三、四片','质感柔软')]
        if t==12308:
            quote='要分别含有二十多段、三十多段、四十多段、七十多段'
            c['specifications']=[r for r in s if not (r['subtype']=='form_factor' and '要分别含有' in str(r['expected']))
                and not (r['subtype']=='components' and any(i['name']=='单品' for i in r['expected'].get('items',[])))]
            c['specifications'].append({'subtype':'composition','attribute':'member_groups','subject':{'kind':'primary_product','name':'主商品'},
                'importance':'required','operator':'matches_members','expected':{'type':'member_ranges','member_attribute':'segment_count','unit':'count',
                'groups':[{'lower':n+1,'upper':n+9,'count':1} for n in (20,30,40,70)]},'price_basis':None,
                'evidence':[{'text':quote},{'text':'包含4个单品'}],'review_status':'pending'})
            c['pending_mentions']=[p for p in c['pending_mentions'] if p['evidence']['text'] not in (quote,'二十多段','三十多段','四十多段','七十多段')]
        if t in (5598,11018):
            quote='不需要开刃' if t==5598 else '无需打磨刀刃'
            r=next(r for r in s if r['subtype']=='condition' and any(e['text']==quote for e in r['evidence']))
            r['expected']={'type':'text','value':'未开刃'};r['importance']='required'
        if t==7131:
            removed=[p for p in c['pending_mentions'] if '不要便宜太多' in p['evidence']['text']]
            for p in removed:
                phrase=p['evidence']['text'];pos=c['query'].index(phrase)
                notes.append({'task_id':t,'query':c['query'],'source':{'text':phrase,'start':pos,'end':pos+len(phrase)},
                    'treatment':'用户确认保留为未量化备注，暂不计分；不编价格下限，<=3000仍是硬要求','confirmed_by_user':True})
            c['pending_mentions']=[p for p in c['pending_mentions'] if p not in removed]
        if t==8188:c['pending_mentions']=[p for p in c['pending_mentions'] if p['evidence']['text']!='金属质感']
        changed={k:v for k,v in c.items() if k!='schema_version'}!={k:v for k,v in before.items() if k!='schema_version'}
        if changed:edits.append({'task_id':t,'query':c['query'],'before':before,'after':copy.deepcopy(c),'whole_contract_approved':False})
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
        if t in KNOWN:decisions.append({**identity,'policy_confirmed':True,'target_decision_applied':True,'program_valid':out['program_valid'],
            'scope':'本轮指定未决项已处理，不等于整题所有字段/商品证据已验收','whole_contract_approved':False})
        if len(results)%250==0:print(f'{len(results)}/1000',flush=True)
    for path,h in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
    lines(OUT/'candidate_drafts1000.jsonl',drafts);lines(OUT/'validated1000.jsonl',results);lines(OUT/'draft_edits.jsonl',edits)
    lines(OUT/'non_scoring_notes.jsonl',notes);lines(OUT/'decisions12.jsonl',decisions)
    blocked=[{'task_id':r['task_id'],'error':r.get('error')} for r in results if not r['program_valid']]
    summary={'total_tasks':1000,'program_valid':sum(r['program_valid'] for r in results),'blocked':blocked,
      'newly_blocked':[r['task_id'] for r in results if old[r['task_id']]['program_valid'] and not r['program_valid']],
      'known12_decisions_applied':len(decisions),'awaiting_user_decisions':0,'raw_edited_ids':[r['task_id'] for r in edits],
      'queues':dict(Counter(r['queue'] for r in results)),'source_sha256':hashes,'api_calls':0,'training_runs':0,'whole_contracts_approved':0}
    write(OUT/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False))
    if blocked:raise SystemExit('Replay has blockers; inspect summary before using outputs')
    b.write_schema()

if __name__=='__main__':main()
