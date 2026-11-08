"""Deterministic closeout of known draft issues plus full offline replay."""
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
import semantic_contract_v2_r7_closeout as b

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'analysis/semantic_targeted_review106_20261107'
OUT=ROOT/'analysis/semantic_closeout_20261107'

def rows(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
def lines(path,records):path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def requirement(subtype,attribute,value,evidence,*,subject=None,importance='required',operator='eq'):
    return {'subtype':subtype,'attribute':attribute,'subject':subject or {'kind':'primary_product','name':'主商品'},
            'importance':importance,'operator':operator,'expected':value,'price_basis':None,
            'evidence':[{'text':x} for x in ([evidence] if isinstance(evidence,str) else evidence)],'review_status':'pending'}
def text(st,attr,value,evidence,**kw):return requirement(st,attr,{'type':'text','value':value},evidence,**kw)
def drop_pending(c,phrases):c['pending_mentions']=[p for p in c['pending_mentions'] if p['evidence']['text'] not in phrases]

def main():
    OUT.mkdir(exist_ok=True)
    paths=[SOURCE/'candidate_drafts1000.jsonl',SOURCE/'validated1000.jsonl']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    previous={r['task_id']:r for r in rows(paths[1])}
    drafts=[];normalized=[];edits=[];changes=[]
    for row in rows(paths[0]):
        c=copy.deepcopy(row['raw_draft']);before=copy.deepcopy(c);t=row['task_id'];s=c['specifications']
        c['schema_version']=b.VERSION
        if t==20520:
            # Do not infer a quantity for each listed component; the explicit
            # 三件套 label independently retains the total package structure.
            for r in s:
                if r['expected']['type']=='components':
                    for item in r['expected']['items']:item.update(quantity=None,unit=None)
        if t in (18823,19437):
            # Revoke an unreviewed mass scope; do not guess per-pack or total.
            for r in s:
                if r['subtype']=='net_content':r['importance']='uncertain'
            quote='两袋五百克' if t==18823 else '需要120g，共10包'
            c['pending_mentions'].append({'evidence':{'text':quote},'reason':'ambiguous_meaning','importance':'required'})
        if t==5615:
            quote='NVME和NGFF标准理论传输速率分别可达10和6千兆比特每秒'
            for name,rate in [('NVME',10),('NGFF',6)]:
                c['functions'].append(text('function','function',f'标准理论传输速率可达{rate}千兆比特每秒',quote,
                    subject={'kind':'compatible_target','name':name}))
            drop_pending(c,{quote})
        if t==10591:
            r=next(r for r in s if r['subtype']=='form_factor' and '双面' in str(r['expected']))
            r['subject']={'kind':'component','name':'729普及套'};r['expected']={'type':'text','value':'双面配置'}
            drop_pending(c,{'配备双面729普及套'})
        if t==12321:
            r=next(r for r in s if r['subtype']=='customization' and '多退少补' in str(r['expected']))
            r.update(subtype='transaction_terms',attribute='settlement_adjustment',expected={'type':'text','value':'按克计价多退少补'})
        if t==14635:
            s.append(text('transaction_terms','return_shipping_reimbursement','退货可报销运费','退货可报销运费'))
            drop_pending(c,{'退货可报销运费'})
        if t==16184:
            for attr,quote in [('billing_method','按克算价'),('billing_method','以实际重量收费'),('settlement_adjustment','要支持超额部分返还，差额部分补齐')]:
                s.append(text('transaction_terms',attr,quote,quote))
            # The two billing clauses describe one combined settlement basis;
            # merge to avoid two separate rewards for the same billing fact.
            billing=[r for r in s if r['subtype']=='transaction_terms' and r['attribute']=='billing_method']
            billing[0]['expected']={'type':'text','value':'按实际重量按克计价'}
            billing[0]['evidence']+=billing[1]['evidence'];s.remove(billing[1])
            drop_pending(c,{'按克算价','以实际重量收费','要支持超额部分返还，差额部分补齐'})
        changed={k:v for k,v in c.items() if k!='schema_version'}!={k:v for k,v in before.items() if k!='schema_version'}
        if changed:edits.append({'task_id':t,'query':row['query'],'before':before,'after':copy.deepcopy(c),'whole_contract_approved':False})
        identity={k:row[k] for k in ('task_id','query','source_attempt_line')}
        drafts.append({**identity,'raw_draft':c,'changed_in_closeout':changed,'reward_approved':False})
        out={**identity,'program_valid':False,'whole_contract_approved':False,'reward_approved':False}
        try:n,a=b.normalize(c,{'task_id':t,'query':row['query']})
        except Exception as error:out.update(error=getattr(error,'message',str(error)),queue='program_blocked')
        else:
            findings=[f for e in a if e['kind']=='semantic_review_findings' for f in e['findings']]
            pending=n['pending_mentions'] or any(r['importance']=='uncertain' for d in b.DIMENSIONS for r in n[d])
            out.update(program_valid=True,normalized=n,audit=a,review_findings=findings,queue='semantic_review_pending' if pending or findings else 'no_static_signals')
            for _,e in b._evidences(n):assert row['query'][e['start']:e['end']]==e['text']
            prior=previous[t]
            strip_ids=lambda rs:[{k:v for k,v in r.items() if k!='id'} for r in rs]
            if prior['program_valid'] and strip_ids(prior['normalized']['budget'])!=strip_ids(n['budget']):
                changes.append({'task_id':t,'query':row['query'],'before':prior['normalized']['budget'],'after':n['budget']})
        normalized.append(out)
        if len(normalized)%250==0:print(f'{len(normalized)}/1000',flush=True)
    assert len(drafts)==len(normalized)==1000
    for path,h in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
    result={r['task_id']:r for r in normalized}
    for t,amount in {373:19,8885:49,7920:49,21469:19,21733:49}.items():
        assert result[t]['program_valid'],result[t]
        r=result[t]['normalized']['budget'][0];assert r['expected']['value']==amount and r['operator']=='at_most'
    assert result[20520]['program_valid'],result[20520]
    newly=[r['task_id'] for r in normalized if previous[r['task_id']]['program_valid'] and not r['program_valid']]
    assert not newly,newly
    lines(OUT/'candidate_drafts1000.jsonl',drafts);lines(OUT/'validated1000.jsonl',normalized)
    lines(OUT/'draft_edits.jsonl',edits);lines(OUT/'budget_changes.jsonl',changes)
    lines(OUT/'remaining_blockers.jsonl',[r for r in normalized if not r['program_valid']])
    summary={'total_tasks':1000,'program_valid':sum(r['program_valid'] for r in normalized),'newly_blocked':newly,
             'raw_edited_ids':[r['task_id'] for r in edits],'budget_changed_ids':[r['task_id'] for r in changes],
             'queues':dict(Counter(r['queue'] for r in normalized)),'source_sha256':hashes,
             'api_calls':0,'training_runs':0,'whole_contracts_approved':0}
    write(OUT/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False))
    b.write_schema()

if __name__=='__main__':main()
