"""Source-grounded small correction set; original 1000 tasks stay intact.

Fixes four explicit approximate-money parses, seven gender omissions/two
gender scopes, two lighting scopes, and incorporates seven actually reviewed
integration contracts. No API calls or broad automatic semantic approvals.
"""
import copy
import hashlib
import json
from pathlib import Path
import semantic_contract_v2_r7_final12 as b
from build_train_reward_smoke_pack import main as make_smoke,OUT as SMOKE

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/semantic_training_revision_20261107'
GENDER={390:'女款',4538:'女款',11840:'女款',15673:'男款',18114:'男款',21586:'女款',22405:'男款',22931:'女款',23153:'女款'}


def raw_from_normalized(c):
    c=copy.deepcopy(c)
    for d in b.DIMENSIONS:
        for r in c[d]:
            r.pop('id',None);r['review_status']='pending'
            if d=='budget':r.update(operator='eq',expected={'type':'text','value':r['evidence'][0]['text']})
            for e in r['evidence']:e.pop('start',None);e.pop('end',None)
    for p in c['pending_mentions']:p['evidence'].pop('start',None);p['evidence'].pop('end',None)
    return c


def main():
    make_smoke();OUT.mkdir(exist_ok=True)
    source=ROOT/'analysis/semantic_final12_20261107/validated1000.jsonl';sha=hashlib.sha256(source.read_bytes()).hexdigest()
    records=[json.loads(s) for s in source.read_text(encoding='utf-8').splitlines()]
    source_raw=ROOT/'analysis/semantic_final12_20261107/candidate_drafts1000.jsonl'
    raw_records={r['task_id']:r for r in map(json.loads,source_raw.read_text(encoding='utf-8').splitlines())}
    signed={r['task_id']:r['normalized'] for r in map(json.loads,(SMOKE/'reviewed_contracts7.jsonl').read_text(encoding='utf-8').splitlines())}
    changes=[];result=[];raw_out=[];notes=[]
    for row in records:
        task=row['task_id'];row=copy.deepcopy(row);old=copy.deepcopy(row['normalized']);c=row['normalized']
        if task in signed:c=copy.deepcopy(signed[task]);row['normalized']=c
        if task in GENDER:
            gender=GENDER[task]
            existing=[r for r in c['functions'] if r['subtype']=='target_user' and r['expected'].get('value')==gender]
            if existing:
                for r in existing:r['subject']={'kind':'primary_product','name':'主商品'};r['review_status']='approved'
            else:
                recipient=next((r for r in c['functions'] if task==21586 and r['subtype']=='target_user' and r['expected'].get('value')=='妹妹'),None)
                if recipient:
                    notes.append({'task_id':task,'source':'送给妹妹','role':'recipient background; explicit female-product requirement retained'})
                    c['functions'].remove(recipient)
                pos=c['query'].index(gender)
                c['functions'].append({'id':'review_gender','subtype':'target_user','attribute':'target_user',
                    'subject':{'kind':'primary_product','name':'主商品'},'importance':'required','operator':'eq',
                    'expected':{'type':'text','value':gender},'price_basis':None,'review_status':'approved',
                    'evidence':[{'text':gender,'start':pos,'end':pos+len(gender)}]})
        if task==9425:
            r=next(r for r in c['specifications'] if r['id']=='r005');r['subject']={'kind':'component','name':'光源'}
            text='LED光源类型';pos=c['query'].index(text)
            r['evidence'].append({'text':text,'start':pos,'end':pos+len(text)});r['review_status']='approved'
        for r in c['budget']:
            if r['expected']['type']=='text':continue
            for e in r['evidence']:
                try:op,value,rule=b.budget_for_role(b.budget_source_clause(c['query'],e['text']),r['subtype'])
                except ValueError:continue
                if (r['operator'],r['expected'])!=(op,value):
                    changes.append({'task_id':task,'kind':'budget_approximate_marker','id':r['id'],'before':copy.deepcopy(r['expected']),'after':value,'source':e['text']})
                    r.update(operator=op,expected=value)
                break
        b.validate(c,source_query=row['query'])
        row['whole_contract_approved']=task in signed;row['reward_approved']=task in signed
        if task in signed:b.validate(c,source_query=row['query'],require_reviewed=True)
        if c!=old:
            changes.append({'task_id':task,'kind':'source_review_revision','before_contract':old,'after_contract':c})
            rr=copy.deepcopy(raw_records[task]);rr['raw_draft']=raw_from_normalized(c)
            # Check that the corrected raw form is also valid for future replay.
            b.normalize(rr['raw_draft'],{'task_id':task,'query':row['query']})
        else:rr=raw_records[task]
        result.append(row);raw_out.append(rr)
    assert len(result)==1000 and [r['task_id'] for r in result]==[r['task_id'] for r in records]
    assert all(a['query']==z['query'] for a,z in zip(result,records))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==sha
    for name,data in [('validated1000.jsonl',result),('candidate_drafts1000.jsonl',raw_out),('edits.jsonl',changes),('non_scoring_context.jsonl',notes)]:
        (OUT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data),encoding='utf-8')
    summary={'tasks':1000,'program_valid':1000,'whole_source_reviewed':sum(r['whole_contract_approved'] for r in result),
             'changed_tasks':sorted({r['task_id'] for r in changes}),'original_file_unchanged':True,'original_sha256':sha,
             'budget_changes':[r for r in changes if r['kind']=='budget_approximate_marker'],
             'actor_queries_changed':0,'tasks_removed':0,'api_calls':0,'training_runs':0}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
