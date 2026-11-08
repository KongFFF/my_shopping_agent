"""Offline, source-bound closure of the residual queue. No API or training."""
from __future__ import annotations
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
import migrate_training_semantics as m
import semantic_remaining_policy as policy

ROOT=m.ROOT
SOURCE=ROOT/'analysis/semantic_policy_migration_20261107/validated1000.jsonl'
OUT=ROOT/'analysis/semantic_remaining_closeout_20261108'
LEDGER=ROOT/'experiments/semantic_remaining_closeout_v1/source_review.jsonl'

def pending_key(p):return (p['reason'],p['evidence']['text'],p['evidence']['start'],p['evidence']['end'])

def apply_review(c,decision):
    if hashlib.sha256(c['query'].encode()).hexdigest()!=decision['query_sha256']:
        raise ValueError('Source-bound review query changed')
    for op in decision.get('changes',[]):
        if op['op']=='add':
            r=copy.deepcopy(op['requirement']);d=op['dimension']
            r['evidence']=[m.evidence(c['query'],t) for t in r.pop('source_texts')]
            m.add_unique(c,d,r);continue
        hits=[(d,r) for d,r in m.allreq(c) if r['id']==op['id']]
        if len(hits)!=1:raise ValueError('Review predicate ID missing or ambiguous')
        d,r=hits[0]
        if r!=op['before']:raise ValueError('Review predicate changed since source inspection')
        if op['op']=='remove':c[d].remove(r);continue
        r.update(copy.deepcopy(op['updates']))
        if 'source_texts' in op:r['evidence']=[m.evidence(c['query'],t) for t in op['source_texts']]
        if op.get('dimension') and op['dimension']!=d:c[d].remove(r);c[op['dimension']].append(r)
    resolved={(p['reason'],p['text']) for p in decision.get('resolves',[])}
    c['pending_mentions']=[p for p in c['pending_mentions'] if (p['reason'],p['evidence']['text']) not in resolved]

def close(row,decisions):
    result=copy.deepcopy(row);c=result['normalized'];audit=[];failures=[]
    if row.get('whole_contract_approved'):
        m.b.validate(c,source_query=row['query'],require_reviewed=True)
        return result,audit,failures
    for decision in decisions:
        if decision['task_id']!=row['task_id']:continue
        before=copy.deepcopy(c)
        try:
            apply_review(c,decision)
            m.b.validate(c,source_query=row['query'])
        except ValueError as exc:
            c.clear();c.update(before)
            failures.append({'task_id':row['task_id'],'review_id':decision['id'],'error':str(exc)})
        else:audit.append({'task_id':row['task_id'],'kind':'source_bound_review','review_id':decision['id'],'before':before,'after':copy.deepcopy(c)})
    # Numeric uncertainty alone does not imply uncertain requirement strength.
    # Actual weak/hope/permission clauses still need a source-bound decision.
    for rid in [r['id'] for _,r in m.allreq(c)]:
        r=next(r for _,r in m.allreq(c) if r['id']==rid)
        if r['operator'] in ('approximately','matches_label'):continue
        word=policy.colloquial_target(r,m.b.chinese_number)
        v=word[1] if word else policy.approximate_target(r,m.b.chinese_number)
        if v is None:continue
        operator=word[0] if word else 'approximately'
        importance=r['importance'] if r['importance']!='uncertain' else m.strength_for(c,r)
        if importance is None:continue
        before=copy.deepcopy(c)
        r.update(operator=operator,expected=v,review_status='approved',importance=importance)
        try:m.b.validate(c,source_query=row['query'])
        except ValueError as exc:
            c.clear();c.update(before)
            failures.append({'task_id':row['task_id'],'requirement_id':rid,'kind':'numeric_candidate_held','error':str(exc)})
        else:audit.append({'task_id':row['task_id'],'kind':'confirmed_approximate_target','requirement_id':rid,'target':v})
    return finalize(result,audit,failures)

def finalize(result,audit,failures):
    c=result['normalized'];kept=[]
    for p in c['pending_mentions']:
        if p in kept:continue
        rs=[r for _,r in m.allreq(c) if any(m.overlap(e,p['evidence']) for e in r['evidence'])]
        numeric=[r for r in rs if policy.family_for(r)]
        resolved=(p['reason']=='undefined_tolerance' and bool(numeric)
                  and all((r['operator']=='approximately' or policy.colloquial_target(r,m.b.chinese_number)==(r['operator'],r['expected'])) and r['review_status']=='approved' for r in numeric))
        # A parsed budget already follows the separately confirmed monetary policy.
        budget=[r for r in rs if r in c['budget']]
        if p['reason']=='undefined_tolerance' and budget and not numeric:
            resolved=all(r['expected']['type'] in ('number','range') and r['importance']!='uncertain' for r in budget)
        if resolved:audit.append({'task_id':result['task_id'],'kind':'resolved_pending_with_retained_requirement','pending':p})
        else:kept.append(p)
    c['pending_mentions']=kept
    m.b.validate(c,source_query=result['query'])
    _,findings=m.b.split_flags(c,m.b.lint(c));result['review_findings']=findings
    result['remaining_closeout_audit']=[{k:v for k,v in a.items() if k not in ('before','after')} for a in audit]
    result['whole_contract_approved']=False;result['reward_approved']=False
    result['program_valid']=True
    return result,audit,failures

def main():
    import semantic_reviewed_scopes
    semantic_reviewed_scopes.CAPTURE_RAW_DIAGNOSTICS=True
    OUT.mkdir(parents=True,exist_ok=True)
    before_sha=hashlib.sha256(SOURCE.read_bytes()).hexdigest();source=m.rows(SOURCE)
    decisions=m.rows(LEDGER) if LEDGER.exists() else []
    results=[];audit=[];failures=[]
    for r in source:
        try:row,a,f=close(r,decisions)
        except ValueError as exc:
            raise ValueError(f"task={r['task_id']}: {exc}") from exc
        results.append(row);audit.extend(a);failures.extend(f)
    assert [(r['task_id'],r['query']) for r in source]==[(r['task_id'],r['query']) for r in results]
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==before_sha
    queue=[{'task_id':r['task_id'],'query':r['query'],'pending':p} for r in results for p in r['normalized']['pending_mentions']]
    summary={'tasks':len(results),'program_valid':len(results),'source_sha256':before_sha,'source_unchanged':True,
             'actor_queries_changed':0,'tasks_removed':0,'before_pending_tasks':sum(bool(r['normalized']['pending_mentions']) for r in source),
             'before_pending_records':sum(len(r['normalized']['pending_mentions']) for r in source),
             'remaining_pending_tasks':sum(bool(r['normalized']['pending_mentions']) for r in results),'remaining_pending_records':len(queue),
             'remaining_reasons':dict(Counter(x['pending']['reason'] for x in queue)),
             'changed_tasks':sum(a['normalized']!=z['normalized'] for a,z in zip(source,results)),
             'held_candidates':len(failures),'source_review_records':len(decisions),'api_calls':0,'training_runs':0,'formal_training_ready':False}
    for name,rows in [('validated1000.jsonl',results),('edits.jsonl',audit),('remaining_review.jsonl',queue),('held_candidates.jsonl',failures)]:m.jl(OUT/name,rows)
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
