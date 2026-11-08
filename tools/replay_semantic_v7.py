"""Offline, fixed-answer r7 replay. No API client, retries or answer reselection.

Compare original selected answers and the existing reviewed drafts separately.
Only a confirmed decision sidecar may add a reviewed exclusion requirement.
"""
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
import semantic_contract_v2_r7 as b


def sha(value): return hashlib.sha256(value).hexdigest()
def object_sha(value): return sha(json.dumps(value,sort_keys=True,ensure_ascii=False).encode('utf-8'))
def rows(path): return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
def write(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def lines(path,values): path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in values),encoding='utf-8')


def apply_decisions(raw, decisions):
    c=copy.deepcopy(raw); audit=[]
    for decision in decisions:
        if decision['task_id']!=c['task_id']: continue
        if decision['status']!='confirmed_by_user' or decision['constraint']['relation']!='exclude_component':
            raise ValueError('Unsupported/unconfirmed decision cannot be auto-applied')
        e=decision['evidence']; q=c['query']; name=decision['constraint']['object']
        if not 0<=e['start']<e['end']<=len(q) or q[e['start']:e['end']]!=e['text']:
            raise ValueError('Decision evidence differs from task source')
        # An explicit decision is an annotation, not an extraction heuristic.
        # Refuse to silently delete a still-positive input assertion.
        if any(r['expected']['type']=='components' and r['operator'] in ('eq','contains_all')
               and any(i['name']==name for i in r['expected']['items']) for r in c['specifications']):
            raise ValueError('Resolve contradictory positive component in the reviewed draft first')
        added={'subtype':'components','attribute':'included_components','subject':{'kind':'primary_product','name':'主商品'},
               'importance':decision['constraint']['importance'],'operator':'excludes',
               'expected':{'type':'components','items':[{'name':name,'quantity':None,'unit':None}]},
               'price_basis':None,'evidence':[{'text':e['text']}],'review_status':'pending'}
        c['specifications'].append(added)
        resolved=[p for p in c['pending_mentions'] if p['reason']=='unsupported_type'
                  and isinstance(p['evidence'],dict) and p['evidence']['text']==e['text']]
        c['pending_mentions']=[p for p in c['pending_mentions'] if p not in resolved]
        audit.append({'kind':'explicit_user_decision_adapter','decision':decision,'added':added,
                      'resolved_representation_pending':resolved,'target_schema_support':True,
                      'reward_runtime_enabled':False,'whole_contract_approved':False})
    return c,audit


def flags_and_queue(normalized,audit):
    findings=[f for a in audit if a['kind']=='semantic_review_findings' for f in a['findings']]
    pending=normalized['pending_mentions']
    uncertain=[r['id'] for d in b.DIMENSIONS for r in normalized[d] if r['importance']=='uncertain']
    return findings,('semantic_review_pending' if findings or pending or uncertain else 'no_static_signals')


def replay(original_cards, reviewed_drafts, decisions_path, output):
    paths=[original_cards.resolve(),reviewed_drafts.resolve(),decisions_path.resolve()]
    output=output.resolve()
    if any(output==p or output.is_relative_to(p.parent) for p in paths):
        raise ValueError('Keep derived output outside input directories')
    if output.exists() and any(output.iterdir()): raise ValueError('Output must be absent or empty')
    hashes={str(p):sha(p.read_bytes()) for p in paths}
    old=rows(paths[0]);drafts=rows(paths[1]);decision_doc=json.loads(paths[2].read_text(encoding='utf-8'))
    old_by={r['task_id']:r for r in old};draft_by={r['task_id']:r for r in drafts}
    if len(old)!=len(old_by) or len(drafts)!=len(draft_by) or old_by.keys()!=draft_by.keys():
        raise ValueError('Task IDs differ or are duplicated')
    if len(old)!=1000: raise ValueError('This release replay requires the fixed original 1000 tasks')
    output.mkdir(parents=True,exist_ok=True);reports={};tracks={}
    for track in ('original_selected','reviewed_drafts'):
        result=[];baseline={};new_valid=set();baseline_valid=set()
        for index,card in enumerate(old,1):
            tid=card['task_id'];draft=draft_by[tid]
            if card['query']!=draft['query'] or card['selected_attempt_line']!=draft['source_attempt_line']:
                raise ValueError('Answer reselection or task query change is forbidden')
            raw=card['raw_selected'] if track=='original_selected' else draft['raw_draft']
            before_hash=object_sha(raw);probe=copy.deepcopy(raw);source_version=probe['schema_version']
            if source_version not in {f'shopping-semantic-requirements-v2-r{i}-draft' for i in range(3,8)}:
                raise ValueError('Unsupported source schema version')
            probe['schema_version']=b.VERSION;decision_audit=[]
            valid_before=card['valid_r6'] if track=='original_selected' else draft['program_valid']
            baseline[tid]=valid_before
            if valid_before: baseline_valid.add(tid)
            entry={'task_id':tid,'query':card['query'],'source_attempt_line':card['selected_attempt_line'],
                   'input_raw_sha256':before_hash,'input_version':source_version,'target_version':b.VERSION,
                   'baseline_valid_r6':valid_before,'program_valid':False,'queue':'program_blocked',
                   'whole_contract_approved':False,'reward_approved':False}
            try:
                if track=='reviewed_drafts':probe,decision_audit=apply_decisions(probe,decision_doc['confirmed'])
                normalized,audit=b.normalize(probe,{'task_id':tid,'query':card['query']})
            except Exception as exc:
                entry['error']=getattr(exc,'message',str(exc))
            else:
                findings,queue=flags_and_queue(normalized,audit)
                entry.update(program_valid=True,normalized=normalized,audit=decision_audit+audit,
                             review_findings=findings,queue=queue)
                new_valid.add(tid)
                for _,e in b._evidences(normalized):
                    if card['query'][e['start']:e['end']]!=e['text']: raise ValueError('Evidence invariant failed')
            assert object_sha(raw)==before_hash,'Normalizer mutated a source input'
            entry['validity_change']='newly_passed' if not valid_before and entry['program_valid'] else 'newly_blocked' if valid_before and not entry['program_valid'] else 'unchanged'
            entry['decision_adapters']=decision_audit
            result.append(entry)
            if index%250==0: print(f'{track}: {index}/1000 replayed',flush=True)
        lines(output/f'{track}.jsonl',result)
        lines(output/f'{track}.queues.jsonl',[{k:r[k] for k in ('task_id','source_attempt_line','queue','program_valid','validity_change','reward_approved')}
                                             |({'error':r['error']} if 'error' in r else {}) for r in result])
        reports[track]={'total_tasks':len(result),'baseline_program_valid':len(baseline_valid),
                        'r7_program_valid':len(new_valid),'r7_program_blocked':len(result)-len(new_valid),
                        'newly_passed_ids':sorted(new_valid-baseline_valid),'newly_blocked_ids':sorted(baseline_valid-new_valid),
                        'queues':dict(Counter(r['queue'] for r in result)),
                        'review_finding_counts':dict(Counter(f['code'] for r in result for f in r.get('review_findings',[]))),
                        'normalization_rules':dict(Counter(rule for r in result for a in r.get('audit',[]) if a['kind']=='r7_representation_adapter' for rule in a['rules'])),
                        'explicit_decisions_applied':sum(len(r['decision_adapters']) for r in result)}
        tracks[track]=result
    for p in paths:
        if sha(p.read_bytes())!=hashes[str(p)]:raise ValueError('Input source file changed')
    source_files=[Path(b.__file__),Path(__file__),*[Path(__file__).with_name(name) for name in
        ('semantic_coverage_v7.py','semantic_scope_v7.py','semantic_repairs_v7.py','semantic_relations_v7.py',
         'semantic_repairs_v6.py','semantic_coverage_v6.py','semantic_contract_v2_r6.py',
         'semantic_scope_v6.py','semantic_scope_v5.py','semantic_coverage_v2_r3.py',
         'semantic_contract_v1.py','semantic_policy_v4.py')]]
    report={'scope':'Fixed-answer offline replay, not extraction accuracy or model performance',
            'target_version':b.VERSION,'input_sha256':hashes,'tracks':reports,
            'api_calls':0,'answer_reselection':False,'whole_contracts_approved':0,'reward_approved':0,
            'pending_user_policy_decisions':decision_doc.get('pending',[]),
            'checks':{'input_files_unchanged':True,'task_id_set_preserved':True,'queries_unchanged':True,
                      'same_selected_answer_lines':True,'source_evidence_coordinates_exact':True,'no_automatic_reward_approval':True},
            'code_sha256':{str(p):sha(p.read_bytes()) for p in source_files}}
    write(output/'report.json',report)
    return report,tracks


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-cards',type=Path,required=True);p.add_argument('--reviewed-drafts',type=Path,required=True)
    p.add_argument('--decisions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();report,_=replay(args.original_cards,args.reviewed_drafts,args.decisions,args.output)
    print(json.dumps(report,ensure_ascii=False,indent=2))
