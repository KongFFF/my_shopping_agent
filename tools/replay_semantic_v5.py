"""Offline r5 regression probe of saved r3/r4/r5 answers; never calls an API.

Only schema_version is adapted in memory. All other model fields are unchanged.
The source protocol is not claimed to have generated r5 answers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import semantic_contract_v2_r5 as backend
from extract_semantic_requirements import loads


def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def lines(path,values):path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in values),encoding='utf-8')


def replay(source, output):
    source,output=source.resolve(),output.resolve()
    if output==source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Use a separate derived directory outside source and its ancestors')
    if output.exists() and any(output.iterdir()):raise ValueError('Derived output must be empty')
    required=('tasks.jsonl','attempts.jsonl','report.json','run_manifest.json')
    hashes={name:sha((source/name).read_bytes()) for name in required}
    tasks=[loads(l) for l in (source/'tasks.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    records=[loads(l) for l in (source/'attempts.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    by={t['task_id']:t for t in tasks}
    if len(by)!=len(tasks):raise ValueError('Duplicate source tasks')
    results=[]; candidates={}; raw_candidates={}
    for line,r in enumerate(records,1):
        tid=r['task_id']
        if tid not in by:raise ValueError('Attempt outside task set')
        entry={'task_id':tid,'source_attempt_line':line,'original_accepted':bool(r.get('accepted')),
               'http_status':r.get('http_status'),'valid_r5':False}
        if r.get('http_status')==200:
            try:
                envelope=loads(r['raw_response']);choice=envelope['choices'][0];message=choice['message']
                if choice.get('finish_reason')!='stop' or message.get('tool_calls') or message.get('refusal'):
                    raise ValueError('Incomplete/refused/tool-call response cannot be reused')
                raw=loads(message['content']);version=raw.get('schema_version')
                if version not in ('shopping-semantic-requirements-v2-r3-draft','shopping-semantic-requirements-v2-r4-draft',backend.VERSION):
                    raise ValueError('Unsupported source version')
                raw_candidates[line]=raw
                probe=dict(raw);probe['schema_version']=backend.VERSION
                entry['version_adapter']={'from':version,'to':backend.VERSION,'only_version_field_changed':True}
                c,audit=backend.normalize(probe,by[tid])
            except Exception as exc:
                entry['error']=getattr(exc,'message',str(exc))
            else:
                findings=[f for a in audit if a['kind']=='semantic_review_findings' for f in a['findings']]
                entry.update(valid_r5=True,audit=audit,review_findings=findings)
                candidates[line]=c
        else:entry['error']=r.get('error','No HTTP 200 model reply')
        results.append(entry)
    cards=[];normalized=[];findings=[]
    for t in tasks:
        own=[x for x in results if x['task_id']==t['task_id']]
        accepted=next((x for x in own if x['original_accepted']),None)
        # Prefer the actual selected original answer, so a favorable historical
        # answer cannot silently replace one that was really accepted.
        selected=(accepted if accepted and accepted['valid_r5'] else
                  next((x for x in own if x['valid_r5']),None))
        selected_valid=selected is not None
        if selected is None:selected=next((x for x in reversed(own) if x['source_attempt_line'] in raw_candidates),None)
        index=selected['source_attempt_line'] if selected else None
        c=candidates.get(index)
        if c is not None:normalized.append(c)
        if selected_valid and selected.get('review_findings'):
            findings.append({'task_id':t['task_id'],'source_attempt_line':index,'findings':selected['review_findings']})
        cards.append({**t,'valid_r5':selected_valid,'original_selected_line':accepted['source_attempt_line'] if accepted else None,
                      'original_selected_passes_r5':bool(accepted and accepted['valid_r5']),
                      'selected_attempt_line':index,'selection_changed':bool(accepted and index!=accepted['source_attempt_line']),
                      'raw_selected':raw_candidates.get(index),'normalized':c,
                      'review_findings':selected.get('review_findings',[]) if selected_valid else [],
                      'errors':[x.get('error') for x in own if not x['valid_r5']]})
    output.mkdir(parents=True,exist_ok=True)
    lines(output/'attempt_revalidation.jsonl',results);lines(output/'review_inputs.jsonl',cards)
    lines(output/'responses.normalized.jsonl',normalized);lines(output/'review_findings.jsonl',findings)
    original_ids={r['task_id'] for r in records if r.get('accepted')}
    valid_ids={c['task_id'] for c in cards if c['valid_r5']}
    report={'scope':'Offline version-adapted regression only, not a new model evaluation or semantic accuracy measurement',
            'source':str(source),'target_version':backend.VERSION,'source_sha256':hashes,
            'total_tasks':len(tasks),'attempts_replayed':len(records),'original_program_valid':len(original_ids),
            'valid_r5':len(valid_ids),'original_selected_still_valid':sum(c['original_selected_passes_r5'] for c in cards),
            'newly_accepted_ids':sorted(valid_ids-original_ids),'newly_blocked_ids':sorted(original_ids-valid_ids),
            'selected_changed_ids':[c['task_id'] for c in cards if c['selection_changed']],
            'first_attempt_valid':sum(next((r['valid_r5'] for r in results if r['task_id']==t['task_id']),False) for t in tasks),
            'failed_task_ids':sorted(by.keys()-valid_ids),'review_tasks':len(findings),
            'review_findings':sum(len(x['findings']) for x in findings),'approved_for_reward':0,'api_calls':0,
            'code_sha256':{str(p):sha(p.read_bytes()) for p in (Path(__file__),Path(backend.__file__),Path(__file__).with_name('semantic_scope_v5.py'),Path(__file__).with_name('semantic_relations_v2_r3.py'),Path(__file__).with_name('semantic_coverage_v2_r3.py'),Path(__file__).with_name('semantic_policy_v4.py'))}}
    for name,digest in hashes.items():
        if sha((source/name).read_bytes())!=digest:raise ValueError('Source changed during replay')
    write(output/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(replay(args.source_run,args.output),ensure_ascii=False,indent=2))
