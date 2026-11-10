"""CPU/API experiment for source-checked purchase facts, never a training entry."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import threading
import time

import purchase_judge_benchmark as old
import purchase_fact_proofs as proof

ROOT=old.ROOT
OUTPUT=ROOT/'analysis/purchase_facts_v2_20261110'
PROMPT=ROOT/'experiments/purchase_facts_v2/system_prompt.txt'


def additional_reference(cases,previous):
    used={r['case_id'] for r in previous};result=[]
    def add(task,contains,passes,unknown=(),fails=None):
        candidates=[c for c in cases.values() if c['first_cohort']=='holdout_v5' and c['case_id'] not in used and c['task_id']==task
                    and (contains is None or contains in old.canonical(c['selected_options']))]
        if len(candidates)!=1:raise ValueError('Ambiguous reference selector: '+str((task,contains)))
        c=candidates[0];rs={r['id']:r for r in c['request']['requirements']}
        for status,entries in [('pass',passes),('unknown',{rid:[] for rid in unknown}),('fail',fails or {})]:
            for rid,pointers in entries.items():
                if isinstance(pointers,str):pointers=[pointers]
                pointers=[next(p for p in c['request']['evidence_table'] if p.startswith('/customization_options/'+x[1:]+'/')) if x.startswith('@') else x for x in pointers]
                result.append({'case_id':c['case_id'],'task_id':task,'requirement_id':rid,'status':status,'requirement':rs[rid],
                    'evidence':[{'pointer':p,'value':c['request']['evidence_table'][p]} for p in pointers],
                    'reason':'source-reviewed before v2 output; unknown retains missing/ambiguous evidence',
                    'reviewed_before_model':True,'reviewed_by':'assistant','scope':'additional source audit on old saved purchases; not unseen-product blind test'})
    add(23082,None,{'r005':'/title','r003':'@颜色分类','r006':'/attribute/6'},('r007',))
    add(9413,None,{'r002':'/title','r001':'/attribute/3','r005':'/attribute/1','r006':'/attribute/2'},('r008',))
    add(7764,None,{'r003':'@颜色分类','r005':'@尺码','r006':'/attribute/5','r010':'@颜色分类'},('r009','r001'))
    add(9408,'200升',{'r005':'@颜色分类'},('r001',))
    add(9408,'{}',{},('r001',),{'r005':'/title'})
    add(20307,None,{'r002':'/title','r004':'@尺码','r005':'@颜色分类','migration001':'/attribute/0'},('migration002',))
    add(7681,None,{'r008':'@颜色分类','r009':'@颜色分类','r010':'/attribute/2'},('r006','r007'))
    add(1167,'喜上眉梢',{'r006':'@颜色分类','r004':'/attribute/0','r001':'/title'},('r003','r008'))
    add(1167,'{}',{'r006':'/attribute/0'},('r004','r003','r008'))
    add(3486,None,{'r003':'/title','r002':'/attribute/2','r004':'/attribute/1','r005':'@颜色分类','r007':'@颜色分类','r008':'/attribute/0'})
    add(18532,None,{'r003':'@颜色分类','r004':'/attribute/0','r007':'/attribute/2'})
    add(21475,None,{'r006':'@鞋码','r007':'@颜色分类','r005':'/attribute/3','r004':'/attribute/2'},('migration002','migration001'))
    add(15719,None,{'r002':'/attribute/0','r003':'/attribute/1','r005':'/attribute/3','r008':['/attribute/4','/attribute/5']},('r001',))
    add(7397,None,{'r005':'/attribute/4','r007':'@颜色分类','r003':['/attribute/0','/attribute/1']},('r008',))
    add(10383,None,{'r002':'@香味','r004':'/attribute/1','r001':'/title'},('r003',))
    add(20567,None,{'r003':'/attribute/1','r004':'/attribute/2','r005':'/attribute/3','r006':'@颜色分类','r007':'@套餐类型','r001':'/attribute/0'})
    return result


def prepare(folder):
    folder.mkdir(parents=True,exist_ok=True)
    cases,events,source,_=old.load_dataset(old.DEFAULT)
    backend=old.TrainingEvidenceBackend(old.MANIFEST)
    prompt=PROMPT.read_text(encoding='utf-8')
    requests={};mapping={}
    for c in cases.values():
        if c['first_cohort']!='holdout_v5':continue
        product=backend.products[c['product_asin']]
        public={'selected_options':c['selected_options'],'evidence_table':c['request']['evidence_table'],'output_schema':proof.schema()}
        key=old.digest({'product':c['product_sha256'],'selected_options':c['selected_options'],'prompt':old.digest(prompt),'schema':old.digest(proof.schema())})
        mapping[c['case_id']]=key
        requests[key]={'request_id':key,'public':public,'checker':{**public,'_checks':{'product':product}},'product_asin':c['product_asin']}
    original=old.rows(old.DEFAULT/'reference_atoms.jsonl')
    additional=additional_reference(cases,original)
    policy={'source_manifest_sha256':old.digest(source),'compiler_sha256':old.sha256(Path(proof.__file__)),
            'runner_sha256':old.sha256(Path(__file__)),'prompt_sha256':old.digest(prompt),'schema_sha256':old.digest(proof.schema()),
            'reference_original_sha256':old.digest(original),'reference_additional_sha256':old.digest(additional),
            'requests_sha256':old.digest(list(requests.values())),'mapping':mapping,
            'objective':'same_frozen_evidence_reward; no_training_release','schema_only_not_semantic_approval':True,
            'known_cases_are_regression_not_blind':True,'llm_sees_requirements':False,'cache_identity_excludes_query':True}
    lock=folder/'protocol.json'
    if lock.exists() and json.loads(lock.read_text(encoding='utf-8'))!=policy:
        raise ValueError('Experiment inputs/code changed; use a new output directory')
    old.write(lock,policy);old.lines(folder/'requests.jsonl',requests.values())
    old.lines(folder/'reference_original.jsonl',original);old.lines(folder/'reference_additional.jsonl',additional)
    old.write(folder/'output.schema.json',proof.schema());(folder/'system_prompt.txt').write_text(prompt,encoding='utf-8')
    summary={'source_purchase_cases':len(mapping),'unique_product_sku_requests':len(requests),'original_reference_atoms':len(original),
             'additional_reference_atoms':len(additional),'additional_reference_statuses':dict(Counter(r['status'] for r in additional)),
             'api_calls':0,'source_review_before_v2_model':True,'same_saved_purchases_not_new_blind_distribution':True}
    old.write(folder/'preparation_summary.json',summary);print(old.canonical(summary),flush=True)
    report(folder)


def load(folder):
    policy=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    requests={r['request_id']:r for r in old.rows(folder/'requests.jsonl')}
    prompt=(folder/'system_prompt.txt').read_text(encoding='utf-8')
    if old.sha256(Path(proof.__file__))!=policy['compiler_sha256'] or old.sha256(Path(__file__))!=policy['runner_sha256'] or old.digest(prompt)!=policy['prompt_sha256'] or old.digest(list(requests.values()))!=policy['requests_sha256']:
        raise ValueError('Frozen experiment code/data changed')
    for name in ('original','additional'):
        if old.digest(old.rows(folder/f'reference_{name}.jsonl'))!=policy[f'reference_{name}_sha256']:
            raise ValueError('Reference changed after freeze')
    return policy,requests,prompt


def run(folder,config,limit=None,concurrency=4,timeout=60):
    policy,requests,prompt=load(folder)
    url=old.endpoint(config['LLM_BASE_URL'],allow_school_http=True)
    if not config['LLM_API_KEY'] or not config['LLM_MODEL']:raise ValueError('Missing API configuration')
    settings={'model':config['LLM_MODEL'],'endpoint':url,'concurrency':concurrency,'timeout':timeout,'max_tokens':4096,'thinking':'disabled','protocol_sha256':old.digest(policy)}
    path=folder/'api_config.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=settings:raise ValueError('API configuration changed')
    old.write(path,settings);namespace=old.digest(settings)
    done={r['request_id'] for r in old.rows(folder/'answers.jsonl') if r['namespace']==namespace}
    pending=[r for key,r in requests.items() if key not in done]
    if limit is not None:pending=pending[:limit]
    lock=folder/'.run.lock'
    with lock.open('x',encoding='utf-8') as h:h.write(str(__import__('os').getpid()))
    mutex=threading.Lock();stop=threading.Event()
    def extract(r):
        if stop.is_set():return
        start=time.perf_counter()
        for attempt in (1,2):
            log={'request_id':r['request_id'],'attempt':attempt,'namespace':namespace};t=time.perf_counter()
            try:
                body={'model':config['LLM_MODEL'],'messages':[{'role':'system','content':prompt},{'role':'user','content':old.canonical(r['public'])}],
                      'temperature':0,'stream':False,'thinking':{'type':'disabled'},'max_tokens':4096,'response_format':{'type':'json_object'}}
                _,raw=old.post(url,config['LLM_API_KEY'],body,timeout)
                envelope=old.loads(raw);log['usage']=envelope.get('usage',{});choice=envelope['choices'][0]
                if choice.get('finish_reason')!='stop':raise ValueError('incomplete_model_output')
                answer=old.loads(choice['message']['content']);log['model_answer']=answer
                compiled=proof.compile_answer(answer,r['checker'])
                result={'request_id':r['request_id'],'namespace':namespace,'answer':answer,'compiled':compiled,
                        'seconds_including_retries':time.perf_counter()-start,'usage':log['usage']}
                log.update(valid=True,accepted_facts=len(compiled['facts']),rejected_facts=len(compiled['rejected']))
                with mutex:
                    old.append(folder/'answers.jsonl',result)
                    log['seconds']=time.perf_counter()-t;old.append(folder/'attempts.jsonl',log)
                    print(f"product={r['product_asin']} VALID facts={len(compiled['facts'])} rejected={len(compiled['rejected'])}",flush=True)
                return
            except old.HTTPFailure as error:
                log.update(valid=False,error=f'HTTP {error.status}')
                if error.status in (401,402,403,404):stop.set()
            except Exception as error:
                log.update(valid=False,error=type(error).__name__+': '+str(error)[:160])
            log['seconds']=time.perf_counter()-t
            log['error']=log['error'].replace(config['LLM_API_KEY'],'[redacted]')
            with mutex:
                old.append(folder/'attempts.jsonl',log);print('FAILED '+log['error'],flush=True)
            if stop.is_set():return
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for future in as_completed([pool.submit(extract,r) for r in pending]):future.result()
    finally:lock.unlink()
    return report(folder)


def accuracy(items,field):
    pass_refs=[r for r in items if r['status']=='pass'];nonpass=[r for r in items if r['status']!='pass']
    return {'atoms':len(items),'agreement':sum(r[field]==r['status'] for r in items)/len(items) if items else None,
            'true_passes':sum(r[field]=='pass' for r in pass_refs),'reference_passes':len(pass_refs),
            'pass_recall':sum(r[field]=='pass' for r in pass_refs)/len(pass_refs) if pass_refs else None,
            'false_pass':sum(r[field]=='pass' for r in nonpass),'reference_nonpass':len(nonpass),
            'confusion':dict(Counter(r['status']+'->'+r[field] for r in items))}


def report(folder):
    policy,requests,_=load(folder)
    cases,events,_,_=old.load_dataset(old.DEFAULT)
    settings=json.loads((folder/'api_config.json').read_text(encoding='utf-8')) if (folder/'api_config.json').exists() else None
    answers={}
    for a in old.rows(folder/'answers.jsonl'):
        if settings is None or a['namespace']!=old.digest(settings):raise ValueError('Mixed model settings')
        answers[a['request_id']]=proof.compile_answer(a['answer'],requests[a['request_id']]['checker'])
    scores={};records=[]
    v1={r['case_id']:r for r in old.rows(old.DEFAULT.parent/'school_flash_v1/purchase_rewards.jsonl')}
    for key,request_id in policy['mapping'].items():
        c=cases[key];r=requests[request_id]
        cpu=proof.compile_answer(proof.automatic_proposals(r['checker']),r['checker'])
        combined={'facts':cpu['facts']+answers.get(request_id,{}).get('facts',[]),'rejected':cpu['rejected']+answers.get(request_id,{}).get('rejected',[])}
        cpu_score,cpu_v=proof.score_case(c,cpu);score,v=proof.score_case(c,combined)
        scores[key]={'rule':c['rule_verification']['statuses'],'cpu':cpu_v['statuses'],'v2':v['statuses'],
                     'v1':v1[key]['hybrid_statuses'],'cpu_reward':cpu_score['reward'],'v2_reward':score['reward'],
                     'rule_reward':c['rule_reward']['reward'],'llm_available':request_id in answers}
        records.append({'case_id':key,'task_id':c['task_id'],'selected_options':c['selected_options'],**scores[key],
                        'cpu_facts':cpu,'combined_proof_comparisons':v['comparisons']})
    comparisons=[];metrics={}
    for group in ('original','additional'):
        refs=old.rows(folder/f'reference_{group}.jsonl');evaluated=[]
        for ref in refs:
            s=scores[ref['case_id']];rid=ref['requirement_id']
            evaluated.append({**ref,'reference_group':group,**{name:s[name][rid] for name in ('rule','cpu','v1','v2')},'llm_available':s['llm_available']})
        comparisons.extend(evaluated)
        metrics[group]={name:accuracy(evaluated,name) for name in ('rule','cpu','v1','v2')}
    ranking=[]
    for pair in old.rows(old.DEFAULT/'reference_rankings.jsonl'):
        left,right=scores[pair['left_case_id']],scores[pair['right_case_id']]
        def compare(name):
            a,b=left[name+'_reward'],right[name+'_reward']
            return a>b+1e-8 if pair['relation']=='gt' else abs(a-b)<=1e-8
        ranking.append({**pair,'cpu_pass':compare('cpu'),'v2_pass':compare('v2'),'cpu_values':[left['cpu_reward'],right['cpu_reward']],
                        'v2_values':[left['v2_reward'],right['v2_reward']]})
    group_rows=[]
    for event in events:
        if event['cohort']!='holdout_v5':continue
        for uid in dict.fromkeys(m['uid'] for m in event['members']):
            members=[m for m in event['members'] if m['uid']==uid]
            row={'task_id':members[0]['task_id'],'uid':uid}
            for variant in ('rule','cpu','v2'):
                values=[scores[m['case_id']][variant+'_reward'] if m['case_id'] else m['original_reward'] for m in members]
                row[variant]='invalid' if any(not m['original_valid'] for m in members) else 'tie' if max(values)-min(values)<1e-8 else 'eligible'
                row[variant+'_rewards']=values
            group_rows.append(row)
    attempted=old.rows(folder/'attempts.jsonl');usage=Counter()
    for a in attempted:
        for k,v in a.get('usage',{}).items():
            if isinstance(v,(int,float)):usage[k]+=v
    original=metrics['original'];additional=metrics['additional']
    summary={'request_count':len(requests),'valid_extractions':len(answers),'attempts':len(attempted),
             'all_requests_attempted':len({r['request_id'] for r in attempted})==len(requests),
             'metrics':metrics,'rankings':ranking,'holdout_group_counts':{name:dict(Counter(r[name] for r in group_rows)) for name in ('rule','cpu','v2')},
             'facts':{'accepted_model':sum(len(a['facts']) for a in answers.values()),'rejected_model':sum(len(a['rejected']) for a in answers.values()),
                      'rejection_reasons':dict(Counter(f['reason'] for a in answers.values() for f in a['rejected']))},
             'attempt_seconds':old.describe([a['seconds'] for a in attempted]),'reported_usage':dict(usage),
             'old_25_false_passes_now':{name:sum(r[name]=='pass' for r in comparisons if r['reference_group']=='original' and r['status']!='pass' and r['v1']=='pass') for name in ('cpu','v2')},
             'new_false_passes':{name:sum(r[name]=='pass' and r['status']!='pass' and r['v1']!='pass' for r in comparisons) for name in ('cpu','v2')},
             'known_cases_are_regression_not_blind':True,'training_assets_modified':False,'api_sees_requirements':False,
             'compiler_only_coverage_is_reported':True,'formal_training_ready':False,
             'decision':'not_released; assess_false_pass_and_true_pass_retention_together'}
    old.lines(folder/'purchase_comparisons.jsonl',records);old.lines(folder/'reference_comparisons.jsonl',comparisons)
    old.lines(folder/'group_comparisons.jsonl',group_rows);old.write(folder/'report.json',summary)
    print(old.canonical(summary),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','report']);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--env-file',type=Path);p.add_argument('--limit',type=int);p.add_argument('--concurrency',type=int,default=4);p.add_argument('--timeout',type=int,default=60)
    args=p.parse_args()
    if args.concurrency<1 or args.timeout<1 or args.limit is not None and args.limit<1:p.error('positive numeric settings required')
    if args.action=='prepare':prepare(args.output)
    elif args.action=='report':report(args.output)
    else:run(args.output,old.env_config(args.env_file),args.limit,args.concurrency,args.timeout)


if __name__=='__main__':main()
