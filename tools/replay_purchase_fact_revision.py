"""Replay frozen fact proposals with an explicitly versioned comparator. No API."""
from collections import Counter
from pathlib import Path
import json
import time

import run_purchase_fact_experiment as experiment
import purchase_fact_comparison_r1 as revised


def replay():
    start=time.perf_counter()
    old=experiment.old;proof=experiment.proof;folder=experiment.OUTPUT;target=folder/'replay_r1'
    target.mkdir(exist_ok=True)
    policy,requests,_=experiment.load(folder)
    cases,events,_,_=old.load_dataset(old.DEFAULT)
    raw_answers=old.rows(folder/'answers.jsonl')
    settings=json.loads((folder/'api_config.json').read_text(encoding='utf-8'))
    answers={}
    for a in raw_answers:
        if a['namespace']!=old.digest(settings):raise ValueError('Mixed original namespaces')
        answers[a['request_id']]=proof.compile_answer(a['answer'],requests[a['request_id']]['checker'])
    scores={};records=[]
    for key,request_id in policy['mapping'].items():
        c=cases[key];request=requests[request_id]['checker']
        cpu=proof.compile_answer(proof.automatic_proposals(request),request)
        all_facts=cpu['facts']+answers.get(request_id,{}).get('facts',[])
        reward,v=revised.score(c,{'facts':all_facts})
        scores[key]={'statuses':v['statuses'],'reward':reward['reward']}
        records.append({'case_id':key,'task_id':c['task_id'],'statuses':v['statuses'],'reward':reward,
                        'comparisons':v['comparisons'],'model_facts_available':request_id in answers})
    evaluated=[]
    for r in old.rows(folder/'reference_comparisons.jsonl'):
        evaluated.append({**r,'r1':scores[r['case_id']]['statuses'][r['requirement_id']]})
    metrics={group:{name:experiment.accuracy([r for r in evaluated if r['reference_group']==group],name)
                    for name in ('rule','cpu','v1','v2','r1')} for group in ('original','additional')}
    rankings=[]
    for pair in old.rows(old.DEFAULT/'reference_rankings.jsonl'):
        a,b=[scores[pair[key]]['reward'] for key in ('left_case_id','right_case_id')]
        passed=a>b+1e-8 if pair['relation']=='gt' else abs(a-b)<1e-8
        rankings.append({**pair,'r1_pass':passed,'rewards':[a,b]})
    groups=[]
    for e in events:
        if e['cohort']!='holdout_v5':continue
        for uid in dict.fromkeys(m['uid'] for m in e['members']):
            members=[m for m in e['members'] if m['uid']==uid]
            values=[scores[m['case_id']]['reward'] if m['case_id'] else m['original_reward'] for m in members]
            status='invalid' if any(not m['original_valid'] for m in members) else 'tie' if max(values)-min(values)<1e-8 else 'eligible'
            groups.append({'task_id':members[0]['task_id'],'status':status,'rewards':values})
    manifest={'parent_protocol_sha256':old.digest(policy),'original_answers_sha256':old.sha256(folder/'answers.jsonl'),
              'original_reference_comparisons_sha256':old.sha256(folder/'reference_comparisons.jsonl'),
              'comparison_revision':revised.VERSION,'comparison_revision_sha256':old.sha256(Path(revised.__file__)),
              'replay_script_sha256':old.sha256(Path(__file__)),'new_api_calls':0,'same_proposals_replayed_not_new_inference':True}
    summary={'metrics':metrics,'rankings':rankings,'holdout_group_counts':dict(Counter(g['status'] for g in groups)),
             'new_api_calls':0,'original_25_false_passes_remaining':sum(r['r1']=='pass' for r in evaluated if r['reference_group']=='original' and r['status']!='pass' and r['v1']=='pass'),
             'new_false_passes_vs_v1':sum(r['r1']=='pass' for r in evaluated if r['status']!='pass' and r['v1']!='pass'),
             'replay_wall_seconds':time.perf_counter()-start,'training_assets_modified':False,'formal_training_ready':False,
             'decision':'reject_as_complete_semantic_training_reward; retain_numeric_set_scope_proofs_only'}
    old.lines(target/'reference_comparisons.jsonl',evaluated);old.lines(target/'purchase_scores.jsonl',records)
    old.lines(target/'groups.jsonl',groups);old.write(target/'manifest.json',manifest);old.write(target/'report.json',summary)
    print(old.canonical(summary),flush=True)
    return summary


if __name__=='__main__':replay()
