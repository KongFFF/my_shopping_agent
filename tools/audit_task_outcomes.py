"""CPU task-level outcome audit on existing online generations and certified pages."""
from collections import Counter, defaultdict
from pathlib import Path
import gzip
import hashlib
import json
import re
import statistics as st
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/task_outcome_audit_20261117'
LABELS=('reference351_500','late651_800')
def dump(x): return json.dumps(x,ensure_ascii=False,sort_keys=True)
def sha(b): return hashlib.sha256(b).hexdigest()
def save(n,x): (OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def ok(r): return bool(r['reward_valid'] and not r.get('infrastructure_invalid') and not r['reward'].get('sampling_invalid'))
def win(r): return bool(ok(r) and r['reward']['purchase_success'])
def key(r): return sha(dump({'task_id':r['task_id'],'actions':r['actions']}).encode())
def fingerprint(r): return dump({k:v for k,v in r.items() if k!='rollout_index'})
def bought(r):
    asin=None
    for a in r['actions']:
        if a['tool']=='open_product': asin=str(a['parameters'].get('asin'))
        if a['tool']=='buy_now': return asin
    return None


def main():
    protocol_hash=sha((OUT/'核查口径.md').read_bytes())
    old=json.loads((ROOT/'analysis/grpo_structural_analysis_20261116/manifest.json').read_text(encoding='utf-8'))
    for p,h in old['inputs_sha256'].items(): assert sha((ROOT/p).read_bytes())==h,p
    certified={r['key'] for r in json.loads((ROOT/'analysis/online_window_audit_20261116/certified_index.json').read_text(encoding='utf-8'))}
    with gzip.open(ROOT/'analysis/online_window_audit_20261116/restored_observations.jsonl.gz','rt',encoding='utf-8')as f:
        restored={r['key']:r for r in map(json.loads,f)}
    groups=[]; source={}; prior=defaultdict(set)
    with zipfile.ZipFile(ROOT/'analysis/online_training_window_return_20261116/evidence.zip')as z:
        train={json.loads(l)['task_id'] for l in z.read('inputs/train.jsonl').splitlines()}
        for label in LABELS:
            es=list(map(json.loads,z.read(label+'/training_diagnostics.jsonl').splitlines()))
            consumed={}; generations={}
            for e in es:
                if e['event']!='optimizer_step':continue
                for r in e['rollouts']:consumed.setdefault(r['uid'],[]).append(r)
            window=[]
            for e in es:
                if e['event']!='generation_batch':continue
                batch=defaultdict(list);meta={g['uid']:g for g in e['groups']}
                for r in e['rollouts']:batch[r['uid']].append(r)
                for uid,rs in batch.items():
                    assert len(rs)==4 and len({r['task_id'] for r in rs})==1 and uid not in generations
                    generations[uid]=rs;task=rs[0]['task_id']
                    if uid in consumed:assert Counter(map(fingerprint,rs))==Counter(map(fingerprint,consumed[uid]))
                    good=all(ok(r) for r in rs) and not meta[uid]['sampling_invalid']
                    g={'window':label,'step':e['global_step'],'batch':e['generation_batch'],'uid':uid,'task_id':task,
                        'valid':good,'consumed':uid in consumed,'drop_reason':meta[uid].get('drop_reason'),
                        'successes':sum(win(r)for r in rs),'prior_success_groups':len(prior[task]),'rollouts':rs}
                    window.append(g)
                # Earlier batches only; a current-batch peer is never historical evidence.
                for uid,rs in batch.items():
                    if all(ok(r) for r in rs) and not meta[uid]['sampling_invalid'] and any(win(r)for r in rs):prior[rs[0]['task_id']].add(uid)
            groups+=window
            source[label]={'all_generated_groups':len(window),'all_trajectories':len(window)*4,
                'valid_groups':sum(g['valid']for g in window),'valid_tasks':len({g['task_id']for g in window if g['valid']}),
                'valid_group_success_histogram':dict(Counter(g['successes']for g in window if g['valid'])),
                'consumed_groups':len(consumed),'invalid_groups':sum(not g['valid']for g in window)}
    assert [source[w]['all_generated_groups']for w in LABELS]==[1220,1370]
    bytask=defaultdict(list)
    for g in groups:
        if g['valid']:bytask[g['task_id']].append(g)
    task_rows=[];category={};gold=defaultdict(set)
    for i in sorted(train):
        gg=bytask[i];early=[g for g in gg if g['window']==LABELS[0]];late=[g for g in gg if g['window']==LABELS[1]]
        late_s=sum(g['successes']for g in late);late_n=4*len(late)
        cls='no_valid_late_observation' if not late else 'late_observed_all_success' if late_s==late_n else 'late_mixed' if late_s else 'early_success_late_all_failure' if any(g['successes']for g in early) else 'no_observed_success_either_window'
        category[i]=cls
        for g in gg:
            gold[i].update(bought(r)for r in g['rollouts']if win(r)and bought(r))
        def describe(gs):
            return {'groups':len(gs),'successes':sum(g['successes']for g in gs),'trajectories':4*len(gs),
                'success_groups':sum(g['successes']>0 for g in gs),'consumed_groups':sum(g['consumed']for g in gs)}
        task_rows.append({'task_id':i,'category':cls,'early':describe(early),'late':describe(late),
            'known_success_asins':sorted(gold[i]),'valid_action_sequence_count':len({key(r)for g in gg for r in g['rollouts']}),
            'group_success_sequence':[{'window':g['window'],'step':g['step'],'uid':g['uid'],'successes':g['successes'],'consumed':g['consumed']}for g in gg]})
    trace_pages=[]; stage_rows=[]
    for g in groups:
        if not g['valid']:continue
        i=g['task_id'];cat=category[i]
        for r in g['rollouts']:
            k=key(r)
            if k not in certified:continue
            rr=restored[k];assert [s['action']for s in rr['steps']]==r['actions']
            seen=set();opened=set();queries=[];sets=[];newcounts=[];repeat_sets=0;searches=0;unknown=0
            for s in rr['steps']:
                a=s['action'];obs=s.get('observation') or ''
                if a['tool']=='open_product':opened.add(str(a['parameters'].get('asin')))
                if a['tool']=='search_products':queries.append(a['parameters'].get('query'))
                # Only search-result rows in the actor-visible projection.
                if a['tool'] in ('search_products','next_page','prev_page') and 'page_type: search_results' in obs:
                    searches+=1;asins=set(re.findall(r'^\d+\|(\d+)\|',obs,re.M))
                    if not asins:unknown+=1;continue
                    frozen=tuple(sorted(asins));repeat_sets+=frozen in sets;sets.append(frozen)
                    newcounts.append(len(asins-seen));seen.update(asins)
            known=gold[i];p=bought(r)
            stage='no_observed_success_reference' if not known else 'purchased_known_success_product' if p in known else 'opened_known_success_product' if opened&known else 'saw_known_success_product' if seen&known else 'known_success_product_not_seen'
            trace_pages.append({'task_id':i,'window':g['window'],'uid':g['uid'],'key':k,'success':win(r),'category':cat,
                'consumed':g['consumed'],'reward_type':r['reward_type'],'search_result_visits':searches,'unparsed_or_empty_visits':unknown,
                'parsed_visits':len(sets),'repeat_result_set_visits':repeat_sets,'new_asins_by_visit':newcounts,
                'unique_queries':len(set(queries)),'query_calls':len(queries),'unique_seen_asins':len(seen),'stage':stage})
        if g['consumed'] and g['successes']==0:
            top=max(r['reward']['terminal_utility']for r in g['rollouts']);best=[r for r in g['rollouts']if abs(r['reward']['terminal_utility']-top)<1e-7]
            partial=[r for r in best if r['reward_type']=='partial_alternative_purchase']
            stage_rows.append({'task_id':i,'window':g['window'],'uid':g['uid'],'step':g['step'],'category':cat,
                'prior_success_groups':g['prior_success_groups'],'top_reward':top,'top_types':sorted({r['reward_type']for r in best}),
                'top_has_partial':bool(partial),'top_partial_asins':sorted({bought(r)for r in partial if bought(r)}),
                'has_any_success_reference':bool(gold[i]),'top_partial_bought_known_success_product':any(bought(r)in gold[i]for r in partial)})
    def page_summary(rs):
        n=sum(r['parsed_visits']for r in rs);new=[x for r in rs for x in r['new_asins_by_visit'][1:]]
        return {'trajectories':len(rs),'tasks':len({r['task_id']for r in rs}),'parsed_search_result_visits':n,
            'repeat_result_set_visits':sum(r['repeat_result_set_visits']for r in rs),'additional_visits':len(new),
            'additional_visits_without_new_asins':sum(x==0 for x in new),
            'repeated_exact_queries':sum(r['query_calls']-r['unique_queries']for r in rs),'query_calls':sum(r['query_calls']for r in rs),
            'median_unique_seen_asins':st.median(r['unique_seen_asins']for r in rs)if rs else None,
            'failure_stages':dict(Counter(r['stage']for r in rs if not r['success']))}
    transitions={}
    for w in LABELS:
        gs=[g for g in groups if g['window']==w and g['valid']and g['successes']==0];used=[g for g in gs if g['consumed']]
        transitions[w]={'valid_zero_success_groups':len(gs),'with_prior_success':sum(g['prior_success_groups']>0 for g in gs),
            'with_two_prior_success_groups':sum(g['prior_success_groups']>=2 for g in gs),'consumed_zero_success_groups':len(used),
            'consumed_with_prior_success':sum(g['prior_success_groups']>0 for g in used),
            'consumed_with_two_prior_success_groups':sum(g['prior_success_groups']>=2 for g in used),
            'top_partial_groups':sum(r['top_has_partial']for r in stage_rows if r['window']==w),
            'top_partial_known_success_product_groups':sum(r['top_partial_bought_known_success_product']for r in stage_rows if r['window']==w)}
    page_index={(r['uid'],r['key']):r for r in trace_pages}
    paired=[]
    stage_order={'known_success_product_not_seen':0,'saw_known_success_product':1,
        'opened_known_success_product':2,'purchased_known_success_product':3}
    for g in groups:
        if not(g['valid']and g['consumed']and g['successes']==0 and gold[g['task_id']]):continue
        partial=[r for r in g['rollouts']if r['reward_type']=='partial_alternative_purchase']
        top=max(r['reward']['terminal_utility']for r in g['rollouts']);directions=[]
        for a in partial:
            if abs(a['reward']['terminal_utility']-top)>1e-7:continue
            pa=page_index.get((g['uid'],key(a)))
            if pa is None:continue
            for b in g['rollouts']:
                if b['reward']['terminal_utility']>=top-1e-7:continue
                pb=page_index.get((g['uid'],key(b)))
                if pb is None:continue
                delta=stage_order[pa['stage']]-stage_order[pb['stage']]
                directions.append('deeper'if delta>0 else 'shallower'if delta<0 else 'equal')
        observed=set(directions)-{'equal'}
        classification='unavailable'if not directions else 'mixed'if len(observed)>1 else next(iter(observed))if observed else 'equal_only'
        paired.append({'window':g['window'],'uid':g['uid'],'task_id':g['task_id'],
            'top_partial_vs_lower_stage':classification,'pair_counts':dict(Counter(directions))})
    summary={'sources':source,'total_valid_observed_tasks':sum(bool(bytask[i])for i in train),'tasks':dict(Counter(category.values())),
        'no_success_task_observation_support':dict(Counter(len(bytask[t['task_id']])for t in task_rows if t['category']=='no_observed_success_either_window')),
        'transitions':transitions,
        'pages_by_category':{cat:page_summary([r for r in trace_pages if r['category']==cat])for cat in sorted(set(category.values()))},
        'late_pages_by_category':{cat:page_summary([r for r in trace_pages if r['category']==cat and r['window']==LABELS[1]])for cat in sorted(set(category.values()))},
        'late_failure_reference_stage':page_summary([r for r in trace_pages if r['window']==LABELS[1] and not r['success']]),
        'paired_progress_proxy':{w:dict(Counter(r['top_partial_vs_lower_stage']for r in paired if r['window']==w))for w in LABELS},
        'top_partial_stage_coverage':{w:dict(Counter(next((p['stage']for p in trace_pages if p['uid']==g['uid']and p['key']==key(r)), 'not_certified')
            for g in groups if g['window']==w and g['valid']and g['consumed']and g['successes']==0
            for r in g['rollouts']if r['reward_type']=='partial_alternative_purchase' and abs(r['reward']['terminal_utility']-max(x['reward']['terminal_utility']for x in g['rollouts']))<1e-7))for w in LABELS}}
    save('summary.json',summary);save('task_history.json',task_rows);save('zero_success_update_groups.json',stage_rows);save('public_path_statistics.json',trace_pages);save('partial_progress_pairs.json',paired)
    assert sha((OUT/'核查口径.md').read_bytes())==protocol_hash
    save('manifest.json',{'protocol_sha256':protocol_hash,'script_sha256':sha(Path(__file__).read_bytes()),
        'upstream_manifest_sha256':sha((ROOT/'analysis/grpo_structural_analysis_20261116/manifest.json').read_bytes()),
        'outputs_sha256':{n:sha((OUT/n).read_bytes())for n in ['summary.json','task_history.json','zero_success_update_groups.json','public_path_statistics.json','partial_progress_pairs.json']}})
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
