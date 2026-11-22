"""Unified CPU-only endpoint and logged advantage audit; no inference or reward repair."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
import re
from pathlib import Path
import statistics
import zipfile

from analyze_b500_dev50_return import strict
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/group_weight_unified_20261122'
PATHS={
 'B500dev':'analysis/b500_independent_dev50_return/20261121T031644Z/evidence.zip',
 'B500final':'analysis/b500_final200_return/20261121T040953Z/evidence.zip',
 'B650eval':'analysis/b650_evaluation_return/20261122T011337Z/evidence.zip',
 'B800eval':'analysis/b800_evaluation_return/20261122T134221Z/evidence.zip',
 'B500train':'analysis/b500_training_return_20261118/20261120T150105Z/evidence.zip',
 'B650train':'analysis/b650_training_return/20261121T144812Z/evidence.zip',
 'Atrain':'analysis/online_training_window_return_20261116/evidence.zip'}

def sha(b):return hashlib.sha256(b).hexdigest()
def rows(b):return[json.loads(l)for l in b.decode('utf-8-sig').splitlines()if l.strip()]
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def key(t):return json.dumps([(s['tool_name'],s['parameters'])for s in t['steps']],ensure_ascii=False,sort_keys=True)
def typ(t):return((t.get('terminal_result')or{}).get('reward_detail')or{}).get('reward_type','unknown')
def query(t):return next((m['content']for m in t['messages']if m['role']=='user'),t['initial_result']['instruction'])
def action_key(s):return json.dumps((s['tool_name'],s['parameters']),sort_keys=True,ensure_ascii=False)

def load():
    manifest={};blobs={}
    for name,rel in PATHS.items():
        path=ROOT/rel;raw=path.read_bytes();manifest[rel]={'sha256':sha(raw),'bytes':len(raw)}
        if path.with_name('manifest.json').exists():
            m=json.loads(path.with_name('manifest.json').read_text(encoding='utf-8'))
            if 'archive_sha256'in m:assert sha(raw)==m['archive_sha256'],name
        with zipfile.ZipFile(path)as z:
            assert z.testzip()is None
            needed=[n for n in z.namelist()if (n.endswith('trajectories.jsonl')and(n.startswith('evaluation/')or n.startswith('reference_A')))
                or n in ['weights.jsonl','weights_before_rollback.jsonl','training_diagnostics.jsonl','training_evidence/weights.jsonl','training_evidence/training_diagnostics.jsonl',
                    'reference351_500/training_diagnostics.jsonl','late651_800/training_diagnostics.jsonl']
                or ('A500'in n and n.endswith('summary.json'))]
            blobs[name]={n:z.read(n)for n in needed}
    data={'dev50':{},'final200_clean':{}}
    for label,folder in [('SFT','sft_dev50'),('A500','grpo_dev50')]:
        path=ROOT/'analysis/grpo_baseline_20261022_202816'/folder/'trajectories.jsonl';raw=path.read_bytes();manifest[str(path.relative_to(ROOT))]={'sha256':sha(raw),'bytes':len(raw)}
        frozen=json.loads((ROOT/'analysis/grpo_baseline_20261022_202816/manifest.json').read_text(encoding='utf-8'))['files'][folder+'/trajectories.jsonl']['sha256']
        assert frozen in [sha(raw),sha(raw.replace(b'\r\n',b'\n'))],label
        data['dev50'][label]=rows(raw)
    for suite in data:
        source='B500dev'if suite=='dev50'else'B500final'
        data[suite]['B500']=rows(blobs[source]['evaluation/'+suite+'/trajectories.jsonl'])
        for step in [650,800]:
            source='B'+str(step)+'eval'
            data[suite]['A'+str(step)]=rows(blobs[source]['reference_A'+str(step)+'/'+suite+'/trajectories.jsonl'])
            data[suite]['B'+str(step)]=rows(blobs[source]['evaluation/'+suite+'/trajectories.jsonl'])
        ref=data[suite]['B500'];ids=[t['task_id']for t in ref];proto=None
        for label,rs in data[suite].items():
            assert len(rs)==(50 if suite=='dev50'else 200)and ids==[t['task_id']for t in rs]and len(set(ids))==len(ids),(suite,label)
            assert all(query(a)==query(b)for a,b in zip(ref,rs)),(suite,label,'query')
    old_summary=json.loads(next(b for n,b in blobs['B500final'].items()if'A500'in n and n.endswith('summary.json')))
    statuses={t['task_id']:t['task_id']in old_summary['strict_success_task_ids']for t in data['final200_clean']['B500']}
    allw={}
    for name,filename,lo,hi in [('B500train','weights_before_rollback.jsonl',1,150),('B500train','weights.jsonl',151,500),
        ('B650train','weights.jsonl',501,650),('B800eval','training_evidence/weights.jsonl',651,800)]:
        for w in rows(blobs[name][filename]):
            if w.get('event')!='actor_update_returned':continue
            step=w['driver_global_step_before_update']+1
            if not lo<=step<=hi:continue
            assert step not in allw,step
            allw[step]=w
    assert sorted(allw)==list(range(1,801))
    diag=rows(blobs['B800eval']['training_evidence/training_diagnostics.jsonl'])
    update_events=[e for e in diag if e['event']=='optimizer_step']
    updates={e['global_step']:e for e in update_events};assert len(update_events)==800 and sorted(updates)==list(range(1,801))
    A={name:rows(blobs['Atrain'][name+'/training_diagnostics.jsonl'])for name in ['reference351_500','late651_800']}
    return data,statuses,allw,updates,A,manifest,diag


def feature(t,target):
    steps=t['steps'];calls=[action_key(s)for s in steps];goal=str(target)if target else None
    seen=[i+1 for i,s in enumerate(steps)if goal and re.search(r'(?<!\d)'+re.escape(goal)+r'(?!\d)',s.get('observation',''))]
    opened=[i+1 for i,s in enumerate(steps)if s['tool_name']=='open_product'and str(s['parameters'].get('asin'))==goal]
    purchase=(t.get('terminal_result')or{}).get('purchase')or{}
    if strict(t):stage='success'
    elif purchase.get('asin')and str(purchase['asin'])==goal:stage='annotated_item_bought_not_gold'
    elif opened:stage='annotated_item_opened_no_gold_purchase'
    elif seen:stage='annotated_item_seen_not_opened'
    else:stage='annotated_item_not_seen'
    return {'success':bool(strict(t)),'type':typ(t),'steps':len(steps),'status':t['status'],'reward':t['final_reward'],
        'stage_proxy':stage,'target_first_seen':min(seen)if seen else None,'target_first_open':min(opened)if opened else None,
        'searches':sum(s['tool_name']=='search_products'for s in steps),'opens':sum(s['tool_name']=='open_product'for s in steps),
        'distinct_opened_items':len({s['parameters'].get('asin')for s in steps if s['tool_name']=='open_product'}),
        'duplicate_action_occurrences':len(calls)-len(set(calls)),'blocked_calls':len(t.get('blocked_tool_calls')or[]),
        'purchase':{k:v for k,v in purchase.items()if k in ['asin','options','price','name']},'action_signature_sha256':sha(key(t).encode())}


def endpoint_analysis(data,old_statuses):
    summary={};tasks={};changes={};review=[]
    for suite,models in data.items():
        maps={m:{t['task_id']:t for t in rs}for m,rs in models.items()};tasks[suite]=[];summary[suite]={}
        for tid in maps['B500']:
            target=next(((t.get('terminal_result')or{}).get('goal',{}).get('asin')for rs in maps.values()if(t:=rs[tid]).get('terminal_result')and(t.get('terminal_result')or{}).get('goal',{}).get('asin')),None)
            targets={(t.get('terminal_result')or{}).get('goal',{}).get('asin')for rs in maps.values()if(t:=rs[tid]).get('terminal_result')and(t.get('terminal_result')or{}).get('goal',{}).get('asin')}
            assert len(targets)<=1,(suite,tid,'annotation')
            feats={m:feature(rs[tid],target)for m,rs in maps.items()}
            states={m:f['success']for m,f in feats.items()}
            if suite=='final200_clean':states['A500']=old_statuses[tid]
            pattern=''.join('1'if states['B'+str(step)]else'0'for step in [500,650,800])
            ab=''.join('+'if states['B'+str(step)]and not states['A'+str(step)]else'-'if states['A'+str(step)]and not states['B'+str(step)]else'='for step in [500,650,800])
            row={'task_id':tid,'B_success_pattern500650800':pattern,'B_minus_A_pattern500650800':ab,'states':states,'models':feats};tasks[suite].append(row)
            changed=any(states['A'+str(s)]!=states['B'+str(s)]for s in [500,650,800])or len(set(pattern))>1
            if changed:
                review.append({'suite':suite,'task_id':tid,'query':query(maps['B500'][tid]),'B_pattern':pattern,'AB_pattern':ab,
                    'models':{m:{'success':feats[m]['success'],'type':feats[m]['type'],'steps':feats[m]['steps'],'purchase':feats[m]['purchase'],
                        'calls':[{'turn':i+1,'tool':s['tool_name'],'args':s['parameters']}for i,s in enumerate(rs[tid]['steps'])]}for m,rs in maps.items()}})
        for model,rs in models.items():
            fs=[r['models'][model]for r in tasks[suite]]
            systems=Counter(sha(next(m['content']for m in t['messages']if m['role']=='system').encode())for t in rs)
            summary[suite][model]={'n':len(rs),'successes':sum(strict(t)for t in rs),'reward_types':dict(Counter(typ(t)for t in rs)),
                'stage_proxy':dict(Counter(f['stage_proxy']for f in fs)),'mean_steps':statistics.mean(len(t['steps'])for t in rs),
                'system_prompt_hashes':dict(systems),'mean_blocked_calls':statistics.mean(f['blocked_calls']for f in fs)}
        systems={h for m in summary[suite].values()for h in m['system_prompt_hashes']};assert len(systems)==1,'System prompt drift'
        changes[suite]={'B_success_patterns':dict(Counter(r['B_success_pattern500650800']for r in tasks[suite])),
            'B_minus_A_patterns':dict(Counter(r['B_minus_A_pattern500650800']for r in tasks[suite])),'review_tasks':[r['task_id']for r in review if r['suite']==suite],
            'persistent_B_failure_tasks':[r['task_id']for r in tasks[suite]if r['B_success_pattern500650800']=='000'],
            'persistent_B_failures_identical_actions':[r['task_id']for r in tasks[suite]if r['B_success_pattern500650800']=='000'and len({r['models']['B'+str(step)]['action_signature_sha256']for step in [500,650,800]})==1],
            'AB800_identical_action_tasks':[r['task_id']for r in tasks[suite]if r['models']['A800']['action_signature_sha256']==r['models']['B800']['action_signature_sha256']],
            'AB800_reward_type_transitions':{a+'->'+b:[r['task_id']for r in tasks[suite]if r['models']['A800']['type']==a and r['models']['B800']['type']==b]for a,b in sorted({(r['models']['A800']['type'],r['models']['B800']['type'])for r in tasks[suite]})},
            'step_transitions':{},'AB_transitions':{}}
        for a,b in [('B500','B650'),('B650','B800'),('B500','B800'),('A650','B650'),('A800','B800')]+([('A500','B500')]if suite=='dev50'else[]):
            cr=[]
            for row in tasks[suite]:
                fa=row['models'][a];fb=row['models'][b]
                if fa['success']!=fb['success']:
                    aa=maps[a][row['task_id']];bb=maps[b][row['task_id']];ca=[action_key(s)for s in aa['steps']];cb=[action_key(s)for s in bb['steps']]
                    first=next((i+1 for i,(x,y)in enumerate(zip(ca,cb))if x!=y),min(len(ca),len(cb))+1)
                    cr.append({'task_id':row['task_id'],'change':'gain'if fb['success']else'loss','from_type':fa['type'],'to_type':fb['type'],
                        'from_steps':fa['steps'],'to_steps':fb['steps'],'first_action_divergence':first,'from_stage_proxy':fa['stage_proxy'],'to_stage_proxy':fb['stage_proxy']})
            target='AB_transitions'if a[0]=='A'else'step_transitions';changes[suite][target][a+'->'+b]=cr
    return summary,tasks,changes,review


def describe(rs):
    masses=defaultdict(lambda:Counter());groups=defaultdict(set);tasks=defaultdict(set)
    for r in rs:
        cohort=r['kind']+'/'+r['sign'];groups[cohort].add(r['uid']);tasks[cohort].add(r['task_id']);c=masses[cohort]
        c['trajectories']+=1;c['tokens']+=r['tokens'];c['trajectory_abs_adv']+=abs(r['adv'])
        c['raw_token_abs_adv']+=r['raw_token_abs_adv'];c['weighted_token_abs_adv']+=r['weighted_token_abs_adv']
        c['batch_normalized_abs_adv']+=r['raw_token_abs_adv']/r['batch_tokens'];c['weighted_batch_normalized_abs_adv']+=r['weighted_token_abs_adv']/r['batch_tokens']
    return {k:{**dict(v),'groups':len(groups[k]),'tasks':len(tasks[k])}for k,v in masses.items()}


def training_analysis(weights,updates,A):
    trajectory_rows=[];group_rows=[];errors=[]
    for step,w in sorted(weights.items()):
        e=updates[step];assert e['metrics']['training/optimizer_updated']==1
        native=defaultdict(list)
        for r in e['rollouts']:native[r['uid']].append(r)
        assert len(w['groups'])==2 and len(w['trajectories'])==8 and len(native)==2
        N=w['original_token_count'];assert N==e['metrics']['b500/actor_tokens']
        denom=sum(t['effective_tokens']*t['raw_weight']for t in w['trajectories'])
        assert N==sum(t['effective_tokens']for t in w['trajectories'])and math.isclose(denom,w['weighted_token_count'])
        for g in w['groups']:
            ts=[t for t in w['trajectories']if t['uid']==g['uid']];assert len(ts)==4
            mean=statistics.mean(t['terminal_utility']for t in ts);kind='S'if any(t['success']for t in ts)else'F';assert kind==g['kind']
            assert all(t['task_id']==g['task_id']for t in ts)
            assert all(r['task_id']==g['task_id']for r in native[g['uid']])
            assert Counter((t['reward_type'],round(t['terminal_utility'],6),bool(t['success']))for t in ts)==Counter((r['reward_type'],round(r['reward']['terminal_utility'],6),bool(r['reward']['purchase_success']))for r in native[g['uid']])
            assert all(t['raw_weight']==(0.5 if kind=='F'else 1.0)for t in ts)
            gold=sum(t['reward_type']=='gold_purchase'for t in ts)
            group_rows.append({'step':step,'uid':g['uid'],'task_id':g['task_id'],'kind':kind,'composition':w['composition'],'gold_count':gold,
                'native_successes':sum(t['success']for t in ts),'coefficient':g['coefficient'],'reward_range':max(t['terminal_utility']for t in ts)-min(t['terminal_utility']for t in ts)})
            for t in ts:
                a=t['original_advantage_mean'];expected=t['terminal_utility']-mean
                if not math.isclose(a,expected,abs_tol=3e-6):errors.append({'step':step,'uid':g['uid'],'adv':a,'expected':expected})
                assert t['original_positive_tokens']+t['original_negative_tokens']in (0,t['effective_tokens'])
                if abs(a)>1e-7:assert (t['original_positive_tokens']if a>0 else t['original_negative_tokens'])==t['effective_tokens']
                assert math.isclose(t['coefficient'],t['raw_weight']*N/denom,rel_tol=1e-12)
                sign='positive'if a>1e-7 else'negative'if a<-1e-7 else'numerical_zero'
                trajectory_rows.append({'step':step,'uid':t['uid'],'task_id':t['task_id'],'kind':kind,'composition':w['composition'],
                    'type':t['reward_type'],'native_success':t['success'],'terminal_utility':t['terminal_utility'],'adv':a,'sign':sign,
                    'tokens':t['effective_tokens'],'batch_tokens':N,'coefficient':t['coefficient'],'tool_steps':t['tool_steps'],
                    'raw_token_abs_adv':abs(a)*t['effective_tokens'],'weighted_token_abs_adv':abs(a)*t['effective_tokens']*t['coefficient']})
    assert not errors,errors[:3]
    out={}
    for name,lo,hi in [('all1_800',1,800),('early1_150',1,150),('middle151_350',151,350),('late351_500',351,500),('continuation501_650',501,650),('endpoint651_800',651,800)]:
        ww=[weights[i]for i in range(lo,hi+1)];gg=[g for g in group_rows if lo<=g['step']<=hi];tt=[r for r in trajectory_rows if lo<=r['step']<=hi]
        ss=[g for g in gg if g['kind']=='S'];ff=[g for g in gg if g['kind']=='F'];fs=[r for r in tt if r['composition']=='FS']
        behaviors={}
        for kind in ['F','S']:
            for sign in ['positive','negative']:
                sel=[r for r in tt if r['kind']==kind and r['sign']==sign]
                behaviors[kind+'/'+sign]={typ:{'n':sum(r['type']==typ for r in sel),
                    'raw_batch_normalized_abs_adv':sum(r['raw_token_abs_adv']/r['batch_tokens']for r in sel if r['type']==typ),
                    'weighted_batch_normalized_abs_adv':sum(r['weighted_token_abs_adv']/r['batch_tokens']for r in sel if r['type']==typ)}for typ in sorted({r['type']for r in sel})}
        out[name]={'updates':len(ww),'composition':dict(Counter(w['composition']for w in ww)),
            'F_groups':len(ff),'S_groups':len(ss),'S_without_gold_groups':sum(g['gold_count']==0 for g in ss),
            'F_mean_reward_range':statistics.mean(g['reward_range']for g in ff)if ff else None,
            'S_mean_reward_range':statistics.mean(g['reward_range']for g in ss)if ss else None,
            'all_update_cohorts':describe(tt),'mixed_FS_only_cohorts':describe(fs),'behavior_by_sign':behaviors,
            'F_distinct_tasks':len({g['task_id']for g in ff}),'S_distinct_tasks':len({g['task_id']for g in ss})}
    a_out={}
    for name,es in A.items():
        tr=[];comp=Counter();taskset=set()
        for e in es:
            if e['event']!='optimizer_step':continue
            grouped=defaultdict(list)
            for r in e['rollouts']:grouped[r['uid']].append(r)
            kinds=[]
            for uid,rs in grouped.items():
                assert len(rs)==4;kind='S'if any(r['reward']['purchase_success']for r in rs)else'F';kinds.append(kind)
                mean=statistics.mean(r['reward']['terminal_utility']for r in rs);taskset.add(rs[0]['task_id'])
                for r in rs:
                    a=r['reward']['terminal_utility']-mean;tr.append({'kind':kind,'sign':'positive'if a>1e-7 else'negative'if a<-1e-7 else'numerical_zero','type':r['reward_type'],'adv':a})
            comp[''.join(sorted(kinds))]+=1
        assert sum(comp.values())==150,(name,'historical window update count')
        a_out[name]={'updates':sum(comp.values()),'composition':dict(comp),'tasks':len(taskset),
            'F_trajectory_abs_adv_share':sum(abs(r['adv'])for r in tr if r['kind']=='F')/sum(abs(r['adv'])for r in tr),
            'behavior_by_sign':{k+'/'+s:dict(Counter(r['type']for r in tr if r['kind']==k and r['sign']==s))for k in ['F','S']for s in ['positive','negative']}}
    return {'B':out,'A_original_windows':a_out,'advantage_validation':{'checked_trajectories':len(trajectory_rows),'reward_centering_mismatches':len(errors),'sign_tolerance':1e-7,
        'token_advantage_proxy_scope':'Exact logged constant trajectory advantage times Actor-effective tokens; batch-normalized exposure, not actual gradients or counterfactual endpoint scores'}},trajectory_rows,group_rows


def generation_coverage(diag,heldout):
    generated=defaultdict(list);trained=set();uids=set()
    for e in diag:
        if e['event']=='optimizer_step':trained.update(r['task_id']for r in e['rollouts'])
        if e['event']!='generation_batch':continue
        task={r['uid']:r['task_id']for r in e['rollouts']}
        for g in e['groups']:
            assert g['uid']not in uids;uids.add(g['uid']);generated[task[g['uid']]].append(g)
    assert len(generated)==1000 and not(set(generated)&heldout)
    category={}
    for tid,gs in generated.items():
        if tid in trained:kind='updated'
        elif any(g['kept']for g in gs):kind='valid_varying_observed_but_never_updated'
        elif all(all(g['purchase_success'])for g in gs):kind='never_updated_observed_all_native_success'
        elif all(not any(g['purchase_success'])for g in gs):kind='never_updated_observed_no_native_success'
        else:kind='never_updated_mixed_across_groups'
        category[tid]=kind
    group_types=Counter(g['drop_reason']or'kept'for gs in generated.values()for g in gs)
    return {'scope':'Canonical effective training path1..800, excluding rolled-back attempts; native purchase_success includes valid alternatives',
        'observed_training_tasks':len(generated),'Actor_updated_tasks':len(trained),'generated_groups':len(uids),'Actor_updated_groups':1600,
        'generation_group_types':dict(group_types),'task_categories':dict(Counter(category.values())),
        'task_ids_by_category':{c:sorted(t for t,k in category.items()if k==c)for c in sorted(set(category.values()))},
        'training_task_id_overlap_with_dev_final':0,'no_observed_success_is_not_a_proof_of_unsolvability':True}


def public_review_evidence(data,reviews,tasks):
    result=[]
    for review in reviews:
        suite=review['suite'];tid=review['task_id'];by_id={m:next(t for t in rs if t['task_id']==tid)for m,rs in data[suite].items()}
        annotation=next(((t.get('terminal_result')or{}).get('goal',{}).get('asin')for t in by_id.values()if(t.get('terminal_result')or{}).get('goal',{}).get('asin')),None)
        models={}
        for model,t in by_id.items():
            turns=[]
            for i,step in enumerate(t['steps']):
                obs=step['observation'];fields=[]
                if not step.get('done'):
                    fields=[line for line in obs.splitlines()if line.startswith(('page_type:','query:','asin:','title:','brand:','price:','key_attributes:','selected_options:','available_options:','subpage:','content:'))]
                    if annotation:fields += [line for line in obs.splitlines()if str(annotation)in line and re.match(r'^\d+\|',line)]
                turns.append({'turn':i+1,'tool':step['tool_name'],'args':step['parameters'],'observation_sha256':sha(obs.encode()),
                    'public_field_excerpt':fields,'terminal_observation_excluded':bool(step.get('done'))})
            models[model]={'success_under_original_metric':bool(strict(t)),'original_reward_type':typ(t),'public_query':query(t),'turns':turns}
        result.append({'suite':suite,'task_id':tid,'review':review,'models':models,
            'scope':'Only actual messages.user and nonterminal Actor observation excerpts. Annotated-item search rows locate paths, not semantic truth; no hidden goal options are used as public requirements.'})
    return result


def conclusion_data(tasks,training,groups):
    cs=training['B']['all1_800']['all_update_cohorts'];by_behavior=training['B']['all1_800']['behavior_by_sign']
    effects={}
    for cohort,v in cs.items():
        before=v['batch_normalized_abs_adv'];after=v['weighted_batch_normalized_abs_adv']
        effects[cohort]={'before':before,'after':after,'relative_percent':100*(after/before-1)}
    total_before=sum(v['before']for v in effects.values());total_after=sum(v['after']for v in effects.values())
    negative_totals={}
    for typ in ['repeat_loop','wrong_purchase','max_steps','partial_alternative_purchase']:
        x=[by_behavior[k].get(typ,{})for k in ['F/negative','S/negative']]
        raw=sum(v.get('raw_batch_normalized_abs_adv',0)for v in x);weighted=sum(v.get('weighted_batch_normalized_abs_adv',0)for v in x)
        negative_totals[typ]={'before':raw,'after':weighted,'relative_percent':100*(weighted/raw-1)}
    fs=training['B']['all1_800']['mixed_FS_only_cohorts'];f_tasks={g['task_id']for g in groups if g['kind']=='F'};s_tasks={g['task_id']for g in groups if g['kind']=='S'}
    return {'normalized_advantage_exposure':effects,'F_exposure_share_before':sum(v['before']for k,v in effects.items()if k.startswith('F/'))/total_before,
        'F_exposure_share_after':sum(v['after']for k,v in effects.items()if k.startswith('F/'))/total_after,
        'negative_feedback_proxy_by_behavior':negative_totals,'F_and_S_overlapping_training_tasks':len(f_tasks&s_tasks),
        'persistent_AB_gains':{suite:[r['task_id']for r in rows if r['B_minus_A_pattern500650800']=='+++']for suite,rows in tasks.items()},
        'B500_to_B800_gain_loss':{suite:{'gains':[r['task_id']for r in rows if not r['states']['B500']and r['states']['B800']],
            'losses':[r['task_id']for r in rows if r['states']['B500']and not r['states']['B800']]}for suite,rows in tasks.items()},
        'cohort_measure_scope':'Sum across updates of Actor-token-mean absolute original advantage, before/after logged coefficient. Same collected B batches; no policy ratio, clipping derivative or parameter gradient attribution.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=int);p.add_argument('--suite',default='dev50');p.add_argument('--models',default='A500,A650,A800,B500,B650,B800');a=p.parse_args()
    data,statuses,w,up,A,manifest,diag=load()
    if a.case is not None:
        for model in a.models.split(','):
            rs=data[a.suite].get(model,[]);t=next((t for t in rs if t['task_id']==a.case),None)
            if t is None:continue
            print('MODEL',model,'SUCCESS',bool(strict(t)),'TYPE',typ(t),'QUERY',query(t),'PURCHASE',(t.get('terminal_result')or{}).get('purchase'))
            for i,s in enumerate(t['steps']):
                obs=s['observation'];fields=[l for l in obs.splitlines()if l.startswith(('asin:','title:','price:','selected_options:','available_options:','subpage:','content:'))]
                print('TURN',i+1,s['tool_name'],json.dumps(s['parameters'],ensure_ascii=False),'STATE',' | '.join(fields)[:3500])
        return
    OUT.mkdir(parents=True,exist_ok=True)
    summary,tasks,changes,review=endpoint_analysis(data,statuses);training,tr,gr=training_analysis(w,up,A)
    coverage=generation_coverage(diag,{r['task_id']for rs in tasks.values()for r in rs})
    save('generation_coverage.json',coverage)
    save('source_manifest.json',{'script_sha256':sha(Path(__file__).read_bytes()),'dependency_sha256':{'tools/analyze_b500_dev50_return.py':sha((ROOT/'tools/analyze_b500_dev50_return.py').read_bytes())},'inputs':manifest,'raw_endpoint_trajectories':sum(len(rs)for m in data.values()for rs in m.values()),
        'A500_Final':'Only historical success IDs available; no action-level A500 Final inference','not_used':'legacy4192 trajectories, new sampling, model weights, reward modifications'})
    save('conclusions_data.json',conclusion_data(tasks,training,gr))
    review_path=OUT/'case_reviews.json'
    if review_path.exists():
        reviews=json.loads(review_path.read_text(encoding='utf-8'))
        for suite in data:assert {r['task_id']for r in reviews if r['suite']==suite and r['review_group']=='all_status_changed_tasks'}==set(changes[suite]['review_tasks'])
        for suite,byid in tasks.items():
            expected={r['task_id']for r in byid if not r['states']['A800']and not r['states']['B800']and r['models']['A800']['type']!=r['models']['B800']['type']}
            actual={r['task_id']for r in reviews if r['suite']==suite and r['review_group']=='supplemental_AB800_failure_type_change'}
            assert expected==actual,(suite,expected,actual)
        save('review_evidence.json',public_review_evidence(data,reviews,tasks))
    save('endpoint_summary.json',summary);save('task_matrix.json',tasks);save('changes.json',changes);save('review_index.json',review);save('training_summary.json',training)
    for n,rs in [('training_trajectories.jsonl',tr),('training_groups.jsonl',gr)]:
        (OUT/n).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in rs),encoding='utf-8',newline='\n')
    print(json.dumps({'review_tasks':{s:changes[s]['review_tasks']for s in changes},'B_success_patterns':{s:changes[s]['B_success_patterns']for s in changes},
        'AB_patterns':{s:changes[s]['B_minus_A_patterns']for s in changes},'raw_endpoint_trajectories':sum(len(rs)for m in data.values()for rs in m.values())},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
