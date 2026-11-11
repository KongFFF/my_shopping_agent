"""Task-paired analysis of the existing 500-step GRPO logs; CPU/files only."""
import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'analysis/grpo_baseline_20261022_202816'
OUT = ROOT/'analysis/grpo_learnability_20261111'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(data): return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))


def convert(raw):
    r = dict(raw)
    for k in ('step','task_id','gold_rollouts','group_size'):
        r[k] = int(r[k])
    for k in ('kept','accepted','trained'):
        if r[k] not in ('True','False'): raise ValueError('Invalid boolean: '+k)
        r[k] = r[k]=='True'
    for k in ('utility_min','utility_max','utility_std','centered_utility_mean_abs','mean_steps'):
        r[k] = float(r[k])
    return r


def period(rs, low, high, valid_only=False):
    z = [r for r in rs if low<=r['step']<=high and (not valid_only or r['drop_reason']!='sampling_invalid')]
    n = sum(r['group_size'] for r in z)
    return {'groups':len(z),'rollouts':n,'gold':sum(r['gold_rollouts'] for r in z),
            'p':sum(r['gold_rollouts'] for r in z)/n if n else None,
            'invalid_groups':sum(r['drop_reason']=='sampling_invalid' for r in z),
            'trained_groups':sum(r['trained'] for r in z),
            'mixed_gold_groups':sum(0<r['gold_rollouts']<r['group_size'] for r in z),
            'mean_steps':statistics.fmean(r['mean_steps'] for r in z) if z else None}


def early_mixed(rs):
    # Membership is determined exclusively before observing steps 101..500.
    return any(r['step']<=100 and r['kept'] and 0<r['gold_rollouts']<r['group_size'] for r in rs)


def describe(items, bootstrap=5000):
    if not items: return {'tasks':0}
    early = [a['p'] for a,b in items]
    late = [b['p'] for a,b in items]
    delta = [b-a for a,b in zip(early,late)]
    rng = random.Random(20261111)
    draws = sorted(statistics.fmean(rng.choices(delta,k=len(delta))) for _ in range(bootstrap))
    return {'tasks':len(items),'early_task_mean':statistics.fmean(early),
            'late_task_mean':statistics.fmean(late),'delta_task_mean':statistics.fmean(delta),
            'delta_task_median':statistics.median(delta),
            'delta_task_bootstrap95':[draws[int(.025*bootstrap)],draws[min(bootstrap-1,int(.975*bootstrap))]],
            'improved':sum(d>1e-12 for d in delta),'unchanged':sum(abs(d)<=1e-12 for d in delta),
            'regressed':sum(d< -1e-12 for d in delta),
            'early_rollouts':sum(a['rollouts'] for a,b in items),'late_rollouts':sum(b['rollouts'] for a,b in items),
            'early_gold':sum(a['gold'] for a,b in items),'late_gold':sum(b['gold'] for a,b in items),
            'early_invalid_groups':sum(a['invalid_groups'] for a,b in items),
            'late_invalid_groups':sum(b['invalid_groups'] for a,b in items),
            'early_rollout_rate':sum(a['gold'] for a,b in items)/sum(a['rollouts'] for a,b in items),
            'late_rollout_rate':sum(b['gold'] for a,b in items)/sum(b['rollouts'] for a,b in items)}


def selection_permutation(by, trials=5000):
    """Shuffle whole groups within tasks and reselect the early-mixed cohort.

    Diagnostic null only: groups must be exchangeable under no temporal change.
    Reselecting is essential; keeping the observed selected cohort is biased.
    """
    relevant=[]
    for rs in by.values():
        e=[r for r in rs if r['step']<=100]
        l=[r for r in rs if r['step']>=401]
        if e and l and any(r['kept'] and 0<r['gold_rollouts']<r['group_size'] for r in e+l):
            relevant.append((e+l,len(e)))
    rng=random.Random(20261112);stats=[];sizes=[];all_stats=[]
    all_paired=[]
    for rs in by.values():
        e=[r for r in rs if r['step']<=100];l=[r for r in rs if r['step']>=401]
        if e and l: all_paired.append((e+l,len(e)))
    for _ in range(trials):
        changes=[];whole=[]
        for pool,ne in all_paired:
            shuffled=rng.sample(pool,len(pool));a,b=shuffled[:ne],shuffled[ne:]
            d=sum(r['gold_rollouts'] for r in b)/sum(r['group_size'] for r in b)-sum(r['gold_rollouts'] for r in a)/sum(r['group_size'] for r in a)
            whole.append(d)
            if any(r['kept'] and 0<r['gold_rollouts']<r['group_size'] for r in a):changes.append(d)
        stats.append(statistics.fmean(changes));sizes.append(len(changes));all_stats.append(statistics.fmean(whole))
    observed=[];observed_all=[]
    for rs in by.values():
        e,l=period(rs,1,100),period(rs,401,500)
        if e['rollouts'] and l['rollouts']:
            observed_all.append(l['p']-e['p'])
            if early_mixed(rs):observed.append(l['p']-e['p'])
    def summary(values,obs):
        z=sorted(values)
        return {'observed':obs,'null_mean':statistics.fmean(z),
                'null95':[z[int(.025*trials)],z[min(trials-1,int(.975*trials))]],
                'one_sided_ge_observed':(1+sum(v>=obs-1e-12 for v in z))/(trials+1)}
    return {'trials':trials,'seed':20261112,'early_mixed_reselected':summary(stats,statistics.fmean(observed)),
            'all_paired_unselected':summary(all_stats,statistics.fmean(observed_all)),
            'null_selected_tasks_mean':statistics.fmean(sizes),
            'limit':'Exploratory within-task whole-group exchangeability null, not a causal test or training-seed uncertainty.'}


def run(output=OUT, bootstrap=5000):
    output.mkdir(parents=True, exist_ok=True)
    manifest=json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))
    receipts={}
    for name in ('training_diagnosis/groups.csv','training_diagnosis/tasks.csv','training_diagnosis/updates.csv'):
        raw=(SOURCE/name).read_bytes();expected=manifest['files'][name]['sha256']
        assert sha(raw)==expected or sha(raw.replace(b'\r\n',b'\n'))==expected, name
        receipts[name]=sha(raw)
    gs=[convert(r) for r in read_csv(SOURCE/'training_diagnosis/groups.csv')]
    assert len(gs)==4296 and sum(r['trained'] for r in gs)==1000
    assert len({(r['source'],r['uid']) for r in gs})==len(gs)
    assert {int(r['step']) for r in read_csv(SOURCE/'training_diagnosis/updates.csv')}==set(range(1,501))
    by=defaultdict(list)
    for r in gs: by[r['task_id']].append(r)
    assert len(by)==1000 and all(r['group_size']==4 for r in gs)
    # Cross-check generated/trained accounting against the independently saved task table.
    for t in read_csv(SOURCE/'training_diagnosis/tasks.csv'):
        rs=by[int(t['task_id'])]
        assert len(rs)==int(t['generated_groups']) and sum(r['trained'] for r in rs)==int(t['trained_groups'])
        assert sum(r['gold_rollouts'] for r in rs)==int(t['generated_gold_rollouts'])
    audit_path=SOURCE/'derived/gold_annotation_audit.json'
    audit={r['task_id']:r for r in json.loads(audit_path.read_text(encoding='utf-8')) if r['split']=='train'}
    assert set(audit)==set(by)
    feasible={t for t,r in audit.items() if r['reward_type']=='gold_purchase' or r['variant_search'].get('gold_found')}
    assert len(feasible)==910
    early_ids={t for t,rs in by.items() if period(rs,1,100)['rollouts']}
    late_ids={t for t,rs in by.items() if period(rs,401,500)['rollouts']}
    mixed={t for t,rs in by.items() if early_mixed(rs)}
    cohorts={'all_tasks':set(by),'static_gold_feasible':feasible,'static_gold_unavailable':set(by)-feasible,
             'early_valid_mixed':mixed,
             'early_all_observed_gold':{t for t in early_ids if period(by[t],1,100)['p']==1},
             'early_no_observed_gold':{t for t in early_ids if period(by[t],1,100)['p']==0},
             'early_between_0_and_1':{t for t in early_ids if 0<period(by[t],1,100)['p']<1}}
    summaries={}
    for label,ids in cohorts.items():
        pairs=[(period(by[t],1,100),period(by[t],401,500)) for t in sorted(ids & early_ids & late_ids)]
        summaries[label]={'eligible_tasks':len(ids),**describe(pairs,bootstrap)}
    sensitivities={}
    for name,e,l,valid_only in [('primary_valid_groups_only',(1,100),(401,500),True),
                               ('wider150',(1,150),(351,500),False),
                               ('wider200',(1,200),(301,500),False)]:
        pairs=[]
        for t in sorted(mixed):
            a,b=period(by[t],*e,valid_only),period(by[t],*l,valid_only)
            if a['rollouts'] and b['rollouts']:pairs.append((a,b))
        sensitivities[name]=describe(pairs,bootstrap)
    doses={}
    for dose in ('0','1','2','3+'):
        pairs=[];tids=[]
        for t in sorted(mixed & early_ids & late_ids):
            n=sum(r['trained'] and 101<=r['step']<=400 for r in by[t])
            tag=str(n) if n<3 else '3+'
            if tag==dose:
                tids.append(t);pairs.append((period(by[t],1,100),period(by[t],401,500)))
        doses[dose]={'task_ids':tids,**describe(pairs,bootstrap)}
    details=[]
    for t,rs in sorted(by.items()):
        e,l=period(rs,1,100),period(rs,401,500)
        n=sum(r['trained'] and 101<=r['step']<=400 for r in rs)
        details.append({'task_id':t,'query':audit[t]['query'],'static_gold_feasible':t in feasible,
            'early_valid_mixed':t in mixed,'paired':bool(e['rollouts'] and l['rollouts']),
            'early':e,'late':l,'delta':l['p']-e['p'] if e['p'] is not None and l['p'] is not None else None,
            'trained_groups_between_windows':n,'total_trained_groups':sum(r['trained'] for r in rs),
            'first_observed_step':min(r['step'] for r in rs),'last_observed_step':max(r['step'] for r in rs)})
    result={'version':'grpo-existing-run-learnability-v1','windows':{'early':[1,100],'late':[401,500]},
            'cohort_rule':'early_valid_mixed uses only kept mixed-gold groups in steps 1..100',
            'success_definition':'gold_rollouts from original diagnostic gold() predicate; not newly judged semantic success',
            'generated_groups':len(gs),'generated_rollouts':sum(r['group_size'] for r in gs),
            'trained_groups':sum(r['trained'] for r in gs),'all_tasks':len(by),'early_tasks':len(early_ids),
            'late_tasks':len(late_ids),'paired_tasks':len(early_ids & late_ids),
            'primary_cohort_missing_late_ids':sorted(mixed-late_ids),
            'cohorts':summaries,'sensitivity_same_early_cohort':sensitivities,
            'selection_aware_permutation':selection_permutation(by,bootstrap),
            'middle_update_dose_early_mixed':doses,
            'sources_sha256':{**receipts,'derived/gold_annotation_audit.json':sha(audit_path.read_bytes())},
            'uncertainty':'Task-paired percentile bootstrap, 5000 resamples, seed 20261111; exploratory single-run uncertainty, not training-seed/causal uncertainty',
            'limits':['Only gold counts, utility range and validity proxies are available per group; mean utility and detailed failure types cannot be reconstructed from this CSV.',
                      'Early/late windows contain changing policies, not exact SFT/step500 reevaluations.',
                      'No-update tasks can improve via shared parameters; update dose is endogenous, not randomized.',
                      'Early cohort selection is noisy and can cause regression to the mean; missing late tasks are not failures.']}
    save(output/'summary.json',result);save(output/'task_details.json',details)
    with (output/'task_table.csv').open('w',encoding='utf-8-sig',newline='') as f:
        names=['task_id','static_gold_feasible','early_valid_mixed','paired','early_rollouts','early_gold','early_p',
               'late_rollouts','late_gold','late_p','delta','trained_groups_between_windows','total_trained_groups']
        w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        for d in details:
            z={k:d[k] for k in ('task_id','static_gold_feasible','early_valid_mixed','paired','delta','trained_groups_between_windows','total_trained_groups')}
            z.update({prefix+'_'+k:d[prefix][k] for prefix in ('early','late') for k in ('rollouts','gold','p')});w.writerow(z)
    print(json.dumps({'cohorts':summaries,'sensitivities':sensitivities,'selection_permutation':result['selection_aware_permutation'], 'dose':{k:{kk:vv for kk,vv in v.items() if kk!='task_ids'} for k,v in doses.items()}},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT)
    a=p.parse_args();run(a.output)
