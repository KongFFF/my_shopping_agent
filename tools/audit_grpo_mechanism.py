"""Reconcile actual online updates and audit a bounded public-budget subset on CPU."""
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import zipfile
from public_decision_rules import requirements

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/grpo_mechanism_audit_20261116'
ARCHIVE=ROOT/'analysis/online_training_window_return_20261116/evidence.zip'
CONTRACTS=ROOT/'analysis/semantic_pilot30_evidence_20261109/contracts1000.jsonl'


def digest(x):return hashlib.sha256(x).hexdigest()
def dump(x):return json.dumps(x,ensure_ascii=False,sort_keys=True)
def fingerprint(r):
    # rollout_index is regenerated after length balancing; it is not stable
    # between generation and optimizer diagnostics.
    return digest(dump({k:v for k,v in r.items()if k!='rollout_index'}).encode())
def action_key(r):return digest(dump({'task_id':r['task_id'],'actions':r['actions']}).encode())
def reindexed_content_count(generated,consumed):
    original={r['rollout_index']:fingerprint(r)for r in generated}
    return sum(original[r['rollout_index']]!=fingerprint(r)for r in consumed)
def scalar_advantages(rewards):
    mean=statistics.fmean(rewards)
    return [r-mean for r in rewards]
def sign(x):return 'positive'if x>1e-7 else 'negative'if x<-1e-7 else 'zero'
def save(name,x):
    (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for name,h in json.loads((OUT/'source_snapshot_manifest.json').read_text(encoding='utf-8'))['files'].items():
        assert digest((OUT/'sources'/name).read_bytes())==h
    package=json.loads((OUT/'package_reference/manifest.json').read_text(encoding='utf-8'))
    for name,h in package['files_sha256'].items():
        assert digest((OUT/'package_reference'/name).read_bytes())==h
    parser_path=OUT/'project_reference/constraints.py'
    spec=importlib.util.spec_from_file_location('frozen_budget_parser',parser_path)
    parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
    assert digest(parser_path.read_bytes())=='b4155708dd7006ae4afe42c535ee4d2abe138585fa79ce4631faa03370dc0ae5'
    contracts=[json.loads(l)for l in CONTRACTS.read_text(encoding='utf-8-sig').splitlines()if l.strip()]
    assert len(contracts)==len({r['task_id']for r in contracts})==1000
    budget_rows=[]
    for r in contracts:
        for req in requirements(r):
            if req['kind']!='explicit_budget':continue
            actual=parser.explicit_budget_from_instruction(r['query']);expected=float(req['expected']['value'])
            status='missing'if actual is None else 'equal'if abs(actual-expected)<1e-7 else 'different'
            budget_rows.append({'task_id':r['task_id'],'query':r['query'],'requirement':req,'native_upper':actual,'status':status})
    # Two simultaneous ceilings on6883:3900 is correctly tighter than4000.
    different=[r for r in budget_rows if r['status']=='different']
    assert all(r['task_id']==6883 and r['native_upper']==3900 for r in different)
    missing=[r for r in budget_rows if r['status']=='missing']
    flags=json.loads((ROOT/'analysis/online_window_audit_20261116/screen_flags.json').read_text(encoding='utf-8'))
    window_summaries={};flag_rows=[];examples=[];sources=[]
    with zipfile.ZipFile(ARCHIVE)as z:
        for label in ('reference351_500','late651_800'):
            es=[json.loads(l)for l in z.read(label+'/training_diagnostics.jsonl').splitlines()]
            generations=[e for e in es if e['event']=='generation_batch'];updates=[e for e in es if e['event']=='optimizer_step']
            generated={};groupmeta={};generated_lookup={};update_groups={};upstep={};reindexed=0
            for e in generations:
                by=defaultdict(list)
                for r in e['rollouts']:
                    by[r['uid']].append(r);generated_lookup[r['uid'],r['rollout_index']]=r
                for uid,rs in by.items():
                    assert uid not in generated and len(rs)==4
                    generated[uid]=rs
                for g in e['groups']:groupmeta[g['uid']]=g
            stats=Counter();metrics_check=[]
            for e in updates:
                assert e['metrics']['training/optimizer_updated']==1
                by=defaultdict(list)
                for r in e['rollouts']:by[r['uid']].append(r)
                assert len(by)==2 and len(e['rollouts'])==8
                all_adv=[]
                for uid,rs in by.items():
                    assert uid not in update_groups and len(rs)==4
                    assert Counter(map(fingerprint,rs))==Counter(map(fingerprint,generated[uid]))
                    assert groupmeta[uid]['kept']and groupmeta[uid]['accepted']
                    update_groups[uid]=rs;upstep[uid]=e['global_step']
                    reindexed+=reindexed_content_count(generated[uid],rs)
                    values=[r['reward']['terminal_utility']for r in rs];advs=scalar_advantages(values);all_adv+=advs
                    stats['groups']+=1;stats['groups_without_purchase_success']+=not any(r['reward']['purchase_success']for r in rs)
                    for r,a in zip(rs,advs):
                        stats['rollouts']+=1;stats['positive_'+r['reward_type']]+=a>1e-7
                        if not r['reward']['purchase_success']and a>1e-7:stats['positive_non_success']+=1
                        if r['reward_type']in ('repeat_loop','partial_alternative_purchase')and a>0.15 and sum(x['window']==label for x in examples)<8:
                            examples.append({'window':label,'step':e['global_step'],'task_id':r['task_id'],'uid':uid,
                                'reward_type':r['reward_type'],'reward':r['reward']['terminal_utility'],'group_rewards':values,'advantage':a})
                metrics=e['metrics'];good=abs(min(all_adv)-metrics['critic/advantages/min'])<1e-6 and abs(max(all_adv)-metrics['critic/advantages/max'])<1e-6
                assert good,e['global_step'];metrics_check.append(good)
                assert math.isfinite(metrics['actor/grad_norm'])and metrics['actor/grad_norm']>0
                assert metrics['actor/lr']==1e-6
            for f in flags:
                if f['window']!=label:continue
                r=generated_lookup[f['uid'],f['rollout_index']]
                assert action_key(r)==f['key']
                if f['uid']in update_groups:
                    assert f['consumed']
                    peers=update_groups[f['uid']]
                    # Match immutable contents, not the renumbered ordinal.
                    matched=[q for q in peers if fingerprint(q)==fingerprint(r)];assert matched
                    mean=statistics.fmean(q['reward']['terminal_utility']for q in peers)
                    adv=r['reward']['terminal_utility']-mean
                else:assert not f['consumed'];mean=adv=None
                upper=parser.explicit_budget_from_instruction(f['query'])if f['kind']=='explicit_budget'else None
                flag_rows.append({**{k:f[k]for k in ('window','step','uid','rollout_index','key','task_id','kind','consumed','query')},
                    'reward_type':r['reward_type'],'reward':r['reward']['terminal_utility'],
                    'group_mean':mean,'advantage':adv,'advantage_sign':sign(adv)if adv is not None else None,
                    'native_budget_upper':upper,'state':f['state'],'requirement':f.get('requirement')})
            consumed_flags=[f for f in flag_rows if f['window']==label and f['consumed']]
            drop=Counter((g.get('drop_reason')or'kept')for g in groupmeta.values())
            window_summaries[label]={'generated_groups':len(generated),'optimizer_updates':len(updates),
                'consumed_groups':len(update_groups),'reindexed_rollouts':reindexed,'advantage_extrema_matched_updates':sum(metrics_check),
                'stats':dict(stats),'group_drop_reasons':dict(drop),
                'consumed_flag_signs':dict(Counter(f['kind']+'/'+f['advantage_sign']for f in consumed_flags)),
                'explicit_budget_flags':sum(f['kind']=='explicit_budget'for f in flag_rows if f['window']==label),
                'missing_parser_budget_flags':sum(f['kind']=='explicit_budget'and f['native_budget_upper']is None for f in flag_rows if f['window']==label)}
            for s in json.loads(z.read(label+'/source_checks.json')):
                sources.append({'window':label,'path':s['path'],'expected_sha256':s['expected_sha256'],'archived':bool(s.get('archive_path'))})
    summary={'schema':'grpo-mechanism-audit-v1','inputs':{
        'online_windows_sha256':digest(ARCHIVE.read_bytes()),'contracts1000_sha256':digest(CONTRACTS.read_bytes()),
        'budget_parser_sha256':digest(parser_path.read_bytes()),'legacy4192_used':False,
        'package_reference_manifest':json.loads((OUT/'package_reference/manifest.json').read_text(encoding='utf-8'))},
        'budget_subset':{'requirements':len(budget_rows),'tasks':len({r['task_id']for r in budget_rows}),
            'same_numeric_bound':sum(r['status']=='equal'for r in budget_rows),'missing_requirements':len(missing),
            'missing_tasks':len({r['task_id']for r in missing}),'benign_tighter_other_bound':different},
        'windows':window_summaries,'relative_advantage_examples':examples,
        'limits':['Extrema matching reconstructs scalar group advantages, not per-token gradient attribution.',
            'Approved strict-budget screening is a bounded subset, not complete semantic ground truth.',
            'rollout_index changes under length balancing; consumption must use contents and group UID.',
            'Installed token-mask construction/core_algos were not in the returned launch binding; audit remains incomplete on those bytes.']}
    save('audit.json',summary);save('flag_advantages.json',flag_rows);save('budget_parser_candidates.json',missing)
    save('source_binding_index.json',sources)
    native=OUT/'native_case_replays.json'
    if native.exists():
        enriched=json.loads(native.read_text(encoding='utf-8'))
        for r in enriched['cases']:
            f=next(f for f in flag_rows if f['window']=='late651_800'and f['uid']==r['uid']and f['rollout_index']==r['generation_rollout_index'])
            r['reconstructed_scalar_advantage']=f['advantage'];r['advantage_sign']=f['advantage_sign']
        save('native_case_replays_with_advantages.json',enriched)
    print(json.dumps({'budget_tasks_screened':summary['budget_subset']['tasks'],
        'budget_tasks_missing':summary['budget_subset']['missing_tasks'],
        'window_summary':{k:{n:v[n]for n in ('optimizer_updates','consumed_groups','advantage_extrema_matched_updates','consumed_flag_signs')}for k,v in window_summaries.items()}},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
