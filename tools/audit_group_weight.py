"""Static, CPU-only audit of fail-group weighting on historical optimizer batches."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import statistics as st
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/group_weight_audit_20261117'
def dump(x):return json.dumps(x,ensure_ascii=False,sort_keys=True)
def sha(b):return hashlib.sha256(b).hexdigest()
def fp(r):return dump({k:v for k,v in r.items()if k!='rollout_index'})
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sign(a):return 'positive'if a>1e-7 else 'negative'if a<-1e-7 else 'zero'
def desc(v):
    v=sorted(v)
    return {'n':len(v),'mean':st.fmean(v)if v else None,'median':st.median(v)if v else None,
        'p90_nearest_rank':v[max(0,(__import__('math').ceil(.9*len(v))-1))]if v else None,'sum':sum(v)}


def main():
    old=json.loads((ROOT/'analysis/grpo_structural_analysis_20261116/manifest.json').read_text(encoding='utf-8'))
    for n,h in old['inputs_sha256'].items():assert sha((ROOT/n).read_bytes())==h,n
    protocol=sha((OUT/'核查口径.md').read_bytes())
    summary={};update_rows=[];group_rows=[];trace_rows=[]
    with zipfile.ZipFile(ROOT/'analysis/online_training_window_return_20261116/evidence.zip')as z:
        for label in ('reference351_500','late651_800'):
            es=list(map(json.loads,z.read(label+'/training_diagnostics.jsonl').splitlines()));generated=defaultdict(list)
            for e in es:
                if e['event']=='generation_batch':
                    for r in e['rollouts']:generated[r['uid']].append(r)
            local_u=[];local_g=[];local_t=[]
            for e in es:
                if e['event']!='optimizer_step':continue
                batch=defaultdict(list)
                for r in e['rollouts']:batch[r['uid']].append(r)
                assert len(batch)==2 and len(e['rollouts'])==8
                assert e['metrics']['training/optimizer_updated']==1
                gs=[];advs=[]
                for uid,rs in batch.items():
                    assert len(rs)==4 and Counter(map(fp,rs))==Counter(map(fp,generated[uid]))
                    assert all(r['reward_valid']and not r['reward'].get('sampling_invalid')for r in rs)
                    nsuccess=sum(bool(r['reward']['purchase_success'])for r in rs);kind='S'if nsuccess else 'F'
                    rewards=[r['reward']['terminal_utility']for r in rs];mean=st.fmean(rewards)
                    aa=[x-mean for x in rewards];assert abs(sum(aa))<1e-6
                    g={'window':label,'step':e['global_step'],'uid':uid,'task_id':rs[0]['task_id'],'kind':kind,
                        'successes':nsuccess,'rewards':rewards,'mean_reward':mean,'reward_range':max(rewards)-min(rewards),
                        'mean_abs_adv':st.fmean(abs(a)for a in aa),'sum_abs_adv':sum(abs(a)for a in aa),
                        'sum_square_adv':sum(a*a for a in aa),'tool_steps':sum(r['steps']for r in rs),
                        'step_weighted_abs_adv':sum(r['steps']*abs(a)for r,a in zip(rs,aa))}
                    gs.append(g);local_g.append(g);advs+=aa
                    for r,a in zip(rs,aa):
                        local_t.append({'window':label,'step':e['global_step'],'uid':uid,'task_id':r['task_id'],'kind':kind,
                            'reward_type':r['reward_type'],'success':bool(r['reward']['purchase_success']),
                            'advantage':a,'sign':sign(a),'tool_steps':r['steps']})
                assert abs(min(advs)-e['metrics']['critic/advantages/min'])<1e-6
                assert abs(max(advs)-e['metrics']['critic/advantages/max'])<1e-6
                nf=sum(g['kind']=='F'for g in gs);composition='FF'if nf==2 else 'FS'if nf==1 else 'SS'
                mean_w=(.5*nf+2-nf)/2
                u={'window':label,'step':e['global_step'],'composition':composition,'group_ids':list(batch),
                    'unnormalized_coefficients':{g['uid']:.5 if g['kind']=='F'else 1. for g in gs},
                    'group_normalized_coefficients':{g['uid']:(.5 if g['kind']=='F'else 1.)/mean_w for g in gs},
                    'raw_abs_adv_proxy':sum(g['sum_abs_adv']for g in gs),
                    'normalized_abs_adv_proxy':sum(g['sum_abs_adv']*(.5 if g['kind']=='F'else 1.)/mean_w for g in gs)}
                local_u.append(u)
            assert len(local_u)==150 and len(local_g)==300 and len(local_t)==1200
            fs_steps={u['step']for u in local_u if u['composition']=='FS'}
            def cohort(kind, mixed_only=False):
                gg=[g for g in local_g if g['kind']==kind and (not mixed_only or g['step']in fs_steps)]
                tt=[r for r in local_t if r['kind']==kind and (not mixed_only or r['step']in fs_steps)]
                return {'groups':len(gg),'tasks':len({g['task_id']for g in gg}),'trajectories':len(tt),
                    'group_mean_abs_adv':desc([g['mean_abs_adv']for g in gg]),
                    'group_reward_range':desc([g['reward_range']for g in gg]),
                    'trace_tool_steps':desc([r['tool_steps']for r in tt]),
                    'sum_abs_adv':sum(g['sum_abs_adv']for g in gg),'sum_square_adv':sum(g['sum_square_adv']for g in gg),
                    'step_weighted_abs_adv':sum(g['step_weighted_abs_adv']for g in gg),
                    'sign_by_behavior':{s:dict(Counter(r['reward_type']for r in tt if r['sign']==s))for s in ('positive','negative','zero')},
                    'abs_adv_by_behavior':{s:{typ:sum(abs(r['advantage'])for r in tt if r['sign']==s and r['reward_type']==typ)
                        for typ in sorted({r['reward_type']for r in tt if r['sign']==s})}for s in ('positive','negative','zero')}}
            counts=Counter(u['composition']for u in local_u);f=cohort('F');s=cohort('S')
            summary[label]={'updates':len(local_u),'composition':dict(counts),'F':f,'S':s,
                'mixed_update_F':cohort('F',True),'mixed_update_S':cohort('S',True),
                'mixed_update_all_tasks':len({g['task_id']for g in local_g if g['step']in fs_steps}),
                'unnormalized_changed_updates':counts['FF']+counts['FS'],
                'unnormalized_pure_scalar_updates':counts['FF'],
                'normalized_changed_updates':counts['FS'],'normalized_unchanged_updates':counts['FF']+counts['SS'],
                'F_groups_in_mixed_updates':counts['FS'],'F_groups_in_FF_updates':2*counts['FF'],
                'fail_group_abs_adv_proxy_share':f['sum_abs_adv']/(f['sum_abs_adv']+s['sum_abs_adv']),
                'fail_group_square_adv_proxy_share':f['sum_square_adv']/(f['sum_square_adv']+s['sum_square_adv']),
                'fail_group_step_weighted_abs_adv_proxy_share':f['step_weighted_abs_adv']/(f['step_weighted_abs_adv']+s['step_weighted_abs_adv']),
                'group_normalized_total_abs_adv_proxy_ratio':sum(u['normalized_abs_adv_proxy']for u in local_u)/sum(u['raw_abs_adv_proxy']for u in local_u)}
            update_rows+=local_u;group_rows+=local_g;trace_rows+=local_t
    # Algebraic checks independent of historical sample outcomes.
    assert .5/((.5+.5)/2)==1 and .5/((.5+1)/2)==2/3 and 1/((.5+1)/2)==4/3
    assert sha((OUT/'核查口径.md').read_bytes())==protocol
    save('summary.json',summary);save('updates.json',update_rows);save('groups.json',group_rows);save('trajectories.json',trace_rows)
    save('manifest.json',{'protocol_sha256':protocol,'script_sha256':sha(Path(__file__).read_bytes()),
        'upstream_manifest_sha256':sha((ROOT/'analysis/grpo_structural_analysis_20261116/manifest.json').read_bytes()),
        'outputs_sha256':{n:sha((OUT/n).read_bytes())for n in ('summary.json','updates.json','groups.json','trajectories.json')}})
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
