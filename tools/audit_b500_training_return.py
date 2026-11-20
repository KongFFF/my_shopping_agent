"""Recompute B500 learning-chain weights and verify returned training evidence."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile


def audit(path):
    errors=[]
    def check(ok,label):
        if not ok:errors.append(label)
    with zipfile.ZipFile(path)as z:
        check(z.testzip()is None,'ZIP CRC')
        j=lambda n:json.loads(z.read(n))
        rows=lambda n:[json.loads(l)for l in z.read(n).splitlines()if l.strip()]
        manifest=json.loads(path.with_name('manifest.json').read_text(encoding='utf-8'))
        sha=hashlib.sha256(path.read_bytes()).hexdigest();check(sha==manifest['archive_sha256'],'Archive SHA256')
        run=j('run.json');done=j('training_complete.json');cpu=j('resume_cpu_acceptance.json')
        package=Path(__file__).resolve().parents[1]/'experiments/group_weight_b500_v1'
        check(run['protocol_sha256']==done['protocol_sha256']==hashlib.sha256((package/'protocol.json').read_bytes()).hexdigest(),'Frozen protocol identity')
        check(run['implementation_manifest_sha256']==hashlib.sha256((package/'implementation_manifest.json').read_bytes()).hexdigest(),'Frozen method manifest identity')
        check(done['completed_updates']==500 and 'stopped.json'not in z.namelist(),'Completion/stopped state')
        check(cpu['passed']and not cpu['gpu_initialized']and not cpu['model_weights_loaded'],'Native CPU acceptance')
        check(cpu['implementation_manifest_sha256']==run['implementation_manifest_sha256'],'Method identity')
        for n,h in run['config_hashes'].items():check(hashlib.sha256(z.read(f'configs/phase{n}.yaml')).hexdigest()==h,'Config '+n)
        updates=[e for e in rows('training_diagnostics.jsonl')if e['event']=='optimizer_step']
        check([e['global_step']for e in updates]==list(range(1,501)),'Continuous1..500')
        returned=lambda n:[r for r in rows(n)if r['event']=='actor_update_returned']
        old=returned('weights_before_rollback.jsonl');new=returned('weights.jsonl')
        check([r['driver_global_step_before_update']for r in old]==list(range(200)),'Original200 update attempts')
        check([r['driver_global_step_before_update']for r in new]==list(range(150,500)),'Resumed350 updates')
        weights=old[:150]+new;check(len(weights)==500,'Weight records500')
        composition=Counter();coeffs=[];tokens=0;F_tokens=0
        for e,w in zip(updates,weights):
            s=str(e['global_step']);m=e['metrics'];gs=w['groups'];ts=w['trajectories']
            check(len(gs)==2 and len(ts)==8,'Batch '+s)
            check(w['composition']==''.join(sorted(g['kind']for g in gs)),'Composition '+s)
            N=sum(t['effective_tokens']for t in ts);nw=sum(t['raw_weight']*t['effective_tokens']for t in ts)
            check(N==w['original_token_count']and math.isclose(nw,w['weighted_token_count']),'Denominator '+s)
            for g in gs:
                members=[t for t in ts if t['uid']==g['uid']];successes=sum(t['success']for t in members)
                raw=.5 if successes==0 else 1.;kind='F'if successes==0 else'S'
                check(len(members)==4 and g['kind']==kind and g['raw_weight']==raw,'Group labels '+s)
                check(successes==sum(bool(r['reward']['purchase_success'])for r in e['rollouts']if r['uid']==g['uid']),'Native success labels '+s)
                check(g['effective_tokens']==sum(t['effective_tokens']for t in members),'Group tokens '+s)
                check(math.isclose(g['coefficient'],raw*N/nw,rel_tol=1e-12),'Coefficient '+s)
                for t in members:check(t['raw_weight']==raw and math.isclose(t['coefficient'],raw*N/nw,rel_tol=1e-12),'Trajectory weight '+s)
            composition[w['composition']]+=1;coeffs.extend(g['coefficient']for g in gs);tokens+=N
            F_tokens+=sum(t['effective_tokens']for t in ts if t['raw_weight']==.5)
            check(math.isclose(m['b500/actor_tokens'],N),'Applied tokens '+s)
            check(math.isclose(m['b500/coefficient_min'],min(g['coefficient']for g in gs),rel_tol=1e-6),'Applied coefficient '+s)
            check(all(math.isfinite(m[k])for k in ('actor/lr','actor/grad_norm','actor/loss')),'Finite metrics '+s)
            check(m['training/optimizer_updated']==1 and m['actor/grad_norm']>0,'Actual update '+s)
            check(math.isclose(m['actor/lr'],1e-6*min(e['global_step']/15,1),rel_tol=1e-5),'LR '+s)
            check(Counter((r['uid'],r['task_id'])for r in e['rollouts'])==Counter((t['uid'],next(g['task_id']for g in gs if g['uid']==t['uid']))for t in ts),'Batch identity '+s)
        curves=[{'step':e['global_step'],'mean_reward':e['metrics']['val-core/shopsimulator/reward/mean@1']}for e in updates if 'val-core/shopsimulator/reward/mean@1'in e['metrics']]
        text=z.read('prior_attempt__train_to_350.log').decode(errors='replace')
        match=re.search(r'step:0 - val-core/shopsimulator/reward/mean@1:np.float64\(([-\d.eE]+)\)',text)
        if match:curves.insert(0,{'step':0,'mean_reward':float(match[1])})
        check([x['step']for x in curves]==list(range(0,501,50)),'Internal validation points')
        gen=rows('generation_tokens.jsonl');counts={'train':sum(r['reserved_trajectories']for r in gen if not r['validation']),'validation_physical':sum(r['reserved_trajectories']for r in gen if r['validation'])}
        check(counts==j('generation_budget.json')==done['generation_budget'],'Generation counters')
        retention=rows('checkpoint_retention.jsonl');plans=[r for r in retention if r['event']=='retention_plan']
        check(plans[-1]['keep']==[350,450,500]and plans[-1]['new_step']==500,'Retention final plan')
        check(not any(r.get('step')==350 for r in retention if r['event']=='checkpoint_removed'),'350 protected')
        restore={str(s):f'Setting global step to {s}'in z.read(n).decode(errors='replace')for n,s in [('train_to_350.log',150),('train_to_500.log',350)]}
        check(all(restore.values()),'Restored150/350')
        historical=path.parents[3]/'analysis/online_training_window_return_20261116/evidence.zip'
        A500=None
        if historical.exists():
            with zipfile.ZipFile(historical)as az:
                A500=next(json.loads(l)['metrics']['val-core/shopsimulator/reward/mean@1']for l in az.read('reference351_500/training_diagnostics.jsonl').splitlines()if json.loads(l).get('event')=='optimizer_step'and json.loads(l)['global_step']==500)
        return {'passed':not errors,'errors':errors,'archive_sha256':sha,'completed_updates':len(updates),'native_cpu_tests':cpu['tests_run'],
            'restored_steps':restore,'composition':dict(composition),'relative_reweight_fraction':composition['FS']/500,
            'coefficient_range':[min(coeffs),max(coeffs)],'committed_Actor_tokens':tokens,'F_Actor_token_fraction':F_tokens/tokens,
            'generation_budget':counts,'rolled_back_original_steps':[151,200],'internal_mean_reward_curve':curves,
            'retained_checkpoints_by_log':plans[-1]['keep'],'new_validation_trajectories_in_archive':any(n.startswith('validation/')for n in z.namelist()),
            'independent_Dev_done':done['independent_Dev_done'],'Final_done':done['Final_done'],'new_supervisor_elapsed_hours':done['elapsed_seconds']/3600,
            'historical_A500_internal_reward':A500,'B500_minus_A500_internal_reward':curves[-1]['mean_reward']-A500 if A500 is not None else None}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.archive);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(0 if result['passed']else 1)
