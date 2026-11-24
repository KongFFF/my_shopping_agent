"""CPU-only C500 returned-log audit; no tensor loading or model evaluation."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile
import yaml

ROOT=Path(__file__).resolve().parents[1]
def sha(data):return hashlib.sha256(data).hexdigest()

def audit(path):
    errors=[];gaps=[]
    def check(ok,label):
        if not ok:errors.append(label)
    with zipfile.ZipFile(path)as z:
        check(z.testzip()is None,'Archive CRC')
        j=lambda n:json.loads(z.read(n))
        rows=lambda n:[json.loads(l)for l in z.read(n).splitlines()if l.strip()]
        meta=json.loads(path.with_name('manifest.json').read_text(encoding='utf-8'))
        check(sha(path.read_bytes())==meta['archive_sha256'],'Archive SHA')
        run=j('run.json');done=j('training_complete.json');pkg=ROOT/'experiments/group_weight_c500_v1'
        check(run['protocol_sha256']==done['protocol_sha256']==meta['protocol_sha256']==sha((pkg/'protocol.json').read_bytes()),'Frozen protocol')
        check(run['implementation_manifest_sha256']==sha((pkg/'implementation_manifest.json').read_bytes()),'Frozen implementation')
        check(run['experiment']==done['experiment']=='C500'and run['gamma']==done['gamma']==4,'C gamma4 identity')
        check(done['completed_updates']==done['new_updates']==500 and 'stopped.json'not in z.namelist(),'Completion state')
        configs={n:yaml.safe_load(z.read('configs/phase'+n+'.yaml'))for n in ['350','500']}
        from c500_run import remap
        frozen=remap(json.loads((pkg/'historical_a_composed_config.json').read_text(encoding='utf-8')),Path(run['project']),Path(run['runtime']))
        frozen['actor_rollout_ref']['rollout']['trace']['experiment_name']='c500-gamma4-v1'
        # Operational paths and worker hook differ; all learning/sampling settings must match B/A.
        a=configs['350'];b=configs['500']
        check(a['trainer']['total_training_steps']==b['trainer']['total_training_steps']==500,'Total steps')
        check(a['trainer']['resume_mode']=='disable'and a['trainer']['resume_from_path']is None,'Fresh SFT start')
        check(b['trainer']['resume_mode']=='resume_path'and b['trainer']['resume_from_path']==run['output']+'/train/global_step_350'and b['trainer']['val_before_train']is False,'350 full restore config')
        for name,c in configs.items():
            check(sha(z.read('configs/phase'+name+'.yaml'))==run['config_hashes'][name],'Config hash '+name)
            for key in ['actor_rollout_ref','data','algorithm','shopping_dynamic_sampling']:
                check(c[key]==frozen[key],'Original learning config '+name+' '+key)
            operational={'default_local_dir','validation_data_dir','experiment_name','resume_mode','resume_from_path','val_before_train'}
            check({k:v for k,v in c['trainer'].items()if k not in operational}=={k:v for k,v in frozen['trainer'].items()if k not in operational},'Trainer learning contract '+name)
            hook=c['ray_kwargs']['ray_init']['runtime_env']
            check(hook['worker_process_setup_hook']=='c500_run_control.setup_worker'and hook['env_vars']==run['env'],'C hook and env '+name)
        check(run['env']['GRPO_MODEL_PATH']==json.loads((pkg/'protocol.json').read_text())['baseline']['actor_init'],'SFT model path')
        ds=rows('training_diagnostics.jsonl');updates=[e for e in ds if e['event']=='optimizer_step']
        check([e['global_step']for e in updates]==list(range(1,501)),'Updates exactly1..500')
        weights=[w for w in rows('weights.jsonl')if w['event']=='actor_update_returned']
        allweights=rows('weights.jsonl');check(Counter(w['event']for w in allweights)=={'actor_update_started':500,'actor_update_returned':500},'Actor calls without failures')
        check([w['driver_global_step_before_update']for w in weights]==list(range(500)),'Actor log chain')
        comps=Counter();tokens=0;max_adv_error=0.;lo=[]
        for e,w in zip(updates,weights):
            step=e['global_step'];m=e['metrics'];ts=w['trajectories'];gs=w['groups']
            check(w['gamma']==4 and w['experiment']=='C500','Logged gamma '+str(step))
            check(len(ts)==8 and len(gs)==2,'Batch shape '+str(step))
            N=sum(t['effective_tokens']for t in ts);D=sum(t['raw_weight']*t['effective_tokens']for t in ts)
            check(N==w['original_token_count']==m['c500/actor_tokens']and math.isclose(D,w['weighted_token_count']),'Token denominator '+str(step))
            for g in gs:
                members=[t for t in ts if t['uid']==g['uid']];native=[r for r in e['rollouts']if r['uid']==g['uid']]
                success=sum(t['success']for t in members);raw=.25 if not success else 1.;kind='F'if not success else'S'
                check(len(members)==len(native)==4 and g['kind']==kind and g['raw_weight']==raw,'F/S labels '+str(step))
                check(Counter((t['task_id'],t['reward_type'],round(t['terminal_utility'],6),bool(t['success']))for t in members)==Counter((r['task_id'],r['reward_type'],round(r['reward']['terminal_utility'],6),bool(r['reward']['purchase_success']))for r in native),'Native trajectory identity '+str(step))
                check(g['successes']==success and g['effective_tokens']==sum(t['effective_tokens']for t in members),'Group metrics '+str(step))
                check(math.isclose(g['coefficient'],raw*N/D,rel_tol=1e-12),'Group coefficient '+str(step))
                mean=statistics.mean(t['terminal_utility']for t in members)
                for t in members:
                    check(t['raw_weight']==raw and math.isclose(t['coefficient'],raw*N/D,rel_tol=1e-12),'Applied trajectory weight '+str(step))
                    err=abs(t['original_advantage_mean']-(t['terminal_utility']-mean));max_adv_error=max(max_adv_error,err)
                    check(err<3e-6,'Reward centered advantage '+str(step))
            check(w['composition']==''.join(sorted(g['kind']for g in gs)),'Composition '+str(step));comps[w['composition']]+=1
            check(math.isclose(m['c500/coefficient_min'],min(g['coefficient']for g in gs),rel_tol=1e-6)and math.isclose(m['c500/coefficient_max'],max(g['coefficient']for g in gs),rel_tol=1e-6),'Actor applied metrics '+str(step))
            check(all(math.isfinite(m[k])for k in ['actor/lr','actor/loss','actor/grad_norm'])and m['actor/grad_norm']>0 and m['training/optimizer_updated']==1,'Finite real update '+str(step))
            check(math.isclose(m['actor/lr'],1e-6*min(step/15,1),rel_tol=1e-5),'LR '+str(step));tokens+=N;lo.extend(g['coefficient']for g in gs)
        for step in [1,350,500]:
            ack=j('accepted_step_'+str(step)+'.json')
            check(ack['step']==step and ack['diagnostics_written']and ack['checkpoint']==run['output']+'/train/global_step_'+str(step),'Accepted checkpoint '+str(step))
        check(j('phase350_complete.json')['step']==350,'350 boundary receipt')
        log=z.read('train_to_500.log').decode(errors='replace');check('Setting global step to 350'in log,'Actual350 restore')
        plans=[e for e in rows('checkpoint_retention.jsonl')if e['event']=='retention_plan'];removed=[e['step']for e in rows('checkpoint_retention.jsonl')if e['event']=='checkpoint_removed']
        check(plans[-1]['new_step']==500 and plans[-1]['keep']==[350,450,500]and 350 not in removed,'Retention protects350')
        gen=rows('generation_tokens.jsonl');counts={k:sum(e['reserved_trajectories']for e in gen if bool(e['validation'])==v)for k,v in [('train',False),('validation_physical',True)]}
        check(counts==j('generation_budget.json')==done['generation_budget'],'Physical generation ledger')
        curve=[]
        for step in range(0,501,50):
            val=rows('validation/'+str(step)+'.jsonl');check(len(val)==50 and all(r['step']==step for r in val),'Internal validation '+str(step))
            curve.append({'step':step,'strict_score1':sum(r['score']==1 for r in val),'reward_mean':statistics.mean(r['reward']for r in val)})
        native_cpu=[n for n in z.namelist()if n.endswith('.json')and 'cpu'in n]
        if not native_cpu:gaps.append('Native CPU receipt file not included; run.json only names external path')
        if 'zero_update_recovery.json'in z.namelist():gaps.append('Prior disk-preflight zero-update claim present; original prior run evidence not included')
        shutdown=rows('auto_shutdown_events.jsonl')if 'auto_shutdown_events.jsonl'in z.namelist()else[]
        skipped=[e for e in ds if e['event']=='skipped_update']
        return {'training_core_passed':not errors,'full_archive_acceptance_passed':not errors and not gaps,'errors':errors,'evidence_gaps':gaps,
            'return_commit':'72a5011b484257b065fad1530fff42f7383a9edc','archive_sha256':sha(path.read_bytes()),'run':run['output'],'protocol_sha256':run['protocol_sha256'],
            'completed_updates':len(updates),'actual_gamma':4,'composition':dict(comps),'relative_reweighted_updates':comps['FS'],'Actor_tokens':tokens,'coefficient_range':[min(lo),max(lo)],'max_advantage_error':max_adv_error,
            'generation_budget':counts,'skipped_updates':len(skipped),'max_consecutive_skips':max(e['metrics']['shopping_dynamic_sampling/consecutive_skips']for e in skipped),
            'internal_validation':curve,'retained_checkpoints_by_log':plans[-1]['keep'],'elapsed_hours':done['elapsed_seconds']/3600,
            'independent_Dev_done':done['independent_Dev_done'],'Final_done':done['Final_done'],'shutdown_events':shutdown,
            'checkpoint_scope':'Returned synchronous-save/CRC retention receipts only; actual checkpoint bytes remain cloud-side'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.archive);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(0 if result['training_core_passed']else 1)

if __name__=='__main__':main()
