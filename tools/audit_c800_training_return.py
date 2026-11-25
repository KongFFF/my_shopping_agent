"""CPU audit of C650->800 recovery, real weighting and internal validation."""
from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def audit(path, previous_archive=None):
    errors=[]
    def check(ok,label):
        if not ok:errors.append(label)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    with zipfile.ZipFile(path)as z:
        check(z.testzip()is None,'Archive CRC');j=lambda n:json.loads(z.read(n))
        rows=lambda n:[json.loads(l)for l in z.read(n).splitlines()if l.strip()]
        manifest=json.loads(path.with_name('manifest.json').read_text(encoding='utf-8'))
        check(sha(path.read_bytes())==manifest['archive_sha256'],'Archive SHA256')
        run=j('run.json');done=j('training_complete.json');prep=j('c800_preparation.json');cpu=j('native_cpu_acceptance.json')
        check(done['completed_updates']==800 and done['new_updates']==150 and 'stopped.json'not in z.namelist(),'Completed800/no stop')
        check(cpu['passed']and not cpu['gpu_initialized']and not cpu['model_weights_loaded'],'Cloud CPU acceptance')
        check(run['target_step']==800 and run['resume_step']==650 and run['phase_stops']==[800],'Fixed650->800')
        check(run['protocol_sha256']==done['protocol_sha256']==sha((ROOT/'experiments/group_weight_c500_v1/protocol.json').read_bytes()),'Frozen original recipe identity')
        check(run['implementation_manifest_sha256']==sha((ROOT/'experiments/group_weight_c500_v1/implementation_manifest.json').read_bytes())==cpu['implementation_manifest_sha256'],'Frozen weighting identity')
        check(sha(z.read('configs/phase800.yaml'))==run['config_hashes']['800'],'Active config hash')
        check(prep['source_crc_verified']and run['source_model_sha256']==prep['source_model_sha256']=='ca4524e7217dcc64d002421d913e9dbf13561b01aec20df390e1e6b065617eac','Same evaluated C650 full checkpoint')
        continuation=sha((ROOT/'experiments/group_weight_c800_v1/protocol.json').read_bytes())
        check(run['continuation_protocol_sha256']==done['continuation_protocol_sha256']==sha(z.read('continuation_protocol.json'))==continuation,'Frozen C800 continuation supplement')
        check(run['experiment']==done['experiment']=='C800'and run['gamma']==done['gamma']==4,'C800 gamma4 arm')
        check(cpu['version']=='c500-cpu-acceptance-v1'and cpu['protocol_sha256']==run['protocol_sha256']and cpu['training_started']is False and not cpu['errors']and all(c['matched']for c in cpu['checks']),'Original CPU receipt binding/checks')
        cpu_log=z.read('native_cpu_acceptance.log').decode(errors='replace')
        check('Ran 12 tests'in cpu_log and 'Ran 15 tests'in cpu_log and 'Ran 2 tests'in cpu_log and 'FAILED'not in cpu_log,'Native continuation and shutdown CPU tests')
        baseline=j('baseline_identity_preflight.json')
        check(baseline['passed']and not baseline['errors']and baseline['tensor_loading']is False and all(c['matched']for c in baseline['checks']),'Original baseline assets and engine identities')
        for step in [651,800]:
            ack=j('accepted_step_'+str(step)+'.json')
            check(ack['step']==step and ack['diagnostics_written']and math.isclose(ack['lr'],1e-6,rel_tol=1e-5)and ack['grad_norm']>0,'Real accepted update '+str(step))
        check(j('accepted_step_800.json')['checkpoint']==run['output']+'/train/global_step_800','Final800 checkpoint binding')
        import yaml
        a=yaml.safe_load(z.read('configs/parent_phase650.yaml'));b=yaml.safe_load(z.read('configs/phase800.yaml'))
        for k in ['algorithm','data','shopping_dynamic_sampling']:check(a[k]==b[k],'Unchanged '+k)
        for k in ['actor','model']:check(a['actor_rollout_ref'][k]==b['actor_rollout_ref'][k],'Unchanged '+k)
        check(b['trainer']['total_training_steps']==800 and not b['trainer']['val_before_train'],'800 target/no repeated initial Dev')
        check(b['trainer']['resume_mode']=='resume_path'and b['trainer']['resume_from_path']==run['source_checkpoint'],'Actual resume source')
        check(b['ray_kwargs']['ray_init']['runtime_env']['env_vars']==run['env'],'Driver/worker environment identity')
        check(set(b['actor_rollout_ref']['actor']['checkpoint']['load_contents'])=={'model','optimizer','extra'},'Full restore components')
        es=rows('training_diagnostics.jsonl');updates=[e for e in es if e['event']=='optimizer_step']
        check([e['global_step']for e in updates]==list(range(1,801)),'Exactly1..800 update chain')
        previous=previous_archive or ROOT/'analysis/c650_training_return/20261124T114352Z/evidence.zip'
        with zipfile.ZipFile(previous)as pz:
            pe=[json.loads(l)for l in pz.read('training_diagnostics.jsonl').splitlines()]
            check(es[:len(pe)]==pe,'Exact committed C650 prefix preserved')
            check(sha(z.read('parent_C650__weights.jsonl'))==sha(pz.read('weights.jsonl')),'Original650 weight log preserved byte-for-byte')
            previous_budget=json.loads(pz.read('generation_budget.json'))
            check(j('parent_C650__generation_budget.json')==previous_budget,'Original650 generation budget identity')
        wr=[w for w in rows('weights.jsonl')if w['event']=='actor_update_returned']
        check([w['driver_global_step_before_update']for w in wr]==list(range(650,800)),'Exactly150 new Actor returns')
        check(Counter(w['event']for w in rows('weights.jsonl'))==Counter(actor_update_started=150,actor_update_returned=150),'No failed Actor updates')
        composition=Counter();coeff=[];tokens=0;max_adv_error=0.
        for e,w in zip(updates[650:],wr):
            s=str(e['global_step']);m=e['metrics'];gs=w['groups'];ts=w['trajectories']
            check(len(gs)==2 and len(ts)==8,'Batch '+s)
            check(w['gamma']==4 and w['experiment']=='C500','Frozen C method log identity '+s)
            N=sum(t['effective_tokens']for t in ts);nw=sum(t['effective_tokens']*t['raw_weight']for t in ts)
            check(N==w['original_token_count']and math.isclose(nw,w['weighted_token_count']),'Token denominator '+s)
            for g in gs:
                members=[t for t in ts if t['uid']==g['uid']];success=sum(t['success']for t in members);raw=.25 if success==0 else 1.
                check(len(members)==4 and g['kind']==('F'if success==0 else'S')and g['raw_weight']==raw,'Group labels '+s)
                check(success==sum(bool(r['reward']['purchase_success'])for r in e['rollouts']if r['uid']==g['uid']),'Native success '+s)
                check(math.isclose(g['coefficient'],raw*N/nw,rel_tol=1e-12),'Coefficient '+s)
                check(all(t['raw_weight']==raw and math.isclose(t['coefficient'],raw*N/nw,rel_tol=1e-12)for t in members),'Trajectory weight '+s)
                check(Counter((t['task_id'],t['reward_type'],round(t['terminal_utility'],6),bool(t['success']))for t in members)==Counter((r['task_id'],r['reward_type'],round(r['reward']['terminal_utility'],6),bool(r['reward']['purchase_success']))for r in e['rollouts']if r['uid']==g['uid']),'Full native trajectory identity '+s)
                mean=sum(t['terminal_utility']for t in members)/4
                for t in members:
                    error=abs(t['original_advantage_mean']-(t['terminal_utility']-mean));max_adv_error=max(max_adv_error,error)
                    check(error<3e-6,'Original reward-centered advantage '+s)

            check(w['composition']==''.join(sorted(g['kind']for g in gs)),'Composition '+s)
            check(Counter((r['uid'],r['task_id'])for r in e['rollouts'])==Counter((t['uid'],next(g['task_id']for g in gs if g['uid']==t['uid']))for t in ts),'Weight/diagnostic identity '+s)
            check(m['training/optimizer_updated']==1 and m['actor/grad_norm']>0 and all(math.isfinite(m[k])for k in ['actor/lr','actor/grad_norm','actor/loss']),'Finite actual update '+s)
            check(math.isclose(m['actor/lr'],1e-6,rel_tol=1e-5),'Inherited constant LR '+s)
            check(m['c500/actor_tokens']==N and math.isclose(m['c500/coefficient_min'],min(g['coefficient']for g in gs),rel_tol=1e-6),'Applied C metrics '+s)
            check(math.isclose(m['c500/coefficient_max'],max(g['coefficient']for g in gs),rel_tol=1e-6)and math.isclose(m['c500/weighted_actor_tokens'],nw,rel_tol=1e-6),'Applied maximum/weighted token denominator '+s)
            composition[w['composition']]+=1;tokens+=N;coeff.extend(g['coefficient']for g in gs)
        validations=[]
        for step in (700,750,800):
            rs=rows(f'validation/{step}.jsonl');mean=sum(r['score']for r in rs)/50
            check(len(rs)==50 and all(r['step']==step for r in rs),'50 native validation records '+str(step))
            check(math.isclose(mean,updates[step-1]['metrics']['val-core/shopsimulator/reward/mean@1'],abs_tol=1e-12),'Native validation mean '+str(step))
            validations.append({'step':step,'score_equal1_count':sum(math.isclose(r['score'],1.,abs_tol=1e-6,rel_tol=0)for r in rs),'mean_reward':mean})
        gen=rows('generation_tokens.jsonl');counts={'train':sum(r['reserved_trajectories']for r in gen if not r['validation']),
            'validation_physical':sum(r['reserved_trajectories']for r in gen if r['validation'])}
        check(counts==j('generation_budget.json')==done['generation_budget'],'Cumulative generation ledger')
        old=j('parent_C650__generation_budget.json');extra={k:counts[k]-old[k]for k in counts}
        check(extra['validation_physical']==600,'Three padded native Dev passes')
        check(run['budget_record_only'] is True and run['deadline'] is None,'User-managed budget, no automatic time cutoff')
        retention=rows('checkpoint_retention.jsonl');last=[r for r in retention if r['event']=='retention_plan'][-1]
        check(last['new_step']==800 and last['keep']==[750,800],'Latest2 retained')
        plans=[r for r in retention if r['event']=='retention_plan']
        check([r['new_step']for r in plans]==[700,750,800]and not any(r['unmanaged']for r in plans),'All scheduled complete save receipts')
        removed=[r['step']for r in retention if r['event']=='checkpoint_removed']
        check(removed==[700],'Only new run700 removed after acceptance')
        text=z.read('train_to_800.log').decode(errors='replace')
        skipped=[e for e in es if e['event']=='skipped_update'and e['global_step']>650]
        max_skip=max((e['metrics']['shopping_dynamic_sampling/consecutive_skips']for e in skipped),default=0)
        check(max_skip<10,'Original consecutive skip bound')
        check(b['shopping_dynamic_sampling']['max_consecutive_skipped_updates']==10,'Skip threshold10 unchanged')
        trainer_operational={'total_training_steps','resume_mode','resume_from_path','del_local_ckpt_after_load','val_before_train','default_local_dir','validation_data_dir','experiment_name'}
        check({k:v for k,v in a['trainer'].items()if k not in trainer_operational}=={k:v for k,v in b['trainer'].items()if k not in trainer_operational},'Original remaining trainer fields')
        check('Setting global step to 650'in text,'Restored global step650')
        loader_skip='Skipping dataloader state restore: global_steps=650'in text
        check(prep['model_optimizer_extra_restore'] and prep['data_checkpoint_required'],'Prepared full state restore')
        return {'audit_passed':not errors,'errors':errors,'return_commit':'ffe65bb4b6f1e59ee55efab4fabfd0cc5347d048',
            'archive_sha256':manifest['archive_sha256'],'completed_updates':800,'new_updates':150,'native_CPU_tests':cpu['tests_run'],'native_additional_tests':{'continuation':12,'shutdown':15,'C800_shutdown_binding':2},'baseline_identity_checks':len(baseline['checks']),'new_skipped_updates':len(skipped),'max_consecutive_skips':max_skip,'max_advantage_error':max_adv_error,
            'new_composition':dict(composition),'relative_reweight_updates':composition['FS'],'relative_reweight_fraction':composition['FS']/150,
            'coefficient_range':[min(coeff),max(coeff)],'new_Actor_tokens':tokens,'internal_validation':validations,
            'cumulative_generation':counts,'new_generation':extra,'retained_checkpoints_by_log':[750,800],
            'dataloader_epoch_boundary_skip_observed':loader_skip,'new_supervisor_elapsed_hours':done['elapsed_seconds']/3600,
            'C800_independent_Dev_done':done['independent_Dev_done'],'C800_Final_done':done['Final_done'],
            'native_score1_is_not_independent_gold_success':True,'checkpoint_scope':'Returned synchronous-save CRC/retention receipts; checkpoint tensors remain on AutoDL','shutdown_guard_events_returned':'auto_shutdown_events.jsonl'in z.namelist(),
            'shutdown_events':rows('auto_shutdown_events.jsonl')if 'auto_shutdown_events.jsonl'in z.namelist()else []}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.archive);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(0 if result['audit_passed']else 1)
