"""Verify B800 completion, evaluation and paired A800/B650/B500 results."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path,PurePosixPath
import zipfile

from analyze_b500_dev50_return import strict,paired_interval
from step650_evaluation import FINAL_HASH
ROOT=Path(__file__).resolve().parents[1]
POST='preparation/posttrain_step800_v1/'


def pair(A,B):
    a={r['task_id']for r in A if strict(r)};b={r['task_id']for r in B if strict(r)}
    gain=sorted(b-a);loss=sorted(a-b);n=len(gain)+len(loss);size=len(B)
    p=min(1.,2*sum(math.comb(n,k)for k in range(min(len(gain),len(loss))+1))/2**n)if n else 1.
    changed=[]
    for ar,br in zip(A,B):
        if ar['task_id']in gain+loss:
            detail=lambda r:(r.get('terminal_result')or{}).get('reward_detail')or{}
            changed.append({'task_id':ar['task_id'],'change':'B_only'if ar['task_id']in gain else'A_only',
                'reference_type':detail(ar).get('reward_type','unknown'),'B800_type':detail(br).get('reward_type','unknown'),
                'reference_status':ar['status'],'B800_status':br['status'],'reference_steps':len(ar['steps']),'B800_steps':len(br['steps'])})
    return {'reference_successes':len(a),'B800_successes':len(b),'net_gain':len(b)-len(a),'net_percentage_points':100*(len(b)-len(a))/size,
        'B_only':gain,'A_only':loss,'both_success':len(a&b),'both_fail':size-len(a|b),'exact_McNemar_two_sided_p':p,
        'paired_bootstrap_95_percentile_interval':paired_interval(len(gain),len(loss),size),'changed_tasks':changed}


def audit_training(z,check):
    import yaml
    T='training_evidence/';sha=lambda b:hashlib.sha256(b).hexdigest()
    j=lambda n:json.loads(z.read(T+n));rows=lambda n:[json.loads(l)for l in z.read(T+n).splitlines()if l.strip()]
    run=j('run.json');done=j('training_complete.json');prep=j('finish750_preparation.json');cpu=j('native_cpu_acceptance.json')
    cloud=json.loads(z.read(POST+'training_completion_check.json'));meta=json.loads(z.read('posttrain.json'))
    check(run['output']==meta['training']and run['resume_step']==750 and run['target_step']==800 and run['phase_stops']==[800],'Actual750->800 run identity')
    check(done['completed_updates']==800 and done['new_updates']==50 and T+'stopped.json'not in z.namelist()and T+'budget_truncated.json'not in z.namelist(),'Completed800/no stopped marker')
    check(run['protocol_sha256']==done['protocol_sha256']==sha((ROOT/'experiments/group_weight_b500_v1/protocol.json').read_bytes()),'Base training recipe identity')
    check(run['implementation_manifest_sha256']==cpu['implementation_manifest_sha256']==sha((ROOT/'experiments/group_weight_b500_v1/implementation_manifest.json').read_bytes()),'Core weighting identity')
    check(cpu['passed']and not cpu['gpu_initialized']and not cpu['model_weights_loaded'],'Native recovery CPU acceptance')
    supplement=sha((ROOT/'experiments/group_weight_b500_v1/finish750_protocol.json').read_bytes())
    check(sha(z.read(T+'effective_protocol_supplement.json'))==run['effective_protocol_supplement_sha256']==prep['effective_protocol_supplement_sha256']==supplement,'750/skip50 supplement identity')
    check(sha(z.read(T+'configs/phase800.yaml'))==run['config_hashes']['800']==cloud['recovery_config_sha256'],'Actual recovery config binding')
    before=yaml.safe_load(z.read(T+'configs/interrupted__phase800.yaml'));after=yaml.safe_load(z.read(T+'configs/phase800.yaml'))
    for k in ['algorithm','data']:check(before[k]==after[k],'Unchanged recovery '+k)
    for k in ['actor','model']:check(before['actor_rollout_ref'][k]==after['actor_rollout_ref'][k],'Unchanged recovery '+k)
    dynamic=dict(before['shopping_dynamic_sampling']);check(dynamic['max_consecutive_skipped_updates']==10,'Original skip bound10');dynamic['max_consecutive_skipped_updates']=50
    check(dynamic==after['shopping_dynamic_sampling'],'Only stopping bound changes in sampler')
    check(after['trainer']['total_training_steps']==800 and not after['trainer']['val_before_train']and after['trainer']['resume_from_path']==run['source_checkpoint'],'Fixed800/full750 resume/no initial validation')
    check(after['ray_kwargs']['ray_init']['runtime_env']['env_vars']==run['env'],'Actual driver/worker environment')
    old_audit=json.loads((ROOT/'analysis/b800_interruption_audit_20261122/audit.json').read_text(encoding='utf-8'))
    check(sha(z.read(T+'interrupted__training_diagnostics.jsonl'))==old_audit['source_diagnostics_sha256']and sha(z.read(T+'interrupted__weights.jsonl'))==old_audit['source_weights_sha256'],'Complete previously audited failed attempt preserved')
    expected_source=str(PurePosixPath(old_audit['source_run'])/'train/global_step_750')
    check(prep['passed']and prep['source_crc_verified']and prep['source']==run['source_checkpoint']==expected_source and prep['source_model_sha256']==run['source_model_sha256'],'Recorded750 CRC/model source')
    check(prep['rolled_back_updates']==list(range(751,788))and prep['skip_limit_before']==10 and prep['skip_limit_after']==50,'Rollback37 and stopping change disclosed')
    es=rows('training_diagnostics.jsonl');updates=[e for e in es if e['event']=='optimizer_step']
    check([e['global_step']for e in updates]==list(range(1,801)),'Exactly1..800 effective update chain')
    prior=rows('interrupted__training_diagnostics.jsonl');prefix=[e for e in prior if e.get('global_step',0)<=750]
    check(es[:len(prefix)]==prefix,'Committed750 prefix preserved')
    weights=rows('weights.jsonl');wr=[w for w in weights if w['event']=='actor_update_returned']
    committed_weights=[w for w in rows('interrupted__weights.jsonl')if w['driver_global_step_before_update']<750]
    check(weights[:len(committed_weights)]==committed_weights,'Committed651..750 weights preserved')
    check([w['driver_global_step_before_update']for w in wr]==list(range(650,800))and Counter(w['event']for w in weights)==Counter(actor_update_started=150,actor_update_returned=150),'Exactly150 canonical Actor returns')
    composition=Counter();coeff=[];tokens=0
    for e,w in zip(updates[650:],wr):
        s=str(e['global_step']);m=e['metrics'];gs=w['groups'];ts=w['trajectories']
        check(len(gs)==2 and len(ts)==8,'Batch '+s)
        N=sum(t['effective_tokens']for t in ts);nw=sum(t['effective_tokens']*t['raw_weight']for t in ts)
        check(N==w['original_token_count']and math.isclose(nw,w['weighted_token_count']),'Token denominator '+s)
        for g in gs:
            members=[t for t in ts if t['uid']==g['uid']];success=sum(t['success']for t in members);raw=.5 if success==0 else 1.
            check(len(members)==4 and g['kind']==('F'if success==0 else'S')and g['raw_weight']==raw,'Group labels '+s)
            check(success==sum(bool(r['reward']['purchase_success'])for r in e['rollouts']if r['uid']==g['uid']),'Native success '+s)
            check(math.isclose(g['coefficient'],raw*N/nw,rel_tol=1e-12),'Coefficient '+s)
            check(all(t['raw_weight']==raw and math.isclose(t['coefficient'],raw*N/nw,rel_tol=1e-12)for t in members),'Trajectory weight '+s)
        check(w['composition']==''.join(sorted(g['kind']for g in gs)),'Composition '+s)
        check(Counter((r['uid'],r['task_id'])for r in e['rollouts'])==Counter((t['uid'],next(g['task_id']for g in gs if g['uid']==t['uid']))for t in ts),'Weight/diagnostic identity '+s)
        check(m['training/optimizer_updated']==1 and m['actor/grad_norm']>0 and all(math.isfinite(m[k])for k in ['actor/lr','actor/grad_norm','actor/loss']),'Finite actual update '+s)
        check(math.isclose(m['actor/lr'],1e-6,rel_tol=1e-5),'Inherited constant LR '+s)
        check(m['b500/actor_tokens']==N and math.isclose(m['b500/coefficient_min'],min(g['coefficient']for g in gs),rel_tol=1e-6),'Applied B metrics '+s)
        composition[w['composition']]+=1;tokens+=N;coeff.extend(g['coefficient']for g in gs)
    validations=[]
    for step in [700,750,800]:
        rs=rows('validation/'+str(step)+'.jsonl');mean=sum(r['score']for r in rs)/50
        check(len(rs)==50 and all(r['step']==step for r in rs),'Native validation50 '+str(step))
        check(math.isclose(mean,updates[step-1]['metrics']['val-core/shopsimulator/reward/mean@1'],abs_tol=1e-12),'Native validation mean '+str(step))
        validations.append({'step':step,'score_equal1_count':sum(math.isclose(r['score'],1.,abs_tol=1e-6,rel_tol=0)for r in rs),'mean_reward':mean})
    gen=rows('generation_tokens.jsonl');budget={'train':sum(g['reserved_trajectories']for g in gen if not g['validation']),
        'validation_physical':sum(g['reserved_trajectories']for g in gen if g['validation'])}
    check(budget==j('generation_budget.json')==done['generation_budget']==cloud['physical_generation_budget'],'Full physical generation ledger')
    check(prep['carried_generation_budget']==old_audit['cumulative_generation'],'All interrupted costs inherited')
    old_budget=prep['carried_generation_budget'];new_budget={k:budget[k]-old_budget[k]for k in budget}
    check(new_budget['validation_physical']==200,'One new final validation')
    check(sha(z.read(T+'training_diagnostics.jsonl'))==cloud['diagnostics_sha256']and sha(z.read(T+'weights.jsonl'))==cloud['weights_sha256'],'Cloud completion hashes')
    check(sha(z.read(POST+'training_completion_check.json'))==meta['training_completion_check_sha256']and cloud['passed'],'Completion receipt integrity')
    accepted=j('accepted_step_800.json');source800=str(PurePosixPath(run['output'])/'train/global_step_800')
    check(accepted['step']==800 and accepted['checkpoint']==cloud['source800']==source800 and math.isclose(accepted['lr'],1e-6,rel_tol=1e-5)and accepted['grad_norm']>0,'Accepted800 checkpoint/update')
    retention=rows('checkpoint_retention.jsonl');plans=[r for r in retention if r['event']=='retention_plan']
    check(plans[-1]['new_step']==800 and plans[-1]['keep']==[800]and not plans[-1]['remove'],'New800 saved, no cross-run deletion')
    text=z.read(T+'train_to_800.log').decode(errors='replace');check('Setting global step to 750'in text and 'B800_TRAINING_COMPLETE'in z.read(T+'supervisor.log').decode(errors='replace'),'Actual750 restore and completed supervisor')
    tail_weights=[w for w in wr if w['driver_global_step_before_update']>=750]
    skips=[e for e in es if e['event']=='skipped_update'and e['global_step']>750]
    maximum=max([e['metrics']['shopping_dynamic_sampling/consecutive_skips']for e in skips],default=0)
    return {'completed_updates':800,'new_tail_updates':50,'canonical_updates_since650':150,'restored_from':750,
        'rolled_back_updates_count':37,'native_CPU_tests':cpu['tests_run'],'canonical_since650_composition':dict(composition),
        'tail50_composition':dict(Counter(w['composition']for w in tail_weights)),
        'canonical_since650_Actor_tokens':tokens,'tail50_Actor_tokens':sum(w['original_token_count']for w in tail_weights),
        'tail50_skips':len(skips),'tail50_max_consecutive_skips':maximum,'tail50_exceeded_old_limit10':maximum>=10,
        'coefficient_range':[min(coeff),max(coeff)],'internal_validation':validations,
        'cumulative_generation':budget,'new_tail_generation':new_budget,'new_run_retained_checkpoints':[800],
        'supervisor_elapsed_seconds':done['elapsed_seconds'],'source750_model_sha256':run['source_model_sha256'],
        'physical_checkpoint_weights_in_archive':False}


def analyze(path):
    errors=[]
    def check(ok,label):
        if not ok:errors.append(label)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    with zipfile.ZipFile(path)as z:
        check(z.testzip()is None,'ZIP CRC');manifest=json.loads(path.with_name('manifest.json').read_text(encoding='utf-8'))
        check(sha(path.read_bytes())==manifest['archive_sha256'],'Archive SHA256')
        j=lambda n:json.loads(z.read(n));rows=lambda n:[json.loads(l)for l in z.read(n).splitlines()if l.strip()]
        spec=j(POST+'evaluation_spec.json');prep=j(POST+'preparation_receipt.json');export=j(POST+'export_receipt.json');service=j(POST+'services_ready.json')
        check(j(POST+'pipeline_exit_status.json')['exit_code']==0 and j(POST+'evaluation_complete.json')['passed'],'Complete pipeline')
        check(prep['passed']and prep['source_crc_verified']and prep['fixed_step']==800,'Cloud fixed800 preparation acceptance')
        check(set(spec['commands'])=={'smoke3','dev50','final200_clean'}and prep['Final_runs_regardless_of_Dev_score'],'Authorized fixed endpoint suites')
        check(spec['benchmarks']['final200_clean']['sha256']==FINAL_HASH,'Original Final Clean fingerprint')
        meta=j('posttrain.json');source=str(PurePosixPath(meta['training'])/'train/global_step_800')
        check(spec['checkpoint']==export['checkpoint']==source and export['actor_model_sha256']==meta['checkpoint_model_sha256'],'Prepared/exported800 identity')
        check(export['lora_rank']==16 and export['lora_alpha']==32,'LoRA identity')
        m=service['model'];check(service['passed']and m['id']==spec['model_tag']and m['root']==export['merged_model']and m['max_model_len']==24576,'Served model identity')
        summaries={};A_summaries={};trajectories={};comparisons={};execution={}
        for suite,n in [('smoke3',3),('dev50',50),('final200_clean',200)]:
            prefix='evaluation/'+suite+'/';rs=rows(prefix+'trajectories.jsonl');b=j(prefix+'summary.json');passed=j(prefix+'passed.json')
            check(passed['passed']and passed['tasks']==n,'Suite receipt '+suite)
            check(sha(z.read(prefix+'trajectories.jsonl'))==passed['trajectory_sha256']and sha(z.read(prefix+'summary.json'))==passed['summary_sha256'],'Suite hashes '+suite)
            ids=[r['task_id']for r in rs];check(ids==spec['benchmarks'][suite]['task_ids']and len(ids)==len(set(ids))==n,'Task identity/order '+suite)
            success=sorted(r['task_id']for r in rs if strict(r));check(success==b['strict_success_task_ids']and len(success)==b['strict_successes'],'Strict gold '+suite)
            check(b['completed_tasks']==n and not b['missing_tasks'],'Full denominator '+suite)
            check(math.isclose(sum(r['final_reward']for r in rs)/n,b['mean_final_reward'],abs_tol=1e-12),'Reward '+suite)
            check(math.isclose(sum(len(r['steps'])for r in rs)/n,b['average_steps'],abs_tol=1e-12),'Steps '+suite)
            check(b['protocol']['benchmark']==spec['benchmarks'][suite]['path'],'Actual benchmark path '+suite)
            bad=[r['task_id']for r in rs if (r.get('error')and r['error'].get('type')!='ContextBudgetError')or r.get('release_error')]
            check(not bad,'No infrastructure/release failures '+suite)
            execution[suite]={'status_counts':dict(Counter(r['status']for r in rs)),
                'context_error_ids':[r['task_id']for r in rs if (r.get('error')or{}).get('type')=='ContextBudgetError'],
                'invalid_action_limit_ids':[r['task_id']for r in rs if r['status']=='invalid_action_limit']}
            types=Counter(((r.get('terminal_result')or{}).get('reward_detail')or{}).get('reward_type','unknown')for r in rs)
            check(dict(types)==b['reward_type_counts'],'Reward types '+suite)
            summaries[suite]=b;trajectories[suite]=rs
            if suite!='smoke3':
                ar=rows('reference_A800/'+suite+'/trajectories.jsonl');a=j(POST+'A800_'+suite+'_summary.json')
                check([r['task_id']for r in ar]==ids,'A800 task alignment '+suite)
                check(sorted(r['task_id']for r in ar if strict(r))==a['strict_success_task_ids']and sum(strict(r)for r in ar)==a['strict_successes'],'A800 strict result '+suite)
                check(math.isclose(sum(r['final_reward']for r in ar)/n,a['mean_final_reward'],abs_tol=1e-12),'A800 reward '+suite)
                check(math.isclose(sum(len(r['steps'])for r in ar)/n,a['average_steps'],abs_tol=1e-12),'A800 steps '+suite)
                types=Counter(((r.get('terminal_result')or{}).get('reward_detail')or{}).get('reward_type','unknown')for r in ar)
                check(dict(types)==a['reward_type_counts'],'A800 reward types '+suite)
                A_summaries[suite]=a
                pa=dict(a['protocol']);pb=dict(b['protocol']);pa.pop('model');pb.pop('model');check(pa==pb,'Original A/B protocol '+suite)
                c=pair(ar,rs);cloud=j(POST+('dev50_complete.json'if suite=='dev50'else'final200_complete.json'))
                check(c['B_only']==cloud['B_only']and c['A_only']==cloud['A_only']and c['net_gain']==cloud['net_gain'],'Cloud pairs '+suite)
                comparisons['vs_A800_'+suite]=c
        sources={
            'B500':{'dev50':ROOT/'analysis/b500_independent_dev50_return/20261121T031644Z/evidence.zip',
                'final200_clean':ROOT/'analysis/b500_final200_return/20261121T040953Z/evidence.zip'},
            'B650':{suite:ROOT/'analysis/b650_evaluation_return/20261122T011337Z/evidence.zip'for suite in ['dev50','final200_clean']}}
        previous_summaries={}
        for label,archives in sources.items():
            previous_summaries[label]={}
            for suite,p in archives.items():
                old_manifest=json.loads(p.with_name('manifest.json').read_text(encoding='utf-8'))
                check(sha(p.read_bytes())==old_manifest['archive_sha256'],label+' archive SHA256 '+suite)
                with zipfile.ZipFile(p)as bz:
                    check(bz.testzip()is None,label+' ZIP CRC '+suite)
                    old=[json.loads(l)for l in bz.read('evaluation/'+suite+'/trajectories.jsonl').splitlines()if l.strip()]
                    ob=json.loads(bz.read('evaluation/'+suite+'/summary.json'))
                previous_summaries[label][suite]=ob
                check([r['task_id']for r in old]==[r['task_id']for r in trajectories[suite]],label+' task alignment '+suite)
                check(sorted(r['task_id']for r in old if strict(r))==ob['strict_success_task_ids']and sum(strict(r)for r in old)==ob['strict_successes'],label+' strict result '+suite)
                pa=dict(ob['protocol']);pb=dict(summaries[suite]['protocol']);pa.pop('model');pb.pop('model');check(pa==pb,label+'/B800 evaluation protocol '+suite)
                comparisons['vs_'+label+'_'+suite]=pair(old,trajectories[suite])
        training=audit_training(z,check)
        return {'audit_passed':not errors,'errors':errors,'return_commit':'f5156fd447aaffdb79ac18b5d53c69e82c9e879e',
            'archive_sha256':manifest['archive_sha256'],'B800_summaries':{s:{k:v for k,v in b.items()if k in ['strict_successes','expected_tasks','mean_final_reward','average_steps','reward_type_counts']}for s,b in summaries.items()},
            'previous_summaries':{label:{s:{k:v for k,v in b.items()if k in ['strict_successes','mean_final_reward','average_steps','reward_type_counts']}for s,b in by_suite.items()}for label,by_suite in previous_summaries.items()},
            'A800_summaries':{s:{k:v for k,v in b.items()if k in ['strict_successes','mean_final_reward','average_steps','reward_type_counts']}for s,b in A_summaries.items()},
            'comparisons':comparisons,'execution_records':execution,'model_sha256':export['merged_weights_sha256'],'actor_model_sha256':export['actor_model_sha256'],'training_audit':training,
            'interpretation':'Valid fixed800 evaluation; both strict endpoint scores are one task below A800. Current group-weight configuration has not demonstrated superiority; aggregate RL gains over SFT do not isolate method contribution.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(a.archive);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(0 if result['audit_passed']else 1)
