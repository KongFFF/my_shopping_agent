"""Verify interrupted B800 chain, weighting, skip streaks and saved750 evidence."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
FAILURE='shopping dynamic sampling reached max_consecutive_skipped_updates=10 without an optimizer update; stopping a persistently no-signal run'


def audit(path):
    errors=[]
    def check(ok,label):
        if not ok:errors.append(label)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    with zipfile.ZipFile(path)as z:
        j=lambda n:json.loads(z.read(n));rows=lambda n:[json.loads(l)for l in z.read(n).splitlines()if l.strip()]
        manifest=json.loads(path.with_name('manifest.json').read_text(encoding='utf-8'))
        check(sha(path.read_bytes())==manifest['archive_sha256']and z.testzip()is None,'Archive SHA/CRC')
        run=j('run.json');cpu=j('native_cpu_acceptance.json');prep=j('b800_preparation.json')
        check('training_complete.json'not in z.namelist()and 'stopped.json'in z.namelist(),'Incomplete stopped run')
        check(run['resume_step']==650 and run['target_step']==800 and run['phase_stops']==[800],'Fixed650->800')
        check(run['protocol_sha256']==sha((ROOT/'experiments/group_weight_b500_v1/protocol.json').read_bytes()),'Original protocol')
        check(run['implementation_manifest_sha256']==cpu['implementation_manifest_sha256']==sha((ROOT/'experiments/group_weight_b500_v1/implementation_manifest.json').read_bytes()),'Frozen method')
        check(cpu['passed']and not cpu['gpu_initialized']and not cpu['model_weights_loaded'],'Native CPU acceptance')
        check(sha(z.read('configs/phase800.yaml'))==run['config_hashes']['800'],'Source config binding')
        check(prep['source_crc_verified']and prep['source_model_sha256']==run['source_model_sha256']=='dd9fb35c2b2ea37849f3a183d0970e122d10ca40ee5c1e5858ea3092793789f9','Evaluated full650 source')
        import yaml
        a=yaml.safe_load(z.read('configs/parent_phase650.yaml'));b=yaml.safe_load(z.read('configs/phase800.yaml'))
        for k in ['algorithm','data','shopping_dynamic_sampling']:check(a[k]==b[k],'Unchanged '+k)
        for k in ['actor','model']:check(a['actor_rollout_ref'][k]==b['actor_rollout_ref'][k],'Unchanged '+k)
        check(b['trainer']['total_training_steps']==800 and not b['trainer']['val_before_train'],'Fixed800/no initial validation')
        check(b['trainer']['resume_from_path']==run['source_checkpoint']and b['ray_kwargs']['ray_init']['runtime_env']['env_vars']==run['env'],'Actual source/environment')
        check(b['shopping_dynamic_sampling']['max_consecutive_skipped_updates']==10,'Original skip bound10')
        es=rows('training_diagnostics.jsonl');updates=[e for e in es if e['event']=='optimizer_step']
        check([e['global_step']for e in updates]==list(range(1,788)),'Exactly1..787 committed updates')
        with zipfile.ZipFile(ROOT/'analysis/b650_training_return/20261121T144812Z/evidence.zip')as pz:
            previous=[json.loads(l)for l in pz.read('training_diagnostics.jsonl').splitlines()if l.strip()]
            check(es[:len(previous)]==previous,'Exact B650 prefix')
        ws=rows('weights.jsonl');wr=[w for w in ws if w['event']=='actor_update_returned']
        check([w['driver_global_step_before_update']for w in wr]==list(range(650,787)),'Exactly137 new Actor returns')
        check(Counter(w['event']for w in ws)==Counter(actor_update_started=137,actor_update_returned=137),'No failed Actor calls')
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
        for step in (700,750):
            rs=rows(f'validation/{step}.jsonl');mean=sum(r['score']for r in rs)/50
            check(len(rs)==50 and all(r['step']==step for r in rs),'Validation50 '+str(step))
            check(math.isclose(mean,updates[step-1]['metrics']['val-core/shopsimulator/reward/mean@1'],abs_tol=1e-12),'Validation mean '+str(step))
            validations.append({'step':step,'score_equal1_count':sum(math.isclose(r['score'],1.,abs_tol=1e-6,rel_tol=0)for r in rs),'mean_reward':mean})
        gen=rows('generation_tokens.jsonl');counts={'train':sum(r['reserved_trajectories']for r in gen if not r['validation']),
            'validation_physical':sum(r['reserved_trajectories']for r in gen if r['validation'])}
        check(counts==j('generation_budget.json'),'Physical generation ledger')
        retry=j('retry650_receipt.json');check(retry['passed']and retry['rolled_back_updates']==[651,652,653],'Prior retry receipt')
        check(retry['carried_generation_budget']==j('prior_attempt__generation_budget.json')=={'train':24352,'validation_physical':3000},'First failed cost retained')
        retention=rows('checkpoint_retention.jsonl');plans=[r for r in retention if r['event']=='retention_plan']
        check([r['new_step']for r in plans]==[700,750]and plans[-1]['keep']==[700,750]and all(not r['remove']for r in plans),'700/750 saved and retained')
        text=z.read('train_to_800.log').decode(errors='replace')
        check(FAILURE in text and 'Setting global step to 650'in text,'Exact skip-guard failure after restore')
        skips=[e for e in es if e['event']=='skipped_update'and e['global_step']>650]
        streak=0;total=0
        for e in es:
            if e['global_step']<=650:continue
            if e['event']=='optimizer_step':streak=0
            if e['event']=='skipped_update':
                streak+=1;total+=1;m=e['metrics']
                check(m['shopping_dynamic_sampling/consecutive_skips']==streak and m['shopping_dynamic_sampling/skipped_updates_total']==total,'Skip-counter sequence')
                check(m['training/optimizer_updated']==0 and m['group/trained']<2 and m['rollout/generated_total']==24,'No update in skipped batch')
        check(len(skips)==142 and [e['global_step']for e in skips[-10:]]==[788]*10 and streak==10,'Terminal streak10 at attempt788')
        blocks=[]
        for lo,hi in [(651,700),(701,750),(751,787),(788,788)]:
            us=[e for e in updates if lo<=e['global_step']<=hi];ss=[e for e in skips if lo<=e['global_step']<=hi]
            blocks.append({'range':[lo,hi],'updates':len(us),'skips':len(ss),'max_consecutive_skips':max([e['metrics']['shopping_dynamic_sampling/consecutive_skips']for e in ss],default=0),
                'accepted_groups_in_skips':dict(Counter(e['metrics']['group/trained']for e in ss))})
        source_budget=retry['carried_generation_budget']
        return {'audit_passed':not errors,'errors':errors,'return_commit':'1b89daba11a8170e4ebac432e197ac4151d49be2',
            'archive_sha256':manifest['archive_sha256'],'source_run':run['output'],
            'source_run_sha256':sha(z.read('run.json')),'source_diagnostics_sha256':sha(z.read('training_diagnostics.jsonl')),
            'source_weights_sha256':sha(z.read('weights.jsonl')),'source_config_sha256':run['config_hashes']['800'],
            'last_completed_update':787,'failed_attempt':788,'new_completed_updates':137,'skipped_attempts':142,'blocks':blocks,
            'new_composition':dict(composition),'new_Actor_tokens':tokens,'internal_validation':validations,
            'cumulative_generation':counts,'current_attempt_generation':{k:counts[k]-source_budget[k]for k in counts},
            'retained_checkpoints_by_log':[700,750],'checkpoint750_crc_accepted_at_save':True,'physical_checkpoint_files_in_archive':False,
            'recovery_step':750,'discarded_updates_on_recovery':list(range(751,788)),
            'recovery_requires_cloud_CRC_and_full_state_restore':True,'native_score1_is_not_independent_gold_success':True}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.archive);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(0 if result['audit_passed']else 1)
