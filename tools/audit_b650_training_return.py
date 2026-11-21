"""CPU audit of B500->650 recovery, real weighting and internal validation."""
from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def audit(path):
    errors=[]
    def check(ok,label):
        if not ok:errors.append(label)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    with zipfile.ZipFile(path)as z:
        check(z.testzip()is None,'Archive CRC');j=lambda n:json.loads(z.read(n))
        rows=lambda n:[json.loads(l)for l in z.read(n).splitlines()if l.strip()]
        manifest=json.loads(path.with_name('manifest.json').read_text(encoding='utf-8'))
        check(sha(path.read_bytes())==manifest['archive_sha256'],'Archive SHA256')
        run=j('run.json');done=j('training_complete.json');prep=j('b650_preparation.json');cpu=j('native_cpu_acceptance.json')
        check(done['completed_updates']==650 and done['new_updates']==150 and 'stopped.json'not in z.namelist(),'Completed650/no stop')
        check(cpu['passed']and not cpu['gpu_initialized']and not cpu['model_weights_loaded'],'Cloud CPU acceptance')
        check(run['target_step']==650 and run['resume_step']==500 and run['phase_stops']==[650],'Fixed500->650')
        check(run['protocol_sha256']==done['protocol_sha256']==sha((ROOT/'experiments/group_weight_b500_v1/protocol.json').read_bytes()),'Frozen original recipe identity')
        check(run['implementation_manifest_sha256']==sha((ROOT/'experiments/group_weight_b500_v1/implementation_manifest.json').read_bytes())==cpu['implementation_manifest_sha256'],'Frozen weighting identity')
        check(sha(z.read('configs/phase650.yaml'))==run['config_hashes']['650'],'Active config hash')
        check(prep['source_crc_verified']and run['source_model_sha256']==prep['source_model_sha256']=='ffb6a3f4994ad0b1fa72b4c0dbf8db8e57adf430ea0d4e7b97975619b2125275','Same evaluated B500 full checkpoint')
        import yaml
        a=yaml.safe_load(z.read('configs/parent_phase500.yaml'));b=yaml.safe_load(z.read('configs/phase650.yaml'))
        for k in ['algorithm','data','shopping_dynamic_sampling']:check(a[k]==b[k],'Unchanged '+k)
        for k in ['actor','model']:check(a['actor_rollout_ref'][k]==b['actor_rollout_ref'][k],'Unchanged '+k)
        check(b['trainer']['total_training_steps']==650 and not b['trainer']['val_before_train'],'650 target/no repeated initial Dev')
        check(b['trainer']['resume_mode']=='resume_path'and b['trainer']['resume_from_path']==run['source_checkpoint'],'Actual resume source')
        check(b['ray_kwargs']['ray_init']['runtime_env']['env_vars']==run['env'],'Driver/worker environment identity')
        check(set(b['actor_rollout_ref']['actor']['checkpoint']['load_contents'])=={'model','optimizer','extra'},'Full restore components')
        es=rows('training_diagnostics.jsonl');updates=[e for e in es if e['event']=='optimizer_step']
        check([e['global_step']for e in updates]==list(range(1,651)),'Exactly1..650 update chain')
        previous=ROOT/'analysis/b500_training_return_20261118/20261120T150105Z/evidence.zip'
        with zipfile.ZipFile(previous)as pz:
            pe=[json.loads(l)for l in pz.read('training_diagnostics.jsonl').splitlines()]
            check(es[:len(pe)]==pe,'Exact committed B500 prefix preserved')
        wr=[w for w in rows('weights.jsonl')if w['event']=='actor_update_returned']
        check([w['driver_global_step_before_update']for w in wr]==list(range(500,650)),'Exactly150 new Actor returns')
        check(Counter(w['event']for w in rows('weights.jsonl'))==Counter(actor_update_started=150,actor_update_returned=150),'No failed Actor updates')
        composition=Counter();coeff=[];tokens=0
        for e,w in zip(updates[500:],wr):
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
        for step in (550,600,650):
            rs=rows(f'validation/{step}.jsonl');mean=sum(r['score']for r in rs)/50
            check(len(rs)==50 and all(r['step']==step for r in rs),'50 native validation records '+str(step))
            check(math.isclose(mean,updates[step-1]['metrics']['val-core/shopsimulator/reward/mean@1'],abs_tol=1e-12),'Native validation mean '+str(step))
            validations.append({'step':step,'score_equal1_count':sum(math.isclose(r['score'],1.,abs_tol=1e-6,rel_tol=0)for r in rs),'mean_reward':mean})
        gen=rows('generation_tokens.jsonl');counts={'train':sum(r['reserved_trajectories']for r in gen if not r['validation']),
            'validation_physical':sum(r['reserved_trajectories']for r in gen if r['validation'])}
        check(counts==j('generation_budget.json')==done['generation_budget'],'Cumulative generation ledger')
        old=j('parent_B500__generation_budget.json');extra={k:counts[k]-old[k]for k in counts}
        check(extra['validation_physical']==600,'Three padded native Dev passes')
        retention=rows('checkpoint_retention.jsonl');last=[r for r in retention if r['event']=='retention_plan'][-1]
        check(last['new_step']==650 and last['keep']==[600,650],'Latest2 retained')
        text=z.read('train_to_650.log').decode(errors='replace')
        check('Setting global step to 500'in text and 'Skipping dataloader state restore: global_steps=500'in text,'Original500 loader boundary rule')
        return {'audit_passed':not errors,'errors':errors,'return_commit':'7a112074f60c5cb215058ceb2fde5b3f7e11ff08',
            'archive_sha256':manifest['archive_sha256'],'completed_updates':650,'new_updates':150,'native_CPU_tests':cpu['tests_run'],
            'new_composition':dict(composition),'relative_reweight_updates':composition['FS'],'relative_reweight_fraction':composition['FS']/150,
            'coefficient_range':[min(coeff),max(coeff)],'new_Actor_tokens':tokens,'internal_validation':validations,
            'cumulative_generation':counts,'new_generation':extra,'retained_checkpoints_by_log':[600,650],
            'dataloader_epoch_boundary_skip_observed':True,'new_supervisor_elapsed_hours':done['elapsed_seconds']/3600,
            'B650_independent_Dev_done':done['independent_Dev_done'],'B650_Final_done':done['Final_done'],
            'native_score1_is_not_independent_gold_success':True}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.archive);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(0 if result['audit_passed']else 1)
