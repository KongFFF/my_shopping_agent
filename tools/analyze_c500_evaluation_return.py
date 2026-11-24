"""Verify fixed C500 suites and pair against A500 and prior B500."""
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
POST='preparation/posttrain_step500_v1/'


def pair(A,B):
    a={r['task_id']for r in A if strict(r)};b={r['task_id']for r in B if strict(r)}
    gain=sorted(b-a);loss=sorted(a-b);n=len(gain)+len(loss);size=len(B)
    p=min(1.,2*sum(math.comb(n,k)for k in range(min(len(gain),len(loss))+1))/2**n)if n else 1.
    changed=[]
    for ar,br in zip(A,B):
        if ar['task_id']in gain+loss:
            detail=lambda r:(r.get('terminal_result')or{}).get('reward_detail')or{}
            changed.append({'task_id':ar['task_id'],'change':'B_only'if ar['task_id']in gain else'A_only',
                'reference_type':detail(ar).get('reward_type','unknown'),'C500_type':detail(br).get('reward_type','unknown'),
                'reference_status':ar['status'],'C500_status':br['status'],'reference_steps':len(ar['steps']),'C500_steps':len(br['steps'])})
    return {'reference_successes':len(a),'C500_successes':len(b),'net_gain':len(b)-len(a),'net_percentage_points':100*(len(b)-len(a))/size,
        'B_only':gain,'A_only':loss,'both_success':len(a&b),'both_fail':size-len(a|b),'exact_McNemar_two_sided_p':p,
        'paired_bootstrap_95_percentile_interval':paired_interval(len(gain),len(loss),size),'changed_tasks':changed}


def public_cases(reference,current,ids,suite):
    cards=[]
    for a,c in zip(reference,current):
        if c['task_id']not in ids:continue
        models={}
        for label,r in [('B500',a),('C500',c)]:
            turns=[]
            for i,step in enumerate(r['steps']):
                obs=step['observation']
                fields=[]if step.get('done')else[line for line in obs.splitlines()if line.startswith(('asin:','title:','price:','key_attributes:','selected_options:','available_options:','subpage:','content:'))]
                turns.append({'turn':i+1,'tool':step['tool_name'],'args':step['parameters'],'public_fields':fields,
                    'observation_sha256':hashlib.sha256(obs.encode()).hexdigest(),'terminal_observation_excluded':bool(step.get('done'))})
            models[label]={'query':next(m['content']for m in r['messages']if m['role']=='user'),'status':r['status'],'turns':turns}
        cards.append({'suite':suite,'task_id':c['task_id'],'direction':'C_gain'if strict(c)else'C_loss','models':models})
    return cards


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
        check(prep['passed']and prep['source_crc_verified']and prep['fixed_step']==500,'Cloud fixed500 preparation acceptance')
        check(set(spec['commands'])=={'smoke3','dev50','final200_clean'}and prep['Final_runs_regardless_of_Dev_score'],'Authorized fixed endpoint suites')
        check(spec['benchmarks']['final200_clean']['sha256']==FINAL_HASH,'Original Final Clean fingerprint')
        meta=j('posttrain.json');source=str(PurePosixPath(meta['training'])/'train/global_step_500')
        check(spec['checkpoint']==export['checkpoint']==source and export['actor_model_sha256']==meta['checkpoint_model_sha256'],'Prepared/exported500 identity')
        check(export['lora_rank']==16 and export['lora_alpha']==32,'LoRA identity')
        m=service['model'];check(service['passed']and m['id']==spec['model_tag']and m['root']==export['merged_model']and m['max_model_len']==24576,'Served model identity')
        check(spec['model_tag']=='shopping-agent-c500-gamma4-v1'and prep['gamma']==4,'C500 gamma4 model tag')
        from audit_c500_training_return import audit as training_audit
        import tempfile
        packet='training_evidence/'
        combined=j(packet+'combined_acceptance.json');supp=j(packet+'supplement_acceptance.json');cpu=j(packet+'native_cpu_acceptance.json')
        check(combined['passed']and combined['training_core_passed']and combined['supplement_verified'],'Combined training acceptance')
        check(supp['passed']and supp['native_CPU_original_received']and supp['prior_disk_zero_update_verified']and supp['CPU_tests_rerun']is False,'Original supplement')
        check(cpu['version']=='c500-cpu-acceptance-v1'and cpu['passed']and cpu['tests_run']==36 and not cpu['errors']and all(c['matched']for c in cpu['checks']),'Native CPU checks')
        check(cpu['training_started']is False and cpu['model_weights_loaded']is False and cpu['gpu_initialized']is False,'CPU no models/CUDA/training')
        pkg=ROOT/'experiments/group_weight_c500_v1'
        check(cpu['protocol_sha256']==sha((pkg/'protocol.json').read_bytes())and cpu['implementation_manifest_sha256']==sha((pkg/'implementation_manifest.json').read_bytes()),'Native CPU identity')
        previous=j(packet+'prior/run.json');previous_stop=j(packet+'prior/stopped.json')
        check(previous['training_started']is False and 'disk'in previous_stop['error'].lower(),'Prior disk preflight before training')
        check(previous['protocol_sha256']==cpu['protocol_sha256']and previous['implementation_manifest_sha256']==cpu['implementation_manifest_sha256'],'Prior same protocol')
        with tempfile.TemporaryDirectory(prefix='c500-return-audit-')as directory:
            temp=Path(directory);(temp/'evidence.zip').write_bytes(z.read(packet+'training_snapshot.zip'));(temp/'manifest.json').write_bytes(z.read(packet+'manifest.json'))
            training=training_audit(temp/'evidence.zip')
        check(training['training_core_passed']and not training['errors'],'Recomputed complete training packet')
        cloud_training=j(packet+'training_audit.json')
        for key in ['completed_updates','actual_gamma','composition','Actor_tokens','generation_budget','max_advantage_error']:
            check(cloud_training[key]==training[key],'Cloud/local training audit '+key)
        check(training['run']==meta['training']==combined['source_run'],'Training/export/evaluation binding')
        packet_acceptance={'passed':not errors,'native_CPU_tests':cpu['tests_run'],'native_torch':cpu['torch_version'],
            'original_gaps_closed':True,'training_core_recomputed':training['training_core_passed'],
            'completed_updates':training['completed_updates'],'gamma':training['actual_gamma'],'composition':training['composition'],
            'native_snapshot_sha256':sha(z.read(packet+'training_snapshot.zip')),
            'note':'Historical archive audit still labels prior gap by filename; original CPU and prior scene are now separately present and native prepare accepted them.'}
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
            frozen= json.loads((ROOT/'experiments/group_weight_c500_v1/protocol.json').read_text(encoding='utf-8'))['evaluation']['protocol']
            check(b['protocol'].get('model')==spec['model_tag']and all(b['protocol'].get(k)==v for k,v in frozen.items()if k!='benchmark'),'Frozen C actual protocol '+suite)
            bad=[r['task_id']for r in rs if (r.get('error')and r['error'].get('type')!='ContextBudgetError')or r.get('release_error')]
            check(not bad,'No infrastructure/release failures '+suite)
            execution[suite]={'status_counts':dict(Counter(r['status']for r in rs)),
                'context_error_ids':[r['task_id']for r in rs if (r.get('error')or{}).get('type')=='ContextBudgetError'],
                'invalid_action_limit_ids':[r['task_id']for r in rs if r['status']=='invalid_action_limit']}
            types=Counter(((r.get('terminal_result')or{}).get('reward_detail')or{}).get('reward_type','unknown')for r in rs)
            check(dict(types)==b['reward_type_counts'],'Reward types '+suite)
            summaries[suite]=b;trajectories[suite]=rs
            if suite!='smoke3':
                ar=rows('reference_A500/'+suite+'/trajectories.jsonl');a=j(POST+'A500_'+suite+'_summary.json')
                check([r['task_id']for r in ar]==ids,'A500 task alignment '+suite)
                check(sorted(r['task_id']for r in ar if strict(r))==a['strict_success_task_ids']and sum(strict(r)for r in ar)==a['strict_successes'],'A500 strict result '+suite)
                check(math.isclose(sum(r['final_reward']for r in ar)/n,a['mean_final_reward'],abs_tol=1e-12),'A500 reward '+suite)
                check(math.isclose(sum(len(r['steps'])for r in ar)/n,a['average_steps'],abs_tol=1e-12),'A500 steps '+suite)
                types=Counter(((r.get('terminal_result')or{}).get('reward_detail')or{}).get('reward_type','unknown')for r in ar)
                check(dict(types)==a['reward_type_counts'],'A500 reward types '+suite)
                A_summaries[suite]=a
                pa=dict(a['protocol']);pb=dict(b['protocol']);pa.pop('model');pb.pop('model');check(pa==pb,'Original A/B protocol '+suite)
                c=pair(ar,rs);cloud=j(POST+('dev50_complete.json'if suite=='dev50'else'final200_complete.json'))
                check(c['B_only']==cloud['B_only']and c['A_only']==cloud['A_only']and c['net_gain']==cloud['net_gain'],'Cloud pairs '+suite)
                comparisons['vs_A500_'+suite]=c
        archives={'dev50':ROOT/'analysis/b500_independent_dev50_return/20261121T031644Z/evidence.zip',
            'final200_clean':ROOT/'analysis/b500_final200_return/20261121T040953Z/evidence.zip'}
        B500_summaries={};review_cases=[]
        for suite,p in archives.items():
            old_manifest=json.loads(p.with_name('manifest.json').read_text(encoding='utf-8'))
            check(sha(p.read_bytes())==old_manifest['archive_sha256'],'B500 archive SHA256 '+suite)
            with zipfile.ZipFile(p)as bz:
                check(bz.testzip()is None,'B500 ZIP CRC '+suite)
                old=[json.loads(l)for l in bz.read('evaluation/'+suite+'/trajectories.jsonl').splitlines()if l.strip()]
                B500_summaries[suite]=json.loads(bz.read('evaluation/'+suite+'/summary.json'))
            check([r['task_id']for r in old]==[r['task_id']for r in trajectories[suite]],'B500 task alignment '+suite)
            ob=B500_summaries[suite]
            check(sorted(r['task_id']for r in old if strict(r))==ob['strict_success_task_ids']and sum(strict(r)for r in old)==ob['strict_successes'],'B500 strict result '+suite)
            pa=dict(ob['protocol']);pb=dict(summaries[suite]['protocol']);pa.pop('model');pb.pop('model');check(pa==pb,'B500/C500 evaluation protocol '+suite)
            comparisons['vs_B500_'+suite]=pair(old,trajectories[suite])
            comp=comparisons['vs_B500_'+suite]
            review_cases.extend(public_cases(old,trajectories[suite],comp['B_only']+comp['A_only'],suite))
            first_user=lambda r:next(m['content']for m in r['messages']if m['role']=='user')
            system=lambda r:[m['content']for m in r['messages']if m['role']=='system']
            ar=rows('reference_A500/'+suite+'/trajectories.jsonl')
            check(all(first_user(a)==first_user(b)==first_user(c)and system(a)==system(b)==system(c)for a,b,c in zip(ar,old,trajectories[suite])),'Actual A/B/C user and system prompts '+suite)

        original_dev=json.loads((ROOT/'analysis/grpo_baseline_20261022_202816/grpo_dev50/summary.json').read_text(encoding='utf-8'))
        check(A_summaries['dev50']['strict_success_task_ids']==original_dev['strict_success_task_ids'],'Archived A500 Dev success IDs')
        with zipfile.ZipFile(archives['final200_clean'])as bz:
            old_A=json.loads(bz.read('preparation/posttrain_step500_v1/A500_final_summary.json'))
        check(A_summaries['final200_clean']['strict_success_task_ids']==old_A['strict_success_task_ids'],'Archived A500 Final success IDs')
        check(A_summaries['dev50']['strict_successes']==27 and A_summaries['final200_clean']['strict_successes']==137,'Original A500 scores')
        check(B500_summaries['dev50']['strict_successes']==28 and B500_summaries['final200_clean']['strict_successes']==141,'Original B500 scores')
        return {'audit_passed':not errors,'errors':errors,'return_commit':'f5632f3dfa2cf43d98247f8d013bc8caa6f9b5be',
            'archive_sha256':manifest['archive_sha256'],'training_packet_acceptance':packet_acceptance,'C500_summaries':{s:{k:v for k,v in b.items()if k in ['strict_successes','expected_tasks','mean_final_reward','average_steps','reward_type_counts']}for s,b in summaries.items()},
            'B500_summaries':{s:{k:v for k,v in b.items()if k in ['strict_successes','mean_final_reward','average_steps','reward_type_counts']}for s,b in B500_summaries.items()},
            'A500_summaries':{s:{k:v for k,v in b.items()if k in ['strict_successes','mean_final_reward','average_steps','reward_type_counts']}for s,b in A_summaries.items()},
            'paired_public_cases':review_cases,'comparisons':comparisons,'execution_records':execution,'model_sha256':export['merged_weights_sha256'],
            'interpretation':'Gamma4 C500 scores below gamma2 B500 by3 tasks on each suite; not a replicated or monotonic weight-response proof, no automatic continuation.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(a.archive);a.output.parent.mkdir(parents=True,exist_ok=True)
    evidence={'scope':'All B500/C500 strict status changes; actual first user message and nonterminal Actor observations. No hidden goal/options as semantic evidence; exploratory assistant review only.','cases':result.pop('paired_public_cases')}
    (a.output.parent/'paired_public_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(0 if result['audit_passed']else 1)
