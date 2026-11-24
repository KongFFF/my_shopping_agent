"""Verify fixed C650 suites and pair against A650 and prior B500."""
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
POST='preparation/posttrain_step650_v1/'


def pair(A,B):
    a={r['task_id']for r in A if strict(r)};b={r['task_id']for r in B if strict(r)}
    gain=sorted(b-a);loss=sorted(a-b);n=len(gain)+len(loss);size=len(B)
    p=min(1.,2*sum(math.comb(n,k)for k in range(min(len(gain),len(loss))+1))/2**n)if n else 1.
    changed=[]
    for ar,br in zip(A,B):
        if ar['task_id']in gain+loss:
            detail=lambda r:(r.get('terminal_result')or{}).get('reward_detail')or{}
            changed.append({'task_id':ar['task_id'],'change':'B_only'if ar['task_id']in gain else'A_only',
                'reference_type':detail(ar).get('reward_type','unknown'),'C650_type':detail(br).get('reward_type','unknown'),
                'reference_status':ar['status'],'C650_status':br['status'],'reference_steps':len(ar['steps']),'C650_steps':len(br['steps'])})
    return {'reference_successes':len(a),'C650_successes':len(b),'net_gain':len(b)-len(a),'net_percentage_points':100*(len(b)-len(a))/size,
        'B_only':gain,'A_only':loss,'both_success':len(a&b),'both_fail':size-len(a|b),'exact_McNemar_two_sided_p':p,
        'paired_bootstrap_95_percentile_interval':paired_interval(len(gain),len(loss),size),'changed_tasks':changed}


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
        check(prep['passed']and prep['source_crc_verified']and prep['fixed_step']==650,'Cloud fixed650 preparation acceptance')
        check(set(spec['commands'])=={'smoke3','dev50','final200_clean'}and prep['Final_runs_regardless_of_Dev_score'],'Authorized fixed endpoint suites')
        check(spec['benchmarks']['final200_clean']['sha256']==FINAL_HASH,'Original Final Clean fingerprint')
        meta=j('posttrain.json');source=str(PurePosixPath(meta['training'])/'train/global_step_650')
        check(spec['checkpoint']==export['checkpoint']==source and export['actor_model_sha256']==meta['checkpoint_model_sha256'],'Prepared/exported650 identity')
        check(export['lora_rank']==16 and export['lora_alpha']==32,'LoRA identity')
        m=service['model'];check(service['passed']and m['id']==spec['model_tag']and m['root']==export['merged_model']and m['max_model_len']==24576,'Served model identity')
        check(spec['model_tag']=='shopping-agent-c650-gamma4-v1'and prep['gamma']==4,'C650 gamma4 model identity')
        from c650_evaluation import audit_training_packet
        import tempfile
        with tempfile.TemporaryDirectory(prefix='c650-eval-local-',dir=ROOT/'outputs')as d:
            temp=Path(d);(temp/'evidence.zip').write_bytes(z.read('training_evidence/training_snapshot.zip'));(temp/'manifest.json').write_bytes(z.read('training_evidence/manifest.json'))
            training=audit_training_packet(temp/'evidence.zip',temp/'checked')
            with zipfile.ZipFile(temp/'evidence.zip')as tz:
                run=json.loads(tz.read('run.json'))
        check(training==j('training_evidence/training_audit.json'),'Cloud/local training snapshot audit equivalence')
        identity=j('training_evidence/training_identity.json')
        check(identity['passed']and identity['fixed_step']==650 and identity['gamma']==4 and identity['training']==run['output']==meta['training'],'Training/evaluation binding')
        packet_acceptance={'passed':training['audit_passed'],'completed_updates':training['completed_updates'],'new_updates':training['new_updates'],
            'snapshot_sha256':sha(z.read('training_evidence/training_snapshot.zip')),'native_CPU_tests':training['native_CPU_tests'],'native_additional_tests':training['native_additional_tests']}
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
            frozen=json.loads((ROOT/'experiments/group_weight_c500_v1/protocol.json').read_text(encoding='utf-8'))['evaluation']['protocol']
            check(b['protocol'].get('model')==spec['model_tag']and all(b['protocol'].get(k)==v for k,v in frozen.items()if k!='benchmark'),'Frozen actual C protocol '+suite)
            bad=[r['task_id']for r in rs if (r.get('error')and r['error'].get('type')!='ContextBudgetError')or r.get('release_error')]
            check(not bad,'No infrastructure/release failures '+suite)
            execution[suite]={'status_counts':dict(Counter(r['status']for r in rs)),
                'context_error_ids':[r['task_id']for r in rs if (r.get('error')or{}).get('type')=='ContextBudgetError'],
                'invalid_action_limit_ids':[r['task_id']for r in rs if r['status']=='invalid_action_limit']}
            types=Counter(((r.get('terminal_result')or{}).get('reward_detail')or{}).get('reward_type','unknown')for r in rs)
            check(dict(types)==b['reward_type_counts'],'Reward types '+suite)
            summaries[suite]=b;trajectories[suite]=rs
            if suite!='smoke3':
                ar=rows('reference_A650/'+suite+'/trajectories.jsonl');a=j(POST+'A650_'+suite+'_summary.json')
                check([r['task_id']for r in ar]==ids,'A650 task alignment '+suite)
                check(sorted(r['task_id']for r in ar if strict(r))==a['strict_success_task_ids']and sum(strict(r)for r in ar)==a['strict_successes'],'A650 strict result '+suite)
                check(math.isclose(sum(r['final_reward']for r in ar)/n,a['mean_final_reward'],abs_tol=1e-12),'A650 reward '+suite)
                check(math.isclose(sum(len(r['steps'])for r in ar)/n,a['average_steps'],abs_tol=1e-12),'A650 steps '+suite)
                types=Counter(((r.get('terminal_result')or{}).get('reward_detail')or{}).get('reward_type','unknown')for r in ar)
                check(dict(types)==a['reward_type_counts'],'A650 reward types '+suite)
                A_summaries[suite]=a
                pa=dict(a['protocol']);pb=dict(b['protocol']);pa.pop('model');pb.pop('model');check(pa==pb,'Original A/B protocol '+suite)
                c=pair(ar,rs);cloud=j(POST+('dev50_complete.json'if suite=='dev50'else'final200_complete.json'))
                check(c['B_only']==cloud['B_only']and c['A_only']==cloud['A_only']and c['net_gain']==cloud['net_gain'],'Cloud pairs '+suite)
                comparisons['vs_A650_'+suite]=c
        old_summaries={};review_cases=[]
        source_archives={'B650':ROOT/'analysis/b650_evaluation_return/20261122T011337Z/evidence.zip',
            'C500':ROOT/'analysis/c500_evaluation_return/20261124T052234Z/evidence.zip'}
        for label,p in source_archives.items():
            check(sha(p.read_bytes())==json.loads(p.with_name('manifest.json').read_text(encoding='utf-8'))['archive_sha256'],'Reference archive SHA '+label)
            with zipfile.ZipFile(p)as oldz:
                check(oldz.testzip()is None,'Reference ZIP CRC '+label)
                old_summaries[label]={}
                for suite in ['dev50','final200_clean']:
                    old=[json.loads(line)for line in oldz.read('evaluation/'+suite+'/trajectories.jsonl').splitlines()if line.strip()]
                    summary=json.loads(oldz.read('evaluation/'+suite+'/summary.json'));old_summaries[label][suite]=summary
                    if label=='B650':
                        old_A=json.loads(oldz.read('preparation/posttrain_step650_v1/A650_'+suite+'_summary.json'))
                        check(old_A['strict_success_task_ids']==A_summaries[suite]['strict_success_task_ids'],'Archived A650 success IDs '+suite)
                    check([r['task_id']for r in old]==[r['task_id']for r in trajectories[suite]],'Task order '+label+' '+suite)
                    check(sorted(r['task_id']for r in old if strict(r))==summary['strict_success_task_ids'],'Reference strict '+label+' '+suite)
                    pp=dict(summary['protocol']);cc=dict(summaries[suite]['protocol']);pp.pop('model');cc.pop('model');check(pp==cc,'Reference protocol '+label+' '+suite)
                    comparison=pair(old,trajectories[suite]);comparisons['vs_'+label+'_'+suite]=comparison
                    from analyze_c500_evaluation_return import public_cases
                    cards=public_cases(old,trajectories[suite],comparison['B_only']+comparison['A_only'],suite)
                    for card in cards:
                        card['reference_model']=label
                        reference_model=card['models'].pop('B500');current_model=card['models'].pop('C500')
                        card['models']={label:reference_model,'C650':current_model}
                        review_cases.append(card)
                    user=lambda r:next(m['content']for m in r['messages']if m['role']=='user')
                    system=lambda r:[m['content']for m in r['messages']if m['role']=='system']
                    ar=rows('reference_A650/'+suite+'/trajectories.jsonl')
                    check(all(user(a)==user(b)==user(c)and system(a)==system(b)==system(c)for a,b,c in zip(ar,old,trajectories[suite])),'Actual prompt equality '+label+' '+suite)
        check(A_summaries['dev50']['strict_successes']==26 and A_summaries['final200_clean']['strict_successes']==141,'Expected A650 scores')
        check(old_summaries['B650']['dev50']['strict_successes']==25 and old_summaries['B650']['final200_clean']['strict_successes']==143,'Expected B650 scores')
        check(old_summaries['C500']['dev50']['strict_successes']==25 and old_summaries['C500']['final200_clean']['strict_successes']==138,'Expected C500 scores')
        return {'audit_passed':not errors,'errors':errors,'return_commit':'6a91906f9e73482afe5c43d18d8bf34aeae60d55',
            'archive_sha256':manifest['archive_sha256'],'training_packet_acceptance':packet_acceptance,'C650_summaries':{s:{k:v for k,v in b.items()if k in ['strict_successes','expected_tasks','mean_final_reward','average_steps','reward_type_counts']}for s,b in summaries.items()},
            'reference_summaries':{label:{s:{k:v for k,v in b.items()if k in ['strict_successes','mean_final_reward','average_steps','reward_type_counts']}for s,b in sm.items()}for label,sm in old_summaries.items()},
            'A650_summaries':{s:{k:v for k,v in b.items()if k in ['strict_successes','mean_final_reward','average_steps','reward_type_counts']}for s,b in A_summaries.items()},
            'public_cases':review_cases,'comparisons':comparisons,'execution_records':execution,'model_sha256':export['merged_weights_sha256'],
            'interpretation':'C650 below A650 and B650 on both suites and below C500; no consistent continuation benefit or automatic further training.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(a.archive);a.output.parent.mkdir(parents=True,exist_ok=True)
    (a.output.parent/'paired_public_evidence.json').write_text(json.dumps({'scope':'Exploratory actual user and nonterminal Actor observation excerpts; no hidden goals as semantic truth.','cases':result.pop('public_cases')},ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(0 if result['audit_passed']else 1)
