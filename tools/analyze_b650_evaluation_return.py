"""Verify fixed B650 suites and pair against A650 and prior B500."""
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
                'reference_type':detail(ar).get('reward_type','unknown'),'B650_type':detail(br).get('reward_type','unknown'),
                'reference_status':ar['status'],'B650_status':br['status'],'reference_steps':len(ar['steps']),'B650_steps':len(br['steps'])})
    return {'reference_successes':len(a),'B650_successes':len(b),'net_gain':len(b)-len(a),'net_percentage_points':100*(len(b)-len(a))/size,
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
        archives={'dev50':ROOT/'analysis/b500_independent_dev50_return/20261121T031644Z/evidence.zip',
            'final200_clean':ROOT/'analysis/b500_final200_return/20261121T040953Z/evidence.zip'}
        B500_summaries={}
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
            pa=dict(ob['protocol']);pb=dict(summaries[suite]['protocol']);pa.pop('model');pb.pop('model');check(pa==pb,'B500/B650 evaluation protocol '+suite)
            comparisons['vs_B500_'+suite]=pair(old,trajectories[suite])
        return {'audit_passed':not errors,'errors':errors,'return_commit':'c60b6efa1d1fae1f8c67908ca15f59050497e69b',
            'archive_sha256':manifest['archive_sha256'],'B650_summaries':{s:{k:v for k,v in b.items()if k in ['strict_successes','expected_tasks','mean_final_reward','average_steps','reward_type_counts']}for s,b in summaries.items()},
            'B500_summaries':{s:{k:v for k,v in b.items()if k in ['strict_successes','mean_final_reward','average_steps','reward_type_counts']}for s,b in B500_summaries.items()},
            'A650_summaries':{s:{k:v for k,v in b.items()if k in ['strict_successes','mean_final_reward','average_steps','reward_type_counts']}for s,b in A_summaries.items()},
            'comparisons':comparisons,'execution_records':execution,'model_sha256':export['merged_weights_sha256'],
            'interpretation':'Valid evaluation with mixed effects: worse Dev, modestly better Final; continuation not a consistent gain; no automatic further training.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(a.archive);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(0 if result['audit_passed']else 1)
