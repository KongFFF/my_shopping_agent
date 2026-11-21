"""Recompute fixed B500 independent Dev50 and paired historical A comparison."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path,PurePosixPath
import zipfile

ROOT=Path(__file__).resolve().parents[1]
POST='preparation/posttrain_step500_v1/'


def strict(r):
    t=r.get('terminal_result')or {};d=t.get('reward_detail')or {}
    return (d.get('reward_version')=='shopsimulator-reward-v3'and r.get('status')=='done'and r.get('done')is True
        and t.get('done')is True and t.get('over')is True and d.get('reward_type')=='gold_purchase'
        and d.get('reward_valid')is True and d.get('purchase_success')is True and d.get('termination_reason')=='gold_purchase')


def paired_interval(gain,loss,n):
    # Exact percentile distribution of the nonparametric paired bootstrap.
    dist={0:1.}
    for _ in range(n):
        nxt={}
        for k,v in dist.items():
            for delta,p in [(1,gain/n),(-1,loss/n),(0,1-(gain+loss)/n)]:nxt[k+delta]=nxt.get(k+delta,0)+v*p
        dist=nxt
    def q(target):
        total=0.
        for k in sorted(dist):
            total+=dist[k]
            if total>=target:return k/n
    return [q(.025),q(.975)]


def analyze(archive):
    errors=[]
    def check(ok,label):
        if not ok:errors.append(label)
    with zipfile.ZipFile(archive)as z:
        check(z.testzip()is None,'Archive CRC')
        sha=lambda b:hashlib.sha256(b).hexdigest()
        manifest=json.loads(archive.with_name('manifest.json').read_text(encoding='utf-8'))
        check(sha(archive.read_bytes())==manifest['archive_sha256'],'Archive SHA256')
        j=lambda n:json.loads(z.read(n))
        rows=lambda n:[json.loads(l)for l in z.read(n).splitlines()if l.strip()]
        spec=j(POST+'evaluation_spec.json');export=j(POST+'export_receipt.json');services=j(POST+'services_ready.json')
        check(set(spec['commands'])=={'smoke3','dev50'}and set(spec['benchmarks'])=={'smoke3','dev50'},'Only smoke/Dev')
        check(j(POST+'pipeline_exit_status.json')['exit_code']==0 and j(POST+'dev50_complete.json')['passed'],'Completed pipeline')
        check(j(POST+'preparation_receipt.json')['source_crc_verified'],'Cloud checkpoint CRC receipt')
        check(export['lora_rank']==16 and export['lora_alpha']==32,'LoRA identity')
        training=PurePosixPath(j('posttrain.json')['training'])
        check(export['checkpoint']==str(training/'train/global_step_500')and spec['checkpoint']==export['checkpoint'],'Fixed500 checkpoint identity')
        check(services['passed']and services['model']['id']==spec['model_tag']and services['model']['root']==export['merged_model']and services['model']['max_model_len']==24576,'Serving model identity')
        for suite,n in [('smoke3',3),('dev50',50)]:
            folder='evaluation/'+suite+'/';rs=rows(folder+'trajectories.jsonl');summary=j(folder+'summary.json');passed=j(folder+'passed.json')
            check(passed['passed']and passed['tasks']==n,'Suite accepted '+suite)
            check(sha(z.read(folder+'trajectories.jsonl'))==passed['trajectory_sha256']and sha(z.read(folder+'summary.json'))==passed['summary_sha256'],'Suite hashes '+suite)
            check([r['task_id']for r in rs]==spec['benchmarks'][suite]['task_ids']and len({r['task_id']for r in rs})==n,'Task identity/order '+suite)
            success=sorted(r['task_id']for r in rs if strict(r))
            check(len(success)==summary['strict_successes']and success==summary['strict_success_task_ids'],'Strict gold recomputation '+suite)
            check(not summary['missing_tasks']and summary['completed_tasks']==n,'Complete denominator '+suite)
            check(math.isclose(sum(r['final_reward']for r in rs)/n,summary['mean_final_reward'],abs_tol=1e-12),'Reward recomputation '+suite)
            check(math.isclose(sum(len(r['steps'])for r in rs)/n,summary['average_steps'],abs_tol=1e-12),'Steps recomputation '+suite)
            check(not any((r.get('error')and r['error'].get('type')!='ContextBudgetError')or r.get('release_error')for r in rs),'Execution failures '+suite)
        B=rows('evaluation/dev50/trajectories.jsonl');Bs=j('evaluation/dev50/summary.json')
        base=ROOT/'analysis/grpo_baseline_20261022_202816'
        A=[json.loads(l)for l in (base/'grpo_dev50/trajectories.jsonl').read_text(encoding='utf-8').splitlines()if l.strip()]
        As=json.loads((base/'grpo_dev50/summary.json').read_text(encoding='utf-8'))
        S=json.loads((base/'sft_dev50/summary.json').read_text(encoding='utf-8'))
        check([r['task_id']for r in A]==[r['task_id']for r in B],'A/B task order')
        pa=dict(As['protocol']);pb=dict(Bs['protocol']);pa.pop('model');pb.pop('model');check(pa==pb,'Original independent protocol')
        frozen=json.loads((ROOT/'experiments/group_weight_b500_v1/protocol.json').read_text(encoding='utf-8'))['evaluation']['protocol']
        check(all(pb.get(k)==v for k,v in frozen.items()),'Frozen B evaluation protocol')
        sa={r['task_id']for r in A if strict(r)};sb={r['task_id']for r in B if strict(r)}
        check(len(sa)==As['strict_successes']==27 and S['strict_successes']==24,'Historical A/SFT reference')
        gain=sorted(sb-sa);loss=sorted(sa-sb);discord=len(gain)+len(loss)
        p=min(1.,2*sum(math.comb(discord,k)for k in range(min(len(gain),len(loss))+1))/2**discord)if discord else 1.
        changed=[];transitions=Counter()
        for a,b in zip(A,B):
            da=(a.get('terminal_result')or{}).get('reward_detail')or{};db=(b.get('terminal_result')or{}).get('reward_detail')or{}
            transitions[da.get('reward_type','unknown')+' -> '+db.get('reward_type','unknown')]+=1
            if a['task_id']in gain+loss:changed.append({'task_id':a['task_id'],'change':'B_only'if a['task_id']in gain else'A_only',
                'A_type':da.get('reward_type'),'B_type':db.get('reward_type'),'A_steps':len(a['steps']),'B_steps':len(b['steps'])})
        cloud=j(POST+'dev50_complete.json');screen=len(sb)>=30 and len(sb)-len(sa)>=3
        check(cloud['B_only']==gain and cloud['A_only']==loss and cloud['continuation_screen_passed']==screen,'Cloud paired-result consistency')
        return {'audit_passed':not errors,'errors':errors,'return_commit':'8f9dd51fa7fc952bff58625e610918feec1bdbc9',
            'archive_sha256':manifest['archive_sha256'],'SFT_successes':S['strict_successes'],'A500_successes':len(sa),'B500_successes':len(sb),
            'net_success_gain':len(sb)-len(sa),'net_percentage_points':100*(len(sb)-len(sa))/50,'B_only':gain,'A_only':loss,
            'both_success':len(sa&sb),'both_fail':50-len(sa|sb),'exact_McNemar_two_sided_p':p,
            'paired_bootstrap_95_percentile_interval':paired_interval(len(gain),len(loss),50),
            'A_mean_reward':As['mean_final_reward'],'B_mean_reward':Bs['mean_final_reward'],
            'A_average_steps':As['average_steps'],'B_average_steps':Bs['average_steps'],
            'A_reward_types':As['reward_type_counts'],'B_reward_types':Bs['reward_type_counts'],
            'changed_tasks':changed,'reward_type_transitions':dict(transitions),'continuation_screen_passed':screen,
            'additional_net_successes_needed_for_frozen_screen':max(0,30-len(sb)),
            'Final_done':False,'recommendation':'Do not advance this fixed0.5 weighting configuration to800 under frozen screen; preserve result and examine paired regressions with existing data.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(a.archive);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(0 if result['audit_passed']else 1)
