"""CPU-only fixed-checkpoint paired report; never fill missing attempts as zero."""
import argparse
import csv
import random
import statistics
from collections import Counter,defaultdict
from pathlib import Path
from checkpoint_pair_common import *
from run_checkpoint_pair import checked,latest_try,selected,slot_dir


def paired_stats(items):
    # Each item is one task with eight completed outcomes for each model.
    delta=[r['grpo_p']-r['sft_p'] for r in items]
    if not delta:return None
    rng=random.Random(20261111)
    draws=sorted(statistics.fmean(rng.choices(delta,k=len(delta))) for _ in range(5000))
    return {'tasks':len(items),'sft_mean':statistics.fmean(r['sft_p'] for r in items),
            'grpo_mean':statistics.fmean(r['grpo_p'] for r in items),'delta_mean':statistics.fmean(delta),
            'task_paired_bootstrap95':[draws[125],draws[4875]],
            'observed_improved':sum(d>0 for d in delta),'observed_same':sum(d==0 for d in delta),
            'observed_regressed':sum(d<0 for d in delta),
            'sft_tasks_any_success':sum(r['sft_successes']>0 for r in items),
            'grpo_tasks_any_success':sum(r['grpo_successes']>0 for r in items)}


def report(run,output):
    cfg=checked(run)
    if output.exists():raise ValueError('Report output must be new')
    groups=defaultdict(list);coverage=Counter();records=[];failure=defaultdict(Counter)
    for task in rows(PACKAGE/'tasks.jsonl'):
        row={'task_id':task['task_id'],'cohort':task['cohort'],'query':task['query']}
        for arm in ('sft','grpo'):
            successes=[];proxies=[]
            for attempt in range(8):
                slot=slot_dir(run,arm,task['task_id'],attempt);value=selected(slot)
                kind=(value or {}).get('classification','not_started');coverage[arm+':'+kind]+=1
                if kind=='completed':
                    if value['model']!=arm or value['task_id']!=task['task_id'] or value['attempt']!=attempt:
                        raise ValueError('Slot/result binding differs')
                    folder=latest_try(slot)
                    if file_hash(folder/'trajectory.json')!=value['trajectory_sha256']:raise ValueError('Trajectory was modified')
                    successes.append(value['strict_success']);proxies.append(value['training_gold_proxy'])
                    failure[arm][value['reward_type'] or value['status']]+=1
            row[arm+'_completed']=len(successes)
            row[arm+'_successes']=sum(successes)
            row[arm+'_p']=sum(successes)/8 if len(successes)==8 else None
            row[arm+'_proxy_successes']=sum(proxies)
        row['paired_complete']=row['sft_completed']==row['grpo_completed']==8
        row['delta']=row['grpo_p']-row['sft_p'] if row['paired_complete'] else None
        groups[task['cohort']].append(row);records.append(row)
    summaries={}
    for cohort,items in groups.items():
        complete=all(i['paired_complete'] for i in items)
        summaries[cohort]={'expected_tasks':len(items),'paired_complete_tasks':sum(i['paired_complete'] for i in items),
                           'complete':complete,'main_statistics':paired_stats(items) if complete else None}
    result={'version':VERSION,'complete':all(x['complete'] for x in summaries.values()),
            'coverage':dict(coverage),'cohorts':summaries,'reward_type_and_policy_failure_counts':{a:dict(c) for a,c in failure.items()},
            'run_manifest_sha256':file_hash(run/'run.json'),'protocol_sha256':cfg['protocol_sha256'],
            'scope':'Training-task checkpoint diagnostic, not held-out generalization or a task-scheduler experiment',
            'uncertainty':'5000 task-paired bootstrap resamples; no training-seed uncertainty; per-task 8 trials are sparse'}
    output.mkdir(parents=True);save(output/'summary.json',result);save(output/'task_pairs.json',records)
    with (output/'task_pairs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    lines=['# 固定checkpoint同题复评','',f"完整主比较：{result['complete']}。各组分开报告，不合成原1000题总体成绩。",'',
           '|队列|完成配对题数/计划|SFT|GRPO|差值|','|---|---:|---:|---:|---:|']
    for name,s in summaries.items():
        v=s['main_statistics']
        vals=f"{v['sft_mean']:.2%}|{v['grpo_mean']:.2%}|{v['delta_mean']:+.2%}" if v else '待完成|待完成|待完成'
        lines.append(f"|{name}|{s['paired_complete_tasks']}/{s['expected_tasks']}|{vals}|")
    lines+=['','策略失败保留在每题8次的分母；基础设施异常和未开始记录不填零，队列未完整时不输出该队列主差值。',
            '原始轨迹、结果、调用用量和补测尝试保留在运行目录；gold、预算和规格证据可回到原始终局明细审阅。',
            '每题8次的升降仅作描述，不能逐题宣布显著修复；best-of-8不是单次成功率。']
    (output/'复评汇总.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(dump({'report':str(output),'complete':result['complete'],'coverage':result['coverage']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.run_dir.resolve(),a.output.resolve())
