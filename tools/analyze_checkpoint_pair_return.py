"""Audit and analyze returned checkpoint-pair artifacts; no inference/API calls."""
import statistics
from collections import Counter,defaultdict
from pathlib import Path
from checkpoint_pair_common import *
from report_checkpoint_pair import paired_stats

RUN=ROOT/'outputs/checkpoint_pair_v1/run_01'
OUT=ROOT/'analysis/checkpoint_pair_result_20261112'


def purchase_signature(trace):
    purchase=(trace.get('terminal_result') or {}).get('purchase') or {}
    if not purchase.get('asin') or not isinstance(purchase.get('price'),(int,float)):return None
    available={}
    for step in trace['steps']:
        s=step['result'].get('observation_state') or {}
        if (s.get('product') or {}).get('asin')==purchase['asin']:
            available=s.get('available_options') or available
    options={str(k).casefold():str(v).casefold() for k,v in (purchase.get('options') or {}).items()}
    single={str(k).casefold():str(v[0]).casefold() for k,v in available.items() if isinstance(v,list) and len(v)==1}
    collapsed={k:v for k,v in options.items() if single.get(k)!=v}
    return {'asin':purchase['asin'],'price':purchase['price'],'options':options,'collapsed':collapsed,'single':single}


def main():
    manifest=check_package();cfg=load(RUN/'run.json');returned=load(RUN/'report_final/summary.json')
    assert cfg['protocol_sha256']==file_hash(PACKAGE/'manifest.json')==returned['protocol_sha256']
    assert file_hash(RUN/'run.json')==returned['run_manifest_sha256']
    assert cfg['settings']==SETTINGS and cfg['vllm_version']=='0.25.1'
    assert cfg['runtime_sha256']==digest(dump(load(PACKAGE/'runtime_receipt.json')))
    cpu=load(ROOT/'outputs/checkpoint_pair_v1/cpu_check_01/cpu_check.json')
    assert cpu==cfg['cpu_check'] and cpu['passed'] and len(cpu['tasks'])==3
    assert load(RUN/'pipeline_exit.json')['exit_code']==0
    for arm in ('sft','grpo'):
        assert Path(cfg['models'][arm]['path']).as_posix().endswith(MODEL_PATHS[arm])
        for phase,expected in [('smoke',3),('full',2096)]:
            proof=load(RUN/f'{phase}_{arm}.json');ready=load(RUN/f'{arm}_{phase}_ready.json')
            assert proof['passed'] and proof['completed_slots']==proof['expected_slots']==expected
            assert ready['identity']['code_sha256']==manifest['code_sha256']
            assert ready['identity']['runtime_sha256']==cfg['runtime_sha256']
            assert ready['identity']['index_sha256']==cfg['index_sha256']
            assert ready['model']['root']==cfg['models'][arm]['path']
            assert ready['model']['id']=='checkpoint-pair-'+arm and ready['model']['max_model_len']==24576
            assert (RUN/f'{arm}_{phase}_services_stopped.json').exists()
    for n in ('tokenizer.json','tokenizer_config.json','chat_template.jinja'):
        assert cfg['models']['sft']['files'][n]==cfg['models']['grpo']['files'][n]
    # Use the independently rebuilt local pinned baseline only for pure scoring.
    local_bundle=ROOT/'outputs/checkpoint_pair_v1/runtime'
    assert runtime_receipt(local_bundle)==load(PACKAGE/'runtime_receipt.json')
    load_agent(local_bundle)
    metrics=defaultdict(lambda:defaultdict(list));counts=Counter();usage_totals=Counter();details=[]
    ids=set();artifact_pairs=[];all_slots=set()
    tasks=rows(PACKAGE/'tasks.jsonl')
    for task in tasks:
        task_records={'sft':[],'grpo':[]}
        row={'task_id':task['task_id'],'cohort':task['cohort'],'query':task['query']}
        for arm in ('sft','grpo'):
            records=[]
            for attempt in range(8):
                slot=RUN/arm/'attempts'/f'{task["task_id"]}_{attempt}';tries=sorted(slot.glob('try_*'))
                assert tries, str(slot)
                counts['all_tries']+=len(tries);counts['repaired_slots']+=len(tries)>1
                last=tries[-1];res=load(last/'result.json');rawpath=last/'trajectory.json'
                assert file_hash(rawpath)==res['trajectory_sha256']
                trace=load(rawpath);start=load(last/'start.json')
                assert (res['model'],res['task_id'],res['attempt'])==(arm,task['task_id'],attempt)
                assert (start['model'],start['task_id'],start['attempt'],start['query_sha256'])==(arm,task['task_id'],attempt,task['query_sha256'])
                assert trace['task_id']==task['task_id'] and trace['attempt_index']==attempt
                assert trace['trajectory_id'] not in ids;ids.add(trace['trajectory_id'])
                assert len(trace['steps'])<=35 and not trace.get('release_error')
                assert trace['initial_result']['instruction'].removeprefix('Instruction: ')==task['query']
                assert trace['initial_result']['environment_version']=='shopsimulator-environment-v2.1'
                recomputed=outcome(trace)
                for k,v in recomputed.items():assert res[k]==v,(arm,task['task_id'],attempt,k)
                assert res['classification']=='completed'
                error=(trace.get('error') or {}).get('type')
                if error:counts[arm+':error:'+error]+=1
                if (last/'usage.jsonl').exists():
                    for u in rows(last/'usage.jsonl'):
                        assert u['seed']==seed_for(task['task_id'],attempt,u['turn'])
                        counts[arm+':model_calls']+=1
                        if u['status']!='ok':counts[arm+':failed_requests']+=1
                        else:
                            for key in ('prompt_tokens','completion_tokens'):
                                usage_totals[arm+':'+key]+=(u.get('usage') or {}).get(key,0)
                counts[arm+':strict_proxy_mismatch']+=res['strict_success']!=res['training_gold_proxy']
                metrics[task['cohort']][arm].append(res)
                sig=purchase_signature(trace)
                task_records[arm].append((res,sig,str(last.relative_to(RUN))))
                records.append(res);all_slots.add((arm,f'{task["task_id"]}_{attempt}'))
            row[arm+'_successes']=sum(r['strict_success'] for r in records);row[arm+'_p']=row[arm+'_successes']/8
        row['delta']=row['grpo_p']-row['sft_p'];details.append(row)
        # Screen only for a potential redundant-single-axis scoring artifact.
        # This is not independent semantic adjudication or causal attribution.
        found=None
        for a,sa,pa in task_records['sft']:
            for b,sb,pb in task_records['grpo']:
                if not sa or not sb or a['strict_success']==b['strict_success']:continue
                if sa['asin']==sb['asin'] and sa['price']==sb['price'] and sa['collapsed']==sb['collapsed'] and sa['options']!=sb['options']:
                    found={'task_id':task['task_id'],'cohort':task['cohort'],'sft':{'path':pa,'success':a['strict_success'],**sa},
                           'grpo':{'path':pb,'success':b['strict_success'],**sb}}
                    break
            if found:break
        if found:artifact_pairs.append(found)
    actual_slots={(a,p.name) for a in ('sft','grpo') for p in (RUN/a/'attempts').iterdir() if p.is_dir()}
    assert actual_slots==all_slots and len(all_slots)==4192
    summary={}
    for cohort,by_arm in metrics.items():
        ds=[d for d in details if d['cohort']==cohort];stats=paired_stats(ds)
        assert stats==returned['cohorts'][cohort]['main_statistics']
        arms={}
        for arm,rs in by_arm.items():
            rewards=[r['terminal_reward'] for r in rs if r['terminal_reward'] is not None]
            arms[arm]={'episodes':len(rs),'successes':sum(r['strict_success'] for r in rs),
                       'reward_types':dict(Counter(r['reward_type'] or r['status'] for r in rs)),
                       'mean_tool_steps':statistics.fmean(r['tool_steps'] for r in rs),
                       'guard_rejections':sum(r['guard_rejections'] for r in rs),
                       'terminal_reward_episodes':len(rewards),'mean_terminal_reward':statistics.fmean(rewards)}
        summary[cohort]={'statistics':stats,'arms':arms,
                        'zero_success_both':sum(d['sft_successes']==d['grpo_successes']==0 for d in ds),
                        'all_success_both':sum(d['sft_successes']==d['grpo_successes']==8 for d in ds),
                        'sft_zero_grpo_positive':[d['task_id'] for d in ds if d['sft_successes']==0<d['grpo_successes']],
                        'grpo_zero_sft_positive':[d['task_id'] for d in ds if d['grpo_successes']==0<d['sft_successes']]}
    old=load(ROOT/'analysis/grpo_learnability_20261111/task_details.json')
    prior131={d['task_id'] for d in old if d['early_valid_mixed'] and d['paired']}
    prior_stats={'same_prior_131':paired_stats([d for d in details if d['task_id'] in prior131]),
                 'previously_missing_51':paired_stats([d for d in details if d['cohort']=='mixed' and d['task_id'] not in prior131])}
    result={'date':'2026-11-12','complete':True,'cohorts':summary,'prior_log_subcohorts':prior_stats,
            'integrity':dict(counts),'usage':dict(usage_totals),
            'possible_redundant_axis_task_ids':[r['task_id'] for r in artifact_pairs],
            'sources_sha256':{'run':file_hash(RUN/'run.json'),'protocol':file_hash(PACKAGE/'manifest.json')},
            'limits':['One training seed, selected training tasks, 8 episodes per model/task.',
                      'Gold remains original reward semantics; artifact screening is not semantic adjudication.']}
    screened={r['task_id'] for r in artifact_pairs}
    result['posthoc_artifact_screen_sensitivity']={
        'screened_main':paired_stats([d for d in details if d['cohort']=='mixed' and d['task_id'] in screened]),
        'not_screened_main':paired_stats([d for d in details if d['cohort']=='mixed' and d['task_id'] not in screened]),
        'limit':'Post-hoc diagnostic split, not corrected semantic accuracy; original main denominator and scores remain unchanged.'}
    OUT.mkdir(parents=True,exist_ok=True);save(OUT/'audit_and_analysis.json',result)
    save(OUT/'task_pairs.json',details);save(OUT/'redundant_axis_screen.json',artifact_pairs)
    print(dump(result))


if __name__=='__main__':main()
