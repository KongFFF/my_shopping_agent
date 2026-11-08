"""Source-bound historical acceptance, explicitly permitting an unsuccessful gate.

Full rewards use the production-strict approval gate. Atom diagnostics for a
blocked draft cannot be mistaken for a reward or a whole-task success label.
"""
import gzip
import hashlib
import json
from collections import Counter
import semantic_contract_v2_r7_final12 as contract_api
import semantic_product_verifier_v4 as verifier
from semantic_history_adapter import canonicalize_purchase
from semantic_reward_history import score_verified
from semantic_evidence_reward import score_training_verified,score_terminal_failure
from freeze_historical_reward_audit import ARCHIVE, OUT, ROOT, SOURCE, rows, lines, write, digest


def atom_diagnostics(contract,product,options,cache):
    state=verifier.snapshot(product,options)
    extra,rejected=verifier.reviewed_facts(product,state,cache)
    facts=verifier.automatic_facts(product,state)+extra
    results=[]
    for d in contract_api.DIMENSIONS:
        for index,r in enumerate(contract[d]):
            results.append({'id':r.get('id',f'{d}:{index}'),'dimension':d,'subtype':r['subtype'],
                'attribute':r['attribute'],'expected':r['expected'],'importance':r['importance'],
                'requirement_evidence':r['evidence'],**verifier.compare_requirement(r,d,product,state,facts)})
    return results,rejected


def main(directory=OUT):
    OUT=directory
    lock=json.loads((OUT/'sample_lock.json').read_text(encoding='utf-8'))
    assert digest(ARCHIVE)==lock['product_data_sha256']
    for path,expected in lock['source_sha256'].items():assert digest(ROOT/path)==expected
    with gzip.open(ARCHIVE,'rt',encoding='utf-8') as f:products={str(p['asin']):p for p in json.load(f)}
    inventory=rows(OUT/'trajectory_inventory100.jsonl')
    raw_histories={(run,r['trajectory_id']):r for run in ('sft_dev50','grpo_dev50') for r in rows(SOURCE/run/'trajectories.jsonl')}
    contracts={r['task_id']:r for r in rows(OUT/'dev14_contracts.jsonl')}
    drafts={r['task_id']:r['raw_draft'] for r in rows(OUT/'dev14_drafts.jsonl')}
    fact_rows=rows(OUT/'reviewed_catalog_facts.jsonl');by_asin={}
    for f in fact_rows:by_asin.setdefault(f['product_asin'],[]).append(f)
    results=[];unknowns=[];mapping_changes=[];comparison_changes=[];before_counts=Counter();transitions=Counter()
    for event in inventory:
        task=event['task_id'];base={k:event[k] for k in ('task_id','run','trajectory_id','status','purchase_executed')}
        if not event['purchase_executed']:
            failure=score_terminal_failure(raw_histories[event['run'],event['trajectory_id']])
            results.append({**base,'scope':'execution_only','reward':None,'reason':'no_executed_purchase',
                            'terminal_failure_score':failure});continue
        product=products[event['asin']];mapping=canonicalize_purchase(product,event['selected_options'])
        base.update(product_asin=event['asin'],recorded_options=event['selected_options'],mapping=mapping)
        if any(p['method']!='exact' for p in mapping['mapping_evidence']):mapping_changes.append(base)
        if mapping['errors']:
            results.append({**base,'scope':'mapping_blocked','reward':None});continue
        if task not in contracts:
            results.append({**base,'scope':'execution_only','reward':None,'reason':'dev_contract_not_annotated'});continue
        record=contracts[task];c=record.get('normalized') or drafts[task]
        diagnostics,rejected=atom_diagnostics(c,product,mapping['selected_options'],by_asin.get(event['asin'],[]))
        before,_=atom_diagnostics(c,product,mapping['selected_options'],[])
        for old,new in zip(before,diagnostics):
            assert (old['dimension'],old['subtype'],old['attribute'])==(new['dimension'],new['subtype'],new['attribute'])
            before_counts[old['status']]+=1;transitions[old['status']+'->'+new['status']]+=1
            if old['status']!=new['status']:
                comparison_changes.append({'task_id':task,'run':event['run'],'subtype':new['subtype'],
                   'expected':new['expected'],'before':old,'after':new})
        score={'reward':None,'reward_valid':False,'semantic_success':None,'type':'contract_blocked'}
        verification=None;training_score=None
        if record['program_valid']:
            verification=verifier.verify(c,product,mapping['selected_options'],catalog_facts=by_asin.get(event['asin'],[]),purchase_executed=True)
            score=score_verified(c,verification,source_query=c['query'])
            training_score=score_training_verified(c,verification,source_query=c['query'])
            verification['reward_ready']=score['reward_valid']
        for r in diagnostics:
            if r['status']=='unknown':unknowns.append({'task_id':task,'run':event['run'],**r})
        results.append({**base,'scope':'semantic_sample','contract_program_valid':record['program_valid'],
            'whole_contract_approved':record['whole_contract_approved'],
            'contract_blocker':record.get('approval_blocker') or record.get('error'),
            'atom_diagnostics':diagnostics,'rejected_catalog_facts':rejected,'verification':verification,'score':score,
            'training_score':training_score,
            'reward':score['reward'],'diagnostics_are_not_full_rewards':True})
    lines(OUT/'replay100.jsonl',results);lines(OUT/'unknown_requirements.jsonl',unknowns)
    lines(OUT/'purchase_mapping_changes.jsonl',mapping_changes)
    lines(OUT/'comparison_changes.jsonl',comparison_changes)
    lookup={(r['task_id'],r['run']):r for r in results}
    def pair(t):return lookup[t,'sft_dev50'],lookup[t,'grpo_dev50']
    def atoms(t,run,subtype):return [r for r in lookup[t,run].get('atom_diagnostics',[]) if r['subtype']==subtype]
    s,g=pair(5293)
    placeholder_equal=[(r['dimension'],r['subtype'],r['attribute'],r['status']) for r in s['atom_diagnostics']]==[(r['dimension'],r['subtype'],r['attribute'],r['status']) for r in g['atom_diagnostics']]
    bundle=[atoms(8666,run,'bundle')[0]['status'] for run in ('sft_dev50','grpo_dev50')]
    curtain_installation_absent=not any(r['subtype']=='installation_service' for run in ('sft_dev50','grpo_dev50') for r in lookup[12703,run]['atom_diagnostics'])
    checks={
      'all_93_purchases_unique_catalog_mapping':sum(r['purchase_executed'] for r in results)==93 and not any(r.get('mapping',{}).get('errors') for r in results),
      'guard_exhausted_never_counted_as_purchase':all(not r['purchase_executed'] for r in results if r['status']=='purchase_guard_exhausted'),
      '5293_placeholder_does_not_change_semantic_statuses':placeholder_equal,
      '8666_wrong_bundle_fails_correct_bundle_passes':bundle==['fail','pass'],
      '12703_no_unrequested_installation_requirement':curtain_installation_absent,
      '23126_empty_case_is_category_failure':atoms(23126,'grpo_dev50','product_category')[0]['status']=='fail',
      '23126_empty_case_cannot_borrow_herbal_filling':atoms(23126,'grpo_dev50','material')[0]['status']=='fail',
      '652_both_over_budget':all(atoms(652,run,'total_price')[0]['status']=='fail' for run in ('sft_dev50','grpo_dev50')),
      'unsupported_facts_never_receive_numeric_rewards':all(r['reward'] is None for r in results if not r.get('score',{}).get('reward_valid')),
      'reviewed_facts_all_source_bound':not any(r.get('rejected_catalog_facts') for r in results),
      '652_budget_failure_scores_survive_unrelated_pending':all(lookup[652,run]['score'].get('reward')==-0.85 for run in ('sft_dev50','grpo_dev50')),
      '23126_empty_case_scores_category_failure':lookup[23126,'grpo_dev50']['score'].get('reward')==-0.85,
      '23126_preference_budget_failure_is_not_a_hard_failure':lookup[23126,'sft_dev50']['score'].get('type')!='category_budget_failure',
      '8666_quantity_scope_now_program_valid':contracts[8666]['program_valid'],
      '15432_square_dimensions_source_proven':contracts[15432]['whole_contract_approved'],
      '17825_confirmed_budget_upper_495':contracts[17825]['normalized']['budget'][0]['expected'].get('value')==495,
      'unknown_intervals_not_used_as_scalar_rewards':all(r['score']['reward'] is None and not r['score']['interval_is_training_reward'] for r in results if r.get('score',{}).get('reward_interval') is not None),
      '8666_full_reward_ranks_bundle_above_single':lookup[8666,'sft_dev50']['score'].get('type')=='partial_failure' and lookup[8666,'grpo_dev50']['score'].get('reward')==1.0,
    }
    promotion_criteria={'all_regression_checks_passed':all(checks.values()),
        'all_selected_contracts_reviewed':all(r['whole_contract_approved'] for r in contracts.values()),
        'all_sample_purchases_have_scalar_rewards':all(r['score'].get('reward_valid') for r in results if r['scope']=='semantic_sample'),
        'historical_success_and_partial_failure_scores_present':{'semantic_success','partial_failure'}<={r['score']['type'] for r in results if 'score' in r}}
    evidence_criteria={'all_regression_checks_passed':all(checks.values()),
        'all_selected_contracts_reviewed':all(r['whole_contract_approved'] for r in contracts.values()),
        'all_sample_purchase_training_scores_valid':all(bool((r.get('training_score') or {}).get('reward_valid')) for r in results if r['scope']=='semantic_sample'),
        'all_actual_execution_failures_scored':all(r.get('terminal_failure_score',{}).get('reward_valid') for r in results if not r['purchase_executed']),
        'source_claim_unknowns_remain_unknown':all((r.get('training_score') or {}).get('unknown_is_real_failure') is False for r in results if r.get('training_score'))}
    summary={'trajectories':len(results),'actual_purchases':93,'catalog_mappings_changed':len(mapping_changes),
       'scope_counts':dict(Counter(r['scope'] for r in results)),
       'annotated_tasks':len(contracts),'unannotated_dev_tasks':50-len(contracts),
       'annotated_program_valid':sum(r['program_valid'] for r in contracts.values()),
       'whole_contract_approved':sum(r['whole_contract_approved'] for r in contracts.values()),
       'fact_products':len(by_asin),'reviewed_facts':len(fact_rows),
       'atom_comparisons':dict(Counter(a['status'] for r in results for a in r.get('atom_diagnostics',[]))),
       'same_verifier_without_reviewed_facts':dict(before_counts),'fact_addition_transitions':dict(transitions),
       'valid_full_numeric_rewards':sum(r.get('score',{}).get('reward_valid',False) for r in results),
       'user_confirmed_evidence_training_rewards':sum(bool((r.get('training_score') or {}).get('reward_valid')) for r in results),
       'fixed_execution_failure_rewards':sum(bool(r.get('terminal_failure_score',{}).get('reward_valid')) for r in results),
       'evidence_training_blocked':sum(not bool((r.get('training_score') or {}).get('reward_valid')) for r in results if r['scope']=='semantic_sample'),
       'training_objective':'verified_public_evidence_satisfaction; user confirmed 2026-11-07',
       'score_types':dict(Counter(r['score']['type'] for r in results if 'score' in r)),
       'reward_scopes':dict(Counter(r['score'].get('scope','blocked') for r in results if 'score' in r)),
       'semantic_promotion_passed':all(promotion_criteria.values()),'promotion_criteria':promotion_criteria,'regression_checks':checks,'api_calls':0,
       'evidence_history_acceptance_passed':all(evidence_criteria.values()),'evidence_history_criteria':evidence_criteria,
       'limitations':['14 selected query contracts, not all Dev50','known regressions are not a blind test',
                      'atom diagnostics do not produce task rewards','train1000 contracts are unchanged and disjoint from Dev50']}
    assert len(results)==100
    write(OUT/'acceptance_summary.json',summary)
    queue=[]
    for task,record in sorted(contracts.items()):
        samples=[r for r in results if r['task_id']==task and 'score' in r]
        queue.append({'task_id':task,'whole_contract_approved':record['whole_contract_approved'],
          'contract_blocker':record.get('approval_blocker') or record.get('error'),
          'pending_mentions':record.get('normalized',{}).get('pending_mentions',[]),
          'product_unknowns':[{'run':r['run'],'subtype':a['subtype'],'expected':a['expected'],'reason':a['reason']}
                              for r in samples for a in r['atom_diagnostics'] if a['status']=='unknown'],
          'score_outcomes':[{'run':r['run'],'type':r['score']['type'],'reward':r['score']['reward'],
                             'reward_interval':r['score'].get('reward_interval'),
                             'evidence_training_reward':(r.get('training_score') or {}).get('reward'),
                             'evidence_training_valid':(r.get('training_score') or {}).get('reward_valid')} for r in samples]})
    write(OUT/'remaining_review_items.json',queue)
    write(OUT/'artifact_hashes.json',{str(p.relative_to(ROOT)):digest(p) for p in [OUT/'dev14_contracts.jsonl',OUT/'reviewed_catalog_facts.jsonl',OUT/'replay100.jsonl',ROOT/'analysis/semantic_final12_20261107/validated1000.jsonl']})
    print(json.dumps(summary,ensure_ascii=False))
    if not all(checks.values()):raise SystemExit('Historical regression assertion failed; inspect acceptance_summary.json')


if __name__=='__main__':main()
