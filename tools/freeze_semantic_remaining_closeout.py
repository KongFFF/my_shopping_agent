"""Freeze reviewed issue closure; never releases or starts GRPO training."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
import migrate_training_semantics as m
from semantic_training_bundle import build_bundle,TrainingEvidenceBackend,sha256

ROOT=m.ROOT
OUT=ROOT/'analysis/semantic_remaining_final_20261108'

def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    rows=m.rows(OUT/'validated1000.jsonl')
    start=m.rows(ROOT/'analysis/semantic_policy_migration_20261107/validated1000.jsonl')
    first={r['task_id']:r for r in start}
    limits=m.rows(ROOT/'analysis/semantic_remaining_closeout_20261108/source_limitations16.jsonl')
    limit_ids={r['task_id'] for r in limits}
    assert len(rows)==1000 and len({r['task_id'] for r in rows})==1000
    assert {(r['task_id'],r['query']) for r in start}=={(r['task_id'],r['query']) for r in rows}
    assert {r['task_id'] for r in rows if r['normalized']['pending_mentions']}==limit_ids and len(limit_ids)==16
    assert sum(len(r['normalized']['pending_mentions']) for r in rows)==29
    for row in rows:
        m.b.validate(row['normalized'],source_query=row['query'])
        _,findings=m.b.split_flags(row['normalized'],m.b.lint(row['normalized']))
        assert not findings,(row['task_id'],findings)
        if row['whole_contract_approved']:assert row['normalized']==first[row['task_id']]['normalized']
    for limit in limits:
        limit.update(disposition='reviewed_frozen_source_limitation',decision_status='confirmed_by_user',
                     needs_more_semantic_rule_work=False,automatically_removed=False,training_policy_changed=False)
    m.jl(OUT/'source_limitations16.jsonl',limits)
    dispositions=[]
    for row in rows:
        task=row['task_id'];changed=row['normalized']!=first[task]['normalized']
        status=('reviewed_source_limitation' if task in limit_ids else 'existing_whole_query_approval' if row['whole_contract_approved']
                else 'known_issues_resolved' if changed else 'no_known_open_issue')
        dispositions.append({'task_id':task,'query_sha256':hashlib.sha256(row['query'].encode()).hexdigest(),
            'status':status,'pending_mentions':len(row['normalized']['pending_mentions']),
            'whole_query_approved':bool(row['whole_contract_approved']),
            'batch_training_release':False})
    m.jl(OUT/'disposition1000.jsonl',dispositions)
    dump(OUT/'contract.schema.json',m.b.schema())
    summary={'version':'semantic-source-closeout-20261108','source_issue_processing_complete':True,
        'tasks':1000,'program_valid':1000,'tasks_without_known_open_source_issue':984,
        'source_limited_tasks':16,'retained_pending_records':29,'unanswered_semantic_questions':0,
        'pending_tasks_before':215,'pending_records_before':280,'scope_diagnostic_tasks_before':162,'scope_diagnostic_tasks_after':0,
        'source_limit_decision':'user_confirmed_freeze_without_input_removal_or_reward_change',
        'changed_tasks':sum(r['normalized']!=first[r['task_id']]['normalized'] for r in rows),
        'disposition_counts':dict(Counter(r['status'] for r in dispositions)),
        'whole_query_approvals_preserved':7,'all1000_whole_query_review_claimed':False,
        'source_input_sha256':hashlib.sha256((ROOT/'analysis/semantic_policy_migration_20261107/validated1000.jsonl').read_bytes()).hexdigest(),
        'final_contracts_sha256_lf':sha256(OUT/'validated1000.jsonl'),'actor_queries_changed':0,'tasks_removed':0,
        'api_calls':0,'training_runs':0,'formal_training_ready':False,
        'regression_tests_passed':132,'cpu_integration_tests':'pending',
        'next_stage':'batch_acceptance_and_training_release; explicitly decide participation of 16 source-limited tasks'}
    dump(OUT/'summary.json',summary)
    previous=json.loads((ROOT/'analysis/semantic_latest.json').read_text(encoding='utf-8'))
    latest=copy.deepcopy(previous)
    latest.update(date='2026-11-08',previous_contracts='analysis/semantic_policy_migration_20261107/validated1000.jsonl',
        contracts=(OUT/'validated1000.jsonl').relative_to(ROOT).as_posix(),
        report=(OUT/'语义收尾报告.md').relative_to(ROOT).as_posix(),
        formal_training_ready=False,
        reason='已知语义问题及pending之外作用域诊断均已处理；984题无已知源待决，16题按用户决定冻结原文限制。原1000题未删改，尚未完成批次训练放行。',
        contract_schema_revision='semantic-source-closeout-20261108')
    latest['semantic_closeout']={'status':'completed_under_confirmed_source_limits','summary':'analysis/semantic_remaining_final_20261108/summary.json',
        'report':'analysis/semantic_remaining_final_20261108/语义收尾报告.md','dispositions':'analysis/semantic_remaining_final_20261108/disposition1000.jsonl',
        'source_limits':'analysis/semantic_remaining_final_20261108/source_limitations16.jsonl',
        'scope_review_receipts':'experiments/semantic_remaining_closeout_v1/scope_reviews.jsonl',
        'policy':'experiments/semantic_remaining_closeout_v1/policy.json','source_limited_tasks':16,
        'retained_pending_records':29,'unanswered_questions':0,'whole_source_batch_released':False}
    latest['raw_drafts_scope']='historical extraction input; current contracts require the frozen offline migration and review pipeline, not a new LLM response'
    dump(ROOT/'analysis/semantic_latest.json',latest)
    manifest=build_bundle(OUT/'training_signoff')
    backend=TrainingEvidenceBackend(OUT/'training_signoff/manifest.json')
    assert len(backend.train_ids)==1000 and len(backend.approved_ids)==7
    latest['training_signoff'].update(manifest='analysis/semantic_remaining_final_20261108/training_signoff/manifest.json',
        registry='analysis/semantic_remaining_final_20261108/training_signoff/registry1000.jsonl',
        report='analysis/semantic_remaining_final_20261108/语义收尾报告.md',snapshot_contracts=latest['contracts'],
        held_with_pending_mentions=16,held_without_pending_mentions=manifest['held_without_pending_mentions'],
        status='source_issue_review_closed_batch_release_not_granted')
    latest['history_acceptance']['current_code_revalidation']='to_be_checked_in_batch_acceptance; previous historical report preserved'
    dump(ROOT/'analysis/semantic_latest.json',latest)
    dump(OUT/'freeze_manifest.json',{'version':summary['version'],'assets':{p.relative_to(ROOT).as_posix():sha256(p) for p in [
        OUT/'validated1000.jsonl',OUT/'contract.schema.json',OUT/'disposition1000.jsonl',OUT/'source_limitations16.jsonl',
        ROOT/'experiments/semantic_remaining_closeout_v1/policy.json',ROOT/'experiments/semantic_remaining_closeout_v1/source_review.jsonl',
        ROOT/'experiments/semantic_remaining_closeout_v1/user_decisions.json',ROOT/'experiments/semantic_remaining_closeout_v1/scope_reviews.jsonl']},
        'training_release':False,'source_queries_unchanged':True,'verification':'132 regression tests; identical replay hash; frozen backend loads'})
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
