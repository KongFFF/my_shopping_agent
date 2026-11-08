"""Publish the current local batch acceptance after real test receipts exist.

Does not run or authorize full-length training. It preserves the previous
semantic-closeout snapshot and records bounded-pilot readiness separately.
"""
import json
from pathlib import Path

from semantic_training_bundle import ROOT, rows, dump, sha256, build_bundle, TrainingEvidenceBackend
from prepare_semantic_batch_acceptance import OUT


def main():
    receipt=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    if receipt['regression_tests_passed']!=132 or receipt['cpu_integration_and_release_tests_passed']<31:
        raise ValueError('Missing completed regression/CPU test receipt')
    total_tests=receipt['regression_tests_passed']+receipt['cpu_integration_and_release_tests_passed']+receipt.get('environment_sync_regression_tests_passed',0)
    certificate_path=OUT/'pilot_release.json'
    certificate=json.loads(certificate_path.read_text(encoding='utf-8'))
    # Audit metadata name was clarified after the tests. The scoring formulas,
    # accepted contracts, source holds and runtime adapter were not changed.
    ledger_path=OUT/'release_ledger1000.jsonl';ledger=rows(ledger_path)
    for entry in ledger:
        if 'whole_query_user_manual_approval' in entry:
            entry['prior_exact_whole_query_approval']=entry.pop('whole_query_user_manual_approval')
    ledger_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in ledger),encoding='utf-8')
    certificate['files']['ledger']['sha256']=sha256(ledger_path)
    dump(certificate_path,certificate)
    manifest=build_bundle(OUT/'training_bundle',contracts_path=OUT/'contracts1000.jsonl',batch_release_path=certificate_path)
    backend=TrainingEvidenceBackend(OUT/'training_bundle/manifest.json');backend.require_ready()
    receipt.pop('training_entry_preflight_passed',None)
    receipt.update(total_tests_passed=total_tests,frozen_assets_valid=True,cpu_launch_signoff_check_passed=True,
        contracts_sha256_lf=sha256(OUT/'contracts1000.jsonl'),manifest_sha256_lf=sha256(OUT/'training_bundle/manifest.json'),
        original_semantic_source_unchanged=sha256(ROOT/'analysis/semantic_remaining_final_20261108/validated1000.jsonl')==
            json.loads((OUT/'sample_lock.json').read_text(encoding='utf-8'))['source_sha256'],
        pilot_training_ready=True,formal_training_ready=False,training_started=False)
    dump(OUT/'verification.json',receipt)
    latest_path=ROOT/'analysis/semantic_latest.json';latest=json.loads(latest_path.read_text(encoding='utf-8'))
    latest['previous_contracts']=latest.get('previous_contracts') if latest.get('contracts')=='analysis/semantic_batch_acceptance_20261108/contracts1000.jsonl' else latest['contracts']
    latest['contracts']='analysis/semantic_batch_acceptance_20261108/contracts1000.jsonl'
    latest['contract_schema_revision']='semantic-batch-accepted-20261108'
    latest['formal_training_ready']=False
    latest['pilot_training_ready']=True
    latest['reason']='批次验收完成；原1000题保留，981题获批参与最多20步试跑，19题按冻结源限制整组不可评分；尚未启动GPU训练或放行500步正式训练。'
    latest['report']='analysis/semantic_batch_acceptance_20261108/批次验收与训练放行报告.md'
    latest['batch_acceptance']={'status':'passed_for_bounded_pilot','summary':'analysis/semantic_batch_acceptance_20261108/acceptance_summary.json',
        'release':'analysis/semantic_batch_acceptance_20261108/pilot_release.json',
        'manifest':'analysis/semantic_batch_acceptance_20261108/training_bundle/manifest.json',
        'registry':'analysis/semantic_batch_acceptance_20261108/training_bundle/registry1000.jsonl',
        'source_sample':60,'accepted_for_pilot':981,'held':19,'retained_actor_tasks':1000,
        'max_optimizer_steps':20,'full_training_released':False,'tests_passed':total_tests,
        'source_limits':'analysis/semantic_batch_acceptance_20261108/source_limitations.jsonl',
        'sample_reviews':'analysis/semantic_batch_acceptance_20261108/sample_reviews60.jsonl',
        'data_edits':'analysis/semantic_batch_acceptance_20261108/data_edits.jsonl',
        'verification':'analysis/semantic_batch_acceptance_20261108/verification.json'}
    latest['training_signoff'].update(manifest=latest['batch_acceptance']['manifest'],registry=latest['batch_acceptance']['registry'],
        report=latest['report'],whole_source_signed=7,batch_accepted_for_pilot=981,held_source_limitations=19,
        status='bounded_pilot_released',snapshot_contracts=latest['contracts'])
    for old_key in ('held_without_pending_mentions','held_with_pending_mentions'):
        latest['training_signoff'].pop(old_key,None)
    latest['history_acceptance'].update(current_code_revalidation='passed_in_batch_acceptance',
        current_revalidation_summary='analysis/semantic_batch_acceptance_20261108/history/acceptance_summary.json')
    latest['semantic_closeout']['whole_source_batch_released']=False
    latest['semantic_closeout']['snapshot_is_previous_stage']=True
    if 'frozen_assets_valid' in latest['semantic_closeout']:
        latest['semantic_closeout']['frozen_assets_valid_at_previous_stage']=latest['semantic_closeout'].pop('frozen_assets_valid')
    latest['training_integration'].update(verified_against_contracts=latest['contracts'],
        current_reward_regression_tests=132,cpu_integration_and_release_tests=receipt['cpu_integration_and_release_tests_passed'],
        training_started=False,real_verl_ray_gpu_tested=False)
    dump(latest_path,latest)
    print(json.dumps({'pilot_training_ready':True,'released_tasks':len(backend.released_ids),
                      'held_tasks':len(backend.held_ids),'formal_training_ready':False,'total_tests_passed':total_tests}))


if __name__=='__main__':main()
