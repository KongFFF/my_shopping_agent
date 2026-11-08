"""Freeze a passed, user-confirmed evidence-reward history sample audit."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'analysis/semantic_history_representation_20261107'
DECISIONS=ROOT/'experiments/semantic_history_review_20261107/user_decisions.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'analysis/semantic_history_closeout_20261107')
    args=parser.parse_args();out=args.output.resolve()
    if out==SOURCE.resolve():parser.error('Use a separate freeze directory')
    summary=json.loads((SOURCE/'acceptance_summary.json').read_text(encoding='utf-8'))
    assert summary['evidence_history_acceptance_passed'] is True
    assert all(summary['regression_checks'].values()) and summary['evidence_training_blocked']==0
    decisions=json.loads(DECISIONS.read_text(encoding='utf-8'))['decisions']
    required={'dev2898_hope_strength','dev23126_hope_strength','dev12703_hot_selling_snapshot'}
    closed={d['id'] for d in decisions if d['status']=='confirmed_by_user' and d['importance']=='required'}
    assert required<=closed
    contracts=rows(SOURCE/'dev14_contracts.jsonl');assert len(contracts)==14 and all(r['whole_contract_approved'] for r in contracts)
    sample_ids={r['task_id'] for r in contracts}
    sample=[r for r in rows(SOURCE/'replay100.jsonl') if r['task_id'] in sample_ids]
    assert len(sample)==28
    for r in sample:
        score=r.get('training_score') or r.get('terminal_failure_score') or {}
        assert score.get('reward_valid') is True
    out.mkdir(parents=True,exist_ok=True)
    copies={name:SOURCE/name for name in ('sample_lock.json','dev14_contracts.jsonl','reviewed_catalog_facts.jsonl',
             'replay100.jsonl','acceptance_summary.json','repair_manifest.json','artifact_hashes.json')}
    copies['user_decisions.json']=DECISIONS
    for name,path in copies.items():
        target=out/name
        if target.exists() and target.read_bytes()!=path.read_bytes():raise ValueError('Frozen artifact differs; use a new directory: '+name)
    manifest={'date':'2026-11-07','status':'passed','acceptance_basis':'user_confirmed_A_verified_public_evidence_satisfaction',
        'sample_tasks':14,'sample_trajectories':28,'sample_executed_purchases':26,'sample_guard_failures':2,
        'sample_training_scores_valid':28,'source_contracts_approved':14,'pending_user_decisions':0,
        'complete_dev50_evaluation':False,'formal_training_started':False,
        'strict_semantic_all_numeric':summary['semantic_promotion_passed'],
        'strict_semantic_flag_is_not_the_chosen_training_gate':True,
        'files_sha256':{name:sha(path) for name,path in copies.items()},'freeze_script_sha256':sha(Path(__file__))}
    encoded=json.dumps(manifest,ensure_ascii=False,indent=2)+'\n'
    target=out/'freeze_manifest.json'
    if target.exists() and target.read_text(encoding='utf-8')!=encoded:raise ValueError('Frozen manifest differs; use a new directory')
    for name,path in copies.items():(out/name).write_bytes(path.read_bytes())
    target.write_text(encoded,encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('files_sha256','freeze_script_sha256')},ensure_ascii=False))


if __name__=='__main__':main()
