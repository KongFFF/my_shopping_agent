"""Verify explicit, source-bound batch acceptance; not a manual-review claim."""
import copy
import hashlib
import json
from pathlib import Path

from semantic_contract_v2_r7_final12 import DIMENSIONS, validate

VERSION='semantic-batch-release-v1'
REQUIRED_GATES={
    'original1000_and_queries_preserved','eligible_contracts_strictly_valid',
    'no_unresolved_compiler_findings','fixed60_sample_review_closed',
    'history_evidence_reward_and_ordering_passed','static_training_probes_scalar',
    'all_eligible_kernel_stress_finite_in_range','no_invalid_fact_cache_records_in_static_probes',
    'bounded_pilot_not_full_training'}


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def verify_release(path, *, root, contracts, sha256, rows):
    document=json.loads(Path(path).read_text(encoding='utf-8'))
    if document.get('version')!=VERSION or document.get('status')!='passed_for_bounded_pilot':
        raise ValueError('Batch acceptance not passed for bounded pilot')
    if document.get('scope')!='bounded_pilot' or document.get('full_training_released') is not False:
        raise ValueError('Unsupported batch release scope')
    if set(document.get('checks',{}))!=REQUIRED_GATES or not all(value is True for value in document['checks'].values()):
        raise ValueError('Batch acceptance contains failed or missing gates')
    if document.get('max_optimizer_steps')!=20:
        raise ValueError('Only the 20-step pilot has been accepted')
    paths={}
    for name,asset in document['files'].items():
        p=(root/asset['path']).resolve()
        if not p.is_relative_to(root) or sha256(p)!=asset['sha256']:
            raise ValueError('Batch acceptance asset mismatch: '+name)
        paths[name]=p
    if sha256(paths['contracts'])!=sha256(contracts):raise ValueError('Batch release belongs to different contracts')
    policy=json.loads(paths['policy'].read_text(encoding='utf-8'))
    if policy.get('status')!='confirmed' or policy.get('max_optimizer_steps')!=20:
        raise ValueError('Pilot participation policy is not confirmed')
    if policy['source_limited_participation']!='keep1000_whole_group_invalid_no_fallback':
        raise ValueError('Unconfirmed source-limited participation')
    source={r['task_id']:r for r in rows(paths['source_contracts'])}
    current={r['task_id']:r for r in rows(contracts)}
    ledger={r['task_id']:r for r in rows(paths['ledger'])}
    edits={r['task_id']:r for r in rows(paths['data_edits'])}
    held={r['task_id'] for r in rows(paths['source_limits'])}
    if len(source)!=1000 or set(current)!=set(source) or set(ledger)!=set(source):
        raise ValueError('Batch must retain original1000 exactly')
    limits={int(t) for t in policy['original_source_limits']}|{int(t) for t in policy['additional_source_limits']}
    if held!=limits:raise ValueError('Source holds differ from adjudicated policy')
    released=set()
    for task,row in current.items():
        before=source[task]['normalized'];after=row['normalized'];entry=ledger[task]
        if row['query']!=source[task]['query'] or after['query']!=row['query'] or after['task_id']!=task:
            raise ValueError('Original source input changed')
        expected=copy.deepcopy(before)
        if task in edits:
            if edits[task]['before']!=before:raise ValueError('Source edit is not bound to original')
            expected=copy.deepcopy(edits[task]['after'])
        if task not in held:
            for d in DIMENSIONS:
                for r in expected[d]:r['review_status']='approved'
        if expected!=after:raise ValueError('Unrecorded semantic change in batch contracts')
        if row.get('whole_contract_approved')!=source[task].get('whole_contract_approved'):
            raise ValueError('Batch cannot invent whole-query manual approval')
        if digest(after)!=entry['contract_sha256'] or digest(before)!=entry['source_contract_sha256']:
            raise ValueError('Batch contract fingerprint mismatch')
        if entry['status']!=('held_source_limitation' if task in held else 'batch_accepted'):
            raise ValueError('Batch ledger status mismatch')
        validate(after,source_query=row['query'],require_reviewed=task not in held)
        if task not in held:released.add(task)
    if sorted(released)!=document['released_ids'] or sorted(held)!=document['held_ids']:
        raise ValueError('Batch release roster mismatch')
    # A completed sampled audit is necessary; validation alone cannot sign it.
    sample=rows(paths['sample_reviews'])
    lock=json.loads(paths['sample_lock'].read_text(encoding='utf-8'))
    if lock['source_sha256']!=sha256(paths['source_contracts']):
        raise ValueError('Source sample lock is stale')
    if len(sample)!=60 or {r['task_id'] for r in sample}!=set(lock['sample_ids']):
        raise ValueError('Missing fixed60 source acceptance review')
    if any(r['verdict'] not in ('accepted','fixed_and_rechecked','held_source_limitation') for r in sample):
        raise ValueError('Source sample has open findings')
    if any(r['source_contract_sha256']!=digest(source[r['task_id']]['normalized'])
           or r['accepted_contract_sha256']!=digest(current[r['task_id']]['normalized']) for r in sample):
        raise ValueError('Source sample review is stale')
    summary=json.loads(paths['acceptance_summary'].read_text(encoding='utf-8'))
    if summary.get('passed') is not True or summary['checks']!=document['checks']:
        raise ValueError('Release does not match the passed batch acceptance')
    return document,paths,released,held


def require_pilot_config(document,config):
    """Require resolved trainer configuration, including direct Hydra launches."""
    if not document:return
    if config is None:raise ValueError('Bounded pilot requires resolved trainer config')
    def get(path):
        value=config
        for key in path.split('.'):
            try:value=value[key]
            except (KeyError,TypeError):raise ValueError('Missing pilot config '+path)
        return value
    steps=get('trainer.total_training_steps')
    if isinstance(steps,bool) or not isinstance(steps,int) or not 1<=steps<=document['max_optimizer_steps']:
        raise ValueError('Batch released for at most 20 optimizer steps; full training not released')
    exact={'trainer.resume_mode':'disable','data.train_batch_size':2,'actor_rollout_ref.rollout.n':4,
           'shopping_dynamic_sampling.enable':True,'shopping_dynamic_sampling.max_num_gen_batches':3,
           'shopping_dynamic_sampling.max_consecutive_skipped_updates':3}
    for key,wanted in exact.items():
        if get(key)!=wanted:raise ValueError('Pilot config requires '+key+'='+str(wanted))
