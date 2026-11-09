"""Audit saved evidence-reward rollouts without models, APIs or rescoring.

Count generation events once; optimizer events repeat the selected rollouts.
Native reward labels are a diagnostic reference, never semantic ground truth.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = ('category', 'budget', 'functions', 'specifications', 'brand', 'model')


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines()
            if line.strip()]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def dump_rows(path, values):
    path.write_text(''.join(canonical(value) + '\n' for value in values), encoding='utf-8')


def audit(diagnostics, contracts, catalog_facts):
    events = read_rows(diagnostics)
    sources = {r['task_id']: r for r in read_rows(contracts)}
    cached_asins = {str(r['product_asin']) for r in read_rows(catalog_facts)}
    generated = [e for e in events if e['event'] == 'generation_batch']
    updates = [e for e in events if e['event'] == 'optimizer_step']
    trained_uids = {r['uid'] for e in updates for r in e.get('rollouts', [])}
    groups, invalid_rollouts = [], []
    sku_queue = {}
    status_counts, unique_status_counts = defaultdict(Counter), defaultdict(Counter)
    baseline_types, reward_types, unknown_reasons = Counter(), Counter(), Counter()
    scalar_counts = Counter()
    budget_only_count = 0
    seen = set()
    for event in generated:
        for group in event['groups']:
            members = sorted((r for r in event['rollouts'] if r['uid'] == group['uid']),
                             key=lambda r: r['rollout_index'])
            if len(members) != len(group['terminal_utilities']) or not members:
                raise ValueError('Group/rollout alignment mismatch')
            task_ids = {r['task_id'] for r in members}
            if len(task_ids) != 1:
                raise ValueError('A comparison group must contain exactly one task')
            task_id = next(iter(task_ids))
            source = sources[task_id]
            requirements = {r['id']: {'dimension': d, **r}
                            for d in DIMENSIONS for r in source['normalized'][d]}
            outcomes, compact_rollouts = [], []
            for r in members:
                identity = (r['uid'], r['rollout_index'])
                if identity in seen:
                    raise ValueError('Repeated generated rollout identity')
                seen.add(identity)
                evidence = r.get('evidence_reward') or {}
                score = evidence.get('score') or {}
                baseline = evidence.get('baseline') or {}
                verification = score.get('verification') or {}
                snapshot = verification.get('snapshot') or {}
                asin = score.get('purchase_asin')
                options = snapshot.get('selected_options')
                if asin and not isinstance(options, dict):
                    raise ValueError('Scored purchase has no selected-option snapshot')
                outcome = (['purchase', str(asin), options] if asin else
                           ['no_verified_purchase', r.get('termination_reason'), r.get('error')])
                outcomes.append(canonical(outcome))
                baseline_types[str(baseline.get('type'))] += 1
                reward_types[str(score.get('type'))] += 1
                scalar_counts[str(score.get('reward'))] += 1
                support = score.get('required_dimension_support') or {}
                if support.get('budget') == 1 and all(v == 0 for k, v in support.items() if k != 'budget'):
                    budget_only_count += 1
                for c in verification.get('comparisons', {}).values():
                    status_counts[c['dimension']][c['status']] += 1
                    if c['status'] == 'unknown':
                        unknown_reasons[c['reason']] += 1
                if asin:
                    key = canonical([task_id, str(asin), options])
                    if key not in sku_queue:
                        for c in verification['comparisons'].values():
                            unique_status_counts[c['dimension']][c['status']] += 1
                        sku_queue[key] = {
                            'task_id': task_id, 'query': source['query'],
                            'product_asin': str(asin), 'selected_options': options,
                            'product_sha256': snapshot['product_sha256'],
                            'has_reviewed_product_cache': str(asin) in cached_asins,
                            'requirements': [{**requirements[rid], 'comparison': comparison}
                                             for rid, comparison in verification['comparisons'].items()],
                            'rollout_count': 0,
                        }
                    sku_queue[key]['rollout_count'] += 1
                compact = {
                    'rollout_index': r['rollout_index'], 'purchase_asin': asin,
                    'selected_options': options, 'reward': score.get('reward'),
                    'reward_type': score.get('type'),
                    'required_dimension_support': score.get('required_dimension_support'),
                    'statuses': verification.get('statuses'),
                    'baseline_type': baseline.get('type'),
                    'baseline_utility': (baseline.get('reward') or {}).get('terminal_utility'),
                    'termination_reason': r.get('termination_reason'), 'error': r.get('error'),
                }
                compact_rollouts.append(compact)
                if r.get('reward_valid') is False:
                    invalid_rollouts.append({'task_id': task_id, **compact})
            groups.append({
                'task_id': task_id, 'query': source['query'], 'uid': group['uid'],
                'global_step': event['global_step'], 'generation_batch': event['generation_batch'],
                'drop_reason': group['drop_reason'], 'accepted': group['accepted'],
                'actually_trained': group['uid'] in trained_uids,
                'distinct_outcome_count': len(set(outcomes)),
                'same_purchase_or_terminal_outcome': len(set(outcomes)) == 1,
                'baseline_utility_varies': len({r['baseline_utility'] for r in compact_rollouts}) > 1,
                'rollouts': compact_rollouts,
            })
    if trained_uids - {g['uid'] for g in groups}:
        raise ValueError('Optimizer record references a group absent from generation records')
    statuses = Counter()
    for counts in status_counts.values():
        statuses.update(counts)
    tied = [g for g in groups if g['drop_reason'] == 'constant_reward']
    purchased_asins = {row['product_asin'] for row in sku_queue.values()}
    summary = {
        'input': str(Path(diagnostics).resolve()),
        'input_sha256': hashlib.sha256(Path(diagnostics).read_bytes()).hexdigest(),
        'scope': 'generated_rollouts_only; optimizer_rollout_copies_excluded',
        'native_labels_are_semantic_ground_truth': False,
        'event_counts': dict(Counter(e['event'] for e in events)),
        'optimizer_steps': [e['global_step'] for e in updates],
        'generated_groups': len(groups), 'generated_rollouts': len(seen),
        'group_drop_counts': dict(Counter(g['drop_reason'] or 'eligible' for g in groups)),
        'actually_trained_groups': sum(g['actually_trained'] for g in groups),
        'eligible_but_unused_tasks': [g['task_id'] for g in groups
                                    if not g['drop_reason'] and not g['actually_trained']],
        'tied_same_outcome_tasks': [g['task_id'] for g in tied if g['same_purchase_or_terminal_outcome']],
        'tied_different_outcome_tasks': [g['task_id'] for g in tied if not g['same_purchase_or_terminal_outcome']],
        'tied_baseline_utility_varies_tasks': [g['task_id'] for g in tied if g['baseline_utility_varies']],
        'baseline_type_counts': dict(baseline_types), 'reward_type_counts': dict(reward_types),
        'reward_scalar_counts': dict(scalar_counts),
        'only_budget_supported_rollouts': budget_only_count,
        'comparison_counts': dict(statuses),
        'comparison_counts_by_dimension': dict(status_counts),
        'unique_task_sku_comparison_counts_by_dimension': dict(unique_status_counts),
        'unknown_reason_counts': dict(unknown_reasons),
        'unique_purchased_products': len(purchased_asins), 'unique_task_skus': len(sku_queue),
        'products_with_reviewed_cache': sorted(purchased_asins & cached_asins),
        'reviewed_cache_total_products': len(cached_asins),
        'invalid_rollouts': invalid_rollouts,
        'skipped_updates': [e for e in events if e['event'] == 'skipped_update'],
    }
    return summary, groups, list(sku_queue.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostics', type=Path)
    parser.add_argument('--contracts', type=Path, default=ROOT / 'analysis/semantic_batch_acceptance_20261108/contracts1000.jsonl')
    parser.add_argument('--catalog-facts', type=Path, default=ROOT / 'analysis/semantic_product_facts_v2_20261107/catalog_facts.jsonl')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary, groups, queue = audit(args.diagnostics, args.contracts, args.catalog_facts)
    args.output.mkdir(parents=True, exist_ok=True)
    dump(args.output / 'summary.json', summary)
    dump_rows(args.output / 'groups.jsonl', groups)
    dump_rows(args.output / 'purchased_sku_review_queue.jsonl', queue)
    print(json.dumps({k: summary[k] for k in (
        'optimizer_steps', 'generated_groups', 'generated_rollouts', 'group_drop_counts',
        'actually_trained_groups', 'tied_different_outcome_tasks', 'comparison_counts',
        'unique_purchased_products', 'unique_task_skus', 'products_with_reviewed_cache')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
