"""Account for every previously unknown requirement in the limited SKU review."""
import argparse
from collections import Counter
import json
from pathlib import Path


def rows(path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def identity(task, asin, options):
    return json.dumps([task, asin, options], ensure_ascii=False, sort_keys=True)


def audit(directory):
    inputs = rows(directory / 'review_inputs35.jsonl')
    notes = json.loads((directory / 'unresolved_notes.json').read_text(encoding='utf-8'))
    current = {r['task_id']: r['normalized'] for r in rows(directory / 'contracts1000.jsonl')}
    rescored = {}
    for r in rows(directory / 'replay/rescored_rollouts.jsonl'):
        score = r['new_score']
        if 'verification' not in score:
            continue
        v = score['verification']
        key = identity(r['task_id'], score['purchase_asin'], v['snapshot']['selected_options'])
        if key in rescored and rescored[key]['statuses'] != v['statuses']:
            raise ValueError('Repeated identical purchase produced different verification')
        rescored[key] = v
    ledger = []
    sku_reviews = []
    for index, row in enumerate(inputs):
        after = rescored[identity(row['task_id'], row['product_asin'], row['selected_options'])]
        requirements = {r['id']: {'dimension': d, **r}
                        for d in ('category', 'budget', 'functions', 'specifications', 'brand', 'model')
                        for r in current[row['task_id']][d]}
        states = Counter()
        for q in row['requirements']:
            if q['comparison']['status'] != 'unknown':
                continue
            result = after['comparisons'][q['id']]
            if result['status'] != 'unknown':
                disposition, note = ('public_evidence_resolved', '明确公开证据已通过逐项核验；来源见comparison。')
            else:
                disposition, note = notes['overrides'].get(str(index), {}).get(
                    q['id'], [notes['default_reason'], notes['default_note']])
            states[disposition] += 1
            ledger.append({'sku_index': index, 'task_id': row['task_id'],
                           'product_asin': row['product_asin'], 'selected_options': row['selected_options'],
                           'requirement_before_review': {k: v for k, v in q.items() if k != 'comparison'},
                           'current_requirement': requirements[q['id']],
                           'disposition': disposition, 'review_note': note,
                           'comparison': result, 'reviewed_by': 'assistant',
                           'rollout_occurrences': row['rollout_count']})
        sku_reviews.append({'sku_index': index, 'task_id': row['task_id'], 'product_asin': row['product_asin'],
                            'selected_options': row['selected_options'], 'review_complete': True,
                            'unknown_dispositions': dict(states), 'after_statuses': after['statuses']})
    if len(ledger) != 176 or len(sku_reviews) != 35:
        raise ValueError('Incomplete original review inventory')
    for name, data in [('unknown_review176.jsonl', ledger), ('sku_review35.jsonl', sku_reviews)]:
        (directory / name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in data), encoding='utf-8')
    remaining = [r for r in ledger if r['comparison']['status'] == 'unknown']
    summary = {'reviewed_products': 30, 'reviewed_sku_snapshots': 35,
               'previous_unique_unknowns': len(ledger), 'resolved_unique_unknowns': len(ledger)-len(remaining),
               'remaining_unique_unknowns': len(remaining),
               'disposition_counts': dict(Counter(r['disposition'] for r in ledger)),
               'remaining_weighted_occurrences': sum(r['rollout_occurrences'] for r in remaining),
               'coverage_scope': 'limited saved purchases; not an audit of unseen train1000 products',
               'manual_user_whole_query_approvals_added': 0}
    (directory / 'review_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    audit(parser.parse_args().directory.resolve())
