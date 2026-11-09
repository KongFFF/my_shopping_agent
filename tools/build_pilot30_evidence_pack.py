"""Compile reviewed public evidence data; never invent facts from requirements."""
import argparse
import hashlib
import json
from pathlib import Path

from semantic_product_facts_v3 import validate_fact
from semantic_product_verifier_v4 import snapshot, fingerprint, pointer, reviewed_facts

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def build(directory):
    source = directory / 'review_inputs35.jsonl'
    records = rows(source)
    annotations = json.loads((directory / 'reviewed_additions.json').read_text(encoding='utf-8'))
    if len(records) != 35 or len({r['product_asin'] for r in records}) != 30:
        raise ValueError('This review is limited to the original 35 SKU snapshots / 30 products')
    actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_hash != annotations['input_sha256']:
        raise ValueError('Reviewed input differs from the frozen annotation input')
    additions = []
    for annotation in annotations['facts']:
        row = records[annotation['i']]
        product = row['public_product']
        state = snapshot(product, row['selected_options'])
        evidence = []
        for path in annotation['proof']:
            if path.startswith('@'):
                selected = [s for s in state['selected'] if s['axis'] == path[1:]]
                if len(selected) != 1:
                    raise ValueError('Evidence must reference exactly one selected option')
                path = selected[0]['pointer']
            evidence.append({'pointer': path, 'value': pointer(product, path)})
        fact = {
            'product_asin': row['product_asin'], 'product_sha256': fingerprint(product),
            'scope': 'sku', 'options': row['selected_options'], 'selector_mode': 'exact',
            'dimension': annotation['d'], 'subtype': annotation['st'], 'attribute': annotation['a'],
            'subject': annotation.get('subject', {'kind': 'primary_product', 'name': '主商品'}),
            'value': annotation.get('value', {'type': 'text', 'value': annotation.get('text')}),
            'polarity': 'positive', 'complete': annotation.get('complete', False),
            'review_status': 'approved', 'review_note': annotation['note'], 'evidence': evidence,
        }
        validate_fact(fact)
        accepted, rejected = reviewed_facts(product, state, [fact])
        if len(accepted) != 1 or rejected:
            raise ValueError('Reviewed addition failed source/selection checks: ' + str(rejected))
        additions.append(fact)
    # Exact duplicates can occur when independent views of one SKU are listed.
    additions = list({json.dumps(f, ensure_ascii=False, sort_keys=True): f for f in additions}.values())
    original = ROOT / 'analysis/semantic_product_facts_v2_20261107/catalog_facts.jsonl'
    combined = rows(original) + additions
    for name, data in [('additions.jsonl', additions), ('catalog_facts.jsonl', combined)]:
        (directory / name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in data), encoding='utf-8')
    receipt = {
        'reviewed_sku_snapshots': len(records), 'reviewed_products': 30,
        'unique_unknown_requirements_before_review': sum(q['comparison']['status'] == 'unknown' for r in records for q in r['requirements']),
        'new_facts': len(additions), 'new_fact_products': len({f['product_asin'] for f in additions}),
        'combined_facts': len(combined), 'combined_fact_products': len({f['product_asin'] for f in combined}),
        'annotation_source_sha256': actual_hash,
        'actor_queries_changed': 0, 'api_calls': 0, 'training_started': False,
        'scope': 'reviewed limited pilot purchases; not coverage acceptance of all train1000 products',
    }
    (directory / 'fact_pack_summary.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    build(parser.parse_args().directory.resolve())
