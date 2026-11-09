"""Rescore saved training purchases on CPU; never regenerate or update weights.

Purchase execution and selected options come from the original scored snapshot.
Invalid executions remain invalid; fixed terminal penalties remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from semantic_product_verifier_v4 import verify, VERSION
from semantic_evidence_reward import score_training_verified

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def replay(diagnostics, manifest, output):
    document = json.loads(manifest.read_text(encoding='utf-8'))
    paths = {k: ROOT / a['path'] for k, a in document['assets'].items()}
    for name in ('contracts', 'catalog', 'catalog_facts'):
        data = paths[name].read_bytes()
        if paths[name].suffix in ('.json', '.jsonl'):
            data = data.replace(b'\r\n', b'\n')
        if hashlib.sha256(data).hexdigest() != document['assets'][name]['sha256']:
            raise ValueError('Frozen replay input differs: ' + name)
    contracts = {r['task_id']: r for r in rows(paths['contracts'])}
    with gzip.open(paths['catalog'], 'rt', encoding='utf-8') as stream:
        products = {str(p['asin']): {k: v for k, v in p.items() if k not in ('instructions', 'reviews')}
                    for p in json.load(stream)}
    cache = defaultdict(list)
    for fact in rows(paths['catalog_facts']):
        cache[str(fact['product_asin'])].append(fact)
    records, groups = [], []
    transitions, old_statuses, new_statuses = Counter(), Counter(), Counter()
    for event in rows(diagnostics):
        if event['event'] != 'generation_batch':
            continue
        rescored = {}
        for rollout in event['rollouts']:
            old = rollout['evidence_reward']['score']
            source = contracts[rollout['task_id']]
            if old.get('purchase_asin'):
                if rollout['task_id'] in document['held_tasks'] or old['reward_valid'] is not True:
                    raise ValueError('Replay cannot release a held or invalid purchase')
                before = old['verification']
                product = products[str(old['purchase_asin'])]
                selected = before['snapshot']['selected_options']
                # Bind replay to the exact public product snapshot used on GPU.
                verification = verify(source['normalized'], product, selected,
                                      catalog_facts=cache[str(product['asin'])], purchase_executed=True)
                if verification['snapshot']['product_sha256'] != before['snapshot']['product_sha256']:
                    raise ValueError('Public catalog differs from original scored snapshot')
                score = score_training_verified(source['normalized'], verification, source_query=source['query'])
                score.update(verification=verification, purchase_asin=str(product['asin']))
                if not score['reward_valid']:
                    raise ValueError('Previously valid purchase became unscorable: ' + str(rollout['task_id']))
                for rid, status in verification['statuses'].items():
                    previous = before['statuses'][rid]
                    transitions[previous + '->' + status] += 1
                    old_statuses[previous] += 1
                    new_statuses[status] += 1
            else:
                score = old
            identity = (rollout['uid'], rollout['rollout_index'])
            if identity in rescored:
                raise ValueError('Repeated generated rollout')
            rescored[identity] = score
            records.append({'task_id': rollout['task_id'], 'uid': rollout['uid'],
                            'rollout_index': rollout['rollout_index'], 'old_score': old,
                            'new_score': score, 'regenerated': False})
        for g in event['groups']:
            scores = [s for (uid, _), s in rescored.items() if uid == g['uid']]
            invalid = any(s['reward_valid'] is not True for s in scores)
            values = [float(s['reward']) if s['reward_valid'] else None for s in scores]
            reason = ('sampling_invalid' if invalid else 'constant_reward'
                      if max(values)-min(values) <= 1e-8 else None)
            members = [r for r in event['rollouts'] if r['uid'] == g['uid']]
            groups.append({'task_id': members[0]['task_id'], 'uid': g['uid'],
                           'old_drop_reason': g['drop_reason'], 'new_drop_reason': reason,
                           'old_utilities': g['terminal_utilities'], 'new_utilities': values})
    output.mkdir(parents=True, exist_ok=True)
    def lines(name, data):
        (output / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data), encoding='utf-8')
    lines('rescored_rollouts.jsonl', records)
    lines('rescored_groups.jsonl', groups)
    summary = {'verifier_version': VERSION, 'generated_rollouts': len(records),
               'source_diagnostics_sha256': hashlib.sha256(diagnostics.read_bytes()).hexdigest(),
               'scope': 'fixed_saved_rollouts_counterfactual_rescoring_not_new_training',
               'manifest_used_as_asset_locator_only': True,
               'source_manifest': str(manifest.resolve()),
               'input_contracts_sha256_lf': document['assets']['contracts']['sha256'],
               'input_fact_pack_sha256_lf': document['assets']['catalog_facts']['sha256'],
               'old_group_counts': dict(Counter(g['old_drop_reason'] or 'eligible' for g in groups)),
               'new_group_counts': dict(Counter(g['new_drop_reason'] or 'eligible' for g in groups)),
               'recovered_signal_tasks': [g['task_id'] for g in groups if g['old_drop_reason'] == 'constant_reward' and g['new_drop_reason'] is None],
               'old_status_counts': dict(old_statuses), 'new_status_counts': dict(new_statuses),
               'status_transitions': dict(transitions),
               'old_reward_types': dict(Counter(r['old_score']['type'] for r in records)),
               'new_reward_types': dict(Counter(r['new_score']['type'] for r in records)),
               'training_started': False, 'llm_api_called': False}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostics', type=Path)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'analysis/semantic_batch_acceptance_20261108/training_bundle/manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    replay(args.diagnostics, args.manifest, args.output)


if __name__ == '__main__':
    main()
