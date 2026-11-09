"""CPU scoring-kernel coverage only; fixed synthetic purchase, not an eval."""
import argparse
import json
import math
from pathlib import Path

from semantic_training_bundle import ROOT, TrainingEvidenceBackend, rows


def check(manifest, output):
    backend = TrainingEvidenceBackend(manifest)
    backend.require_ready()
    previous = rows(ROOT / 'analysis/semantic_batch_acceptance_20261108/static_train19.jsonl')
    def state(probe):
        return {'_evidence_terminal': {'done': True, 'over': True, 'termination_reason': 'gold_purchase',
                                     'purchase': {'asin': probe['asin'], 'options': probe['selected_options']}},
                'steps': [{'tool': 'buy_now'}], 'error': None, 'infrastructure_invalid': False}
    records = []
    for task in sorted(backend.released_ids):
        score = backend.score(task, state(previous[0]))
        if not score['reward_valid'] or not math.isfinite(score['reward']) or not -.85 <= score['reward'] <= 1:
            raise ValueError('Scoring kernel failed for released task ' + str(task))
        records.append({'task_id': task, 'reward': score['reward'], 'reward_valid': True,
                        'type': score['type'], 'verification_blockers': score['verification']['readiness_blockers']})
    for task in sorted(backend.held_ids):
        score = backend.score(task, state(previous[0]))
        if score['reward_valid'] or score['reward'] is not None:
            raise ValueError('Held source leaked into gradient-eligible scores')
    static = []
    for probe in previous:
        score = backend.score(probe['task_id'], state(probe))
        if not score['reward_valid'] or not math.isfinite(score['reward']) or not -.85 <= score['reward'] <= 1:
            raise ValueError('Previous static SKU probe became invalid')
        static.append({'task_id': probe['task_id'], 'asin': probe['asin'], 'selected_options': probe['selected_options'],
                       'not_a_historical_purchase': True, 'score': score})
    output.mkdir(parents=True, exist_ok=True)
    for name, data in [('kernel_stress981.jsonl', records), ('static_train19.jsonl', static)]:
        (output / name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in data), encoding='utf-8')
    summary = {'released_scoring_kernel_cases_passed': len(records), 'held_source_cases_passed': len(backend.held_ids),
               'static_sku_probes_passed': len(static), 'scope': 'synthetic CPU kernel coverage; not task solvability or model score',
               'api_calls': 0, 'gpu_training_executed': False}
    (output / 'kernel_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(args.manifest, args.output)
