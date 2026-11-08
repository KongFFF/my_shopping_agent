"""Prepare a deterministic source-label acceptance sample; no API or training.

Original contracts are immutable. Review status changes are emitted separately
only after the batch acceptance gate passes, never called whole-query reviews.
"""
import copy
import hashlib
import json
from pathlib import Path

from semantic_contract_v2_r7_final12 import DIMENSIONS, validate
from semantic_training_bundle import ROOT, rows, dump, sha256

OUT = ROOT / 'analysis/semantic_batch_acceptance_20261108'
SOURCE = ROOT / 'analysis/semantic_remaining_final_20261108/validated1000.jsonl'
LIMITS = ROOT / 'analysis/semantic_remaining_final_20261108/source_limitations16.jsonl'
SEED = 'semantic-batch-acceptance-20261108-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def lines(path, records):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')


def approved_copy(contract):
    result = copy.deepcopy(contract)
    for dim in DIMENSIONS:
        for requirement in result[dim]:
            requirement['review_status'] = 'approved'
    return result


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    records = rows(SOURCE)
    held = {r['task_id'] for r in rows(LIMITS)}
    eligible = [r for r in records if r['task_id'] not in held]
    if len(records) != 1000 or len(eligible) != 984 or len(held) != 16:
        raise ValueError('Frozen roster changed')
    failures = []
    # Before any sampling, validate every original contract and prevent a batch
    # status change from hiding uncertainty, pending strings or source errors.
    for row in records:
        validate(row['normalized'], source_query=row['query'])
        if row['task_id'] in held:
            continue
        c = approved_copy(row['normalized'])
        try:
            validate(c, source_query=row['query'], require_reviewed=True)
        except ValueError as exc:
            # Query-level receipts bind review_status too. Preserve only exact
            # existing reviewed diagnostics under the status-only transition;
            # the final compiler exports new exact receipts and revalidates.
            failures.append({'task_id':row['task_id'], 'error':str(exc)})
    rank = lambda r: hashlib.sha256(f'{SEED}:{r["task_id"]}'.encode()).hexdigest()
    uniform = sorted([r for r in eligible if not r.get('whole_contract_approved')], key=rank)[:40]
    def risk(row):
        c = row['normalized']
        reqs = [r for d in DIMENSIONS for r in c[d]]
        return sum(r['operator'] in ('approximately','matches_label','covers_range','matches_bundle','matches_members','one_of','excludes')
                   or r['subject']['kind'] not in ('primary_product','order') for r in reqs)
    seen = {r['task_id'] for r in uniform}
    targeted = sorted([r for r in eligible if r['task_id'] not in seen and not r.get('whole_contract_approved')],
                      key=lambda r: (-risk(r), rank(r)))[:20]
    sample = [{'task_id':r['task_id'], 'query':r['query'], 'normalized':r['normalized'],
               'selection':'uniform_hash40' if r['task_id'] in seen else 'high_risk20',
               'source_contract_sha256':digest(r['normalized'])} for r in uniform+targeted]
    lines(OUT/'source_sample60.jsonl',sample)
    lines(OUT/'status_transition_findings.jsonl',failures)
    lock = {'seed':SEED, 'source':str(SOURCE.relative_to(ROOT)), 'source_sha256':sha256(SOURCE),
            'source_limits_sha256':sha256(LIMITS), 'eligible':984,'held':sorted(held),
            'sample_ids':[r['task_id'] for r in sample], 'uniform40':[r['task_id'] for r in uniform],
            'risk20':[r['task_id'] for r in targeted], 'sample_sha256':sha256(OUT/'source_sample60.jsonl'),
            'source_only_selection':True,'product_results_not_used_for_selection':True}
    dump(OUT/'sample_lock.json',lock)
    print(json.dumps({'sample':60,'eligible':984,'held':16,'status_transition_findings':len(failures)}))


if __name__ == '__main__':
    prepare()
