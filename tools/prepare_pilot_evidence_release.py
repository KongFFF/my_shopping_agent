"""Bind a user-approved source correction to the existing bounded release.

No whole-query manual approval is inferred. Original queries, hold roster,
participation policy and source acceptance sample are retained and checked.
"""
import argparse
import copy
import json
from pathlib import Path

from semantic_training_bundle import ROOT, rows, dump, sha256, build_bundle
from semantic_batch_release import digest, verify_release


def prepare(directory):
    previous_path = ROOT / 'analysis/semantic_batch_acceptance_20261108/pilot_release.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8'))
    paths = {k: ROOT / a['path'] for k, a in previous['files'].items()}
    amendment = json.loads((directory / 'source_amendment1295.json').read_text(encoding='utf-8'))
    if amendment.get('confirmed_by_user') is not True or amendment['whole_query_manual_approval_claimed'] is not False:
        raise ValueError('Missing precise user decision')
    current = {r['task_id']: r for r in rows(directory / 'contracts1000.jsonl')}
    old = {r['task_id']: r for r in rows(paths['contracts'])}
    task = amendment['task_id']
    if set(current) != set(old) or len(current) != 1000:
        raise ValueError('Original1000 task roster changed')
    for t, row in current.items():
        if row['query'] != old[t]['query']:
            raise ValueError('Actor query changed')
        expected = amendment['after'] if t == task else old[t]['normalized']
        if row['normalized'] != expected:
            raise ValueError('Unexpected requirement change')
    if old[task]['normalized'] != amendment['before']:
        raise ValueError('Correction not bound to preceding contract')
    # Preserve the generic existing source-edit ledger, rather than granting
    # a runtime exception by task ID or silently bypassing its fingerprints.
    source = {r['task_id']: r for r in rows(paths['source_contracts'])}
    edits = {r['task_id']: r for r in rows(paths['data_edits'])}
    edits[task] = {'task_id': task, 'before': source[task]['normalized'],
                   'after': amendment['after'], 'reason': 'explicit user-confirmed source relation correction',
                   'decision_file': (directory / 'source_amendment1295.json').relative_to(ROOT).as_posix()}
    edit_path = directory / 'data_edits.jsonl'
    edit_path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in edits.values()), encoding='utf-8')
    ledger = rows(paths['ledger'])
    for row in ledger:
        if row['task_id'] == task:
            row['contract_sha256'] = digest(amendment['after'])
            row['source_relation_correction'] = 'user confirmed; does not create whole-query approval'
    ledger_path = directory / 'release_ledger1000.jsonl'
    ledger_path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in ledger), encoding='utf-8')
    sample = rows(paths['sample_reviews'])
    for row in sample:
        if row['task_id'] == task:
            row['accepted_contract_sha256'] = digest(amendment['after'])
            row['verdict'] = 'fixed_and_rechecked'
            row['source_relation_correction'] = 'explicit user decision 2026-11-09'
    sample_path = directory / 'sample_reviews60.jsonl'
    sample_path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in sample), encoding='utf-8')
    certificate = copy.deepcopy(previous)
    replacements = {'contracts': directory / 'contracts1000.jsonl', 'ledger': ledger_path,
                    'data_edits': edit_path, 'sample_reviews': sample_path,
                    'source_amendment': directory / 'source_amendment1295.json'}
    for name, path in replacements.items():
        certificate['files'][name] = {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha256(path)}
    certificate['revision_scope'] = 'one user-confirmed source relation; original source checks retained; new runtime checks in verification.json'
    cert_path = directory / 'pilot_release.json'
    dump(cert_path, certificate)
    verify_release(cert_path, root=ROOT, contracts=directory / 'contracts1000.jsonl', sha256=sha256, rows=rows)
    build_bundle(directory / 'training_bundle', contracts_path=directory / 'contracts1000.jsonl',
                 batch_release_path=cert_path, catalog_facts_path=directory / 'catalog_facts.jsonl')
    print('Original1000/queries retained; one user-approved relation correction and reviewed facts frozen.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    prepare(parser.parse_args().directory.resolve())
