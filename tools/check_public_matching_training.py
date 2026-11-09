"""Freeze the repaired CPU backend and test the real adapter in isolation.

Model generation/HTTP are scripted by existing CPU tests. No GPU or API calls.
Prior manifests, certificates and source contracts are never overwritten.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest

from semantic_training_bundle import ROOT, build_bundle, TrainingEvidenceBackend, sha256


def check(output, previous=None, catalog_facts=None):
    previous = previous or ROOT / 'analysis/semantic_batch_acceptance_20261108/training_bundle/manifest.json'
    old = json.loads(previous.read_text(encoding='utf-8'))
    contracts = ROOT / old['assets']['contracts']['path']
    input_contract_hash = sha256(contracts)
    approvals = ROOT / old['assets']['whole_source_approvals']['path']
    release = ROOT / old['assets']['batch_release']['path']
    build_bundle(output / 'training_bundle', contracts_path=contracts,
                 approvals_path=approvals, batch_release_path=release, catalog_facts_path=catalog_facts)
    manifest = output / 'training_bundle/manifest.json'
    backend = TrainingEvidenceBackend(manifest)
    backend.require_ready()
    import test_semantic_training_integration as integration
    import test_semantic_batch_release as batch
    temporary_root = ROOT / 'outputs'
    temporary_root.mkdir(exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='public_matching_cpu_', dir=temporary_root) as tmp:
        tmp = Path(tmp).resolve()
        if not tmp.is_relative_to(temporary_root.resolve()):
            raise ValueError('Test workspace outside intended temporary root')
        integration.DEFAULT_OUTPUT = tmp / 'legacy'
        integration.build_bundle = lambda: build_bundle(
            integration.DEFAULT_OUTPUT, contracts_path=contracts, approvals_path=approvals, catalog_facts_path=catalog_facts)
        batch.OUT = tmp / 'batch'
        (batch.OUT / 'training_bundle').mkdir(parents=True)
        shutil.copyfile(manifest, batch.OUT / 'training_bundle/manifest.json')
        shutil.copyfile(release, batch.OUT / 'pilot_release.json')
        shutil.copyfile(contracts, batch.OUT / 'contracts1000.jsonl')
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(module)
                                   for module in (integration, batch))
        result = unittest.TextTestRunner(verbosity=1).run(suite)
    receipt = {
        'passed': result.wasSuccessful(), 'cpu_integration_tests': result.testsRun,
        'duration_seconds': round(time.monotonic()-started, 3),
        'manifest': str(manifest.relative_to(ROOT)),
        'input_contracts_preserved_during_tests': sha256(contracts) == input_contract_hash,
        'released_tasks': len(backend.released_ids), 'held_tasks': len(backend.held_ids),
        'generation_and_http': 'scripted; real backend, adapter and group selector',
        'gpu_training_executed': False, 'api_calls': 0,
    }
    (output / 'cpu_integration.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous-manifest', type=Path)
    parser.add_argument('--catalog-facts', type=Path)
    args = parser.parse_args()
    check(args.output.resolve(), args.previous_manifest, args.catalog_facts)
