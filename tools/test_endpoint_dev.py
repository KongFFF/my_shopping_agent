"""Evidence-pipeline tests with historical rows relabeled in TEMP fixtures only."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from export_endpoint_dev import collect, export, RUN, strict
from analyze_endpoint_dev import analyze, ROOT, BASE


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.parent = Path(self.temp.name)
        self.project = self.parent / 'upstream/shopping-grpo-longhorizon'
        self.rows = [json.loads(line) for line in (BASE / 'grpo_dev50/trajectories.jsonl').read_text(encoding='utf-8-sig').splitlines()]
        self.summary = json.loads((BASE / 'grpo_dev50/summary.json').read_text(encoding='utf-8-sig'))
        benchmark = self.project / 'data/dev50.jsonl'
        benchmark.parent.mkdir(parents=True)
        benchmark.write_text('\n'.join(json.dumps({'task_id': row['task_id']}) for row in self.rows), encoding='utf-8')
        for step in (750, 800):
            folder = self.folder(step)
            folder.mkdir(parents=True)
            summary = copy.deepcopy(self.summary)
            summary['protocol']['model'] = f'shopping-agent-grpo-step{step}'
            summary['protocol']['benchmark'] = str(benchmark)
            self.write(folder / 'summary.json', summary)
            (folder / 'trajectories.jsonl').write_text('\n'.join(json.dumps(row) for row in self.rows), encoding='utf-8')
            self.write(folder / 'passed.json', {'passed': True})
            self.write(folder / 'exit_status.json', {'exit_code': 0})

    def tearDown(self):
        self.temp.cleanup()

    def folder(self, step):
        return self.project / RUN / f'evaluation/step{step}_independent_v1/dev50'

    def write(self, path, value):
        path.write_text(json.dumps(value), encoding='utf-8')

    def test_success_definition_and_read_only_collection(self):
        original = {p: p.read_bytes() for p in self.project.rglob('*') if p.is_file()}
        _, manifest = collect(self.parent)
        self.assertEqual(manifest['counts']['step800']['strict_successes'], 27)
        self.assertEqual(original, {p: p.read_bytes() for p in self.project.rglob('*') if p.is_file()})
        self.assertEqual(sum(map(strict, self.rows)), 27)
        gold = copy.deepcopy(next(row for row in self.rows if strict(row)))
        gold['terminal_result']['over'] = False
        self.assertFalse(strict(gold))

    def test_duplicate_task_rejected(self):
        rows = self.rows[:-1] + [self.rows[0]]
        (self.folder(800) / 'trajectories.jsonl').write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '50 unique'):
            collect(self.parent)

    def test_summary_mismatch_rejected(self):
        summary = copy.deepcopy(self.summary)
        summary['strict_successes'] = 28
        self.write(self.folder(800) / 'summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'strict-success'):
            collect(self.parent)

    def test_protocol_difference_rejected(self):
        path = self.folder(800) / 'summary.json'
        summary = json.loads(path.read_text())
        summary['protocol']['temperature'] = 0.7
        self.write(path, summary)
        with self.assertRaisesRegex(ValueError, 'protocols differ'):
            collect(self.parent)

    def test_query_difference_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['initial_result']['instruction'] += ' changed'
        (self.folder(800) / 'trajectories.jsonl').write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'queries differ'):
            collect(self.parent)

    def test_receipt_failure_rejected(self):
        self.write(self.folder(800) / 'exit_status.json', {'exit_code': 1})
        with self.assertRaisesRegex(ValueError, 'receipt'):
            collect(self.parent)

    def test_archive_audit_and_hash_tamper(self):
        # Rewrite fixture protocol benchmark path to the historical identity;
        # collect must resolve it locally, so instead patch archive after collect
        # and regenerate its manifest hashes. No real endpoint data is implied.
        blobs, manifest = collect(self.parent)
        for step in (750, 800):
            name = f'step{step}/summary.json'
            summary = json.loads(blobs[name])
            summary['protocol']['benchmark'] = self.summary['protocol']['benchmark']
            blobs[name] = json.dumps(summary).encode()
            import hashlib
            manifest['files'][name]['sha256'] = hashlib.sha256(blobs[name]).hexdigest()
            manifest['files'][name]['bytes'] = len(blobs[name])
        archive = self.parent / 'fixture.zip'
        with zipfile.ZipFile(archive, 'x') as handle:
            for name, raw in blobs.items():
                handle.writestr(name, raw)
            handle.writestr('manifest.json', json.dumps(manifest))
        output = self.parent / 'analysis_fixture_only'
        analyze(archive, output)
        result = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(result['step800']['strict_successes'], 27)
        self.assertEqual(sum(result['step800']['stages'].values()), 50)
        self.assertEqual(result['sft']['strict_successes'], 24)
        with zipfile.ZipFile(self.parent / 'tampered.zip', 'x') as handle:
            for name, raw in blobs.items():
                handle.writestr(name, raw + b' ' if name.endswith('summary.json') else raw)
            handle.writestr('manifest.json', json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            analyze(self.parent / 'tampered.zip', self.parent / 'bad_output')


if __name__ == '__main__':
    unittest.main()
