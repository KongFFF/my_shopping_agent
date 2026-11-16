import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from export_training_windows import collect, inspect_log, WINDOWS, RUN, digest, publish


def sample_events(start, end):
    result = []
    for step in range(start, end + 1):
        uids = [f'{step}-a', f'{step}-b']
        rollouts = [{'uid': uid, 'rollout_index': i, 'task_id': step % 1000,
                     'actions': [{'tool': 'buy_now', 'parameters': {}}]} for uid in uids for i in range(4)]
        result.extend([{'event': 'generation_batch', 'global_step': step,
                        'groups': [{'uid': uid} for uid in uids], 'rollouts': rollouts},
                       {'event': 'optimizer_step', 'global_step': step,
                        'metrics': {'training/optimizer_updated': 1}, 'rollouts': rollouts}])
    return result


def encode(events):
    return '\n'.join(json.dumps(e) for e in events).encode()


class ExportTests(unittest.TestCase):
    def test_complete_window(self):
        summary = inspect_log(encode(sample_events(651, 800)), 651, 800)
        self.assertEqual(summary['optimizer_updates'], 150)
        self.assertEqual(summary['generated_rollouts'], 1200)
        self.assertEqual(summary['rollouts_with_observation_like_field'], 0)

    def test_missing_and_repeated_update(self):
        events = sample_events(651, 800)
        for modified in (events[:-1], events + [events[-1]]):
            with self.assertRaises(ValueError):
                inspect_log(encode(modified), 651, 800)

    def test_wrong_group_size(self):
        events = sample_events(651, 800)
        events[0]['rollouts'].pop()
        with self.assertRaisesRegex(ValueError, '4 diagnostic'):
            inspect_log(encode(events), 651, 800)

    def test_uid_not_generated(self):
        events = sample_events(651, 800)
        for r in events[1]['rollouts']:
            r['uid'] = 'unknown-a' if r['rollout_index'] < 2 else 'unknown-b'
        with self.assertRaises(ValueError):
            inspect_log(encode(events), 651, 800)

    def test_outside_window_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            inspect_log(encode(sample_events(650, 800)), 651, 800)

    def test_collect_scope_and_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            project = parent / 'upstream/shopping-grpo-longhorizon'
            for _, (folder, start, end) in WINDOWS.items():
                source = project / RUN / folder
                prep = project / RUN / 'preparation' / folder
                source.mkdir(parents=True)
                prep.mkdir(parents=True)
                (source / 'training_diagnostics.jsonl').write_bytes(encode(sample_events(start, end)))
                bound = prep / 'example.py'
                bound.write_text('x = 1\n')
                (prep / 'launch_spec.json').write_text(json.dumps({'project': str(project), 'env': {'API_KEY': 'fixture-do-not-export'},
                    'source_sha256': {str(bound): digest(bound.read_bytes())}}))
                (prep / 'exit_status.json').write_text('{"exit_code":0}')
                (prep / 'composed_config.yaml').write_text('data:\n  x: 1\nray_kwargs:\n  env:\n    API_KEY: fixture-secret\ntrainer:\n  logger: console\n')
                weights = source / 'global_step_800'
                weights.mkdir()
                (weights / 'model.pt').write_bytes(b'fixture-not-a-model')
            (project / 'data/grpo').mkdir(parents=True)
            (project / 'data/grpo/train.jsonl').write_text('\n'.join(json.dumps({'task_id': i}) for i in range(1000)))
            (project / 'configs').mkdir()
            (project / 'configs/agent_loop.yaml').write_text('max_steps: 35\n')
            before = {p: digest(p.read_bytes()) for p in project.rglob('*') if p.is_file()}
            blobs, manifest = collect(parent)
            self.assertEqual(before, {p: digest(p.read_bytes()) for p in project.rglob('*') if p.is_file()})
            self.assertFalse(any(name.endswith('.pt') for name in blobs))
            self.assertNotIn(b'fixture-secret', b''.join(blobs.values()))
            self.assertNotIn(b'fixture-do-not-export', b''.join(blobs.values()))
            self.assertEqual(manifest['windows']['late651_800']['optimizer_updates'], 150)
            for name, record in manifest['sources'].items():
                self.assertEqual(digest(blobs[name]), record['archive_sha256'])

    def test_git_publish_preserves_original_branch_and_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            remote, parent = root / 'remote.git', root / 'parent'
            def git(*args, cwd=root):
                return subprocess.check_output(['git', *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
            git('init', '--bare', str(remote))
            parent.mkdir()
            git('init', cwd=parent)
            git('config', 'user.name', 'Fixture', cwd=parent)
            git('config', 'user.email', 'fixture@example.invalid', cwd=parent)
            (parent / 'code.txt').write_text('fixture')
            git('add', 'code.txt', cwd=parent)
            git('commit', '-m', 'fixture', cwd=parent)
            git('remote', 'add', 'origin', str(remote), cwd=parent)
            (parent / 'outputs').mkdir()
            archive = parent / 'outputs/evidence.zip'
            archive.write_bytes(b'fixture-archive')
            (parent / 'unrelated.txt').write_text('keep staged')
            git('add', 'unrelated.txt', cwd=parent)
            before = git('status', '--porcelain', cwd=parent)
            original_head = git('rev-parse', 'HEAD', cwd=parent)
            publish(parent, archive, 'HEAD')
            self.assertEqual(git('rev-parse', 'HEAD', cwd=parent), original_head)
            self.assertEqual(git('diff', '--cached', '--name-only', cwd=parent), 'unrelated.txt')
            refs = git('for-each-ref', '--format=%(refname)', cwd=remote)
            self.assertIn('refs/heads/codex/training-window-results-', refs)


if __name__ == '__main__':
    unittest.main()
