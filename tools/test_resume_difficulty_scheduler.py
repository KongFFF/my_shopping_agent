"""CPU regression checks for explicit crash recovery and cost disclosure."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from difficulty_scheduler import TaskScheduler, canonical, read, sha, write
from resume_difficulty_scheduler import replay_checkpoint, prepare, cost_receipt, pipeline


class RecoveryTests(unittest.TestCase):
    def fixture(self, root):
        folder = root / 'adaptive'
        folder.mkdir()
        scheduler = TaskScheduler([1, 2, 3], 'adaptive', 20261112, 40)
        events = []
        checkpoint = None
        for step in (1, 2):
            selection = scheduler.reserve()
            events.append({'event': 'selection', **selection, 'generated': scheduler.generated})
            infos = [{'task_id': t['task_id'], 'done': True, 'reward_valid': True,
                      'reward_type': 'gold_purchase' if i < 2 else 'wrong_purchase',
                      'infrastructure_invalid': False, 'reward_unverifiable': False, 'overlong': False,
                      'reward': {'sampling_invalid': False, 'terminal_utility': float(i < 2), 'total': float(i < 2)}}
                     for t in selection['tasks'] for i in range(4)]
            scheduler.observe(infos)
            uids = [f'{step}-{i}' for i in range(2)]
            events.append({'event': 'generation', 'rollouts': infos,
                           'response_tokens': 10, 'input_plus_response_tokens': 30,
                           'groups': [{'uid': u, 'kept': True} for u in uids],
                           'rollout_uids': [u for u in uids for _ in range(4)]})
            events.append({'event': 'ready', 'uids': uids})
            scheduler.commit_update(step)
            events.append({'event': 'update', 'update': step, 'consumed_uids': uids})
            if step == 1:
                checkpoint = json.loads(canonical({'binding': '', 'scheduler': scheduler.state_dict(),
                             'runtime': {'skipped_updates': 0, 'consecutive_skips': 0}}))
        selection = scheduler.reserve()
        events.append({'event': 'selection', **selection, 'generated': scheduler.generated})
        spec = {'arm': 'adaptive', 'seed': scheduler.seed}
        write(folder / 'spec.json', spec)
        checkpoint['binding'] = sha(canonical(spec))
        current = {'binding': checkpoint['binding'], 'scheduler': scheduler.state_dict(),
                   'runtime': checkpoint['runtime']}
        write(folder / 'scheduler_state.json', current)
        cp = folder / 'global_step_1'
        write(cp / 'scheduler_state.json', checkpoint)
        (cp / 'actor').mkdir()
        for name in ('model', 'optim', 'extra_state'):
            (cp / 'actor' / f'{name}_world_size_1_rank_0.pt').write_bytes(b'test-only')
        (cp / 'data.pt').write_bytes(b'test-only')
        (folder / 'latest_checkpointed_iteration.txt').write_text('1')
        raw = b''.join((canonical(e) + '\n').encode() for e in events)
        (folder / 'scheduler_events.jsonl').write_bytes(raw)
        write(root / 'uniform/training_complete.json', {'complete': True})
        write(root / 'dev50_uniform/summary.json', {'complete': True})
        write(root / 'pipeline.lock', {'pid': 99999})
        return checkpoint, raw

    def test_exact_prefix_and_torn_tail_accounted(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, raw = self.fixture(Path(tmp))
            plan = replay_checkpoint(raw + b'{"partial":', checkpoint)
            self.assertEqual(plan['tail_completed_rollouts'], 8)
            self.assertEqual(plan['requested_from_events'], 24)
            self.assertTrue(plan['torn_final_event'])
            retained = raw[:plan['prefix_bytes']]
            self.assertEqual(json.loads(retained.splitlines()[-1])['update'], 1)
            tampered = raw.replace(b'"probability":', b'"probability":0.0,"wrong":', 1)
            with self.assertRaises(ValueError):
                replay_checkpoint(tampered, checkpoint)

    def test_prepare_archives_and_restores_without_touching_weights_or_uniform(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, raw = self.fixture(root)
            original_state = (root / 'adaptive/scheduler_state.json').read_bytes()
            uniform = (root / 'uniform/training_complete.json').read_bytes()
            with patch('resume_difficulty_scheduler.assert_idle'), \
                 patch('run_difficulty_scheduler.checked'), \
                 patch('run_difficulty_scheduler.assert_free'), \
                 contextlib.redirect_stdout(io.StringIO()):
                plan = prepare(root, 1, confirm=False)
                self.assertFalse((root / 'crash_recovery.json').exists())
                plan = prepare(root, 1, confirm=True)
                with self.assertRaisesRegex(ValueError, 'already prepared'):
                    prepare(root, 1, confirm=True)
            archive = root / plan['archive']
            self.assertEqual((archive / 'adaptive/scheduler_events.jsonl').read_bytes(), raw)
            self.assertEqual((archive / 'adaptive/scheduler_state.json').read_bytes(), original_state)
            self.assertEqual(read(root / 'adaptive/scheduler_state.json'), checkpoint)
            self.assertEqual((root / 'uniform/training_complete.json').read_bytes(), uniform)
            self.assertFalse((root / 'pipeline.lock').exists())
            self.assertTrue((archive / 'pipeline.lock').exists())
            self.assertTrue(read(root / 'crash_recovery.json')['preparation_complete'])
            cost = cost_receipt(plan)
            self.assertEqual(cost['requested_rollouts_including_interruption'], 56)
            self.assertEqual(cost['completed_rollouts_with_saved_results'], 48)
            self.assertEqual(cost['interrupted_unconfirmed_rollouts'], 8)
            self.assertFalse(cost['strict_equal_actual_cost'])

    def test_recovered_pipeline_discloses_extra_cost_and_report_failure_is_not_success(self):
        import resume_difficulty_scheduler as recovery
        from prepare_difficulty_scheduler import file_hash
        for fail_report in (False, True):
            with self.subTest(fail_report=fail_report), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                plan = {'preparation_complete': True, 'backup_sha256': {},
                        'recovery_tool_sha256': file_hash(recovery.__file__),
                        'active_chain_rollout_target': 4000, 'tail_requested_rollouts': 152,
                        'tail_completed_rollouts': 144, 'tail_response_tokens': 10,
                        'tail_input_tokens': 20, 'checkpoint_step': 100, 'archive': 'recovery_history/test'}
                write(root / 'crash_recovery.json', plan)
                def report(run):
                    if fail_report:
                        raise RuntimeError('audit failed')
                    (run / 'report').mkdir()
                    (run / 'report/分析报告.md').write_text('实际训练rollout', encoding='utf-8')
                    return {'complete': True, 'limits': 'pilot', 'arms': {'adaptive': {
                        'response_tokens': 100, 'input_tokens': 200, 'counts': {'consumed_groups': 240}}}}
                with patch('run_difficulty_scheduler.checked'), \
                     patch('run_difficulty_scheduler.train') as train, \
                     patch('run_difficulty_scheduler.evaluate') as evaluate, \
                     patch('report_difficulty_scheduler.report', side_effect=report), \
                     contextlib.redirect_stdout(io.StringIO()):
                    if fail_report:
                        with self.assertRaisesRegex(RuntimeError, 'audit failed'):
                            pipeline(root)
                    else:
                        pipeline(root)
                        costs = read(root / 'report/summary.json')['crash_recovery']['adaptive_cost']
                        self.assertEqual(costs['requested_rollouts_including_interruption'], 4152)
                        self.assertEqual(costs['total_known_response_tokens'], 110)
                        self.assertFalse(costs['strict_equal_actual_cost'])
                    self.assertEqual([call.args[1] for call in train.call_args_list], ['uniform', 'adaptive'])
                    self.assertEqual([call.args[1] for call in evaluate.call_args_list], ['sft', 'uniform', 'adaptive'])
                status = read(root / 'pipeline_exit.json')
                self.assertEqual(status['complete'], not fail_report)
                self.assertEqual(status['exit_code'], 1 if fail_report else 0)
                self.assertFalse((root / 'pipeline.lock').exists())

    def test_missing_optimizer_or_wrong_pointer_refused_before_any_writes(self):
        for failure in ('optimizer', 'pointer'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root)
                if failure == 'optimizer':
                    (root / 'adaptive/global_step_1/actor/optim_world_size_1_rank_0.pt').unlink()
                else:
                    (root / 'adaptive/latest_checkpointed_iteration.txt').write_text('2')
                with patch('resume_difficulty_scheduler.assert_idle'), \
                     patch('run_difficulty_scheduler.checked'), \
                     patch('run_difficulty_scheduler.assert_free'):
                    with self.assertRaises(ValueError):
                        prepare(root, 1, confirm=True)
                self.assertFalse((root / 'crash_recovery.json').exists())


if __name__ == '__main__':
    unittest.main()
