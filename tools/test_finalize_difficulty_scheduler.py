"""Regression tests for nullable reward counters and CPU-only finalization."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from difficulty_scheduler import canonical, read, write
from prepare_difficulty_scheduler import file_hash
import finalize_difficulty_scheduler as repair


def summary():
    return {'complete': True, 'limits': 'pilot', 'arms': {
        'uniform': {'reward_types': {'gold_purchase': 10, None: 2}},
        'adaptive': {'reward_types': {None: 3, 'gold_purchase': 11}, 'response_tokens': 100,
                     'input_tokens': 200, 'counts': {'consumed_groups': 242}, 'visits_per_task': {1: 4, 2: 8}}},
        'dev50': {label: {'reward_types': {None: 1, 'gold_purchase': 49}}
                  for label in ('sft', 'uniform', 'adaptive')}}


class ReportTests(unittest.TestCase):
    def test_missing_types_remain_counted_without_mutating_source_or_task_ids(self):
        raw = summary()
        before = copy.deepcopy(raw)
        with self.assertRaises(TypeError):
            canonical(raw)
        fixed = repair.repair_reward_counters(raw)
        canonical(fixed)
        self.assertEqual(raw, before)
        self.assertEqual(sum(fixed['arms']['adaptive']['reward_types'].values()), 14)
        self.assertEqual(fixed['arms']['adaptive']['reward_types'][repair.MISSING], 3)
        self.assertEqual(fixed['arms']['adaptive']['visits_per_task'], {1: 4, 2: 8})
        raw['arms']['uniform']['reward_types'][repair.MISSING] = 5
        with self.assertRaises(ValueError):
            repair.repair_reward_counters(raw)

    def fixture(self, root, incomplete=False):
        write(root / 'crash_recovery.json', {
            'preparation_complete': True, 'recovery_tool_sha256': file_hash(repair.recovery.__file__),
            'backup_sha256': {}, 'active_chain_rollout_target': 4000, 'tail_requested_rollouts': 152,
            'tail_completed_rollouts': 144, 'tail_response_tokens': 10, 'tail_input_tokens': 20,
            'checkpoint_step': 100, 'archive': 'recovery_history/test'})
        for arm in ('uniform', 'adaptive'):
            write(root / arm / 'training_complete.json', {'complete': True, 'generated_rollouts': 4000})
        for label in ('sft', 'uniform', 'adaptive'):
            write(root / ('dev50_' + label) / 'summary.json', {'complete': not incomplete, 'tasks': 50})
        write(root / 'pipeline_exit.json', {'complete': False, 'exit_code': 1, 'error': 'original error'})

    def test_finalize_is_cpu_only_keeps_failure_evidence_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            def fake_report(run):
                result = summary()
                repair.reporter.write(run / 'report/summary.json', result)
                (run / 'report/分析报告.md').write_text('实际训练rollout', encoding='utf-8')
                return result
            with patch.object(repair.recovery, 'assert_idle'), \
                 patch.object(repair.driver, 'assert_free'), patch.object(repair.driver, 'checked'), \
                 patch.object(repair.driver, 'train') as train, patch.object(repair.driver, 'evaluate') as evaluate, \
                 patch.object(repair.reporter, 'report', side_effect=fake_report) as report, \
                 contextlib.redirect_stdout(io.StringIO()):
                original_serializer = repair.reporter.write
                repair.finalize(root)
                repair.finalize(root)
                self.assertIs(repair.reporter.write, original_serializer)
                self.assertEqual(report.call_count, 1)
                train.assert_not_called()
                evaluate.assert_not_called()
            result = read(root / 'report/summary.json')
            self.assertEqual(result['crash_recovery']['adaptive_cost']['requested_rollouts_including_interruption'], 4152)
            status = read(root / 'pipeline_exit.json')
            self.assertTrue(status['complete'])
            self.assertEqual(read(root / status['prior_failure_archive'] / 'pipeline_exit.json')['error'], 'original error')

    def test_incomplete_evaluation_or_failed_audit_does_not_mark_complete(self):
        for incomplete in (True, False):
            with self.subTest(incomplete=incomplete), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root, incomplete)
                before = (root / 'pipeline_exit.json').read_bytes()
                with patch.object(repair.recovery, 'assert_idle'), \
                     patch.object(repair.driver, 'assert_free'), patch.object(repair.driver, 'checked'), \
                     patch.object(repair.reporter, 'report', side_effect=ValueError('audit failed')), \
                     contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(ValueError):
                        repair.finalize(root)
                self.assertEqual((root / 'pipeline_exit.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
