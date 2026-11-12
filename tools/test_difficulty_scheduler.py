"""Stdlib CPU tests, including execution of the pinned trainer's filter block."""
import ast
import contextlib
import copy
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from difficulty_scheduler import (TaskScheduler, bounded_probabilities, canonical,
                                  classify, read, write, sha, append)
from difficulty_scheduler_hook import DriverSchedule
from difficulty_scheduler_patch import build_trainer, DYNAMIC, ORIGINAL, unified_patch
from prepare_difficulty_scheduler import ROOT, PACKAGE


def info(task, gold=False, **changes):
    result = {'task_id': task, 'done': True, 'reward_valid': True,
              'reward_type': 'gold_purchase' if gold else 'wrong_purchase',
              'infrastructure_invalid': False, 'reward_unverifiable': False,
              'overlong': False, 'steps': 3, 'guard_rejections': 0,
              'reward': {'terminal_utility': 1. if gold else 0., 'total': 1. if gold else 0.,
                         'purchase_success': float(gold), 'sampling_invalid': False}}
    result.update(changes)
    return result


def sampled_infos(selection, pattern=(False, False, True, True)):
    return [info(t['task_id'], gold) for t in selection['tasks'] for gold in pattern]


class Array(list):
    def tolist(self):
        return list(self)


class Tensor:
    def __init__(self, values):
        self.values = values
    def sum(self, dim=None):
        if dim is not None:
            return Tensor([sum(v) for v in self.values])
        return Tensor(sum(sum(v) if isinstance(v, list) else v for v in self.values))
    def item(self):
        return self.values
    def detach(self):
        return self
    def cpu(self):
        return self
    def tolist(self):
        return self.values


class Batch:
    def __init__(self, infos, uids=None):
        self.infos = infos
        self.uids = uids or [f'u-{r["task_id"]}' for r in infos]
        self.meta_info = {}
        self.non_tensor_batch = {'shopping': Array(infos), 'uid': Array(self.uids),
                                'multi_modal_inputs': Array({} for _ in infos)}
        self.batch = {'response_mask': Tensor([[1, 1] for _ in infos]),
                      'attention_mask': Tensor([[1, 1, 1] for _ in infos])}
    def select_idxs(self, indices):
        return Batch([self.infos[i] for i in indices], [self.uids[i] for i in indices])
    @staticmethod
    def concat(batches):
        return Batch([r for b in batches for r in b.infos], [u for b in batches for u in b.uids])


def trainer(ids, root, step=0):
    def save():
        (root / f'global_step_{result.global_steps}').mkdir(exist_ok=True)
        if hasattr(result, 'difficulty_schedule'):
            result.difficulty_schedule.checkpoint(result)
    result = NS(global_steps=step, train_dataset=NS(dataframe=[{'extra_info': {'task_id': t}} for t in ids]),
                train_dataloader=NS(collate_fn=lambda x: x), _save_checkpoint=save)
    return result


class SchedulerTests(unittest.TestCase):
    def test_probability_boundaries_and_mass(self):
        rng = random.Random(7)
        for k in (1, 2, 17, 999, 1000):
            for weights in ([0.] * k, [1.] * k, [1.] + [0.] * (k - 1),
                            [10 ** rng.uniform(-15, 10) for _ in range(k)]):
                p = bounded_probabilities(weights)
                self.assertAlmostEqual(sum(p), 1)
                self.assertGreaterEqual(min(p), .5 / k - 1e-12)
                self.assertLessEqual(max(p), min(4 / k, 1) + 1e-12)

    def test_bad_weights_rejected(self):
        for weights in ([], [-1], [math.nan], [math.inf]):
            with self.assertRaises(ValueError):
                bounded_probabilities(weights)

    def test_neutral_initialization_and_no_old_labels(self):
        s = TaskScheduler(range(1000), 'adaptive', 42)
        self.assertEqual({s.estimates(t) for t in s.ids}, {(.5, 1., .875)})

    def test_batch_unique_and_conditional_uniform_probabilities(self):
        s = TaskScheduler(range(1000), 'uniform', 42)
        for _ in range(30):
            chosen = s.reserve()
            self.assertNotEqual(chosen['tasks'][0]['task_id'], chosen['tasks'][1]['task_id'])
            self.assertEqual([t['probability'] for t in chosen['tasks']], [1 / 1000, 1 / 999])
            s.observe(sampled_infos(chosen))

    def test_cannot_select_with_inflight_generation(self):
        s = TaskScheduler(range(10), 'adaptive', 42)
        s.reserve()
        with self.assertRaises(ValueError):
            s.reserve()

    def test_all_generated_counts_even_without_update(self):
        s = TaskScheduler(range(10), 'adaptive', 42)
        selected = s.reserve()
        s.observe(sampled_infos(selected, (False,) * 4))
        for t in selected['tasks']:
            self.assertEqual(s.counts[t['task_id']], [0, 4, 4, 0])
        self.assertEqual(s.updates, 0)

    def test_invalid_not_negative_and_alternative_not_gold(self):
        self.assertEqual(classify(info(1, reward_type='valid_alternative_purchase'))[:2], (False, True))
        for change in ({'overlong': True}, {'reward_unverifiable': True}, {'reward_valid': False}):
            self.assertFalse(classify(info(1, True, **change))[1])
        s = TaskScheduler(range(2), 'adaptive', 1)
        selected = s.reserve()
        s.observe([info(t['task_id'], True, reward_unverifiable=True) for t in selected['tasks'] for _ in range(4)])
        self.assertEqual(s.counts[0], [0, 0, 4, 0])

    def test_context_flag_is_policy_unusable_not_infra_halt(self):
        result = classify(info(1, infrastructure_invalid=True, termination_reason='context_hard_limit_exceeded'))
        self.assertFalse(result[1])
        self.assertIn('policy_context_limit', result[2])

    def test_real_infra_stops_without_partial_mutation(self):
        s = TaskScheduler(range(2), 'adaptive', 1)
        selected = s.reserve()
        before = copy.deepcopy(s.counts)
        infos = sampled_infos(selected)
        infos[-1]['infrastructure_invalid'] = True
        infos[-1]['termination_reason'] = 'tool_error:TimeoutError'
        with self.assertRaises(RuntimeError):
            s.observe(infos)
        self.assertEqual(before, s.counts)
        self.assertIsNotNone(s.pending)

    def test_schema_missing_rejected(self):
        value = info(1)
        del value['reward_valid']
        with self.assertRaises(ValueError):
            classify(value)

    def test_wrong_task_and_group_sizes_rejected(self):
        s = TaskScheduler(range(3), 'adaptive', 1)
        chosen = s.reserve()
        with self.assertRaises(ValueError):
            s.observe(sampled_infos(chosen)[:7])
        bad = sampled_infos(chosen)
        bad[0]['task_id'] = next(t for t in s.ids if t not in [x['task_id'] for x in chosen['tasks']])
        with self.assertRaises(ValueError):
            s.observe(bad)

    def test_decay_uses_updates_not_generation_and_prior_fixed(self):
        s = TaskScheduler(range(2), 'adaptive', 1)
        s.counts[0] = [4., 4., 4., 0]
        s.updates = 100
        self.assertEqual(s._aged(0), (2., 2., 2.))
        self.assertAlmostEqual(s.estimates(0)[0], 2.5 / 3)
        s.batches = 1000
        self.assertEqual(s._aged(0), (2., 2., 2.))

    def test_priority_prefers_mixed_over_confident_or_unusable(self):
        s = TaskScheduler(range(4), 'adaptive', 1)
        s.counts[0] = [50., 100., 100., 0]
        s.counts[1] = [100., 100., 100., 0]
        s.counts[2] = [0., 100., 100., 0]
        s.counts[3] = [0., 0., 100., 0]
        weights = [s.estimates(t)[2] for t in s.ids]
        self.assertGreater(weights[0], weights[1])
        self.assertAlmostEqual(weights[1], weights[2])
        self.assertGreater(weights[2], weights[3])

    def test_json_resume_exact_rng_counts_and_budget(self):
        s = TaskScheduler(range(1000), 'adaptive', 123)
        for _ in range(37):
            selected = s.reserve()
            s.observe(sampled_infos(selected))
            s.commit_update(s.updates + 1)
        resumed = TaskScheduler.from_state(json.loads(canonical(s.state_dict())))
        for _ in range(30):
            a, b = s.reserve(), resumed.reserve()
            self.assertEqual(a, b)
            s.observe(sampled_infos(a))
            resumed.observe(sampled_infos(b))
        self.assertEqual(canonical(s.state_dict()), canonical(resumed.state_dict()))

    def test_fixed_budget_includes_all_same_and_no_forced_updates(self):
        s = TaskScheduler(range(20), 'adaptive', 1, budget=4000)
        for _ in range(500):
            selected = s.reserve()
            s.observe(sampled_infos(selected, (False,) * 4))
        self.assertTrue(s.exhausted)
        self.assertEqual(s.generated, 4000)
        self.assertEqual(s.updates, 0)
        with self.assertRaises(ValueError):
            s.reserve()

    def test_update_commits_consecutive_and_consume_eight(self):
        s = TaskScheduler(range(20), 'adaptive', 1)
        with self.assertRaises(ValueError):
            s.commit_update(2)
        with self.assertRaises(ValueError):
            s.commit_update(1)
        chosen = s.reserve()
        s.observe(sampled_infos(chosen))
        s.commit_update(1)
        self.assertEqual(s.consumed_rollouts, 8)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.spec = self.root / 'spec.json'
        write(self.spec, {'task_ids': list(range(20)), 'arm': 'adaptive', 'seed': 42,
                         'budget': 40, 'output': str(self.root)})
        self.env = patch.dict(os.environ, SHOPPING_SCHEDULER_SPEC=str(self.spec))
        self.env.start()
    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def hook(self):
        t = trainer(range(20), self.root)
        t.difficulty_schedule = DriverSchedule(t)
        return t, t.difficulty_schedule

    def test_dataset_roster_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            DriverSchedule(trainer(range(19), self.root))

    def test_stop_between_stages_prevents_new_gpu_work_and_lock_preserves_stop(self):
        import run_difficulty_scheduler as runner
        (self.root / 'STOP').touch()
        with patch.object(runner, 'checked', return_value={}):
            with self.assertRaisesRegex(RuntimeError, 'safely paused'):
                runner.train(self.root, 'uniform')
            with self.assertRaisesRegex(RuntimeError, 'safely paused'):
                runner.evaluate(self.root, 'sft')
            with runner.lock(self.root / 'pipeline.lock'):
                with self.assertRaises(FileExistsError):
                    runner.pipeline(self.root)
        self.assertTrue((self.root / 'STOP').exists())

    def test_safe_checkpoint_resume_and_uncommitted_tail_refused(self):
        t, h = self.hook()
        selected = h.scheduler.reserve()
        h.observe(Batch(sampled_infos(selected)), {'groups': []})
        h.committed(1)
        h.skipped(3, 2)
        t.global_steps = 1
        t._save_checkpoint()
        resumed = DriverSchedule(trainer(range(20), self.root, 1))
        self.assertEqual(canonical(resumed.scheduler.state_dict()), canonical(h.scheduler.state_dict()))
        self.assertEqual(resumed.runtime, {'skipped_updates': 3, 'consecutive_skips': 2})
        self.assertEqual(resumed.last_checkpoint_generated, 8)
        resumed.scheduler.reserve()
        resumed.persist()
        with self.assertRaisesRegex(RuntimeError, 'Uncommitted'):
            DriverSchedule(trainer(range(20), self.root, 1))

    def test_reordered_uids_preserved_in_audit(self):
        t, h = self.hook()
        selected = h.scheduler.reserve()
        records = sampled_infos(selected)
        order = [4, 0, 5, 1, 6, 2, 7, 3]
        batch = Batch([records[i] for i in order])
        h.observe(batch, {'groups': []})
        events = [json.loads(x) for x in h.events.read_text(encoding='utf-8').splitlines()]
        for uid, record in zip(events[-1]['rollout_uids'], events[-1]['rollouts']):
            self.assertEqual(uid, 'u-' + str(record['task_id']))

    def test_residual_budget_stop_saves_without_update(self):
        t, h = self.hook()
        for _ in range(5):
            selected = h.scheduler.reserve()
            h.observe(Batch(sampled_infos(selected, (False,) * 4)), {'groups': []})
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(h.boundary_stop(t, residual_groups=1))
        end = read(self.root / 'training_complete.json')
        self.assertEqual(end['optimizer_updates'], 0)
        self.assertEqual(end['generated_rollouts'], 40)
        self.assertEqual(end['residual_groups_discarded'], 1)

    def test_original_transform_hash_compile_and_placement(self):
        base = (PACKAGE / 'ray_trainer_0_8_0.py').read_text(encoding='utf-8')
        dynamic = (PACKAGE / 'baseline_dynamic.patch').read_text(encoding='utf-8')
        self.assertEqual(sha(base), ORIGINAL)
        self.assertEqual(sha(unified_patch(base, dynamic)), DYNAMIC)
        source = build_trainer(base, dynamic)
        compile(source, '<scheduled ray trainer>', 'exec')
        self.assertLess(source.index('schedule.observe(batch, group_stats)'), source.index('dynamic_num_gen_batches += 1'))
        self.assertLess(source.index('schedule.committed(current_step)'), source.index('# Check if the ESI'))
        self.assertIn('schedule.skipped(dynamic_skipped_updates, dynamic_consecutive_skips)', source)
        with self.assertRaises(ValueError):
            build_trainer(base + '\n# unknown edit', dynamic)

    def test_actual_trainer_dynamic_block_budget_residual_and_group_identity(self):
        # Execute the real transformed AST block, with tensor/worker calls replaced
        # by CPU test doubles. Filtering, concat, ordering and stop code are real.
        base = (PACKAGE / 'ray_trainer_0_8_0.py').read_text(encoding='utf-8')
        dynamic = (PACKAGE / 'baseline_dynamic.patch').read_text(encoding='utf-8')
        tree = ast.parse(build_trainer(base, dynamic))
        blocks = [n for n in ast.walk(tree) if isinstance(n, ast.If) and
                  isinstance(n.test, ast.Name) and n.test.id == 'dynamic_sampling_enabled' and
                  any(isinstance(x, ast.Name) and x.id == 'seq_rewards' for x in ast.walk(n))]
        self.assertEqual(len(blocks), 1)
        from prepare_difficulty_scheduler import UP, BASE
        source = subprocess.check_output(['git', '-C', str(UP), 'show', BASE +
            ':src/shopping_grpo/training/grpo/dynamic_sampling.py']).decode('utf-8')
        helper = NS()
        namespace_helper = {}
        exec(compile(source, '<pinned original group filter>', 'exec'), namespace_helper)
        helper.__dict__.update(namespace_helper)
        # First batch: one mixed + one all-same; second: two mixed (surplus1);
        # Last batch has one usable group, then budget ends before the three-batch
        # cap. That residual must be discarded rather than force an update.
        patterns = [(True, False), (True, True), (True, False)]
        spec_value = read(self.spec)
        spec_value['budget'] = 24
        write(self.spec, spec_value)
        t, h = self.hook()
        t.config = NS(trainer=NS(balance_batch=False))
        t.checkpoint_manager = NS(sleep_replicas=lambda: None)
        t._shutdown_dump_executor = lambda: None
        metrics = {}
        namespace = {'self': t, 'schedule': h, 'dynamic_sampling_enabled': True,
                     'dynamic_rollout_n': 4, 'dynamic_target_prompts': 2,
                     'dynamic_reward_tolerance': 1e-8, 'dynamic_max_gen_batches': 3,
                     'dynamic_max_consecutive_skips': 10, 'curr_step_profile': False,
                     'diagnostics_path': None, 'metrics': metrics,
                     'logger': NS(log=lambda **kw: None), 'progress_bar': NS(close=lambda: None),
                     'DataProto': Batch, 'torch': NS(sum=lambda value, dim: value.sum(dim)),
                     'json': json, 'patterns': patterns,
                     'extract_reward': lambda batch: (Tensor([[r['reward']['total']] for r in batch.infos]), {}),
                     'accepted_uids_by_update': [], 'info': info, 'Batch': Batch}
        for name in ('extract_shopping_group_signals', 'select_reward_varying_groups',
                     'append_training_diagnostic', 'build_rollout_diagnostics'):
            namespace[name] = getattr(helper, name)
        for name in ('dynamic_accepted_prompts', 'dynamic_num_gen_batches', 'dynamic_generated_groups',
                     'dynamic_filtered_groups', 'dynamic_all_equal_groups', 'dynamic_all_zero_utility_groups',
                     'dynamic_all_purchase_success_groups', 'dynamic_no_purchase_success_groups',
                     'dynamic_sampling_invalid_groups', 'dynamic_generated_trajectories',
                     'dynamic_skipped_updates', 'dynamic_consecutive_skips'):
            namespace[name] = 0
        namespace['dynamic_accepted_batches'] = []
        prefix = '''
for generation, mixed in enumerate(patterns):
    selected = schedule.scheduler.reserve()
    schedule.persist()
    values, uids = [], []
    for position, task in enumerate(selected['tasks']):
        for i in range(4):
            values.append(info(task['task_id'], mixed[position] and i>=2))
            uids.append(f'{generation}-{position}')
    batch = Batch(values,uids)
    reward_tensor, reward_extra_infos_dict = extract_reward(batch)
    current_step = self.global_steps + 1
'''
        tail = '''
    accepted_uids_by_update.append(list(batch.non_tensor_batch['uid']))
    self.global_steps += 1
    schedule.committed(self.global_steps)
    if schedule.boundary_stop(self):
        return
'''
        body = ast.parse(prefix).body[0]
        body.body.append(copy.deepcopy(blocks[0]))
        body.body.extend(ast.parse('def tail():\n' + tail).body[0].body)
        # All dynamic locals must be function locals, including counters and buffers.
        names = [k for k in namespace if k.startswith('dynamic_')]
        assignments = ast.parse('\n'.join(f'{k} = initial["{k}"]' for k in names)).body
        namespace['initial'] = {k: namespace[k] for k in names}
        function = ast.FunctionDef(name='execute', args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                                   body=assignments + [body], decorator_list=[])
        compiled = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(compiled, '<actual trainer filter block>', 'exec'), namespace)
            namespace['execute']()
        self.assertEqual(h.scheduler.generated, 24)
        self.assertEqual(h.scheduler.updates, 1)
        self.assertEqual(h.scheduler.consumed_groups, 2)
        self.assertEqual(set(namespace['accepted_uids_by_update'][0]), {'0-0', '1-0'})
        self.assertTrue(read(self.root / 'training_complete.json')['complete'])
        self.assertEqual(read(self.root / 'training_complete.json')['residual_groups_discarded'], 1)

    def test_complete_4000_event_audit_and_consumed_uid_corruption_rejected(self):
        from report_difficulty_scheduler import audit_arm
        from prepare_difficulty_scheduler import UP, BASE
        value = read(self.spec)
        arm_root = self.root / 'adaptive'
        arm_root.mkdir()
        value.update(budget=4000, output=str(arm_root))
        write(self.spec, value)
        t = trainer(range(20), arm_root)
        h = DriverSchedule(t)
        t.difficulty_schedule = h
        source = subprocess.check_output(['git', '-C', str(UP), 'show', BASE +
            ':src/shopping_grpo/training/grpo/dynamic_sampling.py']).decode('utf-8')
        helpers = {}
        exec(compile(source, '<pinned filter>', 'exec'), helpers)
        for generation in range(500):
            selected = h.scheduler.reserve()
            h.persist()
            append(h.events, {'event': 'selection', **selected, 'generated': h.scheduler.generated})
            values = sampled_infos(selected)
            uids = [f'{generation}-{i // 4}' for i in range(8)]
            batch = Batch(values, uids)
            utility, success, invalid, reasons = helpers['extract_shopping_group_signals'](values)
            _, groups = helpers['select_reward_varying_groups'](uids, [r['reward']['total'] for r in values],
                terminal_utilities=utility, purchase_success=success, sampling_invalid=invalid,
                sampling_invalid_reasons=reasons)
            h.observe(batch, groups)
            h.ready(batch)
            t.global_steps += 1
            h.committed(t.global_steps)
        with contextlib.redirect_stdout(io.StringIO()):
            h.boundary_stop(t)
        result = audit_arm(self.root, 'adaptive')
        self.assertTrue(result['event_replay_verified'])
        self.assertEqual(result['counts']['rollouts'], 4000)
        self.assertEqual(result['counts']['consumed_groups'], 1000)
        events = [json.loads(s) for s in h.events.read_text(encoding='utf-8').splitlines()]
        updates = [e for e in events if e['event'] == 'update']
        updates[-1]['consumed_uids'] = updates[0]['consumed_uids']
        h.events.write_text(''.join(canonical(e) + '\n' for e in events), encoding='utf-8')
        with self.assertRaises(ValueError):
            audit_arm(self.root, 'adaptive')


if __name__ == '__main__':
    unittest.main()
