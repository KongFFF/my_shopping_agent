"""CPU contract tests of the production adapter; generation/HTTP are scripted.

The lightweight veRL stub replaces only model generation and output types in
this test process. This is NOT a veRL/vLLM or GPU smoke training run.
"""
import asyncio
import copy
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from semantic_training_bundle import ROOT, UPSTREAM, DEFAULT_OUTPUT, TrainingEvidenceBackend, build_bundle, rows, sha256

sys.path.insert(0, str(UPSTREAM / 'src'))
if importlib.util.find_spec('verl') is None:
    for name in ('verl', 'verl.experimental', 'verl.experimental.agent_loop',
                 'verl.experimental.agent_loop.tool_agent_loop'):
        sys.modules[name] = types.ModuleType(name)
    class ToolAgentLoop:
        async def run(self, *args, **kwargs):
            raise AssertionError('Test must supply scripted generation')
    module = sys.modules['verl.experimental.agent_loop.tool_agent_loop']
    module.ToolAgentLoop = ToolAgentLoop
    module.AgentState = types.SimpleNamespace(TERMINATED='TERMINATED')
else:
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

from shopping_grpo.training.grpo.adapter.agent_loop import ShoppingToolAgentLoop
from shopping_grpo.training.grpo.adapter.evidence import capture_terminal
from shopping_grpo.training.grpo.adapter.runtime import current_runtime_state, make_runtime_state
from shopping_grpo.training.grpo.adapter.tools import ShopSimulatorTool
from shopping_grpo.training.grpo.dynamic_sampling import extract_shopping_group_signals, select_reward_varying_groups


def terminal(purchase=None, kind='gold_purchase', reward=1.0):
    valid = kind != 'reward_unverifiable'
    return {'done': True, 'over': True, 'termination_reason': kind, 'reward': reward,
            'purchase': purchase, 'goal': {'secret': 'never retained'},
            'reward_detail': {'reward_version': 'shopsimulator-reward-v3', 'reward_type': kind,
                'termination_reason': kind, 'reward_valid': valid, 'terminal_utility': reward,
                'purchase_success': kind == 'gold_purchase', 'sampling_invalid': not valid,
                'hard_gates': {}, 'weighted_score': 1.0, 'evidence_coverage': 1.0}}


def tool():
    schema = {'type': 'function', 'function': {'name': 'buy_now', 'description': 'purchase',
             'parameters': {'type': 'object', 'properties': {}}}}
    try:
        from verl.tools.schemas import OpenAIFunctionToolSchema
    except ImportError:
        pass
    else:
        schema = OpenAIFunctionToolSchema.model_validate(schema)
    return ShopSimulatorTool({}, schema)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build_bundle()
        cls.backend = TrainingEvidenceBackend(DEFAULT_OUTPUT / 'manifest.json')
        cls.probes = {r['task_id']: r for r in rows(ROOT / 'analysis/semantic_product_verifier_20261107/train20_target_probes.jsonl')}
        cls.static = {r['task_id']: r for r in rows(ROOT / 'analysis/semantic_train_smoke7_20261107/static_integration7.jsonl')}

    def receipt(self, task=181, **kwargs):
        probe = self.probes[task]
        return terminal({'asin': str(probe['asin']), 'options': probe['selected_options'],
                         'instruction_text': 'hidden purchase instruction', 'price': -999}, **kwargs)

    def rollout(self, task=181, result=None, enabled=True, infrastructure=False):
        result = result if result is not None else self.receipt(task)
        events = []
        class ScriptedEnv:
            def __init__(self, **kwargs):
                pass
            def reset(self, task_id):
                events.append(('reset', task_id))
                return {'environment_version': 'shopsimulator-environment-v2.1',
                        'instruction': '商品页\n可点击的按钮: ["Buy Now"]'}
            def step(self, action):
                events.append(('step', action))
                return copy.deepcopy(result)
            def release(self):
                events.append(('release', None))
        async def generated(_loop, params, **kwargs):
            # Executes production action guard, tool, terminal capture, and session.
            response, _, _ = await tool().execute('test', {})
            self.assertEqual(response.text, 'Environment terminated.')
            if infrastructure:
                current_runtime_state.get()['infrastructure_invalid'] = True
            return types.SimpleNamespace(reward_score=None, extra_fields={}, prompt_ids=[10], response_ids=[20])
        async def run():
            loop = object.__new__(ShoppingToolAgentLoop)
            loop.base_url = 'http://scripted.test'
            loop.timeout = 1; loop.max_steps = 35
            loop.required_environment_version = 'shopsimulator-environment-v2.1'
            loop.reward_mode = 'constraint_aware'
            loop.env_factory = ScriptedEnv
            loop.evidence_backend = self.backend if enabled else None
            with patch.object(ToolAgentLoop, 'run', generated):
                return await loop.run({}, extra_info={'task_id': task})
        out = asyncio.run(run())
        self.assertEqual(events[-1], ('release', None))
        self.assertIsNone(current_runtime_state.get())
        self.assertEqual(out.prompt_ids, [10]); self.assertEqual(out.response_ids, [20])
        return out

    def test_signoff_keeps_original1000_and_does_not_approve_program_valid_rows(self):
        self.assertEqual(len(self.backend.contracts), 1000)
        self.assertEqual(self.backend.approved_ids, {181, 313, 315, 366, 390, 396, 460})
        with self.assertRaisesRegex(ValueError, '7/1000 approved, 993 held'):
            self.backend.require_ready()

    def test_real_adapter_output_receives_new_reward(self):
        out = self.rollout(396)
        self.assertEqual(out.reward_score, -.85)  # catalog 89 > approved budget 88
        info = out.extra_fields['shopping']
        self.assertTrue(info['reward_valid'])
        self.assertEqual(info['evidence_reward']['baseline']['reward']['total'], 1)
        self.assertEqual(info['reward']['terminal_utility'], out.reward_score)
        self.assertEqual(extract_shopping_group_signals([info])[2], [False])

    def test_all_seven_match_independent_static_scores(self):
        for task in sorted(self.backend.approved_ids):
            with self.subTest(task=task):
                out = self.rollout(task)
                self.assertAlmostEqual(out.reward_score, self.static[task]['training_score']['reward'])

    def test_old_unverifiable_flag_is_replaced_for_supported_new_score(self):
        out = self.rollout(181, result=self.receipt(181, kind='reward_unverifiable', reward=0))
        self.assertEqual(out.reward_score, 1)
        info = out.extra_fields['shopping']
        self.assertFalse(info['reward_unverifiable'])
        self.assertFalse(info['reward']['sampling_invalid'])
        self.assertFalse(info['evidence_reward']['baseline']['valid'])

    def test_unsigned_task_is_invalid_instead_of_native_fallback(self):
        out = self.rollout(53, result=self.receipt(181))
        info = out.extra_fields['shopping']
        self.assertEqual(out.reward_score, 0)
        self.assertEqual(info['reward_type'], 'source_contract_unsigned')
        self.assertEqual(extract_shopping_group_signals([info])[2], [True])

    def test_validation_stays_native_without_training_labels(self):
        task = next(iter(self.backend.validation_ids))
        out = self.rollout(task, result=self.receipt(181))
        self.assertEqual(out.reward_score, 1)
        self.assertEqual(out.extra_fields['shopping']['evidence_reward']['branch'], 'validation_native_v3')

    def test_native_default_behavior_unchanged(self):
        out = self.rollout(396, enabled=False)
        self.assertEqual(out.reward_score, 1)
        self.assertNotIn('evidence_reward', out.extra_fields['shopping'])

    def test_proven_loop_and_guard_failure_keep_original_penalties(self):
        for kind, reward in [('repeat_loop', -.65), ('max_steps', -.5), ('wrong_purchase', -.85)]:
            with self.subTest(kind=kind):
                receipt = terminal(None, kind, reward)
                if kind == 'wrong_purchase':
                    receipt['reward_detail']['evidence'] = {'purchase_guard': {'status': 'exhausted'}}
                out = self.rollout(181, receipt)
                self.assertEqual(out.reward_score, reward)
                self.assertTrue(out.extra_fields['shopping']['reward_valid'])

    def test_no_purchase_proof_cannot_be_fabricated_from_logged_reward(self):
        out = self.rollout(181, terminal(None))
        self.assertFalse(out.extra_fields['shopping']['reward_valid'])
        self.assertEqual(out.reward_score, 0)

    def test_infrastructure_error_never_becomes_learning_signal(self):
        out = self.rollout(181, infrastructure=True)
        self.assertFalse(out.extra_fields['shopping']['reward_valid'])
        self.assertEqual(out.reward_score, 0)

    def test_terminal_capture_excludes_hidden_answers_and_purchase_annotations(self):
        captured = capture_terminal(self.receipt(181))
        self.assertNotIn('goal', captured)
        self.assertEqual(set(captured['purchase']), {'asin', 'options'})
        self.assertNotIn('hidden', json.dumps(captured))
        self.assertTrue(all('instructions' not in p and 'reviews' not in p for p in self.backend.products.values()))

    def test_direct_hydra_backend_load_cannot_bypass_signoff_gate(self):
        from shopping_grpo.training.grpo.adapter.evidence import load_backend
        with patch.dict(os.environ, {'SHOPPING_REWARD_TOOLS_DIR': str(ROOT / 'tools')}):
            with self.assertRaisesRegex(ValueError, '7/1000 approved, 993 held'):
                load_backend(DEFAULT_OUTPUT / 'manifest.json')

    def test_text_hash_is_portable_but_does_not_ignore_content_changes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temp:
            p = Path(temp) / 'source.py'
            p.write_bytes(b'first\nsecond\n'); before = sha256(p)
            p.write_bytes(b'first\r\nsecond\r\n'); self.assertEqual(sha256(p), before)
            p.write_bytes(b'changed\r\nsecond\r\n'); self.assertNotEqual(sha256(p), before)

    def test_unknown_required_items_remain_in_denominator(self):
        out = self.rollout(313)
        score = out.extra_fields['shopping']['evidence_reward']['score']
        self.assertTrue(score['reward_valid'])
        self.assertTrue(score['unknown_ids'])
        self.assertFalse(score['evidence_success'])
        self.assertIsNone(score['semantic_success'])
        self.assertLessEqual(out.reward_score, .25)

    def test_group_selection_consumes_new_scores_and_invalid_flags(self):
        a = self.rollout(181).extra_fields['shopping']
        b = self.rollout(396).extra_fields['shopping']
        utilities, successes, invalid, reasons = extract_shopping_group_signals([a, b])
        selected, _ = select_reward_varying_groups(['same', 'same'], utilities,
            terminal_utilities=utilities, purchase_success=successes, sampling_invalid=invalid,
            sampling_invalid_reasons=reasons)
        self.assertEqual(selected, [0, 1])
        # A real same-task group: successful purchase versus proven failed execution.
        failed = self.rollout(181, terminal(None, 'repeat_loop', -.65)).extra_fields['shopping']
        signals = extract_shopping_group_signals([a, failed])
        selected, _ = select_reward_varying_groups(['same', 'same'], signals[0], terminal_utilities=signals[0],
            purchase_success=signals[1], sampling_invalid=signals[2], sampling_invalid_reasons=signals[3])
        self.assertEqual(selected, [0, 1])

    def test_missing_or_ambiguous_catalog_options_cannot_score(self):
        receipt = self.receipt(181)
        receipt['purchase']['options'] = {'fake-axis': 'fake'}
        out = self.rollout(181, receipt)
        self.assertEqual(out.extra_fields['shopping']['reward_type'], 'purchase_options_unproven')
        self.assertFalse(out.extra_fields['shopping']['reward_valid'])

    def test_outside_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside frozen'):
            self.backend.is_validation(-1)

    def test_frozen_asset_tamper_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temp:
            manifest = copy.deepcopy(self.backend.manifest)
            manifest['assets']['contracts']['sha256'] = '0' * 64
            path = Path(temp) / 'bad_manifest.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'asset mismatch: contracts'):
                TrainingEvidenceBackend(path)

    def test_actual_launcher_rejects_unsigned_labels_before_subprocess(self):
        spec = importlib.util.spec_from_file_location('training_launcher', UPSTREAM / 'scripts/train_grpo.py')
        launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temp:
            model = Path(temp) / 'fake-test-model'; model.mkdir()
            (model / 'config.json').write_text('{}'); (model / 'model.safetensors').write_bytes(b'test only')
            args = types.SimpleNamespace(model=model, train_data=self.backend.paths['train_parquet'],
                val_data=self.backend.paths['validation_parquet'], config=UPSTREAM / 'configs/grpo.yaml',
                output=Path(temp) / 'new-output', logger='console', env_url='http://127.0.0.1:5700',
                experiment_name='test', hydra_overrides=[], reward_tools_dir=ROOT / 'tools',
                evidence_reward_manifest=DEFAULT_OUTPUT / 'manifest.json')
            with self.assertRaisesRegex(SystemExit, '7/1000 approved, 993 held'):
                launcher.build_command(args)
            # Command composition only: the release gate is intentionally mocked,
            # never presented as real train1000 signoff or a launched process.
            with patch.object(TrainingEvidenceBackend, 'require_ready'):
                command, environment = launcher.build_command(args)
                self.assertEqual(environment['SHOPPING_EVIDENCE_MANIFEST'], str(DEFAULT_OUTPUT / 'manifest.json'))
                self.assertEqual(environment['SHOPPING_REWARD_TOOLS_DIR'], str(ROOT / 'tools'))
                self.assertIn(str(ROOT / 'tools'), environment['PYTHONPATH'])
            args.evidence_reward_manifest = None; args.reward_tools_dir = None
            command, environment = launcher.build_command(args)
            self.assertIn('verl.trainer.main_ppo', command)
            self.assertEqual(environment['SHOPPING_EVIDENCE_MANIFEST'], '')
            self.assertFalse(args.output.exists())


if __name__ == '__main__':
    unittest.main()
