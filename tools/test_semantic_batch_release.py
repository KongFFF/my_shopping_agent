"""Batch acceptance gates exercised through the real CPU training adapter."""
import copy
import importlib.util
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import test_semantic_training_integration as production_test
from semantic_training_bundle import ROOT, UPSTREAM, rows, sha256, TrainingEvidenceBackend
from semantic_batch_release import verify_release,require_pilot_config

OUT=ROOT/'analysis/semantic_batch_acceptance_20261108'


def pilot_config(steps=20):
    return {'trainer':{'total_training_steps':steps,'resume_mode':'disable'},
        'data':{'train_batch_size':2},'actor_rollout_ref':{'rollout':{'n':4}},
        'shopping_dynamic_sampling':{'enable':True,'max_num_gen_batches':3,'max_consecutive_skipped_updates':3}}


def overrides(steps=20):
    return [f'trainer.total_training_steps={steps}','trainer.resume_mode=disable','data.train_batch_size=2',
            'actor_rollout_ref.rollout.n=4','shopping_dynamic_sampling.enable=true',
            'shopping_dynamic_sampling.max_num_gen_batches=3','shopping_dynamic_sampling.max_consecutive_skipped_updates=3']


class BatchReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend=TrainingEvidenceBackend(OUT/'training_bundle/manifest.json')
        cls.probes={r['task_id']:r for r in rows(ROOT/'analysis/semantic_product_verifier_20261107/train20_target_probes.jsonl')}

    receipt=production_test.IntegrationTests.receipt
    rollout=production_test.IntegrationTests.rollout

    def test_full_pool_retained_but_only_source_clear_tasks_can_update(self):
        b=self.backend;b.require_ready();b.require_run_config(pilot_config())
        self.assertEqual(len(b.train_ids),1000)
        self.assertEqual(len(b.released_ids),981);self.assertEqual(len(b.held_ids),19)
        self.assertEqual(len(b.approved_ids),7)
        self.assertFalse(b.manifest['formal_training_ready'])
        self.assertTrue(b.manifest['pilot_training_ready'])
        self.assertEqual(b.released_ids|b.held_ids,b.train_ids)

    def test_previously_unsigned_task_reaches_production_reward_output(self):
        out=self.rollout(53,result=self.receipt(181))
        self.assertTrue(out.extra_fields['shopping']['reward_valid'])
        self.assertNotEqual(out.extra_fields['shopping']['reward_type'],'source_contract_unsigned')
        self.assertEqual(out.extra_fields['shopping']['reward']['terminal_utility'],out.reward_score)

    def test_every_held_task_invalid_for_purchase_and_fixed_execution_failure(self):
        from shopping_grpo.training.grpo.adapter.evidence import capture_terminal
        for task in self.backend.held_ids:
            for receipt in (self.receipt(181),production_test.terminal(None,'repeat_loop',-.65)):
                s=self.backend.score(task,{'_evidence_terminal':capture_terminal(receipt),'steps':[{'tool':'buy_now'}]})
                self.assertFalse(s['reward_valid']);self.assertIsNone(s['reward'])
                self.assertEqual(s['type'],'source_contract_held')

    def test_held_group_has_no_learning_signal_even_when_old_rewards_vary(self):
        from shopping_grpo.training.grpo.dynamic_sampling import extract_shopping_group_signals,select_reward_varying_groups
        good=self.rollout(181).extra_fields['shopping']
        failed=self.rollout(181,production_test.terminal(None,'repeat_loop',-.65)).extra_fields['shopping']
        held=[self.rollout(12422,result=self.receipt(181,kind=k,reward=v)).extra_fields['shopping']
              for k,v in [('gold_purchase',1),('wrong_purchase',-.85),('partial_alternative_purchase',.2),('gold_purchase',1)]]
        infos=[good,failed]+held;signals=extract_shopping_group_signals(infos)
        selected,stats=select_reward_varying_groups(['valid']*2+['held']*4,signals[0],
            terminal_utilities=signals[0],purchase_success=signals[1],sampling_invalid=signals[2],sampling_invalid_reasons=signals[3])
        self.assertEqual(selected,[0,1]);self.assertEqual(stats['groups'][1]['drop_reason'],'sampling_invalid')
        self.assertTrue(all(r['reward_type']=='source_contract_held' for r in held))

    def test_unknown_product_evidence_keeps_fixed_denominator(self):
        out=self.rollout(313);s=out.extra_fields['shopping']['evidence_reward']['score']
        self.assertTrue(s['reward_valid']);self.assertTrue(s['unknown_ids'])
        self.assertFalse(s['evidence_success']);self.assertIsNone(s['semantic_success'])
        self.assertLessEqual(s['reward'],.25)

    def test_validation_and_native_metrics_remain_separate(self):
        task=next(iter(self.backend.validation_ids))
        out=self.rollout(task,result=self.receipt(181))
        self.assertEqual(out.reward_score,1)
        self.assertEqual(out.extra_fields['shopping']['evidence_reward']['branch'],'validation_native_v3')

    def test_only_explicit_bounded_budget_can_be_launched(self):
        b=self.backend
        for cfg in (None,pilot_config(0),pilot_config(21),pilot_config(500),pilot_config(True)):
            with self.subTest(cfg=cfg),self.assertRaises(ValueError):b.require_run_config(cfg)
        for steps in (1,20):b.require_run_config(pilot_config(steps))
        for path,value in [('resume_mode','auto'),('total_training_steps',None)]:
            cfg=pilot_config();cfg['trainer'][path]=value
            with self.assertRaises(ValueError):b.require_run_config(cfg)
        cfg=pilot_config();cfg['actor_rollout_ref']['rollout']['n']=8
        with self.assertRaises(ValueError):b.require_run_config(cfg)

    def test_worker_direct_hydra_loader_checks_resolved_budget(self):
        from shopping_grpo.training.grpo.adapter.evidence import load_backend
        with patch.dict(os.environ,{'SHOPPING_REWARD_TOOLS_DIR':str(ROOT/'tools')}):
            with self.assertRaisesRegex(ValueError,'resolved trainer config'):
                load_backend(OUT/'training_bundle/manifest.json')
            with self.assertRaisesRegex(ValueError,'at most 20'):
                load_backend(OUT/'training_bundle/manifest.json',trainer_config=pilot_config(500))
            b=load_backend(OUT/'training_bundle/manifest.json',trainer_config=pilot_config())
            self.assertEqual(len(b.released_ids),981)

    def test_constructor_uses_unwrapped_verl08_trainer_config(self):
        # Matches pinned veRL0.8 AgentLoopBase: trainer_config is a wrapper,
        # and self.config is the underlying full DictConfig after super init.
        def base_init(instance,*args,trainer_config=None,**kwargs):
            wrapped=trainer_config if trainer_config is not None else args[0]
            instance.config=wrapped.config
        def backend_load(_manifest,trainer_config=None):
            self.backend.require_run_config(trainer_config)
            return self.backend
        from shopping_grpo.training.grpo.adapter.agent_loop import ShoppingToolAgentLoop
        with patch.object(production_test.ToolAgentLoop,'__init__',base_init), \
             patch('shopping_grpo.training.grpo.adapter.evidence.load_backend',side_effect=backend_load):
            wrapped=types.SimpleNamespace(config=pilot_config())
            loop=ShoppingToolAgentLoop(trainer_config=wrapped,reward_mode='constraint_aware',
                evidence_reward_manifest='source-bound-test-manifest')
            self.assertIs(loop.config,wrapped.config)
            with self.assertRaisesRegex(ValueError,'at most 20'):
                ShoppingToolAgentLoop(trainer_config=types.SimpleNamespace(config=pilot_config(500)),
                    reward_mode='constraint_aware',evidence_reward_manifest='source-bound-test-manifest')

    def test_source_release_certificate_cannot_skip_an_acceptance_gate(self):
        d=json.loads((OUT/'pilot_release.json').read_text(encoding='utf-8'))
        for change in ('missing_gate','failed_gate','full_scope','larger_budget'):
            with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temp:
                candidate=copy.deepcopy(d)
                if change=='missing_gate':candidate['checks'].pop(next(iter(candidate['checks'])))
                if change=='failed_gate':candidate['checks'][next(iter(candidate['checks']))]=False
                if change=='full_scope':candidate['full_training_released']=True
                if change=='larger_budget':candidate['max_optimizer_steps']=500
                p=Path(temp)/'release.json';p.write_text(json.dumps(candidate),encoding='utf-8')
                with self.assertRaises(ValueError):verify_release(p,root=ROOT,contracts=OUT/'contracts1000.jsonl',sha256=sha256,rows=rows)

    def test_launcher_composes_pilot_without_loading_model_or_starting_process(self):
        spec=importlib.util.spec_from_file_location('batch_launcher',UPSTREAM/'scripts/train_grpo.py')
        launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temp:
            model=Path(temp)/'scripted-model';model.mkdir();(model/'config.json').write_text('{}')
            (model/'model.safetensors').write_bytes(b'test only, no actual model')
            args=types.SimpleNamespace(model=model,train_data=self.backend.paths['train_parquet'],
                val_data=self.backend.paths['validation_parquet'],config=UPSTREAM/'configs/grpo.yaml',
                output=Path(temp)/'new-output',logger='console',env_url='http://127.0.0.1:5700',
                experiment_name='pilot-test',hydra_overrides=overrides(),reward_tools_dir=ROOT/'tools',
                evidence_reward_manifest=OUT/'training_bundle/manifest.json')
            cmd,env=launcher.build_command(args)
            self.assertIn('trainer.total_training_steps=20',cmd);self.assertFalse(args.output.exists())
            self.assertEqual(env['GRPO_TRAIN_FILE'],str(self.backend.paths['train_parquet']))
            args.hydra_overrides=overrides(500)
            with self.assertRaisesRegex(SystemExit,'at most 20'):launcher.build_command(args)

    def test_allocation_does_not_reduce_to_either_color(self):
        c=self.backend.contracts[16481]['normalized']
        color=next(r for r in c['specifications'] if r['id']=='r006')
        from semantic_product_verifier_v4 import compare_value
        self.assertEqual(compare_value(color,{'type':'text','value':'灰色'}),'fail')
        relation=next(r for r in c['specifications'] if r['id']=='batch_color_allocation')
        self.assertEqual(relation['expected']['value'],'灰色和黑色各一套')
        self.assertNotEqual(compare_value(relation,{'type':'text','value':'灰黑拼色'}),'pass')

    def test_quantities_of_different_attributes_do_not_cross_support(self):
        from semantic_product_verifier_v4 import snapshot,compare_requirement,fact
        c=self.backend.contracts[22999]['normalized'];req=next(r for r in c['specifications'] if r['id']=='r004')
        p={'asin':'public-test','customization_options':{},'category':'水彩笔'};state=snapshot(p,{})
        wrong=fact('specifications','attribute_count','hole_count',{'type':'number','value':36,'unit':'count'},[])
        right={**wrong,'attribute':'color_count'}
        self.assertEqual(compare_requirement(req,'specifications',p,state,[wrong])['status'],'unknown')
        self.assertEqual(compare_requirement(req,'specifications',p,state,[right])['status'],'pass')


if __name__=='__main__':unittest.main()
