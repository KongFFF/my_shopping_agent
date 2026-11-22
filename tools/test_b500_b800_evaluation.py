import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import b800_evaluation as tool


class Evaluation800(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=tool.ROOT/'outputs',prefix='b800-eval-test-');self.root=Path(self.temp.name).resolve()
    def tearDown(self):
        self.root.relative_to((tool.ROOT/'outputs').resolve());self.temp.cleanup()
    def fixture(self):
        asset=self.root/'asset';runtime=self.root/'runtime';out=self.root/'out'
        for base in (asset,runtime):
            p=base/'scripts/evaluate_shop_benchmark.py';p.parent.mkdir(parents=True);p.write_text('# identical source\n')
        ref={'protocols':{},'benchmarks':{},'commands':{},'source_sha256':{str(asset/'scripts/evaluate_shop_benchmark.py'):tool.digest(asset/'scripts/evaluate_shop_benchmark.py')}}
        for suite,n in [('smoke3',3),('dev50',50),('final200_clean',200)]:
            ref['protocols'][suite]={'temperature':0};ref['benchmarks'][suite]={'path':'frozen-'+suite,'task_ids':list(range(n))}
            ref['commands'][suite]=['python','old-script','--benchmark','frozen-'+suite,'--model','A800','--output','old.jsonl','--summary','old.json','--temperature','0','--max-steps','35']
        spec=tool.derive(ref,asset,runtime,out,self.root/'global_step_800');post=out/tool.POST;post.mkdir(parents=True)
        tool.write(post/'evaluation_spec.json',spec);tool.write(out/'posttrain.json',{'checkpoint_model_sha256':'source800','training':str(self.root/'training')})
        tool.write(post/'A800_dev50_summary.json',{'strict_successes':29,'strict_success_task_ids':list(range(29))})
        tool.write(post/'A800_final200_clean_summary.json',{'strict_successes':143,'strict_success_task_ids':list(range(143))})
        tool.write(post/'training_completion_check.json',{})
        return ref,spec,out,post
    def modules(self):
        mods={n:types.ModuleType(n)for n in ['shopping_grpo','shopping_grpo.evaluation','shopping_grpo.evaluation.summary']}
        mods['shopping_grpo'].__path__=[];mods['shopping_grpo.evaluation'].__path__=[];mods['shopping_grpo.evaluation.summary'].summarize_trajectories=lambda *a:{}
        if sys.platform!='linux':mods['fcntl']=types.SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *a:None)
        return mods
    def test_preserves_protocol_and_fixes800_identity(self):
        ref,spec,out,post=self.fixture();before=copy.deepcopy(ref)
        self.assertEqual(set(spec['commands']),set(tool.SUITES));self.assertEqual(spec['protocols'],ref['protocols']);self.assertEqual(ref,before)
        self.assertTrue(spec['checkpoint'].endswith('global_step_800'))
        for suite in tool.SUITES:
            for opt in ['--benchmark','--temperature','--max-steps']:
                a=spec['commands'][suite];b=ref['commands'][suite];self.assertEqual(a[a.index(opt)+1],b[b.index(opt)+1])
    def test_source_drift_rejected(self):
        ref,spec,out,post=self.fixture();(Path(spec['project'])/'scripts/evaluate_shop_benchmark.py').write_text('changed')
        with self.assertRaises(ValueError):tool.derive(ref,Path(spec['asset_project']),Path(spec['project']),self.root/'another',self.root/'global_step_800')
    def context(self,post,spec):
        def exported(*a):tool.write(post/'export_receipt.json',{'checkpoint':spec['checkpoint'],'actor_model_sha256':'source800'})
        return exported
    def test_bad_Dev_score_does_not_block_authorized_Final(self):
        _,spec,out,post=self.fixture();result={'strict_successes':0,'strict_success_task_ids':[],'expected_tasks':50,'mean_final_reward':0.}
        with patch.dict(sys.modules,self.modules()),patch.object(tool,'check_code'),patch.object(tool,'verify_completed_training',return_value={}),patch.object(tool,'verify_runtime_snapshot'),patch.object(tool.socket,'socket'),patch.object(tool.legacy,'export_model',side_effect=self.context(post,spec))as export,patch.object(tool.legacy,'serve'),patch.object(tool.subprocess,'run')as run,patch.object(tool.base,'cleanup'),patch.object(tool.base,'validate_suite',side_effect=lambda spec,suite,_:dict(result,expected_tasks={'smoke3':3,'dev50':50,'final200_clean':200}[suite]))as validate:
            tool.pipeline(out);self.assertEqual(run.call_count,3);self.assertEqual([c.args[1]for c in validate.call_args_list],list(tool.SUITES));self.assertEqual(export.call_args.args[1],800)
            with self.assertRaises(ValueError):tool.pipeline(out)
        self.assertTrue(tool.read(post/'evaluation_complete.json')['passed'])
    def test_infrastructure_failure_stops_before_Final(self):
        _,spec,out,post=self.fixture()
        with patch.dict(sys.modules,self.modules()),patch.object(tool,'check_code'),patch.object(tool,'verify_completed_training',return_value={}),patch.object(tool,'verify_runtime_snapshot'),patch.object(tool.socket,'socket'),patch.object(tool.legacy,'export_model',side_effect=self.context(post,spec)),patch.object(tool.legacy,'serve'),patch.object(tool.subprocess,'run'),patch.object(tool.base,'cleanup'),patch.object(tool.base,'validate_suite',side_effect=ValueError('execution failed'))as validate:
            with self.assertRaises(ValueError):tool.pipeline(out)
            self.assertEqual(validate.call_count,1)
        self.assertEqual(tool.read(post/'pipeline_exit_status.json')['exit_code'],1);self.assertFalse((post/'evaluation_complete.json').exists())


    def training_fixture(self):
        import json
        import yaml
        from b500_run import PACKAGE
        training=self.root/'training';(training/'configs').mkdir(parents=True)
        cfg={'trainer':{'total_training_steps':800},'shopping_dynamic_sampling':{'max_consecutive_skipped_updates':50}}
        (training/'configs/phase800.yaml').write_text(yaml.safe_dump(cfg))
        supplement=PACKAGE/'finish750_protocol.json'
        (training/'effective_protocol_supplement.json').write_bytes(supplement.read_bytes())
        budget={'train':30944,'validation_physical':3600}
        tool.write(training/'run.json',{'version':'B800-finish750-skip50-v1','resume_step':750,'target_step':800,
            'protocol_sha256':tool.digest(PACKAGE/'protocol.json'),'implementation_manifest_sha256':tool.digest(PACKAGE/'implementation_manifest.json'),
            'effective_protocol_supplement_sha256':tool.digest(supplement),'config_hashes':{'800':tool.digest(training/'configs/phase800.yaml')}})
        tool.write(training/'training_complete.json',{'completed_updates':800,'new_updates':50,'protocol_sha256':tool.digest(PACKAGE/'protocol.json'),'generation_budget':budget})
        tool.write(training/'finish750_preparation.json',{'passed':True,'resume_step':750,'skip_limit_after':50,'rolled_back_updates':list(range(751,788)),
            'carried_generation_budget':{'train':30144,'validation_physical':3400}})
        tool.write(training/'accepted_step_800.json',{'step':800,'checkpoint':str(training/'train/global_step_800'),'diagnostics_written':True,'lr':1e-6,'grad_norm':1.})
        rows=[{'event':'optimizer_step','global_step':n}for n in range(1,801)]
        for row in rows[750:]:row['metrics']={'training/optimizer_updated':1,'actor/lr':1e-6,'actor/grad_norm':1.,'actor/loss':0.,'b500/actor_tokens':8}
        (training/'training_diagnostics.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
        (training/'interrupted__training_diagnostics.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows[:787]))
        weights=[{'event':event,'driver_global_step_before_update':n}for n in range(650,800)for event in ['actor_update_started','actor_update_returned']]
        (training/'weights.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in weights))
        token_rows=[{'validation':False,'reserved_trajectories':30144},{'validation':False,'reserved_trajectories':800},
            {'validation':True,'reserved_trajectories':3400},{'validation':True,'reserved_trajectories':200}]
        (training/'generation_tokens.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in token_rows));tool.write(training/'generation_budget.json',budget)
        (training/'train_to_800.log').write_text('Setting global step to 750')
        return training
    def test_completed800_recovery_and_cost_evidence_accepted(self):
        training=self.training_fixture();r=tool.verify_completed_training(training)
        self.assertTrue(r['passed']);self.assertEqual(r['new_updates'],50);self.assertEqual(r['skip_limit'],50);self.assertEqual(r['physical_generation_budget']['validation_physical'],3600)
    def test_incomplete_training_blocks_evaluation(self):
        training=self.training_fixture();tool.write(training/'stopped.json',{'error':'failed'})
        with self.assertRaisesRegex(ValueError,'Expected completed B800'):tool.verify_completed_training(training)
    def test_missing_effective_update_blocks_evaluation(self):
        training=self.training_fixture();p=training/'training_diagnostics.jsonl';p.write_text('\n'.join(p.read_text().splitlines()[:-1])+'\n')
        with self.assertRaisesRegex(ValueError,'exactly1..800'):tool.verify_completed_training(training)
    def test_nonfinite_tail_update_blocks_evaluation(self):
        import json
        training=self.training_fixture();p=training/'training_diagnostics.jsonl';rs=[json.loads(l)for l in p.read_text().splitlines()];rs[-2]['metrics']['actor/grad_norm']=float('nan');p.write_text(''.join(json.dumps(r)+'\n'for r in rs))
        with self.assertRaisesRegex(ValueError,'Invalid resumed Actor'):tool.verify_completed_training(training)
    def test_generation_ledger_mismatch_blocks_evaluation(self):
        training=self.training_fixture();tool.write(training/'generation_budget.json',{'train':1,'validation_physical':3600})
        with self.assertRaisesRegex(ValueError,'ledger mismatch'):tool.verify_completed_training(training)
    def test_recovered_prefix_change_blocks_evaluation(self):
        import json
        training=self.training_fixture();p=training/'interrupted__training_diagnostics.jsonl';rs=[json.loads(l)for l in p.read_text().splitlines()];rs[0]['changed']=True;p.write_text(''.join(json.dumps(r)+'\n'for r in rs))
        with self.assertRaisesRegex(ValueError,'prefix changed'):tool.verify_completed_training(training)
    def test_training_change_after_preparation_blocks_before_export(self):
        _,spec,out,post=self.fixture()
        with patch.dict(sys.modules,self.modules()),patch.object(tool,'check_code'),patch.object(tool,'verify_completed_training',return_value={'changed':True}),patch.object(tool.legacy,'export_model')as export:
            with self.assertRaisesRegex(ValueError,'changed since'):tool.pipeline(out)
            export.assert_not_called()


if __name__=='__main__':unittest.main()
