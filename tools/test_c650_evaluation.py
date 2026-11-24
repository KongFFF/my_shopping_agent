import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import c650_evaluation as tool


class EvaluationC650(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=tool.ROOT/'outputs',prefix='b650-eval-test-');self.root=Path(self.temp.name).resolve()
    def tearDown(self):
        self.root.relative_to((tool.ROOT/'outputs').resolve());self.temp.cleanup()
    def fixture(self):
        asset=self.root/'asset';runtime=self.root/'runtime';out=self.root/'out'
        for base in (asset,runtime):
            p=base/'scripts/evaluate_shop_benchmark.py';p.parent.mkdir(parents=True);p.write_text('# identical source\n')
        ref={'protocols':{},'benchmarks':{},'commands':{},'source_sha256':{str(asset/'scripts/evaluate_shop_benchmark.py'):tool.digest(asset/'scripts/evaluate_shop_benchmark.py')}}
        for suite,n in [('smoke3',3),('dev50',50),('final200_clean',200)]:
            ref['protocols'][suite]={'temperature':0};ref['benchmarks'][suite]={'path':'frozen-'+suite,'task_ids':list(range(n))}
            ref['commands'][suite]=['python','old-script','--benchmark','frozen-'+suite,'--model','A650','--output','old.jsonl','--summary','old.json','--temperature','0','--max-steps','35']
        spec=tool.derive(ref,asset,runtime,out,self.root/'global_step_650');post=out/tool.POST;post.mkdir(parents=True)
        tool.write(post/'evaluation_spec.json',spec);tool.write(out/'posttrain.json',{'checkpoint_model_sha256':'source650'})
        tool.write(post/'A650_dev50_summary.json',{'strict_successes':26,'strict_success_task_ids':list(range(26))})
        tool.write(post/'A650_final200_clean_summary.json',{'strict_successes':141,'strict_success_task_ids':list(range(141))})
        return ref,spec,out,post
    def modules(self):
        mods={n:types.ModuleType(n)for n in ['shopping_grpo','shopping_grpo.evaluation','shopping_grpo.evaluation.summary']}
        mods['shopping_grpo'].__path__=[];mods['shopping_grpo.evaluation'].__path__=[];mods['shopping_grpo.evaluation.summary'].summarize_trajectories=lambda *a:{}
        if sys.platform!='linux':mods['fcntl']=types.SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *a:None)
        return mods
    def test_preserves_protocol_and_fixes650_identity(self):
        ref,spec,out,post=self.fixture();before=copy.deepcopy(ref)
        self.assertEqual(set(spec['commands']),set(tool.SUITES));self.assertEqual(spec['protocols'],ref['protocols']);self.assertEqual(ref,before)
        self.assertTrue(spec['checkpoint'].endswith('global_step_650'));self.assertEqual(spec['model_tag'],'shopping-agent-c650-gamma4-v1')
        for suite in tool.SUITES:
            for opt in ['--benchmark','--temperature','--max-steps']:
                a=spec['commands'][suite];b=ref['commands'][suite];self.assertEqual(a[a.index(opt)+1],b[b.index(opt)+1])
    def test_source_drift_rejected(self):
        ref,spec,out,post=self.fixture();(Path(spec['project'])/'scripts/evaluate_shop_benchmark.py').write_text('changed')
        with self.assertRaises(ValueError):tool.derive(ref,Path(spec['asset_project']),Path(spec['project']),self.root/'another',self.root/'global_step_650')
    def context(self,post,spec):
        def exported(*a):tool.write(post/'export_receipt.json',{'checkpoint':spec['checkpoint'],'actor_model_sha256':'source650'})
        return exported
    def test_bad_Dev_score_does_not_block_authorized_Final(self):
        _,spec,out,post=self.fixture();result={'strict_successes':0,'strict_success_task_ids':[],'expected_tasks':50,'mean_final_reward':0.}
        with patch.dict(sys.modules,self.modules()),patch.object(tool,'check_code'),patch.object(tool,'verify_runtime_snapshot'),patch.object(tool.socket,'socket'),patch.object(tool.legacy,'export_model',side_effect=self.context(post,spec))as export,patch.object(tool.legacy,'serve'),patch.object(tool.subprocess,'run')as run,patch.object(tool.base,'cleanup'),patch.object(tool.base,'validate_suite',side_effect=lambda spec,suite,summary:{**result,'expected_tasks':{'smoke3':3,'dev50':50,'final200_clean':200}[suite]})as validate:
            tool.pipeline(out);self.assertEqual(run.call_count,3);self.assertEqual([c.args[1]for c in validate.call_args_list],list(tool.SUITES));self.assertEqual(export.call_args.args[1],650)
            with self.assertRaises(ValueError):tool.pipeline(out)
        self.assertTrue(tool.read(post/'evaluation_complete.json')['passed'])
    def test_export_source_mismatch_blocks_serving(self):
        _,spec,out,post=self.fixture()
        def wrong(*a):tool.write(post/'export_receipt.json',{'checkpoint':'other','actor_model_sha256':'wrong'})
        with patch.dict(sys.modules,self.modules()),patch.object(tool,'check_code'),patch.object(tool,'verify_runtime_snapshot'),patch.object(tool.socket,'socket'),patch.object(tool.legacy,'export_model',side_effect=wrong),patch.object(tool.legacy,'serve')as serve,patch.object(tool.base,'cleanup'):
            with self.assertRaisesRegex(ValueError,'Exported source changed'):tool.pipeline(out)
            serve.assert_not_called()
        self.assertEqual(tool.read(post/'pipeline_exit_status.json')['exit_code'],1)

    def test_infrastructure_failure_stops_before_Final(self):
        _,spec,out,post=self.fixture()
        with patch.dict(sys.modules,self.modules()),patch.object(tool,'check_code'),patch.object(tool,'verify_runtime_snapshot'),patch.object(tool.socket,'socket'),patch.object(tool.legacy,'export_model',side_effect=self.context(post,spec)),patch.object(tool.legacy,'serve'),patch.object(tool.subprocess,'run'),patch.object(tool.base,'cleanup'),patch.object(tool.base,'validate_suite',side_effect=ValueError('execution failed'))as validate:
            with self.assertRaises(ValueError):tool.pipeline(out)
            self.assertEqual(validate.call_count,1)
        self.assertEqual(tool.read(post/'pipeline_exit_status.json')['exit_code'],1);self.assertFalse((post/'evaluation_complete.json').exists())


class ParentPacket(unittest.TestCase):
    def setUp(self):
        self.original=tool.ROOT;self.temp=tempfile.TemporaryDirectory(dir=tool.ROOT/'outputs');self.root=Path(self.temp.name)
        self.patch=patch.object(tool,'ROOT',self.root);self.patch.start()
        self.events=[{'event':'generation_batch','global_step':1},{'event':'optimizer_step','global_step':1}]
        self.weights=b'parent weights';self.budget={'train':8,'validation_physical':0}
        import hashlib
        pkg=self.root/'experiments/group_weight_c650_v1';pkg.mkdir(parents=True)
        tool.write(pkg/'evaluation_training_reference.json',{'parent_diagnostic_events':2,
            'parent_canonical_diagnostic_sha256':hashlib.sha256(json.dumps(self.events,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),
            'parent_weights_sha256':hashlib.sha256(self.weights).hexdigest(),'parent_budget':self.budget})
    def tearDown(self):self.patch.stop();self.temp.cleanup()
    def packet(self):
        import zipfile
        path=self.root/'evidence.zip'
        with zipfile.ZipFile(path,'w')as z:
            z.writestr('training_diagnostics.jsonl',''.join(json.dumps(x)+'\n'for x in self.events))
            z.writestr('parent_C500__weights.jsonl',self.weights)
            z.writestr('parent_C500__generation_budget.json',json.dumps(self.budget))
        return path
    def test_valid_parent_view_passes_without_historical_download(self):
        with patch('audit_c650_training_return.audit',return_value={'audit_passed':True,'errors':[]})as audit:
            result=tool.audit_training_packet(self.packet(),self.root/'check');self.assertTrue(result['audit_passed'])
            self.assertTrue(audit.call_args.kwargs['previous_archive'].exists())
    def test_parent_prefix_change_rejected(self):
        self.events[0]['global_step']=2
        with self.assertRaisesRegex(ValueError,'diagnostic prefix'):tool.audit_training_packet(self.packet(),self.root/'check')
    def test_parent_weights_change_rejected(self):
        self.weights=b'other'
        with self.assertRaisesRegex(ValueError,'weight log'):tool.audit_training_packet(self.packet(),self.root/'check')
    def test_parent_budget_change_rejected(self):
        self.budget={'train':16,'validation_physical':0}
        with self.assertRaisesRegex(ValueError,'budget'):tool.audit_training_packet(self.packet(),self.root/'check')


if __name__=='__main__':unittest.main()
