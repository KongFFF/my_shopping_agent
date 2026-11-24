import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import c500_evaluation as tool


class EvaluationC500(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=tool.ROOT/'outputs',prefix='b500-eval-test-');self.root=Path(self.temp.name).resolve()
    def tearDown(self):
        self.root.relative_to((tool.ROOT/'outputs').resolve());self.temp.cleanup()
    def fixture(self):
        asset=self.root/'asset';runtime=self.root/'runtime';out=self.root/'out'
        for base in (asset,runtime):
            p=base/'scripts/evaluate_shop_benchmark.py';p.parent.mkdir(parents=True);p.write_text('# identical source\n')
        ref={'protocols':{},'benchmarks':{},'commands':{},'source_sha256':{str(asset/'scripts/evaluate_shop_benchmark.py'):tool.digest(asset/'scripts/evaluate_shop_benchmark.py')}}
        for suite,n in [('smoke3',3),('dev50',50),('final200_clean',200)]:
            ref['protocols'][suite]={'temperature':0};ref['benchmarks'][suite]={'path':'frozen-'+suite,'task_ids':list(range(n))}
            ref['commands'][suite]=['python','old-script','--benchmark','frozen-'+suite,'--model','A500','--output','old.jsonl','--summary','old.json','--temperature','0','--max-steps','35']
        spec=tool.derive(ref,asset,runtime,out,self.root/'global_step_500');post=out/tool.POST;post.mkdir(parents=True)
        tool.write(post/'evaluation_spec.json',spec);tool.write(out/'posttrain.json',{'checkpoint_model_sha256':'source500'})
        tool.write(post/'A500_dev50_summary.json',{'strict_successes':27,'strict_success_task_ids':list(range(27))})
        tool.write(post/'A500_final200_clean_summary.json',{'strict_successes':137,'strict_success_task_ids':list(range(137))})
        return ref,spec,out,post
    def modules(self):
        mods={n:types.ModuleType(n)for n in ['shopping_grpo','shopping_grpo.evaluation','shopping_grpo.evaluation.summary']}
        mods['shopping_grpo'].__path__=[];mods['shopping_grpo.evaluation'].__path__=[];mods['shopping_grpo.evaluation.summary'].summarize_trajectories=lambda *a:{}
        if sys.platform!='linux':mods['fcntl']=types.SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *a:None)
        return mods
    def test_preserves_protocol_and_fixes500_identity(self):
        ref,spec,out,post=self.fixture();before=copy.deepcopy(ref)
        self.assertEqual(set(spec['commands']),set(tool.SUITES));self.assertEqual(spec['protocols'],ref['protocols']);self.assertEqual(ref,before)
        self.assertTrue(spec['checkpoint'].endswith('global_step_500'))
        for suite in tool.SUITES:
            for opt in ['--benchmark','--temperature','--max-steps']:
                a=spec['commands'][suite];b=ref['commands'][suite];self.assertEqual(a[a.index(opt)+1],b[b.index(opt)+1])
    def test_source_drift_rejected(self):
        ref,spec,out,post=self.fixture();(Path(spec['project'])/'scripts/evaluate_shop_benchmark.py').write_text('changed')
        with self.assertRaises(ValueError):tool.derive(ref,Path(spec['asset_project']),Path(spec['project']),self.root/'another',self.root/'global_step_500')
    def context(self,post,spec):
        def exported(*a):tool.write(post/'export_receipt.json',{'checkpoint':spec['checkpoint'],'actor_model_sha256':'source500'})
        return exported
    def test_bad_Dev_score_does_not_block_authorized_Final(self):
        _,spec,out,post=self.fixture();result={'strict_successes':0,'strict_success_task_ids':[],'expected_tasks':50,'mean_final_reward':0.}
        with patch.dict(sys.modules,self.modules()),patch.object(tool,'check_code'),patch.object(tool,'verify_runtime_snapshot'),patch.object(tool.socket,'socket'),patch.object(tool.legacy,'export_model',side_effect=self.context(post,spec))as export,patch.object(tool.legacy,'serve'),patch.object(tool.subprocess,'run')as run,patch.object(tool.base,'cleanup'),patch.object(tool.base,'validate_suite',side_effect=lambda spec,suite,summary:{**result,'expected_tasks':{'smoke3':3,'dev50':50,'final200_clean':200}[suite]})as validate:
            tool.pipeline(out);self.assertEqual(run.call_count,3);self.assertEqual([c.args[1]for c in validate.call_args_list],list(tool.SUITES));self.assertEqual(export.call_args.args[1],500)
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


class Supplement(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=tool.ROOT/'outputs');self.root=Path(self.temp.name).resolve()
        self.training=self.root/'current';self.training.mkdir();self.pkg=self.root/'pkg';self.pkg.mkdir()
        self.repo=self.root/'code';(self.repo/'experiments/group_weight_c500_v1').mkdir(parents=True)
        for n in ['protocol.json','implementation_manifest.json','launch_manifest.json']:
            tool.write(self.pkg/n,{'files_sha256':{}})
            (self.repo/'experiments/group_weight_c500_v1'/n).write_bytes((self.pkg/n).read_bytes())
        self.cpu=self.root/'cpu.json'
        run={'experiment':'C500','gamma':4,'protocol_sha256':tool.digest(self.pkg/'protocol.json'),
             'implementation_manifest_sha256':tool.digest(self.pkg/'implementation_manifest.json'),'cpu_receipt':str(self.cpu),'repo':str(self.repo)}
        self.cpu_data={'version':'c500-cpu-acceptance-v1','passed':True,'tests_run':36,'gpu_initialized':False,
            'model_weights_loaded':False,'training_started':False,'protocol_sha256':run['protocol_sha256'],
            'implementation_manifest_sha256':run['implementation_manifest_sha256'],'errors':[]}
        tool.write(self.cpu,self.cpu_data);tool.write(self.training/'run.json',run)
        tool.write(self.training/'training_complete.json',{'completed_updates':500,'new_updates':500,'protocol_sha256':run['protocol_sha256']})
        self.patch=patch.object(tool,'PACKAGE',self.pkg);self.patch.start()
    def tearDown(self):self.patch.stop();self.temp.cleanup()
    def collect(self):return tool.collect_supplement(self.training,self.root/'supplement')
    def prior(self):
        old=self.root/'prior';old.mkdir();run=tool.read(self.training/'run.json');run['training_started']=False
        tool.write(old/'run.json',run);tool.write(old/'stopped.json',{'error':'Insufficient disk space'})
        tool.write(self.training/'zero_update_recovery.json',{'previous_run':str(old)})
        return old
    def test_existing_CPU_copied_without_rerun(self):
        with patch.object(tool.subprocess,'run')as run:self.collect();run.assert_not_called()
        self.assertEqual((self.root/'supplement/native_cpu_acceptance.json').read_bytes(),self.cpu.read_bytes())
    def test_failed_CPU_rejected(self):
        self.cpu_data['passed']=False;tool.write(self.cpu,self.cpu_data)
        with self.assertRaises(ValueError):self.collect()
    def test_wrong_gamma_rejected(self):
        run=tool.read(self.training/'run.json');run['gamma']=2;tool.write(self.training/'run.json',run)
        with self.assertRaises(ValueError):self.collect()
    def test_prior_disk_zero_update_scene_copied(self):
        self.prior();self.collect()
        self.assertTrue(tool.read(self.root/'supplement/supplement_acceptance.json')['prior_disk_zero_update_verified'])
    def test_prior_update_rejected(self):
        old=self.prior();(old/'weights.jsonl').write_text('{"event":"actor_update_started"}\n')
        with self.assertRaises(ValueError):self.collect()
    def test_prior_checkpoint_rejected(self):
        old=self.prior();(old/'train/global_step_1').mkdir(parents=True)
        with self.assertRaises(ValueError):self.collect()
    def test_missing_CPU_rejected(self):
        self.cpu.unlink()
        with self.assertRaises(FileNotFoundError):self.collect()
    def test_source_inventory_change_rejected(self):
        (self.repo/'experiments/group_weight_c500_v1/protocol.json').write_text('changed')
        with self.assertRaises(ValueError):self.collect()


if __name__=='__main__':unittest.main()
