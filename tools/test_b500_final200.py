import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import b500_final200 as tool


class Final200(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=tool.ROOT/'outputs',prefix='b500-final-test-');self.root=Path(self.temp.name).resolve()
    def tearDown(self):
        self.root.relative_to((tool.ROOT/'outputs').resolve());self.temp.cleanup()
    def fixture(self):
        out=self.root/'out';merged=self.root/'existing_merged';merged.mkdir();(merged/'model.safetensors').write_bytes(b'previously exported')
        old={'project':str(self.root/'runtime'),'asset_project':str(self.root/'asset'),'model_tag':'B500',
            'checkpoint':str(self.root/'global_step_500'),'merged_model':str(merged),'source_sha256':{},
            'benchmarks':{'smoke3':{'path':'original-smoke','task_ids':[1,2,3]}},'commands':{'dev50':['must not run']}}
        ref={'protocols':{tool.SUITE:{'temperature':0}},'benchmarks':{tool.SUITE:{'path':'original-final','task_ids':list(range(200))}},
            'commands':{tool.SUITE:['python','original-script','--benchmark','original-final','--model','A500','--output','old.jsonl','--summary','old.json','--max-steps','35','--temperature','0']}}
        spec=tool.derive(ref,old,out);post=out/tool.POST;post.mkdir(parents=True);tool.write(post/'evaluation_spec.json',spec)
        receipt={'checkpoint':old['checkpoint'],'merged_model':str(merged),'merged_weights_sha256':{'model.safetensors':tool.digest(merged/'model.safetensors')}}
        tool.write(post/'export_receipt.json',receipt);tool.write(post/'A500_final_summary.json',{'strict_success_task_ids':list(range(137))})
        return old,ref,out,post,spec,receipt
    def modules(self):
        mods={n:types.ModuleType(n)for n in ['shopping_grpo','shopping_grpo.evaluation','shopping_grpo.evaluation.summary']}
        mods['shopping_grpo'].__path__=[];mods['shopping_grpo.evaluation'].__path__=[]
        mods['shopping_grpo.evaluation.summary'].summarize_trajectories=lambda *args:{}
        if sys.platform!='linux':mods['fcntl']=types.SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *a:None)
        return mods
    def test_fixed_model_and_only_Final_command(self):
        old,ref,out,post,spec,_=self.fixture();before=copy.deepcopy(old)
        self.assertEqual(set(spec['commands']),{tool.SUITE});self.assertEqual(old,before)
        self.assertEqual(spec['merged_model'],old['merged_model']);self.assertEqual(spec['checkpoint'],old['checkpoint'])
        for opt in ['--benchmark','--max-steps','--temperature']:
            a=spec['commands'][tool.SUITE];b=ref['commands'][tool.SUITE];self.assertEqual(a[a.index(opt)+1],b[b.index(opt)+1])
    def test_changed_weights_rejected(self):
        _,_,_,_,spec,receipt=self.fixture();(Path(spec['merged_model'])/'model.safetensors').write_bytes(b'changed')
        with self.assertRaises(ValueError):tool.verify_model(spec,receipt)
    def test_pipeline_reuses_export_and_executes_Final_once(self):
        _,_,out,post,_,_=self.fixture();summary={'strict_successes':140,'strict_success_task_ids':list(range(140)),'mean_final_reward':.6}
        with patch.dict(sys.modules,self.modules()),patch.object(tool,'check_code'),patch.object(tool,'verify_runtime_snapshot'),patch.object(tool.socket,'socket'),patch.object(tool.legacy,'serve'),patch.object(tool.legacy,'export_model',side_effect=AssertionError('No re-export'))as export,patch.object(tool.subprocess,'run')as run,patch.object(tool.base,'validate_suite',return_value=summary)as validate,patch.object(tool.base,'cleanup'):
            tool.pipeline(out);self.assertEqual(run.call_count,1);self.assertEqual(validate.call_args.args[1],tool.SUITE);export.assert_not_called()
            with self.assertRaises(ValueError):tool.pipeline(out)
        self.assertEqual(tool.read(post/'final200_complete.json')['B500'],140)
        self.assertTrue(tool.read(post/'final200_complete.json')['Dev_screen_still_failed'])
    def test_failure_retained_without_completion(self):
        _,_,out,post,_,_=self.fixture()
        with patch.dict(sys.modules,self.modules()),patch.object(tool,'check_code'),patch.object(tool,'verify_runtime_snapshot'),patch.object(tool.socket,'socket'),patch.object(tool.legacy,'serve'),patch.object(tool.subprocess,'run',side_effect=RuntimeError('failure')),patch.object(tool.base,'cleanup'):
            with self.assertRaises(RuntimeError):tool.pipeline(out)
        self.assertEqual(tool.read(post/'pipeline_exit_status.json')['exit_code'],1);self.assertFalse((post/'final200_complete.json').exists())


if __name__=='__main__':unittest.main()
