import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import types
import sys

import b500_dev50 as tool


class IndependentDev(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=tool.ROOT/'outputs',prefix='dev50-test-');self.root=Path(self.temp.name).resolve()
    def tearDown(self):
        self.root.relative_to((tool.ROOT/'outputs').resolve());self.temp.cleanup()
    def fixture(self):
        asset=self.root/'asset';runtime=self.root/'runtime';out=self.root/'out'
        for base in (asset,runtime):
            p=base/'scripts/evaluate_shop_benchmark.py';p.parent.mkdir(parents=True);p.write_text('# original source\n')
        ref={'protocols':{},'benchmarks':{},'commands':{},'source_sha256':{str(asset/'scripts/evaluate_shop_benchmark.py'):tool.digest(asset/'scripts/evaluate_shop_benchmark.py')}}
        for s in ['smoke3','dev50','final200_clean']:
            ref['protocols'][s]={'temperature':0};ref['benchmarks'][s]={'path':'frozen-'+s}
            ref['commands'][s]=['python','old-script','--benchmark','frozen-'+s,'--model','A','--output','old.jsonl','--summary','old.json','--temperature','0','--max-steps','35']
        return ref,asset,runtime,out
    def test_derivation_preserves_nonidentity_CLI_and_excludes_Final(self):
        ref,a,r,o=self.fixture();original=copy.deepcopy(ref);s=tool.derive(ref,a,r,o,self.root/'global_step_500')
        self.assertEqual(ref,original);self.assertEqual(set(s['commands']),{'smoke3','dev50'})
        self.assertNotIn('final200_clean',s['benchmarks']);self.assertEqual(s['protocols']['dev50'],ref['protocols']['dev50'])
        for suite in tool.SUITES:
            before=ref['commands'][suite];after=s['commands'][suite]
            for opt in ('--benchmark','--temperature','--max-steps'):self.assertEqual(before[before.index(opt)+1],after[after.index(opt)+1])
    def test_changed_isolated_source_rejected(self):
        ref,a,r,o=self.fixture();(r/'scripts/evaluate_shop_benchmark.py').write_text('changed')
        with self.assertRaises(ValueError):tool.derive(ref,a,r,o,self.root/'global_step_500')
    def test_insufficient_export_disk_rejected_before_model_loading(self):
        (self.root/'model_world_size_1_rank_0.pt').write_bytes(b'x')
        with patch.object(tool.shutil,'disk_usage',return_value=types.SimpleNamespace(free=0)):
            with self.assertRaisesRegex(ValueError,'Insufficient disk'):tool.require_export_space(self.root)
    def test_duplicate_identity_option_rejected(self):
        ref,a,r,o=self.fixture();ref['commands']['dev50']+=['--model','another']
        with self.assertRaises(ValueError):tool.derive(ref,a,r,o,self.root/'global_step_500')
    def results(self):
        folder=self.root/'evaluation/dev50';folder.mkdir(parents=True)
        rows=[{'task_id':1,'error':{'type':'ContextBudgetError'}},{'task_id':2,'error':None}]
        (folder/'trajectories.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
        calc={'strict_successes':0,'expected_tasks':2,'completed_tasks':2,'missing_tasks':[]}
        frozen=tool.read(tool.PACKAGE/'protocol.json')['evaluation']['protocol']
        tool.write(folder/'summary.json',{**calc,'protocol':frozen})
        spec={'evaluation_root':str(self.root/'evaluation'),'benchmarks':{'dev50':{'task_ids':[1,2]}}}
        return folder,rows,calc,spec
    def test_context_failure_kept_in_denominator_and_no_success_gate(self):
        folder,rows,calc,spec=self.results();out=tool.validate_suite(spec,'dev50',lambda ids,rs:calc)
        self.assertEqual(out['strict_successes'],0);self.assertTrue(tool.read(folder/'passed.json')['passed'])
    def test_noncontext_execution_failure_rejected(self):
        folder,rows,calc,spec=self.results();rows[0]['error']['type']='ConnectionError'
        (folder/'trajectories.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
        with self.assertRaises(ValueError):tool.validate_suite(spec,'dev50',lambda ids,rs:calc)
    def test_duplicate_missing_task_rejected(self):
        folder,rows,calc,spec=self.results();rows[1]['task_id']=1
        (folder/'trajectories.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
        with self.assertRaises(ValueError):tool.validate_suite(spec,'dev50',lambda ids,rs:calc)
    def pipeline_fixture(self):
        ref,a,r,o=self.fixture();spec=tool.derive(ref,a,r,o,self.root/'global_step_500')
        post=o/tool.POST_REL;post.mkdir(parents=True);tool.write(post/'evaluation_spec.json',spec)
        mods={n:types.ModuleType(n)for n in ['shopping_grpo','shopping_grpo.evaluation','shopping_grpo.evaluation.summary']}
        mods['shopping_grpo'].__path__=[];mods['shopping_grpo.evaluation'].__path__=[]
        mods['shopping_grpo.evaluation.summary'].summarize_trajectories=lambda *args:{}
        if sys.platform!='linux':
            mods['fcntl']=types.SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *args:None)
        return o,post,mods
    def test_pipeline_runs_only_smoke_then_dev_and_refuses_repeat(self):
        o,post,mods=self.pipeline_fixture()
        result={'strict_successes':0,'expected_tasks':50,'strict_success_task_ids':[],'mean_final_reward':0.}
        with patch.dict(sys.modules,mods),patch.object(tool,'check_code'),patch.object(tool,'verify_runtime_snapshot'),patch.object(tool,'cleanup'),patch.object(tool.socket,'socket'),patch.object(tool.legacy,'export_model'),patch.object(tool.legacy,'serve'),patch.object(tool.subprocess,'run')as run,patch.object(tool,'validate_suite',return_value=result)as val:
            tool.pipeline(o)
            self.assertEqual([c.args[1]for c in val.call_args_list],['smoke3','dev50']);self.assertEqual(run.call_count,2)
            with self.assertRaises(ValueError):tool.pipeline(o)
        self.assertTrue(tool.read(post/'dev50_complete.json')['passed'])
    def test_smoke_execution_failure_stops_before_dev(self):
        o,post,mods=self.pipeline_fixture()
        with patch.dict(sys.modules,mods),patch.object(tool,'check_code'),patch.object(tool,'verify_runtime_snapshot'),patch.object(tool,'cleanup'),patch.object(tool.socket,'socket'),patch.object(tool.legacy,'export_model'),patch.object(tool.legacy,'serve'),patch.object(tool.subprocess,'run'),patch.object(tool,'validate_suite',side_effect=ValueError('infrastructure'))as val:
            with self.assertRaises(ValueError):tool.pipeline(o)
            self.assertEqual(val.call_count,1)
        self.assertEqual(tool.read(post/'pipeline_exit_status.json')['exit_code'],1)


if __name__=='__main__':unittest.main()
