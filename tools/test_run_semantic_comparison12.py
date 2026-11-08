import contextlib
import functools
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import run_semantic_comparison12 as comparison
import extract_semantic_requirements_r5 as runner
from collect_semantic_diagnostics_r5 import collect

CONFIG={'endpoint':'http://aigw.dlut.edu.cn/v1/chat/completions','model':'DeepSeek-V4-Flash-0731-W8A8',
        'thinking':'omit','extraction_profile':'semantic-v2-r5','evidence_offsets':'strict',
        'response_format':'json_object','max_tokens':8192,'temperature':0}
DONOR=comparison.ROOT/'outputs/semantic_school_regression50_r5_v1'


class ComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.fixed=comparison.inputs()

    def prepare(self,path,**kwargs):
        # Fixed inputs themselves are verified once; isolate persistence tests.
        with patch.object(comparison,'inputs',return_value=self.fixed):
            return comparison.prepare(path,kwargs.pop('config',CONFIG),**kwargs)

    def test_fixed_12_are_exact_same_messages_without_rubric(self):
        tasks,requests,files,p=self.fixed
        self.assertEqual(len(tasks),12)
        self.assertEqual(len({t['task_id'] for t in tasks}),12)
        self.assertEqual(p['mode'],'comparison12')
        for t,r in zip(tasks,requests):
            self.assertEqual(t['task_id'],r['task_id'])
            self.assertIn(runner.dump(t),r['messages'][1]['content'])
            self.assertEqual(r['messages'][0]['content'],files['system_prompt.txt'].decode('utf-8'))
        self.assertNotIn('selection_rationale',files['requests.jsonl'].decode('utf-8'))

    def test_both_models_receive_identical_protocol_and_tasks(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'flash',Path(d)/'qwen'
            self.prepare(a);self.prepare(b,config=dict(CONFIG,model='Qwen3.5-122B-A10B'))
            for name in ('tasks.jsonl','requests.jsonl','system_prompt.txt','output.schema.json','scope_rules_source.py'):
                self.assertEqual((a/name).read_bytes(),(b/name).read_bytes())

    def test_actual_flash_first_four_reused_without_rebilling(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run';tasks,rows,schema,m=self.prepare(p,reuse_from=DONOR)
            _,r=comparison.aggregate(p,tasks,schema,m)
            self.assertEqual(m['reuse']['reused_task_ids'],[1167,1588,2711,2995])
            self.assertEqual(r['validated_tasks'],4)
            self.assertEqual(r['reused_tasks'],4)
            self.assertEqual(r['attempts'],0)
            self.assertEqual(r['usage_all_attempts'],{})

    def test_reuse_from_different_model_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'identical model'):
                self.prepare(Path(d)/'run',config=dict(CONFIG,model='Qwen3.5-122B-A10B'),reuse_from=DONOR)

    def test_resume_and_drift_or_source_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run';self.prepare(p);self.prepare(p)
            with self.assertRaisesRegex(ValueError,'NEW output'):
                self.prepare(p,config=dict(CONFIG,max_tokens=16384))
            with self.assertRaisesRegex(ValueError,'independent output'):
                self.prepare(comparison.SAMPLE)
            (p/'requests.jsonl').write_text('changed',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'snapshot changed'):self.prepare(p)

    def test_prepare_only_school_http_reads_no_key_and_sends_no_request(self):
        with tempfile.TemporaryDirectory() as d:
            args=['test','--output',str(Path(d)/'run'),'--model',CONFIG['model'],'--base-url','http://aigw.dlut.edu.cn/v1','--prepare-only']
            with patch.object(sys,'argv',args),patch.dict(os.environ,{},clear=True),patch.object(comparison,'inputs',return_value=self.fixed),patch.object(comparison,'collect',side_effect=AssertionError('No API')):
                with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(comparison.main(),0)

    def test_model_failures_all_collected_and_resume_does_not_repeat(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run';tasks,rows,schema,m=self.prepare(p,attempts=1)
            calls=[]
            def fake(url,key,body,timeout):
                calls.append(body)
                return 200,json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'{}'}}]})
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(collect(p,tasks,rows,schema,m,'synthetic-test-key',runner=functools.partial(runner.execute,transport=fake)),0)
                collect(p,tasks,rows,schema,m,'synthetic-test-key',runner=functools.partial(runner.execute,transport=fake))
            self.assertEqual(len(calls),12)
            r=json.loads((p/'report.json').read_text(encoding='utf-8'))
            self.assertTrue(r['collection_complete'])
            self.assertFalse(r['complete'])
            self.assertEqual(r['validated_tasks'],0)

    def test_http_authorization_error_stops_after_one_request(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run';tasks,rows,schema,m=self.prepare(p)
            calls=[]
            def fake(*args):
                calls.append(1);raise runner.HTTPFailure(401,'Unauthorized')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(collect(p,tasks,rows,schema,m,'synthetic-test-key',runner=functools.partial(runner.execute,transport=fake)),2)
            self.assertEqual(len(calls),1)


if __name__=='__main__':unittest.main()
