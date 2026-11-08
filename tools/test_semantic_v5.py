import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import semantic_contract_v2_r4 as old
import semantic_contract_v2_r5 as new
from semantic_scope_v5 import approximation_evidence


def raw(q, values, version=new.VERSION):
    c={"schema_version":version,"task_id":1,"query":q,**{d:[] for d in new.DIMENSIONS},"pending_mentions":[]}
    for subtype,attribute,expected,evidence,importance in values:
        dim=next(d for d,subtypes in new.DIMENSIONS.items() if subtype in subtypes)
        c[dim].append({"subtype":subtype,"attribute":attribute,"subject":{"kind":"primary_product","name":"主商品"},
                       "importance":importance,"operator":"eq","expected":expected,"price_basis":None,
                       "evidence":[{"text":x} for x in evidence],"review_status":"pending"})
    return c


def text(q,value=None,importance="required",subtype="function",attribute=None,evidence=None):
    return raw(q,[(subtype,attribute or subtype,{"type":"text","value":value or q},evidence or [q],importance)])


def norm(c, backend=new):return backend.normalize(c,{"task_id":c["task_id"],"query":c["query"]})


class ScopeRegressionTest(unittest.TestCase):
    def test_directional_terms_stay_required_without_tolerance(self):
        for q in ("左右自动摇头","上下移动","左右各有一个开关","上下调节高度"):
            c,a=norm(text(q))
            self.assertEqual(c['functions'][0]['importance'],'required')
            self.assertEqual(c['pending_mentions'],[])
            self.assertFalse(any(x['kind']=='pending_policy' for x in a))

    def test_real_numeric_approximation_still_pending(self):
        for q in ('350ml左右','大约200公斤','重量约为5斤','20支上下'):
            c,a=norm(text(q,subtype='net_content',attribute='mass'))
            self.assertEqual(c['specifications'][0]['importance'],'uncertain')
            self.assertEqual(c['pending_mentions'][0]['reason'],'undefined_tolerance')

    def test_broad_quote_does_not_lend_another_attribute_number(self):
        c,_=norm(text('尺寸350ml左右，可以左右摇头',value='左右摇头'))
        self.assertEqual(c['functions'][0]['importance'],'required')
        self.assertEqual(c['pending_mentions'],[])

    def test_separate_subject_and_value_evidence_anchors_pending_to_value(self):
        q='镜头外壳，宽度大约20mm'
        c=text(q,value='20mm',subtype='dimensions',attribute='width',evidence=['镜头外壳','宽度大约20mm'])
        c['specifications'][0]['subject']={'kind':'component','name':'镜头外壳'}
        result,_=norm(c)
        self.assertEqual(result['pending_mentions'][0]['evidence']['text'],'宽度大约20mm')

    def test_structured_components_never_flattened_by_approximation(self):
        c=raw('配件大约三件',[('components','included_components',{'type':'components','items':[{'name':'配件','quantity':None,'unit':None}]},['配件大约三件'],'required')])
        self.assertEqual(approximation_evidence(c['specifications'][0]),[])

    def test_likes_nested_in_capability_is_review_not_retry_demand(self):
        c,a=norm(text('支持自选喜欢的海报'))
        self.assertEqual(c['functions'][0]['importance'],'required')
        findings=[x for x in a if x['kind']=='semantic_review_findings']
        self.assertTrue(findings)
        self.assertIn('soft_clause_mandatory',[x['code'] for x in findings[0]['findings']])

    def test_category_not_weakened_from_broad_hobby_clause(self):
        c,a=norm(text('他喜欢紫檀木雕的鼻烟壶',value='鼻烟壶',subtype='product_category',attribute='product_category'))
        self.assertEqual(c['category'][0]['importance'],'required')
        self.assertTrue(any(x['kind']=='semantic_review_findings' for x in a))

    def test_direct_preference_and_explicit_mandatory_still_checked(self):
        with self.assertRaisesRegex(ValueError,'soft_clause_mandatory'):
            norm(text('最好蓝色',value='蓝色',subtype='color'))
        with self.assertRaisesRegex(ValueError,'explicit_hard_clause_optional'):
            norm(text('必须防水',value='防水',importance='preference'))
        c,_=norm(text('最好蓝色',value='蓝色',importance='preference',subtype='color'))
        self.assertEqual(c['specifications'][0]['importance'],'preference')

    def test_direct_hope_pending_but_other_targets_not_changed(self):
        c,_=norm(text('希望防水',value='防水'))
        self.assertEqual(c['functions'][0]['importance'],'uncertain')
        self.assertEqual(c['pending_mentions'][0]['reason'],'ambiguous_strength')
        c,_=norm(text('希望颜色好看并且可以摇头',value='摇头'))
        self.assertEqual(c['functions'][0]['importance'],'required')

    def test_hope_does_not_override_explicit_inner_modifier(self):
        c,_=norm(text('希望必须防水',value='防水'))
        self.assertEqual(c['functions'][0]['importance'],'required')
        c,_=norm(text('希望最好蓝色',value='蓝色',importance='preference',subtype='color'))
        self.assertEqual(c['specifications'][0]['importance'],'preference')

    def test_unproven_optional_not_promoted_automatically(self):
        c,a=norm(text('颜色想要奶咖棕色',value='奶咖棕色',importance='preference',subtype='color'))
        self.assertEqual(c['specifications'][0]['importance'],'preference')
        self.assertTrue(any(f['code']=='preference_scope_unproven' for x in a if x['kind']=='semantic_review_findings' for f in x['findings']))

    def test_uncertainty_without_explanation_is_visible(self):
        _,a=norm(text('支持自选喜欢的海报',importance='uncertain'))
        self.assertTrue(any(f['code']=='uncertainty_without_reason' for x in a if x['kind']=='semantic_review_findings' for f in x['findings']))

    def test_percentage_not_relabelled_as_count(self):
        c=raw('含金量99.9%',[('material','material',{'type':'number','value':99.9,'unit':'count'},['含金量99.9%'],'required')])
        with self.assertRaisesRegex(ValueError,'percentage_as_count'):norm(c)

    def test_approximation_cannot_hide_percentage_unit_error(self):
        c=raw('含金量约99.9%',[('material','material',{'type':'number','value':99.9,'unit':'count'},['含金量约99.9%'],'required')])
        with self.assertRaisesRegex(ValueError,'percentage_as_count'):norm(c)

    def test_size_label_not_a_count(self):
        c=raw('尺码38.5',[('size','size_label',{'type':'number','value':38.5,'unit':'count'},['尺码38.5'],'required')])
        with self.assertRaisesRegex(ValueError,'size_label_as_count'):norm(c)

    def test_unsupported_percentage_preserved_as_text_pending(self):
        c=text('含金量99.9%',subtype='material')
        c['pending_mentions']=[{'evidence':{'text':c['query']},'reason':'unsupported_unit','importance':'uncertain'}]
        c,_=norm(c)
        self.assertEqual(c['specifications'][0]['expected']['value'],'含金量99.9%')
        self.assertTrue(c['pending_mentions'])

    def test_unconfirmed_budget_relation_not_silently_given_upper_bound(self):
        from test_semantic_v4 import raw as budget_raw
        for q in ('预估价格是40元','价格接近5000元'):
            c=budget_raw(q);c['schema_version']=new.VERSION
            result,_=norm(c)
            self.assertEqual(result['budget'][0]['expected']['type'],'text')
            self.assertEqual(result['budget'][0]['importance'],'uncertain')
            self.assertTrue(result['pending_mentions'])

    def test_diagnostics_cannot_be_used_to_approve_reward(self):
        c,_=norm(text('支持自选喜欢的海报'))
        c['functions'][0]['review_status']='approved'
        with self.assertRaisesRegex(ValueError,'Unresolved semantic review findings'):
            new.validate(c,source_query=c['query'],require_reviewed=True)

    def test_original_not_mutated_and_bad_evidence_still_rejected(self):
        c=text('左右自动摇头');before=copy.deepcopy(c);norm(c);self.assertEqual(c,before)
        c['functions'][0]['evidence'][0]['text']='左右自动摆头'
        with self.assertRaises(ValueError):norm(c)

    def test_legacy_runner_and_sealed_sample_still_verify(self):
        from build_semantic_holdout import verify
        self.assertEqual(len(verify()),50)
        c,_=norm(text('左右自动摇头',importance='required'),new)
        legacy=text('左右自动摇头');legacy['schema_version']=old.VERSION
        c0,_=norm(legacy,old)
        self.assertEqual(c['functions'][0]['importance'],'required')
        self.assertEqual(c0['functions'][0]['importance'],'uncertain')




class R5IntegrationTest(unittest.TestCase):
    def test_snapshots_preserve_50_1000_queries_and_bind_scope_code(self):
        from extract_semantic_requirements_r5 import snapshot
        old_tasks,old_requests,_,_=__import__('extract_semantic_requirements').snapshot('50','semantic-v2-r4')
        for mode,n in (('50',50),('1000',1000),('regression50',50)):
            tasks,requests,files,provenance=snapshot(mode,'semantic-v2-r5')
            self.assertEqual(len(tasks),n)
            self.assertIn('scope_rules_source.py',files)
            self.assertEqual(json.loads(files['output.schema.json'])['properties']['schema_version']['const'],new.VERSION)
            self.assertEqual(provenance['mode'],mode)
            if mode=='50':self.assertEqual(tasks,old_tasks)

    def test_review_only_finding_does_not_trigger_destructive_retry(self):
        import extract_semantic_requirements_r5 as runner
        c=text('支持自选喜欢的海报');task={'task_id':1,'query':c['query']}
        row={'task_id':1,'custom_id':'test-1','messages':[]}
        reply=json.dumps({'choices':[{'finish_reason':'stop','message':{'content':json.dumps(c,ensure_ascii=False)}}]})
        config={'endpoint':'https://example.com/v1/chat/completions','model':'test','thinking':'omit',
                'response_format':'json_object','max_tokens':8192,'temperature':0,'extraction_profile':'semantic-v2-r5'}
        with tempfile.TemporaryDirectory() as d:
            transport=unittest.mock.Mock(return_value=(200,reply))
            self.assertEqual(runner.execute(Path(d),[task],[row],new.schema(),config,'synthetic-test-token',transport=transport),0)
            self.assertEqual(transport.call_count,1)
            report=json.loads((Path(d)/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['semantic_review_tasks'],1)
            self.assertTrue((Path(d)/'review_findings.jsonl').read_text(encoding='utf-8').strip())
            transport.reset_mock()
            runner.execute(Path(d),[task],[row],new.schema(),config,'synthetic-test-token',transport=transport)
            transport.assert_not_called()

    def test_prepare_only_http_requires_no_key_or_transport_and_collector_matches(self):
        import extract_semantic_requirements_r5 as runner
        import collect_semantic_diagnostics_r5 as collector
        import sys
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'prepared';output=Path(d)/'collected';output.mkdir()
            argv=['test','--mode','50','--output',str(source),'--base-url','http://aigw.dlut.edu.cn/v1',
                  '--model','test','--thinking','omit','--prepare-only']
            with patch.object(sys,'argv',argv), patch.dict(runner.os.environ,{},clear=True), patch.object(runner,'post',side_effect=AssertionError('No network')):
                self.assertEqual(runner.main(),0)
            tasks,requests,schema,manifest=collector.prepare_collection(source,output)
            self.assertEqual(len(tasks),50)
            self.assertEqual(schema['properties']['schema_version']['const'],new.VERSION)
            self.assertEqual(manifest['seed_attempts'],0)

    def test_r5_http_boundary_stays_explicit_and_exact(self):
        from extract_semantic_requirements_r5 import endpoint
        school='http://aigw.dlut.edu.cn/v1'
        self.assertEqual(endpoint(school,allow_school_http=True),school+'/chat/completions')
        with self.assertRaises(ValueError):endpoint(school)
        for url in ('http://other.example/v1',school+'?token=x','http://user:pw@aigw.dlut.edu.cn/v1',school+'#x'):
            with self.assertRaises(ValueError):endpoint(url,allow_school_http=True)

    def test_replay_rejects_overwrite_and_preserves_source(self):
        from replay_semantic_v5 import replay
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source';source.mkdir();out=Path(d)/'out'
            c=text('左右自动摇头');c['schema_version']=old.VERSION
            task={'task_id':1,'query':c['query']}
            reply={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(c,ensure_ascii=False)}}]}
            record={'task_id':1,'http_status':200,'accepted':True,'raw_response':json.dumps(reply,ensure_ascii=False)}
            for name,value in (('tasks.jsonl',task),('attempts.jsonl',record),('report.json',{}),('run_manifest.json',{})):
                (source/name).write_text(json.dumps(value,ensure_ascii=False)+'\n',encoding='utf-8')
            before={x.name:x.read_bytes() for x in source.iterdir()}
            with self.assertRaises(ValueError):replay(source,source)
            report=replay(source,out)
            self.assertEqual(report['valid_r5'],1)
            self.assertEqual(report['api_calls'],0)
            self.assertEqual(before,{x.name:x.read_bytes() for x in source.iterdir()})
            with self.assertRaises(ValueError):replay(source,out)


if __name__=='__main__':unittest.main()
