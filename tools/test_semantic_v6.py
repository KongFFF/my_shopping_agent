import copy
import unittest
import semantic_contract_v2_r6 as backend
from test_semantic_v5 import raw as base_raw, text as base_text
from semantic_repairs_v6 import preprocess
from semantic_coverage_v6 import explicit_counts


def adapt(c):c['schema_version']=backend.VERSION;return c
def text(*args,**kw):return adapt(base_text(*args,**kw))
def raw(*args,**kw):return adapt(base_raw(*args,**kw))
def norm(c):return backend.normalize(c,{'task_id':c['task_id'],'query':c['query']})
def count(n,e):return ('quantity','purchase_count',{'type':'number','value':n,'unit':'piece'},[e],'required')


class V6RegressionTest(unittest.TestCase):
    def test_operation_not_order(self):
        q='买3个固定扣，需要两根手指按开'
        c=raw(q,[count(3,'买3个')]);v,_=norm(c)
        self.assertEqual(len(v['specifications']),1)
        self.assertEqual([x['count'] for x in explicit_counts(q,backend.chinese_number)],[3])

    def test_conflicting_order_counts_blocked(self):
        c=raw('买3个固定扣，需要两根手指按开',[count(3,'买3个'),count(2,'需要两根手指按开')])
        with self.assertRaisesRegex(ValueError,'conflicting_purchase_counts'):norm(c)

    def test_real_new_counters_and_verbs_still_covered(self):
        for q in ('买一台电视','入手一条裤子'):
            c=text(q)
            with self.assertRaisesRegex(ValueError,'purchase_quantity_coverage'):norm(c)

    def test_recommendation_count_review_only(self):
        v,a=norm(text('推荐一台机器',value='机器',subtype='product_category',attribute='product_category'))
        self.assertTrue(any(f['code']=='recommendation_quantity_scope' for x in a if x['kind']=='semantic_review_findings' for f in x['findings']))

    def test_finger_goods_not_body_operation(self):
        q='买2个手指套';self.assertEqual(len(explicit_counts(q,backend.chinese_number)),1)

    def test_configure_does_not_force_main_product_into_component(self):
        v,a=norm(text('想配置一台清洗机器',value='清洗机器',subtype='product_category',attribute='product_category'))
        self.assertTrue(any(x['kind']=='semantic_review_findings' for x in a))

    def test_explicit_accessory_count_still_required(self):
        with self.assertRaisesRegex(ValueError,'component_quantity_coverage'):
            norm(text('配有一台推车'))

    def test_feature_and_pack_counts_not_orders(self):
        for q in ('需要5个档位','需要三件套'):
            self.assertFalse(explicit_counts(q,backend.chinese_number))

    def test_storage_count_preserved_not_guessed_gb(self):
        c=raw('要256G存储空间',[('capacity','storage_capacity',{'type':'number','value':256,'unit':'count'},['256G'],'required')])
        before=copy.deepcopy(c);v,a=norm(c)
        self.assertEqual(c,before)
        self.assertEqual(v['specifications'][0]['expected'],{'type':'text','value':'256G'})
        self.assertEqual(v['pending_mentions'][0]['reason'],'unsupported_unit')
        self.assertTrue(any(x['kind']=='unsupported_unit_preserve_source' for x in a))

    def test_unanchored_storage_count_still_blocked(self):
        c=raw('存储容量很大',[('capacity','storage_capacity',{'type':'number','value':256,'unit':'count'},['存储容量很大'],'required')])
        with self.assertRaisesRegex(ValueError,'storage_as_count'):norm(c)

    def test_percentage_keeps_required_fact_and_pending(self):
        c=raw('含金量不低于99.9%',[('composition','composition',{'type':'number','value':99.9,'unit':'count'},['含金量不低于99.9%'],'required')])
        v,_=norm(c)
        self.assertEqual(v['specifications'][0]['importance'],'required')
        self.assertEqual(v['specifications'][0]['expected']['value'],c['query'])
        self.assertEqual(v['pending_mentions'][0]['reason'],'unsupported_unit')

    def test_bare_cun_user_decision(self):
        c=raw('尺寸不超过5寸',[('dimensions','dimensions',{'type':'number','value':5,'unit':'inch'},['不超过5寸'],'required')])
        c['specifications'][0]['operator']='at_most'
        v,_=norm(c)
        self.assertEqual(v['specifications'][0]['expected']['type'],'text')
        self.assertTrue(v['pending_mentions'])

    def test_explicit_inch_not_changed(self):
        c=raw('长度5英寸',[('dimensions','length',{'type':'number','value':5,'unit':'inch'},['5英寸'],'required')])
        v,_=norm(c);self.assertEqual(v['specifications'][0]['expected']['unit'],'inch');self.assertFalse(v['pending_mentions'])

    def test_word_chicun_is_not_a_cun_unit(self):
        v,_=norm(text('尺寸很大',subtype='dimensions',attribute='dimensions'))
        self.assertFalse(v['pending_mentions'])

    def test_embedded_cun_component_keeps_structure(self):
        c=raw('配有5寸低音炮',[('components','included_components',{'type':'components','items':[{'name':'5寸低音炮','quantity':None,'unit':None}]},['配有5寸低音炮'],'required')])
        v,_=norm(c)
        self.assertEqual(v['specifications'][0]['expected']['type'],'components')
        self.assertTrue(v['pending_mentions'])

    def test_count_not_a_dimension(self):
        c=raw('有5档调节',[('attribute_count','gear_count',{'type':'number','value':5,'unit':'count'},['5档调节'],'required')])
        v,_=norm(c);self.assertEqual(v['specifications'][0]['expected']['unit'],'count')

    def test_repeated_order_quote_resolved_by_purchase_span(self):
        c=raw('推荐一盒饮料，一盒装20袋',[('quantity','purchase_count',{'type':'number','value':1,'unit':'box'},['一盒'],'required')])
        v,a=norm(c);self.assertEqual(v['specifications'][0]['evidence'][0]['text'],'推荐一盒饮料')
        self.assertTrue(any(x['kind']=='evidence_unique_context' for x in a))

    def test_repeated_color_resolved_only_by_literal_object(self):
        c=text('上衣黑色，裙子黑色',value='黑色',subtype='color',evidence=['黑色'])
        c['specifications'][0]['subject']={'kind':'component','name':'裙子'}
        v,_=norm(c);self.assertEqual(v['specifications'][0]['evidence'][0]['text'],'裙子黑色')

    def test_repeated_unanchored_color_stays_ambiguous(self):
        c=text('上衣黑色，裙子黑色',value='黑色',subtype='color',evidence=['黑色'])
        with self.assertRaisesRegex(ValueError,'quote must occur once'):norm(c)

    def test_missing_subject_same_clause_only(self):
        c=text('上衣是杏色，裙子是绿色',value='杏色',subtype='color',evidence=['杏色'])
        c['specifications'][0]['subject']={'kind':'component','name':'上衣'}
        v,a=norm(c);self.assertTrue(any(e['text']=='上衣是杏色' for e in v['specifications'][0]['evidence']))
        self.assertTrue(any(x['kind']=='subject_same_clause_evidence' for x in a))

    def test_cross_clause_subject_not_invented(self):
        c=text('上衣是杏色，裙子是绿色',value='绿色',subtype='color',evidence=['绿色'])
        c['specifications'][0]['subject']={'kind':'component','name':'上衣'}
        with self.assertRaisesRegex(ValueError,'subject_evidence'):norm(c)

    def test_absent_quote_never_fabricated(self):
        c=text('颜色黑色',value='黑色',subtype='color',evidence=['颜色墨黑'])
        with self.assertRaisesRegex(ValueError,'quote must occur once'):norm(c)

    def test_directional_and_real_approximation_regressions(self):
        v,_=norm(text('左右自动摇头'));self.assertEqual(v['functions'][0]['importance'],'required')
        v,_=norm(text('350ml左右',subtype='capacity',attribute='liquid_volume'))
        self.assertEqual(v['pending_mentions'][0]['reason'],'undefined_tolerance')


if __name__=='__main__':unittest.main()
