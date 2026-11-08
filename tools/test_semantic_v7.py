import copy
import unittest
import semantic_contract_v2_r7 as b
from test_semantic_v5 import raw as base_raw, text as base_text
from semantic_coverage_v7 import explicit_counts


def raw(q, values):
    c=base_raw(q,values);c['schema_version']=b.VERSION;return c


def text(q, **kwargs):
    c=base_text(q,**kwargs);c['schema_version']=b.VERSION;return c


def norm(c): return b.normalize(c,{'task_id':c['task_id'],'query':c['query']})


def components(q, name, count=None, unit=None, attribute='included_components'):
    return raw(q,[('components',attribute,{'type':'components','items':[{'name':name,'quantity':count,'unit':unit}]},[q],'required')])


class R7Regression(unittest.TestCase):
    def test_delivery_is_included_not_gift(self):
        c=components('配送一条15cm的线','线',1,'piece')
        n,_=norm(c);self.assertEqual(n['specifications'][0]['attribute'],'included_components')

    def test_operation_delivery_is_not_gift(self):
        for q in ('能同时投送两包饲料','运输两包饲料','寄送两包货物'):
            self.assertFalse(explicit_counts(q,b.chinese_number))
        norm(text('能同时投送两包饲料'))

    def test_true_gift_and_accessory_omissions_still_fail(self):
        for q in ('赠送两个刷子','配有一台推车','配送一条线'):
            with self.assertRaisesRegex(ValueError,'component_quantity_coverage'):norm(text(q))

    def test_suffix_bounds_and_exact_distinguished(self):
        for source,operator in [('买4支以上','at_least'),('买4支以下','at_most'),('买4支','eq')]:
            c=raw(source,[('quantity','purchase_count',{'type':'number','value':4,'unit':'piece'},[source],'required')])
            c['specifications'][0]['operator']=operator;norm(c)
        bad=raw('买4支以上',[('quantity','purchase_count',{'type':'number','value':4,'unit':'piece'},['买4支以上'],'required')])
        with self.assertRaisesRegex(ValueError,'purchase_quantity_coverage'):norm(bad)

    def test_one_character_accessory(self):
        norm(components('配有21根弦','弦',21,'piece'))

    def test_holes_are_features_but_plugs_are_components(self):
        self.assertFalse(explicit_counts('带有8个喷孔',b.chinese_number))
        self.assertEqual(len(explicit_counts('配有两个三孔插头',b.chinese_number)),1)

    def test_named_secondary_count_not_forced_on_primary(self):
        q='需要5瓶配套的清洁液'
        n,a=norm(components(q,'清洁液',5,'bottle'))
        self.assertFalse(any(r['subtype']=='quantity' for r in n['specifications']))
        self.assertTrue(any(f['code']=='component_vs_order_quantity_scope' for x in a if x['kind']=='semantic_review_findings' for f in x['findings']))
        with self.assertRaisesRegex(ValueError,'purchase_quantity_coverage'):norm(text(q))

    def test_secondary_count_cannot_be_lent_to_other_item(self):
        q='配有2个夹子和3个盒子'
        c=components(q,'盒子',2,'piece')
        with self.assertRaisesRegex(ValueError,'component_quantity_coverage'):norm(c)

    def test_direct_sales_not_origin(self):
        c=text('生产厂家直接销售',subtype='merchant',attribute='merchant')
        n,_=norm(c);self.assertEqual(n['specifications'][0]['subtype'],'merchant')

    def test_manufacturer_role_not_brand(self):
        c=text('由示例企业生产的',subtype='merchant',attribute='merchant')
        n,a=norm(c);self.assertEqual(n['specifications'][0]['subtype'],'manufacturer');self.assertFalse(n['brand'])

    def test_explicit_best_gift_preference(self):
        c=components('最好可以赠送安装配件','安装配件',attribute='gift_presence')
        c['specifications'][0]['importance']='preference'
        n,_=norm(c);self.assertEqual(n['specifications'][0]['importance'],'preference')

    def test_ambiguous_gift_still_pending(self):
        n,_=norm(components('可以赠送附件','附件',attribute='gift_presence'))
        self.assertEqual(n['specifications'][0]['importance'],'uncertain');self.assertTrue(n['pending_mentions'])

    def test_nested_likes_is_not_purchase_preference(self):
        q='适合喜欢尝试不同风味的咖啡爱好者'
        c=text(q,value='喜欢尝试不同风味的咖啡爱好者',subtype='target_user',attribute='target_user')
        n,_=norm(c);self.assertEqual(n['functions'][0]['importance'],'required')

    def test_component_length_split_without_count_one(self):
        c=components('需要有50米的管道','管道',50,'m');before=copy.deepcopy(c)
        n,a=norm(c);self.assertEqual(c,before)
        component=next(r for r in n['specifications'] if r['subtype']=='components')
        measure=next(r for r in n['specifications'] if r['subtype']=='dimensions')
        self.assertIsNone(component['expected']['items'][0]['quantity'])
        self.assertEqual(measure['subject']['name'],'管道');self.assertEqual(measure['expected']['unit'],'m')

    def test_unanchored_physical_component_not_repaired(self):
        with self.assertRaises(Exception):norm(components('有管道','管道',50,'m'))

    def test_order_volume_not_piece_count(self):
        c=raw('需求量15升以上',[('quantity','purchase_count',{'type':'number','value':15,'unit':'l'},['需求量15升以上'],'required')])
        c['specifications'][0]['operator']='at_least'
        n,_=norm(c);self.assertEqual(n['specifications'][0]['attribute'],'purchase_volume')

    def test_per_flavor_not_total_and_color_not_tone(self):
        q='包含4种不同口味，每种10粒'
        c=raw(q,[('attribute_count','tone_type_count',{'type':'number','value':4,'unit':'count'},['包含4种不同口味'],'required'),
                 ('quantity','purchase_count',{'type':'number','value':10,'unit':'piece'},['每种10粒'],'required')])
        n,_=norm(c)
        self.assertEqual(n['specifications'][0]['attribute'],'other_feature_count')
        self.assertEqual(n['specifications'][1]['attribute'],'per_variant_count')
        self.assertFalse(any(r['attribute']=='purchase_count' for r in n['specifications']))

    def test_missing_unit_does_not_gain_cm(self):
        c=raw('适合身高110的孩子',[('size','wearer_height',{'type':'number','value':110,'unit':'cm'},['身高110'],'required')])
        n,_=norm(c)
        self.assertEqual(n['specifications'][0]['expected'],{'type':'text','value':'身高110'})
        self.assertTrue(any(p['reason']=='unsupported_unit' for p in n['pending_mentions']))

    def test_shared_or_missing_dimension_unit_not_guessed(self):
        c=raw('0.9米宽，1.9长',[('dimensions','length',{'type':'number','value':1.9,'unit':'m'},['1.9长'],'required')])
        n,_=norm(c);self.assertEqual(n['specifications'][0]['expected']['type'],'text');self.assertTrue(n['pending_mentions'])

    def test_duo_retains_strict_lower_and_unknown_upper(self):
        c=raw('包含20多种效果',[('attribute_count','effect_type_count',{'type':'number','value':20,'unit':'count'},['包含20多种效果'],'required')])
        n,_=norm(c);r=n['specifications'][0]
        self.assertEqual(r['operator'],'between');self.assertEqual(r['expected']['lower'],20)
        self.assertFalse(r['expected']['lower_inclusive']);self.assertIsNone(r['expected']['upper'])
        self.assertEqual(r['importance'],'uncertain');self.assertTrue(n['pending_mentions'])

    def test_matching_item_sent_with_package_scope_remains_review(self):
        n,a=norm(components('要配套送一个刷子','刷子',1,'piece'))
        self.assertEqual(n['specifications'][0]['attribute'],'included_components')
        self.assertTrue(any(f['code']=='gift_vs_included_scope' for x in a if x['kind']=='semantic_review_findings' for f in x['findings']))

    def test_confirmed_exclusion_representable_but_not_approved(self):
        c=components('不需要镜头','镜头');c['specifications'][0]['operator']='excludes'
        n,_=norm(c);self.assertEqual(n['specifications'][0]['operator'],'excludes')
        with self.assertRaisesRegex(ValueError,'Unreviewed'):b.validate(n,source_query=c['query'],require_reviewed=True)

    def test_negative_source_not_positive_inclusion(self):
        with self.assertRaisesRegex(ValueError,'negated_component_as_positive'):norm(components('不需要镜头','镜头'))

    def test_negative_scope_not_lent_to_other_component(self):
        c=components('不需要镜头，需要机身','机身');c['specifications'][0]['operator']='excludes'
        with self.assertRaisesRegex(ValueError,'exclusion_source_unproven'):norm(c)

    def test_no_negative_no_exclusion(self):
        c=components('包含镜头','镜头');c['specifications'][0]['operator']='excludes'
        with self.assertRaisesRegex(ValueError,'negative source'):norm(c)

    def test_per_variant_mismatched_amount_rejected(self):
        c=raw('每种20粒',[('quantity','purchase_count',{'type':'number','value':10,'unit':'piece'},['每种20粒'],'required')])
        with self.assertRaisesRegex(ValueError,'per_variant_count_source_mismatch'):norm(c)

    def test_exclusion_cannot_hide_a_bounded_count(self):
        c=components('不需要镜头','镜头',1,'piece');c['specifications'][0]['operator']='excludes'
        with self.assertRaisesRegex(ValueError,'bounded forbidden quantities'):norm(c)

    def test_true_order_and_body_operation_regression(self):
        c=raw('买3个固定扣，需要两根手指按开',[('quantity','purchase_count',{'type':'number','value':3,'unit':'piece'},['买3个'],'required')])
        norm(c)

    def test_find_one_type_not_forced_to_order_one(self):
        n,a=norm(text('找一个能缓解的药'))
        self.assertFalse(any(r['subtype']=='quantity' for r in n['specifications']))
        self.assertTrue(any(f['code']=='selection_quantity_scope' for x in a if x['kind']=='semantic_review_findings' for f in x['findings']))
        with self.assertRaisesRegex(ValueError,'purchase_quantity_coverage'):norm(text('买一个药盒'))

    def test_cun_and_non_budget_tolerance_stay_pending(self):
        n,_=norm(text('5寸左右',subtype='dimensions',attribute='dimensions'))
        self.assertTrue(n['pending_mentions']);self.assertEqual(n['specifications'][0]['importance'],'uncertain')


if __name__=='__main__': unittest.main()
