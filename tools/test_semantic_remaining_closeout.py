import copy
import unittest
import json
from pathlib import Path
import semantic_remaining_policy as p
import migrate_training_semantics as m
from semantic_product_verifier_v4 import compare_value

class RemainingPolicyTests(unittest.TestCase):
    def requirement(self,text,subtype,attribute,**kwargs):
        return m.req(text,text,subtype,attribute=attribute,**kwargs)[1]

    def target(self,text,subtype,attribute):
        r=self.requirement(text,subtype,attribute)
        value=p.approximate_target(r,m.b.chinese_number)
        self.assertIsNotNone(value)
        r.update(operator='approximately',expected=value)
        p.validate_approximate(r,m.b.chinese_number)
        return r

    def test_point_has_no_invented_tolerance(self):
        r=self.target('功率大概1500瓦','function','power')
        self.assertEqual(compare_value(r,{'type':'number','value':1500,'unit':'W'}),'pass')
        for n in (1499,1501):self.assertEqual(compare_value(r,{'type':'number','value':n,'unit':'W'}),'unknown')

    def test_range_inside_pass_outside_unknown(self):
        r=self.target('容量大概在8到9毫升之间','capacity','liquid_volume')
        for n in (8,8.5,9):self.assertEqual(compare_value(r,{'type':'number','value':n,'unit':'ml'}),'pass')
        for n in (7.9,9.1):self.assertEqual(compare_value(r,{'type':'number','value':n,'unit':'ml'}),'unknown')

    def test_per_pack_count_does_not_use_denominator(self):
        r=self.target('一袋大概60颗左右','packaging','pack_count')
        self.assertEqual(r['expected'],{'type':'number','value':60,'unit':'piece'})
        r=self.target('9颗左右一袋','packaging','pack_count')
        self.assertEqual(r['expected']['value'],9)

    def test_feature_count_not_physical_pieces(self):
        r=self.target('大概有648个小孔','attribute_count','other_feature_count')
        self.assertEqual(r['expected']['unit'],'count')

    def test_elided_meter_decimal(self):
        r=self.target('长度在1米4上下','dimensions','length')
        self.assertEqual(r['expected']['value'],1.4)
        self.assertEqual(compare_value(r,{'type':'number','value':140,'unit':'cm'}),'pass')

    def test_ambiguous_amount_and_unsupported_unit_not_guessed(self):
        for text,sub,attr in [('需要大概一100包','quantity','purchase_count'),('大小10寸左右','dimensions','length'),
            ('尺寸大概2250*1160','dimensions','dimensions'),('容量1.5匹左右','capacity','energy_capacity'),
            ('需要五斤多一点','quantity','purchase_mass')]:
            self.assertIsNone(p.approximate_target(self.requirement(text,sub,attr),m.b.chinese_number))

    def test_wrong_attribute_family_rejected(self):
        self.assertIsNone(p.approximate_target(self.requirement('容量在4g左右','capacity','liquid_volume'),m.b.chinese_number))
        r=self.target('净含量在4g左右','net_content','mass')
        r['expected']['unit']='ml'
        with self.assertRaises(ValueError):p.validate_approximate(r,m.b.chinese_number)

    def test_wearer_point_in_complete_fit_interval(self):
        r=self.target('身高在90cm左右','size','wearer_height')
        fit={'type':'range','lower':85,'upper':95,'unit':'cm','lower_inclusive':True,'upper_inclusive':True}
        self.assertEqual(compare_value(r,fit),'pass')
        fit.update(lower=100,upper=110)
        self.assertEqual(compare_value(r,fit),'unknown')

    def test_label_no_conversion_or_axis_permutation(self):
        r=self.requirement('尺寸20x25cm','dimensions','dimensions',value={'type':'text','value':'20x25cm'},operator='matches_label')
        p.validate_label(r)
        for text in ('20×25cm','20*25cm','20x25cm'):
            self.assertEqual(compare_value(r,{'type':'text','value':text}),'pass')
        for text in ('25x20cm','200x250mm','20x24cm'):
            self.assertEqual(compare_value(r,{'type':'text','value':text}),'unknown')

    def test_cun_label_stays_cun(self):
        r=self.requirement('尺寸5寸','dimensions','dimensions',value={'type':'text','value':'5寸'},operator='matches_label')
        p.validate_label(r)
        self.assertEqual(compare_value(r,{'type':'text','value':'5寸'}),'pass')
        self.assertEqual(compare_value(r,{'type':'text','value':'5英寸'}),'unknown')
        r['expected']['value']='5英寸'
        with self.assertRaises(ValueError):p.validate_label(r)

    def test_same_number_other_attribute_not_used_as_target(self):
        r=self.requirement('功率1500瓦，容量大概1升','function','power')
        self.assertIsNone(p.approximate_target(r,m.b.chinese_number))

    def test_colloquial_discrete_and_continuous_ranges(self):
        r=self.requirement('包含20多种失真类效果','attribute_count','effect_type_count')
        op,v=p.colloquial_target(r,m.b.chinese_number);r.update(operator=op,expected=v)
        for n in (21,29):self.assertEqual(compare_value(r,{'type':'number','value':n,'unit':'count'}),'pass')
        for n in (20,30):self.assertEqual(compare_value(r,{'type':'number','value':n,'unit':'count'}),'fail')
        r=self.requirement('要二十多公斤','quantity','purchase_mass')
        op,v=p.colloquial_target(r,m.b.chinese_number);r.update(operator=op,expected=v)
        self.assertEqual(compare_value(r,{'type':'number','value':20.5,'unit':'kg'}),'pass')
        self.assertEqual(compare_value(r,{'type':'number','value':20,'unit':'kg'}),'fail')
        self.assertEqual(compare_value(r,{'type':'number','value':30,'unit':'kg'}),'fail')

    def test_lai_does_not_invent_interval(self):
        r=self.target('希望可以有10来张替换使用','quantity','purchase_count')
        self.assertEqual(r['expected'],{'type':'number','value':10,'unit':'sheet'})
        self.assertEqual(compare_value(r,{'type':'number','value':11,'unit':'sheet'}),'unknown')

    def test_human_height_only_cm_inference(self):
        r=self.requirement('适合身高110的女宝','size','wearer_height')
        self.assertEqual(p.human_height_target(r,m.b.chinese_number)['unit'],'cm')
        self.assertIsNone(p.human_height_target(self.requirement('尺寸110','dimensions','height'),m.b.chinese_number))
        self.assertIsNone(p.human_height_target(self.requirement('身高一米一','size','wearer_height'),m.b.chinese_number))

    def test_range_coverage_accepts_wider_not_fixed_height(self):
        r=self.requirement('升降范围最好是在40到80厘米之间','dimensions','height',operator='covers_range',
            value={'type':'range','lower':40,'upper':80,'unit':'cm','lower_inclusive':True,'upper_inclusive':True})
        p.validate_covering_range(r,m.b.chinese_number)
        actual={'type':'range','lower':350,'upper':850,'unit':'mm','lower_inclusive':True,'upper_inclusive':True}
        self.assertEqual(compare_value(r,actual),'pass')
        actual.update(lower=450,upper=750)
        self.assertEqual(compare_value(r,actual),'fail')
        self.assertEqual(compare_value(r,{'type':'number','value':60,'unit':'cm'}),'unknown')

    def test_label_preserves_bits_vs_bytes(self):
        r=self.requirement('速率10Gbps','function','function',operator='matches_label',value={'type':'text','value':'10Gbps'})
        p.validate_label(r)
        self.assertEqual(compare_value(r,{'type':'text','value':'10Gbps'}),'pass')
        self.assertEqual(compare_value(r,{'type':'text','value':'10GBps'}),'unknown')

    def test_budget_tone_and_exact_offer_price(self):
        for text,amount in [('预估价格是40元',40),('预估50元以内',50),('售价接近100元',110),('价格接近5000元',5500)]:
            self.assertEqual(m.b.budget_for_role(text,'total_price')[1]['value'],amount)
        op,value,_=m.b.budget_for_role('价格为350元的那一款','total_price')
        self.assertEqual((op,value['value']),('eq',350))

    def test_explicit_complete_claim_not_optional_or_other_sku(self):
        from semantic_product_verifier_v4 import snapshot,automatic_facts
        product={'asin':'test','price':10,'attribute':['配件齐全'],
                 'customization_options':{'套餐':[{'value':'单机'},{'value':'全套配件'}]}}
        def claims(selected):
            return [f for f in automatic_facts(product,snapshot(product,selected)) if f['value'].get('value')=='配件齐全']
        self.assertEqual(claims({'套餐':'单机'}),[])
        self.assertEqual(len(claims({'套餐':'全套配件'})),1)
        product={'asin':'test','price':10,'attribute':['配件齐全可选'],'customization_options':{}}
        self.assertFalse(any(f['value'].get('value')=='配件齐全' for f in automatic_facts(product,snapshot(product,{}))))

    def test_only_selected_full_size_label_is_a_fact(self):
        from semantic_product_verifier_v4 import snapshot,automatic_facts
        product={'asin':'test','price':10,'customization_options':{'尺寸':[{'value':'20x25cm'},{'value':'40x50cm'}]}}
        facts=automatic_facts(product,snapshot(product,{'尺寸':'20x25cm'}))
        labels=[f['value'].get('value') for f in facts if f['subtype']=='dimensions']
        self.assertIn('20x25cm',labels);self.assertNotIn('40x50cm',labels)


class FrozenCloseoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows={r['task_id']:r for r in m.rows(m.ROOT/'analysis/semantic_remaining_final_20261108/validated1000.jsonl')}

    def test_original_pool_queries_and_approvals_preserved(self):
        source=m.rows(m.ROOT/'analysis/semantic_policy_migration_20261107/validated1000.jsonl')
        self.assertEqual(len(self.rows),1000)
        for row in source:
            self.assertEqual(self.rows[row['task_id']]['query'],row['query'])
            if row['whole_contract_approved']:self.assertEqual(self.rows[row['task_id']]['normalized'],row['normalized'])

    def test_source_limits_still_block_reward_readiness(self):
        held=[r for r in self.rows.values() if r['normalized']['pending_mentions']]
        self.assertEqual(len(held),16)
        for row in held:
            with self.assertRaises(ValueError):m.b.validate(row['normalized'],source_query=row['query'],require_reviewed=True)

    def test_uncertain_bundle_not_forced_to_one_of(self):
        c=self.rows[17803]['normalized']
        self.assertTrue(c['pending_mentions'])
        self.assertFalse(any(r['operator']=='one_of' and r['expected'].get('values')==['2个装','10个装'] for _,r in m.allreq(c)))

    def test_final_received_quantity_not_paid_order_quantity(self):
        c=self.rows[9294]['normalized']
        self.assertTrue(any(r['attribute']=='received_count' and r['expected']['value']==2 for _,r in m.allreq(c)))
        self.assertFalse(any(r['attribute']=='purchase_count' and r['expected'].get('value')==2 for _,r in m.allreq(c)))
        bundle=next(r for _,r in m.allreq(c) if r['subtype']=='bundle')
        self.assertEqual((bundle['expected']['paid_quantity'],bundle['expected']['gift_quantity']),(1,1))

    def test_cooling_power_and_layer_counts_keep_distinct_slots(self):
        self.assertTrue(any(r['attribute']=='cooling_power' for _,r in m.allreq(self.rows[22198]['normalized'])))
        self.assertTrue(any(r['attribute']=='layer_count' for _,r in m.allreq(self.rows[20383]['normalized'])))
        self.assertFalse(any(r['subtype']=='components' and any(i['name']=='钢化玻璃' and i['quantity']==2 for i in r['expected']['items']) for _,r in m.allreq(self.rows[20383]['normalized'])))

    def test_typo_correction_cannot_transfer_to_other_query(self):
        c=copy.deepcopy(self.rows[9286]['normalized'])
        m.b.validate(c,source_query=c['query'])
        self.assertIn('一100包',c['query'])
        c['query']+='补充说明'
        with self.assertRaises(ValueError):m.b.validate(c,source_query=c['query'])

    def test_scope_receipt_does_not_survive_predicate_or_query_change(self):
        import semantic_reviewed_scopes as scope
        record=next(r for r in scope.records() if r['kind']=='predicate_scope_review')
        c=copy.deepcopy(self.rows[record['task_id']]['normalized'])
        d,r=next((d,r) for d,r in m.allreq(c) if {k:r[k] for k in scope.KEYS}==record['predicate'])
        f={'code':record['code'],'location':f'/{d}/{c[d].index(r)}'}
        self.assertTrue(scope.reviewed(c,f))
        r['importance']='required' if r['importance']=='preference' else 'preference'
        self.assertFalse(scope.reviewed(c,f))
        r.update({k:copy.deepcopy(record['predicate'][k]) for k in scope.KEYS})
        c['query']+='另一个问题'
        self.assertFalse(scope.reviewed(c,f))

    def test_no_unreviewed_scope_diagnostics_hidden_in_final_output(self):
        for row in self.rows.values():
            _,findings=m.b.split_flags(row['normalized'],m.b.lint(row['normalized']))
            self.assertEqual(findings,[],msg=str(row['task_id']))

if __name__=='__main__':unittest.main()
