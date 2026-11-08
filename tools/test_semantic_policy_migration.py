"""Regression tests for source policy migration, not approvals or model tests."""
import copy
import hashlib
import json
import unittest

import semantic_contract_v2_r7_final12 as b
import migrate_training_semantics as m
from semantic_review_decisions import strength_decision
from semantic_migration_policy import adjacent_tens_money, explicit_money_multiplier
from semantic_product_verifier_v4 import compare_value


class SourceMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source={r['task_id']:r for r in m.rows(m.SOURCE)}
        cls.policy=json.loads(m.POLICY.read_text(encoding='utf-8'))
        cls.decisions={(d['task_id'],d['reason'],d['text']):d for d in m.rows(m.LEDGER)}

    def migrated(self,task):
        result,changes,held=m.migrate(self.source[task],self.policy,self.decisions)
        b.validate(result['normalized'],source_query=result['query'])
        return result['normalized'],changes,held

    def test_source_not_mutated_and_actor_query_unchanged(self):
        before=copy.deepcopy(self.source[243])
        c,_,_=self.migrated(243)
        self.assertEqual(self.source[243],before)
        self.assertEqual(c['query'],before['query'])

    def test_missing_qualitative_and_certification_requirements_restored(self):
        c,_,_=self.migrated(243)
        text=json.dumps([r['expected'] for _,r in m.allreq(c)],ensure_ascii=False)
        for word in ('变化多样','品牌、质量有保障','中国强制性产品认证'):self.assertIn(word,text)
        self.assertFalse(c.get('whole_contract_approved',False))

    def test_retained_style_not_added_twice(self):
        c,_,_=self.migrated(14888)
        relevant=[r for _,r in m.allreq(c) if '时尚潮流百搭' in str(r['expected'])]
        self.assertEqual(len(relevant),1)
        self.assertEqual(relevant[0]['importance'],'required')

    def test_latest_statement_not_relative_current_year(self):
        c,_,_=self.migrated(10440)
        latest=[r for _,r in m.allreq(c) if '最新款' in str(r['expected'])]
        self.assertEqual(len(latest),1)
        self.assertEqual(latest[0]['importance'],'required')
        other,_,_=self.migrated(19395)
        self.assertTrue(any(p['reason']=='missing_reference_time' for p in other['pending_mentions']))

    def test_newly_bought_car_is_context_not_new_product_condition(self):
        c,changes,_=self.migrated(2147)
        self.assertFalse(any('新买的' in str(r['expected']) for _,r in m.allreq(c)))
        self.assertTrue(any(a['method']=='reviewed_non_scoring_context' for a in changes))

    def test_hard_and_preferred_hope_decisions_are_source_bound(self):
        c,_,_=self.migrated(3499)
        water=next(r for _,r in m.allreq(c) if r['id']=='r006')
        self.assertEqual(water['importance'],'preference')
        forged=copy.deepcopy(water);forged['expected']={'type':'text','value':'过滤净水'}
        self.assertIsNone(strength_decision(c['query'],forged))
        self.assertIsNone(strength_decision(c['query']+'另一个需求',water))

    def test_later_source_bound_color_decision_preserves_c_table_functions(self):
        c,_,_=self.migrated(3844)
        byid={r['id']:r for _,r in m.allreq(c)}
        self.assertEqual(byid['r008']['importance'],'required')
        self.assertEqual(byid['r009']['importance'],'required')
        # The later A-table decision explicitly approves black as preference.
        self.assertEqual(byid['r006']['importance'],'preference')

    def test_confirmed_shipping_is_zero_buyer_fee_not_product_price_zero(self):
        c,_,_=self.migrated(6323)
        shipping=[r for r in c['budget'] if r['subtype']=='shipping_fee']
        self.assertEqual(len(shipping),1)
        self.assertEqual(shipping[0]['expected']['value'],0)
        self.assertEqual(shipping[0]['importance'],'required')
        self.assertTrue(any(r['subtype']=='total_price' and r['expected']['value']>0 for r in c['budget']))

    def test_hope_does_not_erase_unspecified_accessory_set_or_approximate_count(self):
        for task in (247,11503):
            c,_,_=self.migrated(task)
            self.assertTrue(any(p['reason']=='ambiguous_meaning' for p in c['pending_mentions']))

    def test_hard_hope_budgets_have_expected_bounds(self):
        for task,amount in ((17119,170),(19807,11),(20493,30)):
            c,_,_=self.migrated(task)
            r=c['budget'][0]
            self.assertEqual(r['importance'],'required')
            self.assertEqual(r['expected']['value'],amount)

    def test_mixed_pack_omission_restored_without_order_count(self):
        c,_,_=self.migrated(9648)
        self.assertTrue(any(r['subtype']=='packaging' and r['expected'].get('value')=='混装' for _,r in m.allreq(c)))

    def test_sales_pack_does_not_become_three_ordered_units(self):
        c,_,_=self.migrated(147)
        self.assertFalse(any(r['subtype']=='quantity' and r['attribute']=='purchase_count' and r['expected'].get('value')==3 for _,r in m.allreq(c)))
        self.assertTrue(any(r['subtype']=='packaging' and '三瓶装' in str(r['expected']) for _,r in m.allreq(c)))

    def test_budget_825_fixed_to_385(self):
        c,_,_=self.migrated(9736)
        self.assertEqual(c['budget'][0]['expected']['value'],385)

    def test_hundred_adjacent_tens_does_not_guess_other_intervals(self):
        self.assertEqual(adjacent_tens_money('价格三百四五十左右'),'价格350左右')
        self.assertIsNone(adjacent_tens_money('价格三百四七十左右'))
        self.assertIsNone(adjacent_tens_money('价格三百四五左右'))

    def test_explicit_currency_scales_and_separate_pack_count(self):
        for task,amount in ((4051,40000),(4320,14300),(15490,33000),(19500,3300),(20711,7000),(22458,330)):
            c,_,_=self.migrated(task)
            r=next(r for r in c['budget'] if r['subtype']=='total_price')
            self.assertEqual(r['expected']['value'],amount)
        self.assertEqual(explicit_money_multiplier('预算1.3万元左右'),'预算13000元左右')
        self.assertEqual(explicit_money_multiplier('预算1万3左右'),'预算1万3左右')

    def test_elided_money_rules_preserve_magnitude(self):
        self.assertEqual(b.budget_for_role('预算1万2以内','total_price')[1]['value'],12000)
        self.assertEqual(b.budget_for_role('总价大概1万3左右能拿下','total_price')[1]['value'],14300)

    def test_explicit_open_lower_and_closed_upper_budget(self):
        c,_,_=self.migrated(17918)
        v=c['budget'][0]['expected']
        self.assertEqual((v['lower'],v['upper']),(250,300))
        self.assertFalse(v['lower_inclusive']);self.assertTrue(v['upper_inclusive'])

    def test_round_tens_duo_has_no_second_approximation(self):
        for task,amount in ((2147,89),(3597,139),(3638,379),(20808,259)):
            c,_,_=self.migrated(task)
            self.assertEqual(c['budget'][0]['expected']['value'],amount)

    def test_existing_budget_price_quality_is_note_not_double_weight(self):
        c,changes,_=self.migrated(497)
        self.assertFalse(any('经济实惠' in str(r['expected']) for _,r in m.allreq(c)))
        self.assertTrue(any(a['method']=='confirmed_non_scoring_qualitative_price_note' for a in changes))

    def test_percentage_has_correct_open_boundary_and_no_fake_count_unit(self):
        c,_,_=self.migrated(11435)
        r=next(r for _,r in m.allreq(c) if r['attribute']=='percentage')
        self.assertEqual(r['expected']['unit'],'percent')
        self.assertEqual(compare_value(r,{'type':'number','value':99,'unit':'percent'}),'fail')
        self.assertEqual(compare_value(r,{'type':'number','value':99.1,'unit':'percent'}),'pass')

    def test_unqualified_cun_not_silently_interpreted_as_inches(self):
        c,_,_=self.migrated(2976)
        self.assertTrue(any('5寸' in p['evidence']['text'] for p in c['pending_mentions']))
        self.assertFalse(any(r['expected'].get('unit')=='inch' for _,r in m.allreq(c)))

    def test_mirror_body_and_camera_exclusion_not_weakened(self):
        c,_,_=self.migrated(22795)
        self.assertTrue(any(r['subtype']=='components' and r['operator']=='excludes' and '镜头' in str(r['expected']) for _,r in m.allreq(c)))
        self.assertTrue(any(r['expected'].get('value')=='全画幅' for _,r in m.allreq(c)))

    def test_approximate_center_does_not_invent_tolerance(self):
        c,_,_=self.migrated(497)
        r=next(r for _,r in m.allreq(c) if r['operator']=='approximately')
        self.assertEqual(r['expected']['value'],16)
        self.assertEqual(compare_value(r,{'type':'number','value':16,'unit':'g'}),'pass')
        self.assertEqual(compare_value(r,{'type':'number','value':15.9,'unit':'g'}),'unknown')


    def test_numeric_sales_packs_do_not_create_order_counts(self):
        for task,count in ((11396,1),(18548,20)):
            c,_,_=self.migrated(task)
            self.assertTrue(any(r['subtype']=='packaging' and r['expected'].get('value')==count for _,r in m.allreq(c)))
            self.assertFalse(any(r['subtype']=='quantity' for _,r in m.allreq(c)))
            self.assertFalse(any('quantity_scope_pending'==f.get('code') for f in b.lint(c)))

    def test_softness_is_not_color_and_scratch_resistance_not_duplicated(self):
        c,_,_=self.migrated(21432)
        self.assertFalse(any(r['subtype']=='color' for _,r in m.allreq(c)))
        self.assertEqual(len([r for _,r in m.allreq(c) if r['expected'].get('value') in ('耐抓','耐猫抓')]),1)

    def test_independent_accessory_ambiguity_not_duplicated(self):
        c,_,_=self.migrated(247)
        p=[p for p in c['pending_mentions'] if p['reason']=='ambiguous_meaning']
        self.assertEqual(len(p),1)
        self.assertEqual(p[0]['importance'],'required')


if __name__=='__main__':unittest.main()
