import copy
import json
import unittest
from pathlib import Path
import semantic_contract_v2_r7_final12 as b
from semantic_final12_rules import checked_flags,coordination_proof
from semantic_product_verifier_v3 import snapshot,automatic_facts,compare_requirement,compare_value

PRIMARY={'kind':'primary_product','name':'主商品'}

def requirement(st,attr,expected,quote,op='eq',subject=None):
    return {'subtype':st,'attribute':attr,'subject':subject or PRIMARY,'importance':'required','operator':op,'expected':expected,
            'price_basis':None,'evidence':[{'text':quote}],'review_status':'pending'}
def raw(query,requirements,dim='specifications'):
    return {'schema_version':b.VERSION,'task_id':0,'query':query,**{d:requirements if d==dim else [] for d in b.DIMENSIONS},'pending_mentions':[]}
def small_product(label,*,title='商品',description='',attribute=None,price=12):
    return {'asin':'P','title':title,'category':'商品','full_description':description,'attribute':attribute or [],
            'customization_options':{'规格':[{'value':label,'price':price}]},'pricing':[price]}
def compare(r,p,dimension='specifications'):
    s=snapshot(p,{'规格':p['customization_options']['规格'][0]['value']});facts=automatic_facts(p,s)
    return compare_requirement(r,dimension,p,s,facts)


class ConfirmedContractPolicyTests(unittest.TestCase):
    def test_coordinated_lower_bounds_are_both_source_proven(self):
        q='至少三个usb接口和两个type-c接口'
        requirements=[requirement('attribute_count','other_feature_count',{'type':'number','value':n,'unit':'count'},q,'at_least',
                      {'kind':'component','name':name}) for n,name in [(3,'usb接口'),(2,'type-c接口')]]
        c,_=b.normalize(raw(q,requirements),{'task_id':0,'query':q})
        self.assertTrue(all(r['operator']=='at_least' for r in c['specifications']))
        requirements[1]['operator']='eq'
        with self.assertRaisesRegex(ValueError,'confirmed_coordinated_lower_bound'):b.normalize(raw(q,requirements),{'task_id':0,'query':q})

    def test_coordinated_proof_never_borrows_other_component_count(self):
        q='至少三个usb接口和两个type-c接口'
        r=requirement('attribute_count','other_feature_count',{'type':'number','value':3,'unit':'count'},q,'at_least',{'kind':'component','name':'type-c接口'})
        self.assertFalse(coordination_proof(r,b.chinese_number))

    def test_separate_clause_bound_does_not_leak(self):
        q='至少三个usb接口，但是恰好两个type-c接口'
        r=requirement('attribute_count','other_feature_count',{'type':'number','value':2,'unit':'count'},q,'at_least',{'kind':'component','name':'type-c接口'})
        self.assertFalse(coordination_proof(r,b.chinese_number))

    def test_adjacent_quantity_range_preserves_endpoints(self):
        q='需要三、四片'
        r=requirement('quantity','purchase_count',{'type':'range','lower':3,'upper':4,'unit':'piece','lower_inclusive':True,'upper_inclusive':True},q,'between')
        b.normalize(raw(q,[r]),{'task_id':0,'query':q})
        r['expected']['upper']=5
        with self.assertRaisesRegex(ValueError,'confirmed_adjacent_quantity_pair'):b.normalize(raw(q,[r]),{'task_id':0,'query':q})

    def test_symbolic_sale_basis_is_not_a_guessed_physical_denominator(self):
        q='单价在20块钱左右'
        r=requirement('unit_price','unit_price',{'type':'text','value':q},q)
        c,_=b.normalize(raw(q,[r],'budget'),{'task_id':0,'query':q})
        self.assertEqual(c['budget'][0]['price_basis'],{'amount':1,'unit':'sale_unit'})
        self.assertEqual(c['budget'][0]['expected']['value'],22)
        self.assertEqual(c['budget'][0]['importance'],'required')
        self.assertEqual(b.price_basis_from_phrase('每米单价20元'),{'amount':1,'unit':'m'})
        with self.assertRaises(ValueError):b.price_basis_from_phrase('单价每种颜色20元')

    def test_member_ranges_adhere_to_source_policy(self):
        q='包含4个单品，要分别含有二十多段、三十多段、四十多段、七十多段'
        v={'type':'member_ranges','member_attribute':'segment_count','unit':'count','groups':[{'lower':n+1,'upper':n+9,'count':1} for n in (20,30,40,70)]}
        r=requirement('composition','member_groups',v,q,'matches_members')
        b.normalize(raw(q,[r]),{'task_id':0,'query':q})
        r['expected']['groups'][0]['lower']=20
        with self.assertRaisesRegex(ValueError,'member_range_source'):b.normalize(raw(q,[r]),{'task_id':0,'query':q})


class ConfirmedProductComparisonTests(unittest.TestCase):
    def test_quantity_three_or_four(self):
        r=requirement('quantity','purchase_count',{'type':'range','lower':3,'upper':4,'unit':'piece','lower_inclusive':True,'upper_inclusive':True},'需要三、四片','between')
        for count,status in [(2,'fail'),(3,'pass'),(4,'pass'),(5,'fail')]:
            self.assertEqual(compare(r,small_product(f'保护膜{count}片装'))['status'],status)

    def test_explicit_unsharpened_and_sharpened_are_distinguished(self):
        r=requirement('condition','condition',{'type':'text','value':'未开刃'},'无需打磨刀刃')
        self.assertEqual(compare(r,small_product('白色',title='龙泉宝剑未开刃'))['status'],'pass')
        self.assertEqual(compare(r,small_product('已开刃款'))['status'],'fail')
        self.assertEqual(compare(r,small_product('白色',title='宝剑无需打磨'))['status'],'unknown')

    def test_blade_variant_prevents_common_claim_leak(self):
        p=small_product('已开刃款',title='宝剑未开刃')
        r=requirement('condition','condition',{'type':'text','value':'未开刃'},'未开刃')
        self.assertEqual(compare(r,p)['status'],'fail')

    def test_qualitative_claim_requires_literal_catalog_support(self):
        metal=requirement('style','style',{'type':'text','value':'金属质感'},'金属质感')
        soft=requirement('function','function',{'type':'text','value':'质感柔软'},'质感柔软')
        self.assertEqual(compare(metal,small_product('金色',title='铁艺货架'))['status'],'unknown')
        self.assertEqual(compare(metal,small_product('金色',description='本品表面具有金属质感'))['status'],'pass')
        self.assertEqual(compare(soft,small_product('TPU膜',attribute=['柔软']),'functions')['status'],'pass')
        self.assertEqual(compare(soft,small_product('TPU膜',description='不柔软'),'functions')['status'],'unknown')

    def test_member_matching_requires_all_distinct_groups(self):
        r=requirement('composition','member_groups',{'type':'member_ranges','member_attribute':'segment_count','unit':'count',
            'groups':[{'lower':n+1,'upper':n+9,'count':1} for n in (20,30,40,70)]},'源文','matches_members')
        for values,status in [([24,36,48,72],'pass'),([24,24,24,24],'fail'),([24,36,72],'fail'),([20,36,48,72],'fail'),([24,36,48,72,72],'fail')]:
            actual={'type':'member_values','member_attribute':'segment_count','unit':'count','values':values}
            self.assertEqual(compare_value(r,actual),status)
        self.assertEqual(compare_value(r,actual,complete=False),'unknown')

    def test_member_assignment_handles_overlap_without_greedy_error(self):
        r=requirement('composition','member_groups',{'type':'member_ranges','member_attribute':'segment_count','unit':'count',
             'groups':[{'lower':1,'upper':3,'count':1},{'lower':1,'upper':1,'count':1}]},'源文','matches_members')
        self.assertEqual(compare_value(r,{'type':'member_values','member_attribute':'segment_count','unit':'count','values':[1,2]}),'pass')

    def test_actual_member_list_never_comes_from_title_alone(self):
        title='儿童魔尺24段36段48段72段'
        p=small_product('超值彩虹全套【24+36+48+72】送教程书',title=title)
        s=snapshot(p,{'规格':p['customization_options']['规格'][0]['value']});f=automatic_facts(p,s)
        self.assertTrue(any(x['value']['type']=='member_values' and x['value']['values']==[24,36,48,72] for x in f))
        p=small_product('彩虹24段【送教程书】',title=title);s=snapshot(p,{'规格':p['customization_options']['规格'][0]['value']});f=automatic_facts(p,s)
        self.assertTrue(any(x['value']['type']=='member_values' and x['value']['values']==[24] for x in f))

    def test_symbolic_sale_unit_uses_catalog_evidence_and_handles_lots(self):
        r=requirement('unit_price','unit_price',{'type':'number','value':1,'unit':'CNY'},'单价为1块钱','at_most')
        r['price_basis']={'amount':1,'unit':'sale_unit'}
        good=compare(r,small_product('(20个)贴片AO3415',price=1),'budget')
        self.assertEqual(good['status'],'pass');self.assertEqual(good['actual']['value'],1)
        self.assertEqual(compare(r,small_product('(20个)贴片AO3415',price=2),'budget')['status'],'fail')
        self.assertEqual(compare(r,small_product('未说明销售单位',price=1),'budget')['status'],'unknown')

if __name__=='__main__':unittest.main()
