import copy
import unittest
from semantic_product_verifier_v2 import snapshot,automatic_facts,compare_requirement,compare_value,reviewed_facts,fingerprint
from semantic_catalog_rules_v2 import selected_facts
from semantic_fact_language_v2 import canonical

PRIMARY={'kind':'primary_product','name':'主商品'}
ORDER={'kind':'order','name':'本次订单'}
def req(st,attr,value,op='eq',subject=None):
    return {'subtype':st,'attribute':attr,'subject':subject or PRIMARY,'importance':'required','operator':op,'expected':value,'price_basis':None}
def product(value='12g*10包',axis='净含量'):
    return {'asin':'P','title':'商品','category':'目录','customization_options':{axis:[{'value':value,'price':12}]},'pricing':[12]}
def state_facts(p,options):
    state=snapshot(p,options);return state,automatic_facts(p,state)


class ProductV2Tests(unittest.TestCase):
    def test_explicit_multiplication_yields_total_and_count_without_query(self):
        p=product();s,facts=state_facts(p,{'净含量':'12g*10包'})
        mass=req('quantity','purchase_mass',{'type':'number','value':120,'unit':'g'},subject=ORDER)
        count=req('quantity','purchase_count',{'type':'number','value':10,'unit':'bag'})
        self.assertEqual(compare_requirement(mass,'specifications',p,s,facts)['status'],'pass')
        self.assertEqual(compare_requirement(count,'specifications',p,s,facts)['status'],'pass')
        mass['expected']['value']=1200
        self.assertEqual(compare_requirement(mass,'specifications',p,s,facts)['status'],'fail')

    def test_ambiguous_mass_count_not_multiplied(self):
        p=product('120g共10包');s,f=state_facts(p,{'净含量':'120g共10包'})
        self.assertFalse(any(x['attribute']=='purchase_mass' for x in f))

    def test_multiple_masses_not_multiplied_as_dimensions(self):
        p=product('350g*14条*25克');s,f=state_facts(p,{'净含量':'350g*14条*25克'})
        self.assertFalse(any(x['attribute']=='purchase_mass' for x in f))

    def test_physical_units_and_wearer_range(self):
        p=product('XL 130-150斤','尺码');s,f=state_facts(p,{'尺码':'XL 130-150斤'})
        r=req('size','wearer_weight_limit',{'type':'text','value':'适合140斤的人穿'})
        self.assertEqual(compare_requirement(r,'specifications',p,s,f)['status'],'pass')
        r['expected']['value']='适合160斤的人穿'
        self.assertEqual(compare_requirement(r,'specifications',p,s,f)['status'],'fail')

    def test_explicit_size_alias(self):
        p=product('2XL','尺码');s,f=state_facts(p,{'尺码':'2XL'})
        r=req('size','size_label',{'type':'text','value':'XXL'})
        self.assertEqual(compare_requirement(r,'specifications',p,s,f)['status'],'pass')

    def test_light_color_not_body_color(self):
        p=product('圆球灯【白光】','颜色分类');s,f=state_facts(p,{'颜色分类':'圆球灯【白光】'})
        r=req('color','color',{'type':'text','value':'白色'})
        self.assertEqual(compare_requirement(r,'specifications',p,s,f)['status'],'unknown')

    def test_negative_color_not_positive_fact(self):
        p=product('非黑色款','颜色分类');s,f=state_facts(p,{'颜色分类':'非黑色款'})
        self.assertFalse(any(x['subtype']=='color' for x in f))

    def test_subjective_color_not_disproved_by_literal_label(self):
        for label,want in [('象牙色遮瑕款','偏白'),('02自然色','看起来像是自己皮肤透出来的好气色')]:
            p=product(label,'颜色分类');s,f=state_facts(p,{'颜色分类':label})
            self.assertEqual(compare_requirement(req('color','color',{'type':'text','value':want}),'specifications',p,s,f)['status'],'unknown')
        p=product('黑色','颜色分类');s,f=state_facts(p,{'颜色分类':'黑色'})
        self.assertEqual(compare_requirement(req('color','color',{'type':'text','value':'白色'}),'specifications',p,s,f)['status'],'fail')

    def test_selected_mass_range_is_compatible(self):
        p=product('15g');s,f=state_facts(p,{'净含量':'15g'})
        r=req('net_content','mass',{'type':'range','lower':14,'upper':16,'unit':'g','lower_inclusive':True,'upper_inclusive':True},'between')
        self.assertEqual(compare_requirement(r,'specifications',p,s,f)['status'],'pass')

    def test_square_dimensions_need_explicit_axis_names(self):
        p=product('边长25CM*高20cm','颜色分类');s,f=state_facts(p,{'颜色分类':'边长25CM*高20cm'})
        self.assertTrue(any(x['attribute']=='width' and x['value']['value']==25 for x in f))
        p=product('1500mm*2000mm','尺寸');s,f=state_facts(p,{'尺寸':'1500mm*2000mm'})
        self.assertFalse(any(x['subtype']=='dimensions' for x in f))

    def test_packaging_literal_not_ignored(self):
        p=product('3瓶');s,f=state_facts(p,{'净含量':'3瓶'})
        r=req('packaging','packaging',{'type':'text','value':'三瓶装'})
        self.assertEqual(compare_requirement(r,'specifications',p,s,f)['status'],'pass')

    def test_unit_price_uses_actual_sale_count(self):
        p=product('3瓶');s,f=state_facts(p,{'净含量':'3瓶'})
        r=req('unit_price','unit_price',{'type':'number','value':5,'unit':'CNY'},'at_most',subject=ORDER)
        r['price_basis']={'amount':1,'unit':'bottle'}
        result=compare_requirement(r,'budget',p,s,f)
        self.assertEqual(result['status'],'pass');self.assertEqual(result['actual']['value'],4)
        r['price_basis']=None
        self.assertEqual(compare_requirement(r,'budget',p,s,f)['status'],'unknown')

    def test_weak_claims_not_upgraded(self):
        self.assertNotEqual(canonical('function','防水'),canonical('function','抗水浸'))
        self.assertNotEqual(canonical('function','持久'),canonical('function','持久一整天'))
        self.assertNotEqual(canonical('condition','正品'),canonical('condition','官方认证的正品'))
        self.assertNotEqual(canonical('material','铜'),canonical('material','铜芯'))

    def test_partial_selector_applies_only_when_evidence_axis_selected(self):
        p=product('3瓶');p['customization_options']['颜色']=[{'value':'黑色','price':12}]
        record={'product_asin':'P','product_sha256':fingerprint(p),'scope':'sku','selector_mode':'contains','options':{'净含量':'3瓶'},
          'dimension':'specifications','subtype':'packaging','attribute':'pack_count','subject':PRIMARY,
          'value':{'type':'number','value':3,'unit':'bottle'},'complete':True,'review_status':'approved',
          'evidence':[{'pointer':'/customization_options/净含量/0/value','value':'3瓶'}]}
        s=snapshot(p,{'净含量':'3瓶','颜色':'黑色'});accepted,rejected=reviewed_facts(p,s,[record]);self.assertEqual(len(accepted),1);self.assertFalse(rejected)
        s=snapshot(p,{'颜色':'黑色'});accepted,rejected=reviewed_facts(p,s,[record]);self.assertFalse(accepted);self.assertFalse(rejected)
        bad=copy.deepcopy(record);bad['options']={'颜色':'黑色'}
        accepted,rejected=reviewed_facts(p,s,[bad]);self.assertFalse(accepted);self.assertTrue(rejected)

if __name__=='__main__':unittest.main()
