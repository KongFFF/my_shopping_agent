import copy
import unittest
import semantic_contract_v2_r7_final12 as b
import semantic_product_verifier_v4 as v
from semantic_history_adapter import canonicalize_purchase
from semantic_reward_history import score_verified
from freeze_historical_reward_audit import purchase_event


def product():
    return {'asin':'P','title':'衬衫','category':'衬衫','attribute':[],
            'customization_options':{'颜色分类':[{'value':'白色','price':50}],
                                     '占位':[{'value':'规格','price':50}]},'pricing':[50]}


def contract(query='衬衫，预算100元以内'):
    def r(st,attr,text,subject):
        return {'subtype':st,'attribute':attr,'expected':{'type':'text','value':text},'importance':'required',
                'operator':'eq','price_basis':None,'subject':subject,'evidence':[{'text':text}],'review_status':'pending'}
    c={'schema_version':b.VERSION,'task_id':0,'query':query,**{d:[] for d in b.DIMENSIONS},'pending_mentions':[]}
    c['category']=[r('product_category','product_category','衬衫',v.PRIMARY)]
    c['budget']=[r('total_price','total_price','预算100元以内',v.ORDER)]
    c,_=b.normalize(c,{'task_id':0,'query':query})
    for d in b.DIMENSIONS:
        for requirement in c[d]:requirement['review_status']='approved'
    b.validate(c,source_query=query,require_reviewed=True)
    return c


class HistoricalExecutionTests(unittest.TestCase):
    def test_only_completed_matching_buy_event_is_executed(self):
        p={'asin':'P','options':{'颜色分类':'白色'}}
        row={'terminal_result':{'done':True,'purchase':p},'steps':[{'tool_name':'buy_now','result':{'done':True,'purchase':p}}]}
        self.assertTrue(purchase_event(row)['purchase_executed'])
        row['steps'][0]['result']['purchase']={'asin':'other','options':{}}
        self.assertFalse(purchase_event(row)['purchase_executed'])

    def test_guard_exhaustion_is_not_a_purchase(self):
        row={'terminal_result':{'done':True,'purchase':{'asin':'P','options':{}}},'steps':[{'tool_name':'buy_now',
             'result':{'done':True,'reward_detail':{'evidence':{'purchase_guard':{'status':'exhausted'}}}}}]}
        self.assertFalse(purchase_event(row)['purchase_executed'])
        self.assertEqual(purchase_event({'terminal_result':{}})['status'],'no_purchase')

    def test_unique_environment_normalization_and_ambiguity(self):
        p=product();p['customization_options']['尺码']=[{'value':'M','price':50}]
        mapped=canonicalize_purchase(p,{'尺码':'m'})
        self.assertEqual(mapped['selected_options'],{'尺码':'M'})
        p['customization_options']['尺码'].append({'value':'ｍ','price':50})
        self.assertTrue(canonicalize_purchase(p,{'尺码':'m'})['errors'])

    def test_unknown_value_never_chooses_nearest_sku(self):
        self.assertTrue(canonicalize_purchase(product(),{'颜色分类':'红色'})['errors'])

    def test_completed_purchase_does_not_require_constant_placeholder(self):
        c=contract();p=product()
        results=[]
        for options in ({'颜色分类':'白色'},{'颜色分类':'白色','占位':'规格'}):
            result=v.verify(c,p,options,purchase_executed=True)
            results.append(score_verified(c,result,source_query=c['query']))
        self.assertEqual(results[0]['reward'],1)
        self.assertEqual(results[0],results[1])
        self.assertFalse(v.verify(c,p,{'颜色分类':'白色'},purchase_executed=True)['selection_complete'])

    def test_no_execution_cannot_receive_purchase_reward(self):
        c=contract();result=v.verify(c,product(),{'颜色分类':'白色','占位':'规格'},purchase_executed=False)
        self.assertIsNone(score_verified(c,result,source_query=c['query'])['reward'])

    def test_missing_physical_fact_remains_unknown(self):
        c=contract();r=copy.deepcopy(c['category'][0]);r.update(id='unproven',subtype='function',attribute='function',
           expected={'type':'text','value':'防水'})
        # Compare the product fact atom without weakening the whole-query gate.
        s=v.snapshot(product(),{'颜色分类':'白色'})
        self.assertEqual(v.compare_requirement(r,'functions',product(),s,v.automatic_facts(product(),s))['status'],'unknown')

    def test_changed_contract_cannot_reuse_verification(self):
        c=contract();result=v.verify(c,product(),{'颜色分类':'白色'},purchase_executed=True)
        c['task_id']=1
        with self.assertRaisesRegex(ValueError,'bound'):score_verified(c,result,source_query=c['query'])


class HistoricalProductTests(unittest.TestCase):
    def test_color_shade_is_not_an_unsupported_failure_or_success(self):
        c=contract();r=copy.deepcopy(c['category'][0]);r.update(subtype='color',attribute='color',expected={'type':'text','value':'白色'})
        p=product();p['customization_options']['颜色分类'][0]['value']='米白色'
        s=v.snapshot(p,{'颜色分类':'米白色'})
        self.assertEqual(v.compare_requirement(r,'specifications',p,s,v.automatic_facts(p,s))['status'],'unknown')
        r['expected']['value']='灰色'
        self.assertEqual(v.compare_requirement(r,'specifications',p,s,v.automatic_facts(p,s))['status'],'fail')

    def test_shoe_size_reads_only_explicit_sizing_axis(self):
        p=product();p['customization_options']['鞋码']=[{'value':'40','price':50}]
        s=v.snapshot(p,{'鞋码':'40'})
        self.assertIn({'type':'text','value':'40'},[f['value'] for f in v.automatic_facts(p,s) if f['subtype']=='size'])
        p['customization_options']['颜色分类'][0]['value']='40'
        s=v.snapshot(p,{'颜色分类':'40'})
        self.assertFalse(any(f['subtype']=='size' for f in v.automatic_facts(p,s)))


if __name__=='__main__':unittest.main()
