import copy
import unittest
from semantic_contract_v2_r7_closeout import normalize, VERSION, budget_amount
from semantic_product_verifier import verify as verify_product, compare_value, snapshot, fingerprint
from semantic_reward_r7_closeout import score_verified


def verify(*args,**kwargs):
    # Tests below explicitly simulate a completed purchase. Production entry
    # defaults to diagnostic-only unless the caller proves execution.
    kwargs.setdefault('purchase_executed',True)
    return verify_product(*args,**kwargs)


def product():
    return {'asin':'A','title':'压脚买10送1 黑色 红色','category':'配件›压脚','attribute':['全钢'],
            'customization_options':{'套餐':[{'value':'压脚1个','price':9},{'value':'压脚买10送1','price':90}],
                                     '颜色分类':[{'value':'规格','price':90}]},'pricing':[9,90]}


def contract(*,approve=False):
    query='买全钢压脚，买10送1，预算100元以内。'
    raw={'schema_version':VERSION,'task_id':0,'query':query,**{d:[] for d in ('category','budget','functions','specifications','brand','model')},'pending_mentions':[]}
    def add(d,s,a,e,quote,operator='eq'):
        raw[d].append({'subtype':s,'attribute':a,'subject':{'kind':'order','name':'本次订单'} if d=='budget' else {'kind':'primary_product','name':'主商品'},
                       'importance':'required','operator':operator,'expected':e,'price_basis':None,'evidence':[{'text':quote}],'review_status':'pending'})
    add('category','product_category','product_category',{'type':'text','value':'压脚'},'压脚')
    add('specifications','material','material',{'type':'text','value':'全钢'},'全钢')
    add('specifications','bundle','promotion',{'type':'bundle','paid_quantity':10,'gift_quantity':1,'received_quantity':None,'unit':'piece','same_item':True},'买10送1','matches_bundle')
    add('budget','total_price','total_price',{'type':'text','value':'预算100元以内'},'预算100元以内')
    c,_=normalize(raw,{'task_id':0,'query':query})
    if approve:
        for d in ('category','budget','functions','specifications','brand','model'):
            for r in c[d]:r['review_status']='approved'
    return c


def bundle_fact(p,options):
    return {'product_asin':p['asin'],'product_sha256':fingerprint(p),'scope':'sku','options':options,'review_status':'approved',
            'dimension':'specifications','subtype':'bundle','attribute':'promotion','subject':{'kind':'primary_product','name':'主商品'},
            'value':{'type':'bundle','paid_quantity':10,'gift_quantity':1,'received_quantity':11,'unit':'piece','same_item':True},
            'complete':True,'evidence':[{'pointer':'/customization_options/套餐/1/value','value':'压脚买10送1'}]}


class ProductVerifierTests(unittest.TestCase):
    def test_unselected_promotion_in_title_does_not_pass(self):
        result=verify(contract(),product(),{'套餐':'压脚1个','颜色分类':'规格'})
        promo=next(r for r in result['comparisons'].values() if r['subtype']=='bundle')
        self.assertEqual(promo['status'],'fail')

    def test_promotion_unit_and_same_item_not_invented(self):
        result=verify(contract(),product(),{'套餐':'压脚买10送1','颜色分类':'规格'})
        promo=next(r for r in result['comparisons'].values() if r['subtype']=='bundle')
        self.assertEqual(promo['status'],'unknown')

    def test_placeholder_does_not_change_semantic_status(self):
        c=contract();p=product()
        a=verify(c,p,{'套餐':'压脚1个'});b=verify(c,p,{'套餐':'压脚1个','颜色分类':'规格'})
        self.assertEqual(a['statuses'],b['statuses']);self.assertFalse(a['execution_complete']);self.assertTrue(b['execution_complete'])

    def test_actual_variant_price_not_product_maximum(self):
        result=verify(contract(),product(),{'套餐':'压脚1个','颜色分类':'规格'})
        self.assertEqual(result['snapshot']['price_resolution']['price'],9)

    def test_invalid_option_never_uses_a_price(self):
        result=verify(contract(),product(),{'套餐':'不存在','颜色分类':'规格'})
        self.assertTrue(result['snapshot']['selection_errors']);self.assertEqual(set(result['statuses'].values()),{'unknown'})

    def test_requested_count_not_multiplied_into_sale_variant_price(self):
        state=snapshot(product(),{'套餐':'压脚买10送1','颜色分类':'规格'})
        self.assertEqual(state['price_resolution']['price'],90)

    def test_unit_conversion_and_dimension_mismatch(self):
        r={'operator':'eq','expected':{'type':'number','value':1000,'unit':'g'}}
        self.assertEqual(compare_value(r,{'type':'number','value':1,'unit':'kg'}),'pass')
        self.assertEqual(compare_value(r,{'type':'number','value':1,'unit':'ml'}),'unknown')

    def test_open_world_absence_not_failure(self):
        r={'operator':'contains_all','expected':{'type':'components','items':[{'name':'镜头','quantity':None,'unit':None}]}}
        self.assertEqual(compare_value(r,{'type':'components','items':[]},complete=False),'unknown')
        self.assertEqual(compare_value(r,{'type':'components','items':[]},complete=True),'fail')

    def test_exclusion_is_not_absence_in_open_fact_list(self):
        r={'operator':'excludes','expected':{'type':'text_list','values':['镜头']}}
        self.assertEqual(compare_value(r,{'type':'text_list','values':['机身']},complete=False),'unknown')

    def test_reviewed_fact_is_bound_to_product_and_sku(self):
        p=product();options={'套餐':'压脚买10送1','颜色分类':'规格'};f=bundle_fact(p,options)
        result=verify(contract(),p,options,catalog_facts=[f])
        self.assertEqual(next(r['status'] for r in result['comparisons'].values() if r['subtype']=='bundle'),'pass')
        result=verify(contract(),p,{'套餐':'压脚1个','颜色分类':'规格'},catalog_facts=[f])
        self.assertTrue(result['rejected_catalog_facts'])

    def test_independent_positive_functions_do_not_conflict(self):
        from semantic_product_verifier import compare_requirement
        p=product();s=snapshot(p,{'套餐':'压脚1个','颜色分类':'规格'})
        req={'subtype':'function','attribute':'function','subject':{'kind':'primary_product','name':'主商品'},'importance':'required',
             'operator':'eq','expected':{'type':'text','value':'防水'}}
        facts=[{'dimension':'functions',**{k:req[k] for k in ('subtype','attribute','subject')},
                'value':{'type':'text','value':v},'evidence':[],'origin':'reviewed_public_catalog_fact','complete':True} for v in ('防水','防尘')]
        self.assertEqual(compare_requirement(req,'functions',p,s,facts)['status'],'pass')
        self.assertEqual(compare_requirement(req,'functions',p,s,facts[1:])['status'],'unknown')

    def test_target_instructions_and_fake_evidence_rejected(self):
        p=product();p['instructions']=[{'value':'买10送1'}];opts={'套餐':'压脚买10送1','颜色分类':'规格'}
        f=bundle_fact(p,opts);f['evidence']=[{'pointer':'/instructions/0/value','value':'买10送1'}]
        self.assertTrue(verify(contract(),p,opts,catalog_facts=[f])['rejected_catalog_facts'])

    def test_fact_cannot_cite_an_unselected_option(self):
        p=product();opts={'套餐':'压脚1个','颜色分类':'规格'}
        f=bundle_fact(p,opts)
        self.assertTrue(verify(contract(),p,opts,catalog_facts=[f])['rejected_catalog_facts'])
        f=bundle_fact(p,opts);f['evidence'][0]['value']='错误证据'
        self.assertTrue(verify(contract(),p,opts,catalog_facts=[f])['rejected_catalog_facts'])

    def test_drafts_not_automatically_approved_for_reward(self):
        c=contract();result=verify(c,product(),{'套餐':'压脚买10送1','颜色分类':'规格'})
        self.assertFalse(result['reward_ready']);self.assertIsNone(result['reward'])
        with self.assertRaises(ValueError):score_verified(c,result,source_query=c['query'])

    def test_complete_options_do_not_prove_purchase_execution(self):
        c=contract(approve=True);p=product();opts={'套餐':'压脚买10送1','颜色分类':'规格'}
        result=verify_product(c,p,opts,catalog_facts=[bundle_fact(p,opts)])
        self.assertTrue(result['selection_complete']);self.assertFalse(result['execution_complete'])
        self.assertFalse(result['reward_ready'])
        self.assertFalse(score_verified(c,result,source_query=c['query'])['reward_valid'])

    def test_verified_approved_contract_scores_and_identity_not_penalized(self):
        c=contract(approve=True);p=product();opts={'套餐':'压脚买10送1','颜色分类':'规格'}
        rewards=[]
        for asin in ('A','B'):
            p['asin']=asin;f=bundle_fact(p,opts);r=verify(c,p,opts,catalog_facts=[f])
            rewards.append(score_verified(c,r,source_query=c['query'])['reward'])
        self.assertEqual(rewards,[1,1])

    def test_wrong_bundle_scores_below_complete_bundle(self):
        c=contract(approve=True);p=product();opts={'套餐':'压脚买10送1','颜色分类':'规格'}
        good=score_verified(c,verify(c,p,opts,catalog_facts=[bundle_fact(p,opts)]),source_query=c['query'])
        bad=score_verified(c,verify(c,p,{'套餐':'压脚1个','颜色分类':'规格'}),source_query=c['query'])
        self.assertTrue(good['semantic_success']);self.assertFalse(bad['semantic_success']);self.assertLess(bad['reward'],good['reward'])

    def test_bound_contract_cannot_be_changed_after_verification(self):
        c=contract(approve=True);p=product();opts={'套餐':'压脚买10送1','颜色分类':'规格'}
        result=verify(c,p,opts,catalog_facts=[bundle_fact(p,opts)])
        c['specifications'][0]['expected']['value']='钢'
        with self.assertRaises(ValueError):score_verified(c,result,source_query=c['query'])

    def test_explicit_budget_failure_is_lowest_tier_even_with_other_unknowns(self):
        c=contract(approve=True);p=product();p['customization_options']['套餐'][1]['price']=200
        result=verify(c,p,{'套餐':'压脚买10送1','颜色分类':'规格'})
        score=score_verified(c,result,source_query=c['query'])
        self.assertEqual(score['reward'],-0.85);self.assertFalse(score['semantic_success'])

    def test_preference_changes_success_reward_but_not_success(self):
        c=contract(approve=True);start=len(c['query']);c['query']+='最好黑色。'
        c['specifications'].append({'id':'preference1','subtype':'color','attribute':'color','subject':{'kind':'primary_product','name':'主商品'},
            'importance':'preference','operator':'eq','expected':{'type':'text','value':'黑色'},'price_basis':None,'review_status':'approved',
            'evidence':[{'text':'最好黑色','start':start,'end':start+4}]})
        p=product();p['customization_options']['颜色分类']=[{'value':'黑色','price':90},{'value':'白色','price':90}]
        scores=[]
        for color in ('白色','黑色'):
            opts={'套餐':'压脚买10送1','颜色分类':color}
            scores.append(score_verified(c,verify(c,p,opts,catalog_facts=[bundle_fact(p,opts)]),source_query=c['query']))
        self.assertTrue(all(s['semantic_success'] for s in scores))
        self.assertEqual([s['reward'] for s in scores],[0.55,1.0])


class ColloquialBudgetTests(unittest.TestCase):
    def test_confirmed_upper_no_approximation_factor(self):
        for phrase,limit in [('十几块钱左右',19),('预算四十几块',49),('预算三四十元左右',49)]:
            self.assertEqual(budget_amount(phrase)[1]['value'],limit)

    def test_product_quantity_not_parsed_as_money(self):
        with self.assertRaises(ValueError):budget_amount('需要十几包')

    def test_explicit_budget_policy_unchanged(self):
        self.assertEqual(budget_amount('20元左右')[1]['value'],22)
        self.assertEqual(budget_amount('预算100元以内')[1]['value'],100)

if __name__=='__main__':unittest.main()
