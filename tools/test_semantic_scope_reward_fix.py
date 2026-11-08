import copy
import itertools
import unittest
import semantic_contract_v2_r7_final12 as b
import semantic_product_verifier_v4 as v
from semantic_reward_history import score_verified
from semantic_source_scope_proofs import filter_flags,square_proof,explicit_claim_standard
from semantic_fact_language_v3 import canonical
from test_historical_semantic_reward import contract,product
from test_semantic_final12 import raw,requirement


def bundle(query,paid=10,gift=1,unit='piece',pending=None):
    phrase=query.split('。')[-1] if '。' in query else query
    c=raw(query,[requirement('bundle','promotion',{'type':'bundle','paid_quantity':paid,'gift_quantity':gift,
        'received_quantity':paid+gift,'unit':unit,'same_item':True},phrase,'matches_bundle')])
    if pending:c['pending_mentions']=[{'evidence':{'text':pending},'reason':'ambiguous_meaning','importance':'uncertain'}]
    return c


def add_pending(c,text):
    pos=c['query'].index(text)
    c['pending_mentions'].append({'evidence':{'text':text,'start':pos,'end':pos+len(text)},
                                 'reason':'unverifiable_subjective','importance':'uncertain'})


def priced(amount):
    p=product()
    for entries in p['customization_options'].values():
        for e in entries:e['price']=amount
    p['pricing']=[amount]
    return p


def verify(c,p=None):return v.verify(c,p or product(),{'颜色分类':'白色'},purchase_executed=True)


class ScopeProofTests(unittest.TestCase):
    def test_promotion_paid_count_does_not_require_second_order_count(self):
        q='买10个要能送1个';c,a=b.normalize(bundle(q),{'task_id':0,'query':q})
        self.assertFalse(any(r['subtype']=='quantity' for r in c['specifications']))
        self.assertTrue(any(x.get('rule')=='promotion_paid_count_is_not_order_count' for x in a))

    def test_chinese_promotion_counts_preserve_physical_unit(self):
        q='买十瓶送两瓶';c,_=b.normalize(bundle(q,10,2,'bottle'),{'task_id':0,'query':q})
        self.assertEqual(c['specifications'][0]['expected']['gift_quantity'],2)

    def test_wrong_bundle_values_or_units_cannot_prove_source_count(self):
        q='买10个要能送1个'
        with self.assertRaises(ValueError):b.normalize(bundle(q,9,1),{'task_id':0,'query':q})
        with self.assertRaises(ValueError):b.normalize(bundle(q,10,1,'bottle'),{'task_id':0,'query':q})

    def test_unrelated_buy_count_still_must_be_retained(self):
        q='买5个。买10个要能送1个'
        with self.assertRaisesRegex(ValueError,'purchase_quantity_coverage'):b.normalize(bundle(q),{'task_id':0,'query':q})

    def test_ambiguous_introduction_stays_pending(self):
        q='需要一个压脚。买10个要能送1个'
        c,_=b.normalize(bundle(q,pending='需要一个'),{'task_id':0,'query':q})
        self.assertTrue(c['pending_mentions'])
        with self.assertRaises(ValueError):b.validate(c,source_query=q,require_reviewed=True)

    def test_pending_does_not_hide_unambiguous_buy_count(self):
        q='买5个';p={'reason':'ambiguous_meaning','evidence':{'text':q,'start':0,'end':len(q)}}
        f={'code':'purchase_quantity_coverage','location':'/query','source':{'count':5,'source':q,'start':1,'end':3}}
        c={d:[] for d in b.DIMENSIONS};c.update(query=q,pending_mentions=[p])
        self.assertEqual(filter_flags(c,[f],b.chinese_number)[0]['code'],'purchase_quantity_coverage')

    def test_confirmed_intro_and_bundle_resolves_only_selection_ambiguity(self):
        q='需要一个压脚。买10个要能送1个'
        c=bundle(q,pending='需要一个')
        c['category']=[requirement('product_category','product_category',{'type':'text','value':'压脚'},'压脚')]
        n,a=b.normalize(c,{'task_id':0,'query':q})
        self.assertFalse(n['pending_mentions'])
        self.assertTrue(any(x['kind']=='user_confirmed_selection_intro_resolution' for x in a))
        for d in b.DIMENSIONS:
            for r in n[d]:r['review_status']='approved'
        b.validate(n,source_query=q,require_reviewed=True)

    def test_explicit_only_one_and_different_counter_still_pending(self):
        for q,paid,unit in [('只需要一个压脚。买10个要能送1个',10,'piece'),('需要一个压脚。买10瓶送1瓶',10,'bottle')]:
            c=bundle(q,paid,1,unit,pending='需要一个')
            c['category']=[requirement('product_category','product_category',{'type':'text','value':'压脚'},'压脚')]
            n,_=b.normalize(c,{'task_id':0,'query':q})
            self.assertTrue(n['pending_mentions'])

    def test_additional_order_count_is_not_removed_by_intro_policy(self):
        q='需要一个压脚。买3个。买10个要能送1个'
        c=bundle(q,pending='需要一个')
        c['category']=[requirement('product_category','product_category',{'type':'text','value':'压脚'},'压脚')]
        with self.assertRaisesRegex(ValueError,'purchase_quantity_coverage'):b.normalize(c,{'task_id':0,'query':q})

    def test_equal_two_axis_measurements_have_a_source_proof(self):
        q='20厘米×200毫米';r=requirement('dimensions','width',{'type':'number','value':20,'unit':'cm'},q)
        r['evidence'][0].update(start=0,end=len(q))
        self.assertIsNotNone(square_proof(r,b.chinese_number))
        r['attribute']='height';self.assertIsNone(square_proof(r,b.chinese_number))

    def test_non_square_or_three_axes_not_inferred(self):
        for q in ('20厘米×30厘米','20厘米×20厘米×10厘米','约20厘米×20厘米'):
            r=requirement('dimensions','length',{'type':'number','value':20,'unit':'cm'},q)
            self.assertIsNone(square_proof(r,b.chinese_number))

    def test_confirmed_money_abbreviation_and_existing_budget_policy(self):
        for phrase,upper in [('预算四百五左右',495),('预算三千五左右',3850),('价格两百五以内',250)]:
            op,v,rule=b.budget_amount(phrase)
            self.assertEqual((op,v['value']),('at_most',upper));self.assertIn('user_confirmed_elided_money',rule)
        q='价格别太贵，大概四百五左右。'
        self.assertEqual(b.budget_amount(b.budget_source_clause(q,'大概四百五左右'))[1]['value'],495)

    def test_amount_policy_does_not_change_non_money_quantity(self):
        with self.assertRaises(ValueError):b.chinese_number('四百五')
        with self.assertRaises(ValueError):b.budget_amount('四百五左右')
        self.assertEqual(b.budget_amount('预算四百五十元以内')[1]['value'],450)
        self.assertEqual(b.budget_amount('预算二十出头')[1]['value'],24)

    def test_synonyms_never_turn_quality_or_purity_into_proof(self):
        self.assertEqual(canonical('customization','可定做'),canonical('customization','可定制'))
        self.assertNotEqual(canonical('function','轻量'),canonical('function','轻薄'))
        self.assertNotEqual(canonical('function','加厚'),canonical('function','厚实'))
        self.assertNotEqual(canonical('composition','足金'),canonical('composition','含金量大于99.9%'))

    def test_qualitative_standard_keeps_requirement_and_audits_resolution(self):
        q='复古风';c=raw(q,[requirement('style','style',{'type':'text','value':q},q)])
        c['pending_mentions']=[{'evidence':{'text':q},'reason':'unverifiable_subjective','importance':'uncertain'}]
        n,a=b.normalize(c,{'task_id':0,'query':q})
        self.assertEqual(n['specifications'][0]['importance'],'required')
        self.assertEqual(n['specifications'][0]['review_status'],'pending')
        self.assertFalse(n['pending_mentions'])
        self.assertTrue(any(r['kind']=='user_confirmed_public_claim_standard' for r in a))
        self.assertEqual(v.compare_value(n['specifications'][0],{'type':'text','value':'复古'}),'pass')

    def test_qualitative_standard_does_not_resolve_strength_or_approximation(self):
        q='希望复古风';c=raw(q,[requirement('style','style',{'type':'text','value':'复古风'},q)])
        n,_=b.normalize(c,{'task_id':0,'query':q})
        self.assertEqual(n['specifications'][0]['importance'],'uncertain')
        self.assertTrue(any(p['reason']=='ambiguous_strength' for p in n['pending_mentions']))
        q='规格500g左右';c=raw(q,[requirement('net_content','mass',{'type':'text','value':'500g左右'},q)])
        n,_=b.normalize(c,{'task_id':0,'query':q})
        self.assertEqual(n['specifications'][0]['importance'],'uncertain')
        self.assertTrue(any(p['reason']=='undefined_tolerance' for p in n['pending_mentions']))

    def test_postfix_preference_is_not_silently_promoted_to_required(self):
        q='玻璃钢一类的优先考虑';r=requirement('material','material',{'type':'text','value':'玻璃钢'},q)
        r['importance']='preference';n,_=b.normalize(raw(q,[r]),{'task_id':0,'query':q})
        n['specifications'][0]['review_status']='approved'
        b.validate(n,source_query=q,require_reviewed=True)
        self.assertEqual(n['specifications'][0]['importance'],'preference')

    def test_quality_guarantee_means_explicit_claim_not_verified_quality(self):
        q='质量要有保障';r=requirement('function','function',{'type':'text','value':'质量有保障'},q)
        c=raw(q,[r],'functions')
        c['pending_mentions']=[{'evidence':{'text':q},'reason':'unverifiable_subjective','importance':'uncertain'}]
        n,_=b.normalize(c,{'task_id':0,'query':q})
        self.assertFalse(n['pending_mentions'])
        self.assertEqual(v.compare_value(n['functions'][0],{'type':'text','value':'品质保证'}),'pass')
        self.assertEqual(v.compare_value(n['functions'][0],{'type':'text','value':'加厚'},complete=False),'unknown')

    def test_negative_or_rumored_source_cannot_establish_positive_claim_standard(self):
        for quote in ('不要复古风','不是复古风','听说复古风'):
            r=requirement('style','style',{'type':'text','value':'复古风'},quote)
            self.assertFalse(explicit_claim_standard(r))


class EvidenceRewardTests(unittest.TestCase):
    def test_proven_budget_failure_survives_unrelated_pending(self):
        c=contract('衬衫，预算100元以内，图案要时髦');add_pending(c,'时髦')
        score=score_verified(c,verify(c,priced(150)),source_query=c['query'])
        self.assertEqual(score['reward'],-0.85);self.assertFalse(score['contract_fully_reviewed'])

    def test_pending_on_budget_itself_blocks_certificate(self):
        c=contract();add_pending(c,'预算100元以内')
        self.assertIsNone(score_verified(c,verify(c,priced(150)),source_query=c['query'])['reward'])

    def test_unreviewed_budget_cannot_issue_hard_failure(self):
        c=contract();c['budget'][0]['review_status']='pending'
        self.assertIsNone(score_verified(c,verify(c,priced(150)),source_query=c['query'])['reward'])

    def test_pending_without_proven_hard_failure_never_gets_reward(self):
        c=contract('衬衫，预算100元以内，图案要时髦');add_pending(c,'时髦')
        self.assertIsNone(score_verified(c,verify(c),source_query=c['query'])['reward'])

    def test_invalid_fact_or_no_execution_still_blocks_hard_failure(self):
        c=contract();result=verify(c,priced(150));result['readiness_blockers'].append({'reason':'rejected_catalog_facts'})
        self.assertIsNone(score_verified(c,result,source_query=c['query'])['reward'])
        result=verify(c,priced(150));result['execution_complete']=False
        self.assertIsNone(score_verified(c,result,source_query=c['query'])['reward'])

    def rich_contract(self):
        q='衬衫，预算100元以内，需要防水，需要防尘，最好红色'
        c=contract();c['query']=q
        for index,text in enumerate(('防水','防尘','红色')):
            e={'text':text,'start':q.index(text),'end':q.index(text)+len(text)}
            c['functions' if index<2 else 'specifications'].append({
                'id':'extra'+str(index),'subtype':'function' if index<2 else 'color','attribute':'function' if index<2 else 'color',
                'subject':v.PRIMARY,'expected':{'type':'text','value':text},'operator':'eq','price_basis':None,
                'importance':'required' if index<2 else 'preference','evidence':[e],'review_status':'approved'})
        b.validate(c,source_query=q,require_reviewed=True)
        return c

    def test_reward_bounds_enclose_every_assignment_with_unknown_required(self):
        c=self.rich_contract();result=verify(c)
        for key in ('extra0','extra1','extra2'):result['statuses'][key]='unknown'
        uncertain=score_verified(c,result,source_query=c['query'])
        self.assertIsNone(uncertain['reward']);self.assertFalse(uncertain['interval_is_training_reward'])
        lo,hi=uncertain['reward_interval']
        for values in itertools.product(('pass','fail'),repeat=3):
            concrete=copy.deepcopy(result)
            for key,value in zip(('extra0','extra1','extra2'),values):concrete['statuses'][key]=value
            value=score_verified(c,concrete,source_query=c['query'])['reward']
            self.assertGreaterEqual(value,lo-1e-10);self.assertLessEqual(value,hi+1e-10)

    def test_unknown_preferences_do_not_block_known_partial_failure(self):
        c=self.rich_contract();result=verify(c)
        result['statuses'].update(extra0='fail',extra1='pass',extra2='unknown')
        score=score_verified(c,result,source_query=c['query'])
        self.assertTrue(score['reward_valid']);self.assertAlmostEqual(score['reward'],-0.30+0.55*(1+1+0.5)/3)

    def test_success_known_but_unknown_preference_gives_range(self):
        c=self.rich_contract();result=verify(c);result['statuses'].update(extra0='pass',extra1='pass',extra2='unknown')
        score=score_verified(c,result,source_query=c['query'])
        self.assertTrue(score['semantic_success']);self.assertIsNone(score['reward'])
        self.assertEqual(score['reward_interval'],[0.55,1])


if __name__=='__main__':unittest.main()
