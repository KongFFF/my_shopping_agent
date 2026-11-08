import copy
import unittest
import hashlib
from unittest.mock import patch
from jsonschema import ValidationError
import semantic_contract_v2_r7_final12 as b
import semantic_product_verifier_v4 as v
from semantic_product_facts_v3 import validate_fact
from semantic_evidence_reward import score_training_verified,score_terminal_failure
from semantic_component_group_proofs import group_proof
from semantic_source_scope_proofs import resolve_reviewed_claim_mentions
from semantic_review_decisions import strength_decision,resolved_strength_source_flag
from test_semantic_final12 import raw,requirement,small_product
from test_historical_semantic_reward import contract,product


def compare(r,p,dimension='specifications',extra=()):
    options={axis:es[0]['value'] for axis,es in p['customization_options'].items()}
    state=v.snapshot(p,options)
    return v.compare_requirement(r,dimension,p,state,v.automatic_facts(p,state)+list(extra))


class NumericAndCompositionTests(unittest.TestCase):
    def percent(self,quote='黄金含量要大于99.9%'):
        return requirement('composition','percentage',{'type':'range','lower':99.9,'upper':None,'unit':'percent',
            'lower_inclusive':False,'upper_inclusive':False},quote,'between',{'kind':'component','name':'黄金'})

    def test_percentage_strict_boundary_and_literal_source(self):
        r=self.percent();q=r['evidence'][0]['text'];c,_=b.normalize(raw(q,[r]),{'task_id':0,'query':q})
        self.assertEqual(v.compare_value(c['specifications'][0],{'type':'number','value':99.9,'unit':'percent'}),'fail')
        self.assertEqual(v.compare_value(c['specifications'][0],{'type':'number','value':99.95,'unit':'percent'}),'pass')
        c['specifications'][0]['expected']['lower_inclusive']=True
        with self.assertRaisesRegex(ValueError,'Percentage threshold'):b.validate(c,source_query=q)

    def test_percentage_source_negation_and_unit_not_guessed(self):
        q='黄金含量不要超过99.9%';r=self.percent(q)
        with self.assertRaises(ValueError):b.normalize(raw(q,[r]),{'task_id':0,'query':q})
        r.update(operator='at_most',expected={'type':'number','value':99.9,'unit':'percent'})
        b.normalize(raw(q,[r]),{'task_id':0,'query':q})
        r=self.percent('黄金含量要大于99.9%');r['expected']['unit']='count'
        with self.assertRaises(ValueError):b.normalize(raw(r['evidence'][0]['text'],[r]),{'task_id':0,'query':r['evidence'][0]['text']})

    def test_numeric_purity_never_inferred_from_fine_gold(self):
        r=self.percent();self.assertEqual(compare(r,small_product('60mm',title='黄金手镯足金'))['status'],'unknown')

    def test_forged_percentage_fact_requires_public_number_and_valid_domain(self):
        p=small_product('60mm',title='黄金手镯足金');s=v.snapshot(p,{'规格':'60mm'})
        f={'product_asin':'P','product_sha256':v.fingerprint(p),'scope':'product','dimension':'specifications',
           'subtype':'composition','attribute':'percentage','subject':{'kind':'component','name':'黄金'},
           'value':{'type':'number','value':99.99,'unit':'percent'},'complete':True,'review_status':'approved',
           'evidence':[{'pointer':'/title','value':p['title']}]}
        good,bad=v.reviewed_facts(p,s,[f]);self.assertFalse(good);self.assertTrue(bad)
        f['value']['value']=101
        with self.assertRaises(ValueError):validate_fact(f)

    def test_exact_approximation_center_passes_other_values_stay_unknown(self):
        q='规格500g左右';r=requirement('net_content','mass',{'type':'number','value':500,'unit':'g'},q,'approximately')
        n,a=b.normalize(raw(q,[r]),{'task_id':0,'query':q})
        self.assertFalse(n['pending_mentions']);self.assertEqual(n['specifications'][0]['importance'],'required')
        self.assertEqual(v.compare_value(r,{'type':'number','value':0.5,'unit':'kg'}),'pass')
        for amount in (450,499,501,550):self.assertEqual(v.compare_value(r,{'type':'number','value':amount,'unit':'g'}),'unknown')

    def test_approximation_cannot_borrow_a_different_target_or_unit(self):
        q='规格500g左右';r=requirement('net_content','mass',{'type':'number','value':600,'unit':'g'},q,'approximately')
        with self.assertRaises(ValueError):b.normalize(raw(q,[r]),{'task_id':0,'query':q})
        r['expected'].update(value=500,unit='ml')
        with self.assertRaises(ValueError):b.normalize(raw(q,[r]),{'task_id':0,'query':q})

    def test_component_group_is_one_reward_item_and_quantity_is_not_invented(self):
        q='六套杯碟勺齐全';r=requirement('components','included_components',{'type':'components','items':[
             {'name':n,'quantity':6,'unit':'piece'} for n in ('杯','碟','勺')]},q,'contains_all')
        n,a=b.normalize(raw(q,[r]),{'task_id':0,'query':q});self.assertEqual(len(n['specifications']),1)
        self.assertTrue(any(x.get('rule')=='reviewed_complete_component_sets' for x in a))
        r['expected']['items'][0]['quantity']=5;self.assertIsNone(group_proof(r,b.chinese_number))

    def test_component_presence_without_counts_does_not_prove_six_each(self):
        r=requirement('components','included_components',{'type':'components','items':[
          {'name':n,'quantity':6,'unit':'piece'} for n in ('杯','碟','勺')]},'六套杯碟勺齐全','contains_all')
        present={'type':'components','items':[{'name':n,'quantity':None,'unit':None} for n in ('杯','碟','勺')]}
        self.assertEqual(v.compare_value(r,present,complete=False),'unknown')

    def test_component_facts_can_be_joined_but_duplicates_not_summed(self):
        r=requirement('components','included_components',{'type':'components','items':[
          {'name':n,'quantity':6,'unit':'piece'} for n in ('杯','碟','勺')]},'六套杯碟勺齐全','contains_all')
        fs=[v.fact('specifications','components','included_components',{'type':'components','items':[{'name':n,'quantity':6,'unit':'piece'}]},
                   [{'pointer':'/title','value':n}],complete=False) for n in ('杯','碟','勺')]
        self.assertEqual(compare(r,small_product('套装'),extra=fs)['status'],'pass')
        contradictory=copy.deepcopy(fs[0]);contradictory['value']['items'][0]['quantity']=3
        self.assertEqual(compare(r,small_product('套装'),extra=fs+[contradictory])['status'],'unknown')
        self.assertEqual(compare(r,small_product('套装'),extra=fs+[fs[0]])['status'],'pass')


class PublicClaimTests(unittest.TestCase):
    def r(self):return requirement('function','function',{'type':'text_list','values':['轻','薄']},'轻薄','contains_all')
    def f(self,value,polarity='positive'):
        return {**v.fact('functions','function','function',{'type':'text','value':value},[{'pointer':'/title','value':value}]),'polarity':polarity}

    def test_light_alone_never_proves_light_and_thin(self):
        result=compare(self.r(),small_product('M'),'functions',[self.f('轻量')])
        self.assertEqual(result['status'],'unknown');self.assertEqual(result['satisfied_parts'],['轻']);self.assertEqual(result['unknown_parts'],['薄'])

    def test_conjunction_can_join_independently_evidenced_claims(self):
        self.assertEqual(compare(self.r(),small_product('M'),'functions',[self.f('轻量'),self.f('薄')])['status'],'pass')

    def test_negative_or_conflicting_claims_are_respected(self):
        fs=[self.f('轻量'),self.f('薄','negative')]
        self.assertEqual(compare(self.r(),small_product('M'),'functions',fs)['status'],'fail')
        self.assertEqual(compare(self.r(),small_product('M'),'functions',fs+[self.f('薄')])['status'],'unknown')

    def test_other_component_claim_never_satisfies_primary_product(self):
        f=self.f('薄');f['subject']={'kind':'gift','name':'赠品'}
        self.assertEqual(compare(self.r(),small_product('M'),'functions',[self.f('轻量'),f])['status'],'unknown')

    def test_craft_entailment_is_directional_and_not_negative_contraposition(self):
        r=requirement('craft','craft',{'type':'text','value':'传统工艺'},'传统工艺')
        f=v.fact('specifications','craft','craft',{'type':'text','value':'古法'},[{'pointer':'/title','value':'古法'}],complete=False)
        self.assertEqual(compare(r,small_product('60mm'),extra=[f])['status'],'pass')
        r['expected']['value']='古法';f['value']['value']='传统工艺'
        self.assertEqual(compare(r,small_product('60mm'),extra=[f])['status'],'unknown')
        r['expected']['value']='传统工艺';f['value']['value']='古法';f['polarity']='negative'
        self.assertEqual(compare(r,small_product('60mm'),extra=[f])['status'],'unknown')


class EvidenceObjectiveTests(unittest.TestCase):
    def fixture(self):
        c=contract();q=c['query']+'，需要防水';c['query']=q
        r=requirement('function','function',{'type':'text','value':'防水'},'防水')
        r.update(id='waterproof',review_status='approved');r['evidence'][0].update(start=q.index('防水'),end=q.index('防水')+2)
        c['functions']=[r];b.validate(c,source_query=q,require_reviewed=True)
        z=v.verify(c,product(),{'颜色分类':'白色'},purchase_executed=True)
        return c,z

    def test_unknown_gets_no_credit_but_is_not_changed_to_false(self):
        c,z=self.fixture();before=copy.deepcopy(z);r=score_training_verified(c,z,source_query=c['query'])
        self.assertTrue(r['reward_valid']);self.assertAlmostEqual(r['reward'],-0.30+0.55*2/3)
        self.assertIsNone(r['semantic_success']);self.assertFalse(r['evidence_success']);self.assertFalse(r['unknown_is_real_failure'])
        self.assertEqual(z,before)

    def test_unknown_stays_in_fixed_denominator_and_more_evidence_helps(self):
        c,z=self.fixture();low=score_training_verified(c,z,source_query=c['query'])
        z['statuses']['waterproof']='pass';high=score_training_verified(c,z,source_query=c['query'])
        self.assertEqual(high['reward'],1);self.assertGreater(high['reward'],low['reward'])

    def test_unknown_category_is_not_a_proven_hard_failure(self):
        c,z=self.fixture();z['statuses'][c['category'][0]['id']]='unknown'
        r=score_training_verified(c,z,source_query=c['query']);self.assertGreater(r['reward'],-0.85)
        self.assertFalse(r['evidence_success']);self.assertIsNone(r['semantic_success'])

    def test_unknown_preference_only_loses_bonus_after_required_items_pass(self):
        c,z=self.fixture();q=c['query']+'，最好红色';c['query']=q
        r=requirement('color','color',{'type':'text','value':'红色'},'红色');r.update(id='preferred_color',importance='preference',review_status='approved')
        r['evidence'][0].update(start=q.index('红色'),end=q.index('红色')+2);c['specifications'].append(r)
        z=v.verify(c,product(),{'颜色分类':'白色'},purchase_executed=True)
        z['statuses']['waterproof']='pass';z['statuses']['preferred_color']='unknown'
        result=score_training_verified(c,z,source_query=q)
        self.assertEqual(result['reward'],0.55);self.assertTrue(result['semantic_success']);self.assertTrue(result['evidence_success'])

    def test_unreviewed_contract_invalid_execution_and_changed_hash_still_block(self):
        c,z=self.fixture();c['functions'][0]['review_status']='pending'
        z=v.verify(c,product(),{'颜色分类':'白色'},purchase_executed=True)
        self.assertFalse(score_training_verified(c,z,source_query=c['query'])['reward_valid'])
        c,z=self.fixture();z['execution_complete']=False
        self.assertFalse(score_training_verified(c,z,source_query=c['query'])['reward_valid'])
        c,z=self.fixture();c['task_id']=999
        with self.assertRaises(ValueError):score_training_verified(c,z,source_query=c['query'])

    def test_loop_penalty_uses_proven_event_not_logged_reward(self):
        row={'final_reward':99,'terminal_result':{'done':True,'termination_reason':'repeat_loop'},'steps':[
             {'tool_name':'search_products','result':{'done':True,'termination_reason':'repeat_loop'}}]}
        self.assertEqual(score_terminal_failure(row)['reward'],-0.65)
        row['steps'][0]['result']['done']=False;self.assertFalse(score_terminal_failure(row)['reward_valid'])

    def test_completed_purchase_cannot_receive_a_no_purchase_penalty(self):
        row={'terminal_result':{'done':True,'termination_reason':'repeat_loop','purchase':{'asin':'P'}},'steps':[
             {'tool_name':'buy_now','result':{'done':True,'termination_reason':'repeat_loop'}}]}
        self.assertFalse(score_terminal_failure(row)['reward_valid'])


class ReviewBoundaryTests(unittest.TestCase):
    def test_pending_extraction_cannot_self_resolve_nonliteral_claim(self):
        q='当传家宝传下去';r=requirement('purpose','purpose',{'type':'text','value':'传家用途'},q)
        r.update(id='family');r['evidence'][0].update(start=0,end=len(q))
        c=raw(q,[r],'functions');c['pending_mentions']=[{'evidence':dict(r['evidence'][0]),'reason':'unverifiable_subjective','importance':'uncertain'}]
        self.assertFalse(resolve_reviewed_claim_mentions(c));self.assertTrue(c['pending_mentions'])
        c['functions'][0]['review_status']='approved'
        self.assertTrue(resolve_reviewed_claim_mentions(c));self.assertFalse(c['pending_mentions'])

    def test_reviewed_claim_standard_cannot_resolve_uncertain_strength(self):
        q='希望防水';r=requirement('function','function',{'type':'text','value':'防水'},q)
        r.update(id='waterproof',review_status='approved');r['evidence'][0].update(start=0,end=len(q))
        c=raw(q,[r],'functions');c['pending_mentions']=[{'evidence':dict(r['evidence'][0]),'reason':'ambiguous_strength','importance':'uncertain'}]
        self.assertFalse(resolve_reviewed_claim_mentions(c));self.assertTrue(c['pending_mentions'])

    def test_confirmed_strength_is_bound_to_query_and_predicate(self):
        q='希望防水';r=requirement('function','function',{'type':'text','value':'防水'},q)
        r['importance']='uncertain'
        decision={'id':'test','kind':'requirement_strength','status':'confirmed_by_user','importance':'required',
                  'query_sha256':hashlib.sha256(q.encode()).hexdigest(),'source_text':q,
                  'predicate':{k:r[k] for k in ('subtype','attribute','subject','operator','expected')}}
        with patch('semantic_review_decisions.records',return_value=[decision]):
            c,_=b.normalize(raw(q,[r],'functions'),{'task_id':0,'query':q})
            self.assertEqual(c['functions'][0]['importance'],'required');self.assertFalse(c['pending_mentions'])
            c['functions'][0]['review_status']='approved';b.validate(c,source_query=q,require_reviewed=True)
            c['functions'][0]['importance']='preference'
            with self.assertRaisesRegex(ValueError,'user decision'):b.validate(c,source_query=q)
            changed=copy.deepcopy(r);changed['expected']['value']='防尘'
            self.assertIsNone(strength_decision(q,changed));self.assertIsNone(strength_decision(q+'！',r))
        decision['status']='awaiting_user'
        with patch('semantic_review_decisions.records',return_value=[decision]):self.assertIsNone(strength_decision(q,r))

    def test_bare_approximate_money_uses_same_confirmed_factor(self):
        self.assertEqual(b.budget_amount('价格约90元')[1]['value'],99)
        self.assertEqual(b.budget_amount('其面料每米价格约135元')[1]['value'],148.5)
        self.assertEqual(b.budget_amount('价格90元以内')[1]['value'],90)

    def test_resolved_global_hope_flag_is_limited_to_exact_decided_source(self):
        q='希望防水';r=requirement('function','function',{'type':'text','value':'防水'},q)
        r['importance']='uncertain'
        d={'id':'test_global','kind':'requirement_strength','status':'confirmed_by_user','importance':'required',
           'query_sha256':hashlib.sha256(q.encode()).hexdigest(),'source_text':q,
           'predicate':{k:r[k] for k in ('subtype','attribute','subject','operator','expected')}}
        with patch('semantic_review_decisions.records',return_value=[d]):
            c,_=b.normalize(raw(q,[r],'functions'),{'task_id':0,'query':q})
            f={'code':'soft_clause_in_mixed_uncovered','location':'/query','source':q}
            self.assertTrue(resolved_strength_source_flag(c,f))
            self.assertFalse(resolved_strength_source_flag(c,{**f,'source':'希望防尘'}))
            self.assertFalse(resolved_strength_source_flag(c,{**f,'code':'purchase_quantity_coverage'}))

    def test_intervening_particle_does_not_erase_explicit_preference(self):
        q='内胆最好也是中药填充的'
        r=requirement('material','material',{'type':'text','value':'中药填充'},q,subject={'kind':'component','name':'内胆'})
        r['importance']='preference';c,_=b.normalize(raw(q,[r]),{'task_id':0,'query':q})
        c['specifications'][0]['review_status']='approved'
        b.validate(c,source_query=q,require_reviewed=True)
        self.assertEqual(c['specifications'][0]['importance'],'preference')
        q='内胆必须也是中药填充的';r['evidence']=[{'text':q}]
        c,_=b.normalize(raw(q,[r]),{'task_id':0,'query':q});c['specifications'][0]['review_status']='approved'
        with self.assertRaises(ValueError):b.validate(c,source_query=q,require_reviewed=True)

    def test_snapshot_hot_selling_requires_an_explicit_declaration(self):
        r=requirement('condition','condition',{'type':'text','value':'网络热卖款'},'网络热卖款的')
        p=small_product('白色',attribute=['热卖款'])
        self.assertEqual(compare(r,p)['status'],'pass')
        for declaration in ('抖音','网红款','推荐款','不是热卖款','2020年热卖款'):
            p=small_product('白色',attribute=[declaration])
            self.assertEqual(compare(r,p)['status'],'unknown',declaration)


if __name__=='__main__':unittest.main()
