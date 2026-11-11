import unittest
from purchase_fact_proofs import compile_answer, compile_fact, compare_requirement


def proposal(kind,role,quote,actual,subject=None,pointer='/customization_options/颜色分类/0/value'):
    return dict(kind=kind,role=role,quote=quote,actual=actual,subject=subject,pointer=pointer)


def request(label):
    return {'evidence_table':{'/customization_options/颜色分类/0/value':label},'selected_options':{'颜色分类':label}}


def req(value,subtype='dimensions',attribute='length',op='approximately',subject=None):
    return {'subtype':subtype,'attribute':attribute,'operator':op,'expected':value,
            'subject':subject or {'kind':'primary_product','name':'主商品'},'evidence':[{'text':'五种不同规格'}]}


class ProofTests(unittest.TestCase):
    def test_hallucinated_member_rejected_even_with_real_quote(self):
        label='24+36+48+60段+教程书'
        p=proposal('members','variant_members',label,{'values':[24,36,48,60,72],'unit':'count'})
        with self.assertRaisesRegex(ValueError,'members differ'):
            compile_fact(p,request(label))

    def test_member_list_count_computed_from_entire_sku(self):
        label='24+36+48+60段+教程书'
        p=proposal('members','variant_members',label,{'values':[24,36,48,60],'unit':'count'})
        f=compile_fact(p,request(label))
        r=req({'type':'number','value':5,'unit':'count'},'attribute_count','other_feature_count','eq')
        self.assertEqual(compare_requirement(r,[f])['status'],'fail')
        with self.assertRaises(ValueError):
            compile_fact({**p,'quote':'24+36'},request(label))

    def test_title_members_cannot_be_borrowed(self):
        label='24+36+48+60+72段'
        p=proposal('members','variant_members',label,{'values':[24,36,48,60,72],'unit':'count'},pointer='/title')
        with self.assertRaises(ValueError):
            compile_fact(p,{'evidence_table':{'/title':label},'selected_options':{}})

    def test_exact_approximation_and_unit_conversion(self):
        label='长度2.92米'
        f=compile_fact(proposal('scalar','length',label,{'value':2.92,'unit':'m'}),request(label))
        self.assertEqual(compare_requirement(req({'type':'number','value':2.9,'unit':'m'}),[f])['status'],'unknown')
        self.assertEqual(compare_requirement(req({'type':'number','value':292,'unit':'cm'}),[f])['status'],'pass')

    def test_dimension_axis_cannot_be_invented(self):
        with self.assertRaisesRegex(ValueError,'axis'):
            compile_fact(proposal('scalar','height','2.92米',{'value':2.92,'unit':'m'}),request('2.92米'))

    def test_wrong_unit_or_number_rejected(self):
        for actual in ({'value':2.9,'unit':'m'},{'value':2.92,'unit':'cm'}):
            with self.subTest(actual=actual),self.assertRaises(ValueError):
                compile_fact(proposal('scalar','length','长度2.92米',actual),request('长度2.92米'))

    def test_scope_and_component_cannot_be_clipped_away(self):
        label='调料208g*1袋+蘸料2g*5袋'
        with self.assertRaises(ValueError):
            compile_fact(proposal('scalar','net_content','2g',{'value':2,'unit':'g'}),request(label))
        with self.assertRaises(ValueError):
            compile_fact(proposal('scalar','net_content','蘸料2g*5袋',{'value':2,'unit':'g'}),request(label))
        f=compile_fact(proposal('scalar','net_content','蘸料2g*5袋',{'value':2,'unit':'g'},subject='蘸料'),request(label))
        self.assertEqual(compare_requirement(req({'type':'number','value':2,'unit':'g'},'net_content','mass','eq',{'kind':'component','name':'蘸料'}),[f])['status'],'pass')

    def test_primary_count_not_pack_mass_or_default_one(self):
        label='100g凝胶3支'
        f=compile_fact(proposal('scalar','quantity',label,{'value':3,'unit':'piece'}),request(label))
        self.assertEqual(compare_requirement(req({'type':'number','value':2,'unit':'piece'},'quantity','purchase_count','eq'),[f])['status'],'fail')
        with self.assertRaises(ValueError):
            compile_fact(proposal('scalar','quantity','凝胶',{'value':1,'unit':'piece'}),request('凝胶'))

    def test_implicit_component_or_synonym_not_proof(self):
        with self.assertRaises(ValueError):
            compile_fact(proposal('component','component','可视',{'name':'屏幕','quantity':None,'unit':None}),request('可视'))
        f=compile_fact(proposal('claim','declaration','天然',{'text':'天然'}),request('天然'))
        self.assertEqual(compare_requirement(req({'type':'text','value':'无添加'},'composition','composition','eq'),[f])['status'],'unknown')

    def test_component_count_has_to_bind_its_name(self):
        label='送2本填色本+1支勾线笔'
        good=proposal('component','component','送2本填色本',{'name':'填色本','quantity':2,'unit':'book'})
        compile_fact(good,request(label))
        with self.assertRaises(ValueError):
            compile_fact({**good,'actual':{'name':'填色本','quantity':1,'unit':'piece'}},request(label))

    def test_unselected_and_fake_evidence_rejected(self):
        with self.assertRaises(ValueError):
            compile_fact(proposal('claim','declaration','天然',{'text':'天然'},pointer='/instructions/0/answer'),request('天然'))

    def test_negative_context_not_erased_by_short_quote(self):
        label='不防水'
        f=compile_fact(proposal('claim','declaration','防水',{'text':'防水'}),request(label))
        self.assertEqual(compare_requirement(req({'type':'text','value':'防水'},'function','function','eq'),[f])['status'],'fail')

    def test_accessory_material_not_primary_material(self):
        with self.assertRaises(ValueError):
            compile_fact(proposal('claim','declaration','金属',{'text':'金属'}),request('外壳金属'))

    def test_html_entity_preserves_source_and_has_explicit_normalization(self):
        label='CRUML&amp;KECC'
        f=compile_fact(proposal('claim','declaration','CRUML&KECC',{'text':'CRUML&KECC'}),request(label))
        self.assertEqual(f['evidence']['value'],label)
        self.assertIn('HTML',f['evidence']['quote_normalization'])

    def test_size_alias_requires_size_axis(self):
        pointer='/customization_options/尺码/0/value'
        p=proposal('label','size_label','S',{'text':'S'},pointer=pointer)
        f=compile_fact(p,{'evidence_table':{pointer:'S'},'selected_options':{'尺码':'S'}})
        self.assertEqual(compare_requirement(req({'type':'text','value':'小码'},'size','size_label','eq'),[f])['status'],'pass')
        with self.assertRaises(ValueError):
            compile_fact({**p,'pointer':'/customization_options/颜色分类/0/value'},request('S'))

    def test_partially_rejected_answer_preserves_rejections(self):
        label='24+36+48+60段'
        answer={'facts':[proposal('members','variant_members',label,{'values':[24,36,48,60,72],'unit':'count'}),
                         proposal('members','variant_members',label,{'values':[24,36,48,60],'unit':'count'})]}
        result=compile_answer(answer,request(label))
        self.assertEqual(len(result['rejected']),1)
        self.assertEqual(len(result['facts']),1)

    def test_imitations_are_not_primary_material_proof(self):
        label='仿真发'
        f=compile_fact(proposal('claim','declaration',label,{'text':'真发'}),request(label))
        self.assertEqual(compare_requirement(req({'type':'text','value':'真发'},'material','material','eq'),[f])['status'],'unknown')

    def test_color_shade_cannot_shrink_to_substring(self):
        with self.assertRaises(ValueError):
            compile_fact(proposal('label','color','粉红色',{'text':'红色'}),request('粉红色'))

    def test_money_addition_is_not_variant_inventory(self):
        with self.assertRaises(ValueError):
            compile_fact(proposal('members','variant_members','100+20元',{'values':[100,20],'unit':'count'}),request('100+20元'))


if __name__=='__main__':
    unittest.main()
