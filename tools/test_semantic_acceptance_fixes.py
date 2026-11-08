import unittest
import semantic_contract_v2_r7_review2 as b
from test_semantic_v7 import text


def norm(c):
    c['schema_version']=b.VERSION
    return b.normalize(c,{'task_id':c['task_id'],'query':c['query']})


class AcceptanceFixes(unittest.TestCase):
    def test_accessory_number_not_money(self):
        q='附带5支刷头且价格在400元以内'
        self.assertEqual(b.budget_source_clause(q,'价格在400元以内'),'价格在400元以内')
        self.assertEqual(b.budget_amount(b.budget_source_clause(q,'价格在400元以内'))[1]['value'],400)

    def test_unit_price_preserved(self):
        s=b.budget_source_clause('附带5份配件且单价每瓶20元以内','单价每瓶20元以内')
        self.assertEqual(b.price_basis_from_phrase(s),{'amount':1,'unit':'bottle'})

    def test_structure_not_order(self):
        norm(text('要两个筒的设计',value='两个筒的设计',subtype='form_factor',attribute='form_factor'))

    def test_real_order_not_suppressed(self):
        with self.assertRaisesRegex(ValueError,'purchase_quantity_coverage'):
            norm(text('买两个筒的设计',value='两个筒的设计',subtype='form_factor',attribute='form_factor'))

    def test_missing_structure_count_still_blocked(self):
        with self.assertRaisesRegex(ValueError,'purchase_quantity_coverage'):
            norm(text('要两个筒的设计',value='筒的设计',subtype='form_factor',attribute='form_factor'))


if __name__=='__main__':unittest.main()
