import unittest
import semantic_contract_v2_r7_review3 as b


class MoneySpans(unittest.TestCase):
 def test_price_before_product_number(self):
  self.assertEqual(b.budget_source_clause('在找一款1500元左右的显卡，显存12G','1500元左右'),'1500元左右')
 def test_hard_cap_kept(self):
  s='预算在540块左右，别超了'
  self.assertEqual(b.budget_amount(b.budget_source_clause(s,s))[1]['value'],540)
 def test_denominator_not_clipped(self):
  q='每瓶价格20元左右'
  self.assertEqual(b.budget_source_clause(q,'价格20元左右'),q)
 def test_explicit_double_limit(self):
  op,v,_=b.budget_amount('价格在960元以上，不超过1000元')
  self.assertEqual((op,v['lower'],v['upper']),('between',960,1000))
 def test_undefined_colloquial_not_read_as340(self):
  for phrase in ('三四十元左右','十几块钱左右'):
   with self.assertRaises(ValueError):b.budget_amount(phrase)
 def test_approx_rule_unchanged(self):
  self.assertEqual(b.budget_amount('8300元左右')[1]['value'],9130)


if __name__=='__main__':unittest.main()
