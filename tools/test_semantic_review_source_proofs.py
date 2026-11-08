import unittest
import semantic_contract_v2_r7_review as b
from test_semantic_v7 import components


def norm(c):
    c['schema_version']=b.VERSION
    return b.normalize(c,{'task_id':c['task_id'],'query':c['query']})


class ReviewedCountProof(unittest.TestCase):
    def test_clear_elided_one_accepted_with_audit(self):
        for source,name in [('再加个高清防眩光的钢化膜','钢化膜'),('还要有个保温桶','保温桶')]:
            n,a=norm(components(source,name,1,'piece'))
            self.assertEqual(n['specifications'][0]['expected']['items'][0]['quantity'],1)
            self.assertTrue(any(x['kind']=='user_confirmed_quantity_source_proof' for x in a))

    def test_count_two_not_licensed_by_elided_one(self):
        with self.assertRaisesRegex(ValueError,'invented_component_count'):
            norm(components('再加个钢化膜','钢化膜',2,'piece'))

    def test_unrelated_classifier_not_lent_to_other_object(self):
        with self.assertRaisesRegex(ValueError,'invented_component_count'):
            norm(components('配个盒子，还附带刷子','刷子',1,'piece'))

    def test_count_on_a_person_not_lent_to_their_requested_item(self):
        with self.assertRaisesRegex(ValueError,'invented_component_count'):
            norm(components('有个朋友需要盒子','盒子',1,'piece'))

    def test_unanswered_set_policy_not_inferred(self):
        c=components('包含上衣、马甲、短裤的三件套','上衣',1,'piece')
        c['specifications'][0]['expected']['items'] += [{'name':n,'quantity':1,'unit':'piece'} for n in ('马甲','短裤')]
        c['specifications'][0]['evidence']=[{'text':'包含上衣、马甲、短裤'}]
        with self.assertRaisesRegex(ValueError,'component_count_source_unproven|invented_component_count'):norm(c)

    def test_proof_does_not_approve_contract(self):
        n,_=norm(components('配个盒子','盒子',1,'piece'))
        with self.assertRaisesRegex(ValueError,'Unreviewed'):b.validate(n,source_query=n['query'],require_reviewed=True)


if __name__=='__main__':unittest.main()
