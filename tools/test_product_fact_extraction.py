import copy
import unittest
from extract_product_facts import validate_output
from semantic_product_verifier_v2 import fingerprint
from semantic_public_catalog_v2 import public_product


class FactExtractionTests(unittest.TestCase):
    def fixture(self):
        p={'asin':'P','title':'某品牌商品','category':'商品','instructions':[{'instruction':'不能发送的需求','options':['目标答案']}],
           'reviews':['评价'],'customization_options':{'颜色':[{'value':'黑色','price':3,'is_selected':True,'asin':'SKU','image':'url'}]}}
        f={'product_asin':'P','product_sha256':fingerprint(p),'scope':'sku','selector_mode':'contains','options':{'颜色':'黑色'},
           'dimension':'specifications','subtype':'color','attribute':'color','subject':{'kind':'primary_product','name':'主商品'},
           'value':{'type':'text','value':'黑色'},'complete':True,'review_status':'pending',
           'evidence':[{'pointer':'/customization_options/颜色/0/value','value':'黑色'}]}
        return p,{'facts':[f],'unresolved':[]}

    def test_public_request_does_not_contain_answers_or_initial_selection(self):
        p,_=self.fixture();public=public_product(p)
        self.assertNotIn('instructions',public);self.assertNotIn('reviews',public)
        self.assertNotIn('is_selected',public['customization_options']['颜色'][0])

    def test_pending_source_backed_fact_validates_without_approval(self):
        p,a=self.fixture();self.assertEqual(validate_output(a,p),a)
        self.assertEqual(a['facts'][0]['review_status'],'pending')

    def test_self_approval_rejected(self):
        p,a=self.fixture();a['facts'][0]['review_status']='approved'
        with self.assertRaises(Exception):validate_output(a,p)

    def test_answers_and_unbound_option_sources_rejected(self):
        p,a=self.fixture();a['facts'][0]['evidence']=[{'pointer':'/instructions/0/instruction','value':'不能发送的需求'}]
        with self.assertRaises(ValueError):validate_output(a,p)
        p,a=self.fixture();a['facts'][0]['options']={'颜色':'白色'}
        with self.assertRaises(ValueError):validate_output(a,p)

    def test_duplicate_facts_rejected(self):
        p,a=self.fixture();a['facts'].append(copy.deepcopy(a['facts'][0]))
        with self.assertRaises(ValueError):validate_output(a,p)

if __name__=='__main__':unittest.main()
