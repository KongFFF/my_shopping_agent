import unittest
import purchase_fact_proofs as base
from purchase_fact_comparison_r1 import comparison
from test_purchase_fact_proofs import proposal,request,req


class RevisionTests(unittest.TestCase):
    def test_complete_imitation_phrase_cannot_prove_real_material(self):
        for label,want in [('仿真发','真发'),('人造羊皮','羊皮'),('金属质感','金属'),('原木色','原木')]:
            with self.subTest(label=label):
                f=base.compile_fact(proposal('claim','declaration',label,{'text':label}),request(label))
                self.assertEqual(comparison(req({'type':'text','value':want},'material','material','eq'),[f])['status'],'unknown')

    def test_real_material_still_supported(self):
        label='小羊皮'
        f=base.compile_fact(proposal('claim','declaration',label,{'text':label}),request(label))
        self.assertEqual(comparison(req({'type':'text','value':'羊皮'},'material','material','eq'),[f])['status'],'pass')

    def test_two_shades_do_not_prove_pure_single_color(self):
        label='黑色+白色'
        f=base.compile_fact(proposal('label','color',label,{'text':'黑色'}),request(label))
        self.assertEqual(comparison(req({'type':'text','value':'黑色'},'color','color','eq'),[f])['status'],'unknown')


if __name__=='__main__':unittest.main()
