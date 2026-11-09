import copy
import unittest

import semantic_product_verifier_v4 as v


def requirement(subtype, attribute, expected, operator='eq', subject=None):
    return dict(subtype=subtype, attribute=attribute, expected=expected, operator=operator,
                importance='required', subject=subject or v.PRIMARY, price_basis=None)


def product(label='天蓝色', axis='颜色分类', title='商品', attributes=()):
    return dict(asin='P', title=title, category='目录', attribute=list(attributes),
                customization_options={axis: [dict(value=label, price=10)]}, pricing=[10])


def compare(p, r, dimension='specifications', options=None):
    options = options if options is not None else {a: es[0]['value'] for a, es in p['customization_options'].items()}
    s = v.snapshot(p, options)
    return v.compare_requirement(r, dimension, p, s, v.automatic_facts(p, s))


def text_req(subtype, value, attribute=None, **kwargs):
    return requirement(subtype, attribute or subtype, dict(type='text', value=value), **kwargs)


class PublicMatchingTests(unittest.TestCase):
    def test_exact_shade_is_preserved(self):
        self.assertEqual(compare(product(), text_req('color', '天蓝色'))['status'], 'pass')
        self.assertNotEqual(compare(product(), text_req('color', '红色'))['status'], 'pass')

    def test_pink_alias_is_not_red(self):
        p = product('粉红色')
        self.assertEqual(compare(p, text_req('color', '粉色'))['status'], 'pass')
        self.assertEqual(compare(p, text_req('color', '红色'))['status'], 'fail')

    def test_unknown_compound_does_not_become_plain_color(self):
        self.assertEqual(compare(product('孔雀蓝色'), text_req('color', '蓝色'))['status'], 'unknown')

    def test_multicolor_and_negated_color_do_not_invent_single_color(self):
        for label in ('黑色/白色随机', '非黑色款', '圆球灯【白光】'):
            self.assertNotEqual(compare(product(label), text_req('color', '黑色'))['status'], 'pass')

    def test_wearer_range_binds_only_wearer_role(self):
        p = product('XS胸30背20 约1-2斤左右', '尺码')
        expected = dict(type='range', lower=1, upper=2, unit='jin', lower_inclusive=True, upper_inclusive=True)
        r = requirement('size', 'wearer_weight_limit', expected, 'approximately',
                        {'kind': 'intended_user', 'name': '小型犬'})
        self.assertEqual(compare(p, r)['status'], 'pass')
        r['subject'] = {'kind': 'component', 'name': '附件'}
        self.assertEqual(compare(p, r)['status'], 'unknown')

    def test_narrower_fit_is_not_a_pass(self):
        p = product('XS 1-2斤', '尺码')
        r = requirement('size', 'wearer_weight_limit', dict(type='range', lower=1, upper=3,
                        unit='jin', lower_inclusive=True, upper_inclusive=True), 'approximately',
                        {'kind': 'intended_user', 'name': '小型犬'})
        self.assertEqual(compare(p, r)['status'], 'unknown')

    def test_dimension_units_and_order(self):
        p = product('1500mm*2000mm', '尺寸')
        self.assertEqual(compare(p, text_req('dimensions', '1.5*2米'))['status'], 'pass')
        self.assertEqual(compare(p, text_req('dimensions', '2*1.5米'))['status'], 'unknown')
        self.assertEqual(compare(p, text_req('dimensions', '1.6*2米'))['status'], 'unknown')

    def test_literal_label_policy_remains_literal_and_bare_cun_is_not_converted(self):
        p = product('1500mm*2000mm', '尺寸')
        self.assertEqual(compare(p, text_req('dimensions', '1.5*2米', operator='matches_label'))['status'], 'unknown')
        self.assertEqual(compare(product('127mm', '尺寸'), text_req('dimensions', '5寸'))['status'], 'unknown')

    def test_explicit_attribute_and_grammatical_wrapper(self):
        p = product(title='BOSS人声效果器', attributes=['和声', '便携式', '民谣弹唱'])
        self.assertEqual(compare(p, text_req('product_category', '人声效果器'), 'category')['status'], 'pass')
        for subtype, value in (('function', '带有和声功能'), ('function', '方便携带'), ('purpose', '能够用于民谣弹唱')):
            self.assertEqual(compare(p, text_req(subtype, value), 'functions')['status'], 'pass')

    def test_no_strengthening_and_no_negation_leak(self):
        r = text_req('function', '防水')
        self.assertEqual(compare(product(title='不支持防水'), r, 'functions')['status'], 'fail')
        self.assertEqual(compare(product(title='防水，但是不支持防水'), r, 'functions')['status'], 'unknown')
        self.assertEqual(compare(product(title='不锈钢防水'), r, 'functions')['status'], 'pass')
        self.assertEqual(compare(product(title='并非不防水'), r, 'functions')['status'], 'unknown')
        self.assertEqual(compare(product(title='防水'), text_req('function', '全天防水'), 'functions')['status'], 'unknown')

    def test_therapeutic_keyword_needs_positive_claim(self):
        r = text_req('purpose', '缓解肌肉疼痛')
        self.assertEqual(compare(product(title='止痛贴 肌肉疼痛'), r, 'functions')['status'], 'pass')
        self.assertEqual(compare(product(title='肌肉疼痛记录本'), r, 'functions')['status'], 'unknown')
        self.assertEqual(compare(product(title='无法缓解肌肉疼痛'), r, 'functions')['status'], 'fail')

    def test_title_does_not_lend_unselected_variant_feature(self):
        p = product('普通款', title='防水背包')
        p['customization_options']['颜色分类'].append(dict(value='防水款', price=10))
        r = text_req('function', '防水')
        self.assertEqual(compare(p, r, 'functions')['status'], 'unknown')
        self.assertEqual(compare(p, r, 'functions', {'颜色分类': '防水款'})['status'], 'pass')

    def test_accessory_is_not_its_compatible_primary_product(self):
        for title in ('手机保护套', '适用于手机的收纳袋'):
            self.assertEqual(compare(product(title=title), text_req('product_category', '手机'), 'category')['status'], 'unknown')

    def test_target_annotations_and_reviews_are_not_public_evidence(self):
        p = product()
        p.update(instructions=[dict(attributes=['防水'])], reviews=['防水'])
        self.assertEqual(compare(p, text_req('function', '防水'), 'functions')['status'], 'unknown')

    def test_single_can_mass_and_count(self):
        p = product('熊猫炼乳350g*1罐')
        self.assertEqual(compare(p, requirement('net_content', 'mass', dict(type='number', value=350, unit='g')))['status'], 'pass')
        self.assertEqual(compare(p, requirement('quantity', 'purchase_count', dict(type='number', value=1, unit='piece')))['status'], 'pass')

    def test_pack_quantity_not_borrowed_from_another_variant(self):
        p = product('天蓝色', attributes=['24片'])
        r = requirement('quantity', 'purchase_count', dict(type='number', value=24, unit='piece'))
        self.assertEqual(compare(p, r)['status'], 'pass')
        p['customization_options']['数量'] = [dict(value='12片', price=10), dict(value='24片', price=20)]
        self.assertEqual(compare(p, r, options={'颜色分类': '天蓝色'})['status'], 'unknown')
        self.assertEqual(compare(p, r, options={'颜色分类': '天蓝色', '数量': '12片'})['status'], 'fail')

    def test_promotional_or_ambiguous_pack_does_not_invent_count(self):
        r = requirement('quantity', 'purchase_count', dict(type='number', value=1, unit='piece'))
        for label in ('350g*2罐', '买10送1', '随机1罐或2罐'):
            self.assertNotEqual(compare(product(label), r)['status'], 'pass')

    def test_capacity_or_port_count_is_not_purchase_quantity(self):
        r = requirement('quantity', 'purchase_count', dict(type='number', value=10, unit='bottle'))
        self.assertEqual(compare(product('可装10瓶'), r)['status'], 'unknown')
        r['expected'] = dict(type='number', value=3, unit='piece')
        self.assertEqual(compare(product('3个接口'), r)['status'], 'unknown')

    def test_compatibility_brand_list_is_not_manufacturer_evidence(self):
        p = product(title='叉车配件 杭叉合力龙工', attributes=['合力'])
        self.assertEqual(compare(p, text_req('brand', '合力'), 'brand')['status'], 'unknown')

    def test_generic_gift_only_from_selected_sku_and_without_invented_quantity(self):
        r = requirement('components', 'gift_presence', dict(type='components', items=[dict(name='赠品', quantity=None, unit=None)]), 'contains_all')
        p = product('VE-8+全套赠品')
        self.assertEqual(compare(p, r)['status'], 'pass')
        self.assertEqual(compare(product('电池套装', title='全套赠品'), r)['status'], 'unknown')
        self.assertEqual(compare(product('无赠品'), r)['status'], 'fail')
        numbered = copy.deepcopy(r); numbered['expected']['items'][0]['quantity'] = 3
        self.assertEqual(compare(p, numbered)['status'], 'unknown')

    def test_merchant_evidence_is_the_actual_shop(self):
        p = product(title='推荐甲店')
        r = text_req('merchant', '甲店')
        self.assertEqual(compare(p, r)['status'], 'unknown')
        p['shop_name'] = '甲店'
        self.assertEqual(compare(p, r)['status'], 'pass')

    def test_complete_accessories_do_not_leak_through_public_fallback(self):
        p = product('单机', title='配件齐全', attributes=['配件齐全'])
        p['customization_options']['颜色分类'].append(dict(value='全套配件', price=10))
        self.assertEqual(compare(p, text_req('function', '配件齐全'), 'functions')['status'], 'unknown')
        self.assertEqual(compare(p, text_req('function', '配件齐全'), 'functions', {'颜色分类': '全套配件'})['status'], 'pass')

    def test_optional_claim_does_not_prove_actual_purchase(self):
        self.assertEqual(compare(product(title='防水可选'), text_req('function', '防水'), 'functions')['status'], 'unknown')
        r = requirement('components', 'gift_presence', dict(type='components', items=[dict(name='赠品', quantity=None, unit=None)]), 'contains_all')
        self.assertEqual(compare(product('可加购赠品'), r)['status'], 'unknown')

    def test_imitation_or_visual_material_is_not_real_material(self):
        for title, want in (('人造羊皮', '羊皮'), ('仿真皮', '真皮'), ('原木色', '原木'), ('金属质感', '金属')):
            self.assertEqual(compare(product(title=title), text_req('material', want))['status'], 'unknown')
        self.assertEqual(compare(product(title='小羊皮'), text_req('material', '羊皮'))['status'], 'pass')


if __name__ == '__main__':
    unittest.main()
