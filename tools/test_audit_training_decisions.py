import unittest
from audit_training_decisions import public_state, compare, requirements


class DecisionScreenTests(unittest.TestCase):
    def test_visible_only_and_range_unknown(self):
        state = public_state('price: 10.0 to 20.0\nselected_options: {}\navailable_options: {"颜色": ["黑色", "白色"]}')
        self.assertIsNone(state['numeric_price'])
        self.assertEqual(state['selected_options'], {})

    def test_truncated_options_not_guessed(self):
        state = public_state('selected_options: {"颜色": [TRUNCATED]')
        self.assertNotIn('selected_options', state)

    def test_budget_boundaries(self):
        req = {'kind': 'explicit_budget', 'operator': 'at_most', 'expected': {'value': 20}}
        self.assertTrue(compare({'numeric_price': 20}, req)['passed'])
        req['operator'] = 'less_than'
        self.assertFalse(compare({'numeric_price': 20}, req)['passed'])
        self.assertIsNone(compare({'numeric_price': None}, req))

    def test_mass_units(self):
        req = {'kind': 'explicit_mass', 'expected': {'value': 1, 'unit': 'jin'}}
        self.assertTrue(compare({'selected_options': {'规格': '500g装'}}, req)['passed'])
        self.assertFalse(compare({'selected_options': {'规格': '10斤装'}}, req)['passed'])
        self.assertIsNone(compare({'selected_options': {'规格': '500g*2袋'}}, req))
        self.assertIsNone(compare({'selected_options': {}}, req))

    def test_opaque_and_multiple_colors_unknown(self):
        req = {'kind': 'explicit_color', 'expected': {'value': '白色'}}
        self.assertIsNone(compare({'selected_options': {'颜色': '100074689'}}, req))
        self.assertIsNone(compare({'selected_options': {'颜色': '黑色+白色'}}, req))
        self.assertFalse(compare({'selected_options': {'颜色': '黑色'}}, req)['passed'])

    def test_required_atom_and_source_offsets(self):
        query = '价格20元以内'
        atom = {'id': 'r001', 'review_status': 'approved', 'importance': 'required', 'operator': 'at_most',
                'expected': {'type': 'number', 'value': 20, 'unit': 'CNY'},
                'evidence': [{'text': query, 'start': 0, 'end': len(query)}]}
        contract = {'query': query, 'normalized': {'budget': [atom]}}
        self.assertEqual(len(requirements(contract)), 1)
        atom['importance'] = 'preference'
        self.assertEqual(requirements(contract), [])
        atom['importance'] = 'required'
        atom['evidence'][0]['start'] = 1
        with self.assertRaises(ValueError):
            requirements(contract)

    def test_vague_budget_and_component_color_excluded(self):
        for query in ('在四百元的价位', '价格在九十元出头'):
            atom = {'id': 'r1', 'review_status': 'approved', 'importance': 'required', 'operator': 'at_most',
                    'expected': {'type': 'number', 'value': 400, 'unit': 'CNY'},
                    'evidence': [{'text': query, 'start': 0, 'end': len(query)}]}
            self.assertEqual(requirements({'query': query, 'normalized': {'budget': [atom]}}), [])
        query = '绳子要红色'
        atom = {'id': 'r1', 'review_status': 'approved', 'importance': 'required', 'operator': 'eq',
                'subject': {'kind': 'component', 'name': '绳子'}, 'attribute': 'color',
                'expected': {'type': 'text', 'value': '红色'}, 'evidence': [{'text': query, 'start': 0, 'end': len(query)}]}
        self.assertEqual(requirements({'query': query, 'normalized': {'specifications': [atom]}}), [])


if __name__ == '__main__':
    unittest.main()
