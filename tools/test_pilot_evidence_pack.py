"""Regression checks of reviewed real SKU evidence, not model quality tests."""
import copy
import json
from pathlib import Path
import unittest

from semantic_product_verifier_v4 import snapshot, reviewed_facts
from semantic_product_facts_v3 import validate_fact

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/semantic_pilot30_evidence_20261109'


def rows(path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


class PilotEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = rows(OUT / 'review_inputs35.jsonl')
        cls.facts = rows(OUT / 'additions.jsonl')
        cls.records = rows(OUT / 'replay/rescored_rollouts.jsonl')

    def score(self, task, index=0):
        return [r['new_score'] for r in self.records if r['task_id'] == task][index]

    def test_every_fact_has_public_proof_bound_to_a_reviewed_actual_sku(self):
        self.assertEqual(len(self.inputs), 35)
        for fact in self.facts:
            validate_fact(fact)
            candidates = [r for r in self.inputs if r['product_asin'] == fact['product_asin'] and r['selected_options'] == fact['options']]
            self.assertTrue(candidates)
            p = candidates[0]['public_product']
            accepted, rejected = reviewed_facts(p, snapshot(p, fact['options']), [fact])
            self.assertEqual(len(accepted), 1); self.assertFalse(rejected)
            self.assertTrue(all(not e['pointer'].startswith(('/instructions', '/reviews')) for e in fact['evidence']))

    def test_cold_warm_and_power_facts_cannot_leak_to_unselected_air_conditioner(self):
        r = self.inputs[4]; p = r['public_product']
        facts = [f for f in self.facts if f['product_asin'] == r['product_asin']]
        actual = {'颜色分类': '1.5匹单冷 KY-35NL3a 白'}
        accepted, rejected = reviewed_facts(p, snapshot(p, actual), facts)
        self.assertFalse(accepted); self.assertFalse(rejected)

    def test_effect_processor_same_model_retains_purpose_and_gift_difference(self):
        scores = [self.score(11205, i) for i in range(4)]
        self.assertTrue(all(s['verification']['statuses']['r004'] == 'pass' for s in scores))
        self.assertEqual(scores[0]['verification']['statuses']['r005'], 'unknown')
        self.assertEqual(scores[1]['verification']['statuses']['r005'], 'pass')
        self.assertLess(scores[0]['reward'], scores[1]['reward'])

    def test_valve_cap_is_selected_and_does_not_create_german_origin(self):
        scores = [self.score(15910, i) for i in range(4)]
        self.assertEqual(scores[1]['verification']['statuses']['r005'], 'pass')
        self.assertEqual(scores[0]['verification']['statuses']['r005'], 'unknown')
        self.assertTrue(all(s['verification']['statuses']['r001'] == 'unknown' for s in scores))
        self.assertGreater(scores[1]['reward'], scores[0]['reward'])

    def test_dog_fit_order_and_unknown_softness_survive(self):
        self.assertGreater(self.score(9199, 1)['reward'], self.score(9199, 0)['reward'])
        self.assertEqual(self.score(9199, 1)['verification']['statuses']['r002'], 'unknown')

    def test_user_amendment_is_use_not_free_antifreeze(self):
        amendment = json.loads((OUT / 'source_amendment1295.json').read_text(encoding='utf-8'))
        self.assertTrue(amendment['confirmed_by_user'])
        self.assertEqual(amendment['before']['query'], amendment['after']['query'])
        self.assertFalse(any(r['id'] == 'r006' for r in amendment['after']['specifications']))
        self.assertEqual(self.score(1295)['verification']['statuses']['r006'], 'pass')
        self.assertFalse(any(f['subtype'] == 'components' and '防冻液' in json.dumps(f['value'], ensure_ascii=False) for f in self.facts))

    def test_approximation_units_and_quantity_unknowns_are_not_erased(self):
        for task, rid in ((7085, 'r005'), (21742, 'r005'), (20753, 'r001'), (4979, 'r004')):
            self.assertEqual(self.score(task)['verification']['statuses'][rid], 'unknown')

    def test_brand_compatibility_list_and_unselected_price_remain_unknown(self):
        score = self.score(22255, 2)
        self.assertEqual(score['verification']['statuses']['r002'], 'unknown')
        self.assertEqual(score['verification']['statuses']['r004'], 'unknown')

    def test_original_queries_parquets_and_hold_roster_are_preserved(self):
        old = json.loads((ROOT / 'analysis/semantic_public_matching_fix_20261109/training_bundle/manifest.json').read_text(encoding='utf-8'))
        new = json.loads((OUT / 'training_bundle/manifest.json').read_text(encoding='utf-8'))
        for k in ('train_ids', 'train_parquet', 'validation_parquet', 'original_queries'):
            self.assertEqual(old['assets'][k]['sha256'], new['assets'][k]['sha256'])
        self.assertEqual(old['held_tasks'], new['held_tasks'])
        self.assertEqual(len(new['held_tasks']), 19)


if __name__ == '__main__':
    unittest.main()
