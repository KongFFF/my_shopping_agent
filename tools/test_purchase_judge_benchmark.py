import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import purchase_judge_benchmark as b


class JudgeTests(unittest.TestCase):
    def request(self):
        return {'requirements': [{'id': 'r1'}], 'evidence_table': {'/title': '纯棉上衣', '/customization_options/颜色/1/value': '黑色'}}

    def answer(self, status='pass', evidence=None):
        code = {'pass': 'supported', 'fail': 'contradicted', 'unknown': 'not_stated'}[status]
        return {'decisions': [{'id': 'r1', 'status': status, 'reason_code': code, 'reason': 'test',
                'evidence': evidence if evidence is not None else [{'pointer': '/title', 'value': '纯棉上衣'}]}]}

    def test_unselected_option_and_answer_sources_rejected(self):
        for pointer in ('/customization_options/颜色/0/value', '/instructions/0/answer', '/reviews/0'):
            with self.subTest(pointer=pointer), self.assertRaises(ValueError):
                b.validate_answer(self.answer(evidence=[{'pointer': pointer, 'value': '白色'}]), self.request())

    def test_fabricated_or_shortened_evidence_rejected(self):
        with self.assertRaises(ValueError):
            b.validate_answer(self.answer(evidence=[{'pointer': '/title', 'value': '纯棉'}]), self.request())

    def test_duplicate_missing_and_extra_requirements_rejected(self):
        for ids in ([], ['r1', 'r1'], ['r2']):
            a = self.answer()
            a['decisions'] = [{**a['decisions'][0], 'id': rid} for rid in ids]
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                b.validate_answer(a, self.request())

    def test_unknown_can_have_no_evidence_but_pass_cannot(self):
        b.validate_answer(self.answer('unknown', []), self.request())
        with self.assertRaises(ValueError):
            b.validate_answer(self.answer('pass', []), self.request())

    def test_false_unknown_reason_code_rejected(self):
        a = self.answer('unknown', [])
        a['decisions'][0]['reason_code'] = 'supported'
        with self.assertRaises(ValueError):
            b.validate_answer(a, self.request())

    def test_hybrid_preserves_known_rules_and_unknowns(self):
        v = {'statuses': {'r1': 'unknown', 'r2': 'fail', 'budget': 'pass'},
             'comparisons': {rid: {} for rid in ('r1', 'r2', 'budget')}}
        case = {'rule_verification': v, 'contract': {'query': 'q'}}
        a = self.answer()
        a['decisions'].append({**a['decisions'][0], 'id': 'r2'})
        with patch.object(b, 'score_training_verified', return_value={'reward': 0}):
            _, after = b.hybrid_score(case, a)
        self.assertEqual(after['statuses'], {'r1': 'pass', 'r2': 'fail', 'budget': 'pass'})
        self.assertEqual(v['statuses']['r1'], 'unknown')

    def test_source_filter_hides_unselected_options_and_annotations(self):
        p = {'asin': 'a', 'title': 'test', 'attribute': ['棉'], 'instructions': [{'answer': 'white'}],
             'reviews': ['secret'], 'pricing': [10], 'customization_options': {'颜色': [
                 {'value': '白色', 'price': 10, 'is_selected': True}, {'value': '黑色', 'price': 10}]}}
        table = b.evidence_table(p, {'颜色': '黑色'})
        self.assertEqual(table['/customization_options/颜色/1/value'], '黑色')
        self.assertFalse(any('/instructions' in k or '/reviews' in k or '/0/value' in k for k in table))

    def test_http_exception_is_explicit(self):
        with self.assertRaises(ValueError):
            b.endpoint('http://aigw.dlut.edu.cn/v1')
        self.assertEqual(b.endpoint('http://aigw.dlut.edu.cn/v1', allow_school_http=True),
                         'http://aigw.dlut.edu.cn/v1/chat/completions')
        for url in ('http://aigw.dlut.edu.cn/v1?key=x', 'http://u:p@aigw.dlut.edu.cn/v1', 'http://evil/v1'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                b.endpoint(url, allow_school_http=True)

    def test_parallel_duplicate_rollouts_coalesced_and_resumed(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            case = {'request': self.request(), 'task_id': 1}
            events = [{'event_id': 'e1', 'cohort': 'holdout_v5', 'global_step': 1, 'generation_batch': 1,
                       'members': [{'case_id': 'k'} for _ in range(4)]},
                      {'event_id': 'e2', 'cohort': 'holdout_v5', 'global_step': 1, 'generation_batch': 2,
                       'members': [{'case_id': 'k'} for _ in range(4)]}]
            calls = []
            def fake(url, key, body, timeout):
                calls.append(body)
                return 200, json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(self.answer())}}]})
            with patch.object(b, 'load_dataset', return_value=({'k': case}, events, {}, 'prompt')), patch.object(b, 'report', return_value={}):
                for _ in range(2):
                    b.run(folder, folder / 'run', {'LLM_BASE_URL': 'https://example.com', 'LLM_MODEL': 'test', 'LLM_API_KEY': 'secret'}, transport=fake)
            self.assertEqual(len(calls), 1)
            logs = b.rows(folder / 'run/attempts.jsonl')
            self.assertEqual(len(logs), 1)
            self.assertNotIn('secret', (folder / 'run/config.json').read_text())

    def test_auth_failure_is_not_cached_as_unknown_or_reward(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            case = {'request': self.request(), 'task_id': 1}
            events = [{'event_id': 'e1', 'cohort': 'holdout_v5', 'global_step': 1, 'generation_batch': 1,
                       'members': [{'case_id': 'k'}]}]
            def denied(*args):
                raise b.HTTPFailure(403, 'private gateway response')
            with patch.object(b, 'load_dataset', return_value=({'k': case}, events, {}, 'prompt')), patch.object(b, 'report', return_value={}):
                b.run(folder, folder / 'run', {'LLM_BASE_URL': 'https://example.com', 'LLM_MODEL': 'test', 'LLM_API_KEY': 'secret'}, transport=denied)
            self.assertFalse((folder / 'run/answers.jsonl').exists())
            attempts = b.rows(folder / 'run/attempts.jsonl')
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0]['error'], 'HTTP 403')
            self.assertNotIn('private gateway response', (folder / 'run/attempts.jsonl').read_text())


if __name__ == '__main__':
    unittest.main()
