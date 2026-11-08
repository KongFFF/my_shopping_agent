import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from semantic_contract_v2_r4 import VERSION, DIMENSIONS, normalize, validate, schema, PACKAGE


def raw(query, strength="uncertain", pending=None, quote=None, role="total_price"):
    return {"schema_version":VERSION,"task_id":1,"query":query,**{d:[] for d in DIMENSIONS},
            "budget":[{"subtype":role,"attribute":role,"subject":{"kind":"order","name":"本次订单"},
                       "importance":strength,"operator":"eq","expected":{"type":"text","value":quote or query},
                       "price_basis":None,"evidence":[{"text":quote or query}],"review_status":"pending"}],
            "pending_mentions":pending or []}


def norm(c):
    return normalize(c,{"task_id":c["task_id"],"query":c["query"]})


def mention(text,reason="undefined_tolerance"):
    return {"evidence":{"text":text},"reason":reason,"importance":"uncertain"}


class V4Test(unittest.TestCase):
    def test_confirmed_budget_strength_and_false_tolerance_recovered(self):
        for q,amount in (("预算20元左右",22),("价格50元出头",54),("九百多块钱",999),("价格差不多在1700元以内",1700),("每米价格在50元左右",55)):
            c=raw(q,pending=[mention(q)],role="unit_price" if "每米" in q else "total_price")
            before=copy.deepcopy(c)
            result,audit=norm(c)
            self.assertEqual(c,before)
            self.assertEqual(result["budget"][0]["importance"],"required")
            self.assertEqual(result["budget"][0]["expected"]["value"],amount)
            self.assertEqual(result["pending_mentions"],[])
            self.assertIn("source_budget_strength",[a["kind"] for a in audit])
            self.assertIn("resolved_budget_pending",[a["kind"] for a in audit])

    def test_preference_and_hope_not_promoted_to_required(self):
        result,_=norm(raw("最好价格100元左右",pending=[mention("最好价格100元左右","ambiguous_strength")]))
        self.assertEqual(result["budget"][0]["importance"],"preference")
        result,_=norm(raw("希望价格100元左右",pending=[mention("希望价格100元左右","ambiguous_strength")]))
        self.assertEqual(result["budget"][0]["importance"],"uncertain")
        self.assertTrue(result["pending_mentions"])
        result,_=norm(raw("希望必须预算100元以内"))
        self.assertEqual(result["budget"][0]["importance"],"required")

    def test_other_clause_preference_and_unbound_modality_not_guessed(self):
        result,_=norm(raw("最好蓝色，预算100元左右",quote="预算100元左右"))
        self.assertEqual(result["budget"][0]["importance"],"required")
        from semantic_policy_v4 import budget_strength
        self.assertIsNone(budget_strength("最好蓝色预算100元"))
        self.assertIsNone(budget_strength("如果预算100元"))
        self.assertIsNone(budget_strength("不要求预算100元"))

    def test_real_nonbudget_uncertainty_retained(self):
        c=raw("尺寸350ml左右，预算170元",pending=[mention("尺寸350ml左右"),mention("预算170元")],quote="预算170元")
        result,_=norm(c)
        self.assertEqual([p["evidence"]["text"] for p in result["pending_mentions"]],["尺寸350ml左右"])
        with self.assertRaisesRegex(ValueError,"Unreviewed|Pending|uncertain"):
            validate(result,source_query=c["query"],require_reviewed=True)

    def test_mixed_or_ambiguous_meaning_pending_not_deleted(self):
        for reason in ("ambiguous_meaning","unsupported_unit"):
            result,_=norm(raw("预算100元左右",pending=[mention("预算100元左右",reason)]))
            self.assertEqual(len(result["pending_mentions"]),1)
        from semantic_policy_v4 import money_only
        self.assertFalse(money_only("预算100元左右同时要蓝色"))
        self.assertFalse(money_only("左右"))

    def test_unsupported_currency_and_amount_keep_pending(self):
        for q in ("预算一千多块钱","预算20美元"):
            result,_=norm(raw(q))
            self.assertEqual(result["budget"][0]["importance"],"uncertain")
            self.assertTrue(result["pending_mentions"])

    def test_validation_rejects_manual_strength_or_pending_bypass(self):
        c,_=norm(raw("预算100元左右"))
        c["budget"][0]["importance"]="uncertain"
        with self.assertRaisesRegex(ValueError,"budget_strength_source"):
            validate(c,source_query=c["query"])
        c,_=norm(raw("预算100元左右"))
        c["pending_mentions"]=[{"evidence":{"text":c["query"],"start":0,"end":len(c["query"])},"reason":"undefined_tolerance","importance":"uncertain"}]
        with self.assertRaisesRegex(ValueError,"resolved_budget_still_pending"):
            validate(c,source_query=c["query"])

    def test_pending_object_and_singleton_array_are_equivalent_and_audited(self):
        c=raw("质量要好，预算100元",quote="预算100元",pending=[mention("质量要好","unverifiable_subjective")])
        a,_=norm(c)
        c["pending_mentions"][0]["evidence"]=[c["pending_mentions"][0]["evidence"]]
        b,audit=norm(c)
        self.assertEqual(a,b)
        self.assertIn("singleton_pending_evidence_unwrap",[x["kind"] for x in audit])

    def test_multi_pending_scopes_and_malformed_containers_fail_actionably(self):
        for e in ([],[{"text":"预算100元"},{"text":"质量要好"}],None,"预算100元"):
            c=raw("预算100元",pending=[mention("预算100元")]);c["pending_mentions"][0]["evidence"]=e
            with self.assertRaisesRegex(ValueError,"pending_mentions"):
                norm(c)
        c=raw("预算100元");c["budget"][0]["expected"]=[]
        with self.assertRaisesRegex(ValueError,"expected"):
            norm(c)

    def test_clear_purpose_pending_only_and_background_are_detected(self):
        c=raw("能用于外部墙体，预算100元",quote="预算100元",pending=[mention("能用于外部墙体","ambiguous_meaning")])
        with self.assertRaisesRegex(ValueError,"objective_requirement_pending_only"):
            norm(c)
        for t in ("今天报考了英语四级考试","空调彻底不工作了"):
            c=raw(t+"，预算100元",quote="预算100元",pending=[mention(t,"ambiguous_meaning")])
            with self.assertRaisesRegex(ValueError,"background_as_pending"):
                norm(c)

    def test_unknown_subject_ambiguity_is_not_automatically_removed(self):
        c=raw("同款参照不清楚，预算100元",quote="预算100元",pending=[mention("同款参照不清楚","ambiguous_meaning")])
        result,_=norm(c)
        self.assertEqual(len(result["pending_mentions"]),1)

    def test_selector_and_component_shapes_do_not_crash_before_schema(self):
        c=raw("预算100元");c["budget"][0]["subtype"]=[]
        with self.assertRaisesRegex(ValueError,"subtype"):
            norm(c)
        c=raw("预算100元");c["budget"][0]["expected"]={"type":"components","items":[{"name":[]}]}
        with self.assertRaisesRegex(ValueError,"items"):
            norm(c)

    def test_strong_budget_pending_and_partial_evidence_keep_correct_scope(self):
        q="一定要预算100元以内"
        c,_=norm(raw(q,pending=[mention(q,"ambiguous_strength")]))
        self.assertEqual(c["budget"][0]["importance"],"required")
        self.assertEqual(c["pending_mentions"],[])
        c,_=norm(raw("预算100元左右，颜色不清楚",quote="100元",pending=[mention("颜色不清楚","ambiguous_meaning")]))
        self.assertEqual(c["budget"][0]["expected"]["value"],110)
        self.assertEqual(len(c["pending_mentions"]),1)

    def test_actual_budget_reviewer_regression_examples(self):
        c=raw("售价在460元左右")
        result,_=norm(c)
        self.assertEqual(result["budget"][0]["expected"]["value"],506)
        self.assertEqual(result["budget"][0]["importance"],"required")

    def test_snapshot_binds_new_policy_source_for_both_modes(self):
        from extract_semantic_requirements import snapshot
        for mode,n in (("50",50),("1000",1000)):
            tasks,requests,files,provenance=snapshot(mode,"semantic-v2-r4")
            self.assertEqual(len(tasks),n)
            self.assertIn("budget_review_source.py",files)
            self.assertEqual(json.loads(files["output.schema.json"])["properties"]["schema_version"]["const"],VERSION)
            self.assertIn('pending_mentions每项的evidence是单个对象',files["system_prompt.txt"].decode())


if __name__=="__main__":unittest.main()
