import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from jsonschema import ValidationError
from semantic_contract_v2_r3 import VERSION, DIMENSIONS, normalize, schema, validate, raw_example


def single(query, subtype, attribute, value, dimension="specifications", operator="eq", importance="required", subject=None, evidence=None):
    return {"schema_version": VERSION, "task_id": 1, "query": query, **{d: [] for d in DIMENSIONS}, "pending_mentions": [],
            dimension: [{"subtype": subtype, "attribute": attribute, "expected": value, "operator": operator,
                         "importance": importance, "subject": subject or {"kind": "primary_product", "name": "主商品"},
                         "price_basis": None, "evidence": [{"text": e} for e in (evidence or [query])], "review_status": "pending"}]}


def norm(raw):
    return normalize(raw, {"task_id": raw["task_id"], "query": raw["query"]})


def number(n, unit):
    return {"type": "number", "value": n, "unit": unit}


def text(value):
    return {"type": "text", "value": value}


class R3Test(unittest.TestCase):
    def test_pending_evidence_singleton_wrapper_is_audited_and_input_unchanged(self):
        raw = single("黑色，质量好", "color", "color", text("黑色"), evidence=["黑色"])
        raw["pending_mentions"] = [{"evidence": [{"text": "质量好"}], "reason": "unverifiable_subjective", "importance": "uncertain"}]
        before = copy.deepcopy(raw)
        c, audit = norm(raw)
        self.assertEqual(raw, before)
        self.assertEqual(c["pending_mentions"][0]["evidence"]["text"], "质量好")
        self.assertIn("singleton_pending_evidence_unwrap", [a["kind"] for a in audit])
        with self.assertRaisesRegex(ValueError, "Unreviewed|Pending|uncertain"):
            validate(c, source_query=c["query"], require_reviewed=True)

    def test_pending_evidence_invalid_shapes_report_schema_error_without_crash(self):
        for evidence in ([], [{"text":"质量好"}, {"text":"黑色"}], None, {"text":[]}, "质量好"):
            raw = single("黑色，质量好", "color", "color", text("黑色"), evidence=["黑色"])
            raw["pending_mentions"] = [{"evidence": evidence, "reason": "unverifiable_subjective", "importance": "uncertain"}]
            with self.subTest(evidence=evidence), self.assertRaises(ValidationError):
                norm(raw)

    def test_saved_reply_revalidation_preserves_source_and_recovers_failure(self):
        from revalidate_semantic_collection import revalidate
        raw = single("黑色，质量好", "color", "color", text("黑色"), evidence=["黑色"])
        raw["pending_mentions"] = [{"evidence": [{"text":"质量好"}], "importance":"uncertain", "reason":"unverifiable_subjective"}]
        envelope = json.dumps({"choices":[{"finish_reason":"stop","message":{"content":json.dumps(raw)}}]})
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d)/"source",Path(d)/"derived"
            src.mkdir()
            (src/"tasks.jsonl").write_text(json.dumps({"task_id":1,"query":raw["query"]})+"\n",encoding="utf-8")
            (src/"attempts.jsonl").write_text(json.dumps({"task_id":1,"http_status":200,"raw_response":envelope,"accepted":False,"error":"old type error"})+"\n",encoding="utf-8")
            for name in ("report.json","run_manifest.json"):(src/name).write_text("{}",encoding="utf-8")
            before={p.name:p.read_bytes() for p in src.iterdir()}
            result=revalidate(src,out)
            self.assertEqual(result["validated_tasks"],1)
            self.assertEqual(result["api_calls"],0)
            self.assertEqual({p.name:p.read_bytes() for p in src.iterdir()},before)
            with self.assertRaisesRegex(ValueError,"empty"):
                revalidate(src,out)

    def test_bare_feature_count_is_exact_and_explicit_minimum_is_lower(self):
        norm(single("包含4种音色", "attribute_count", "tone_type_count", number(4, "count")))
        with self.assertRaisesRegex(ValueError, "bare_feature_exact_count"):
            norm(single("包含4种音色", "attribute_count", "tone_type_count", number(4, "count"), operator="at_least"))
        for q in ("预设至少80个", "预设达到80个"):
            norm(single(q, "attribute_count", "preset_count", number(80, "count"), operator="at_least"))
            with self.assertRaisesRegex(ValueError, "explicit_feature_lower_bound"):
                norm(single(q, "attribute_count", "preset_count", number(80, "count")))

    def test_feature_comparison_scope_does_not_leak_between_clauses(self):
        q = "预设音色库至少80个，包含4种音色"
        raw = single(q, "attribute_count", "tone_type_count", number(4, "count"))
        norm(raw)
        raw["specifications"][0]["operator"] = "at_least"
        with self.assertRaisesRegex(ValueError, "bare_feature_exact_count"):
            norm(raw)

    def test_other_feature_count_and_negation_keep_their_scope(self):
        norm(single("包含4个接口", "attribute_count", "other_feature_count", number(4, "count")))
        with self.assertRaisesRegex(ValueError, "bare_feature_exact_count"):
            norm(single("包含4个接口", "attribute_count", "other_feature_count", number(4, "count"), operator="at_least"))
        norm(single("不要4种音色", "attribute_count", "tone_type_count", text("不要4种音色")))

    def test_duo_cannot_be_inclusive_lower_even_when_pending(self):
        q = "包含20多种失真效果"
        raw = single(q, "attribute_count", "effect_type_count", text(q), importance="uncertain")
        raw["pending_mentions"] = [{"evidence": {"text": q}, "reason": "ambiguous_meaning", "importance": "uncertain"}]
        norm(raw)
        raw["specifications"][0].update(operator="at_least", expected=number(20, "count"))
        with self.assertRaisesRegex(ValueError, "feature_duo_boundary"):
            norm(raw)
        raw["specifications"][0].update(operator="between", expected={"type": "range", "lower": 20, "upper": None, "unit": "count", "lower_inclusive": False, "upper_inclusive": False})
        norm(raw)

    def test_closed_route_is_audited_and_does_not_mutate_input(self):
        raw = single("12伏输出", "voltage", "voltage", number(12, "V"))
        before = copy.deepcopy(raw)
        c, audit = norm(raw)
        self.assertEqual(raw, before)
        self.assertEqual(c["functions"][0]["subtype"], "function")
        self.assertEqual(c["specifications"], [])
        self.assertIn("physical_attribute_parent", audit[0]["rules"])
        norm(single("适合室外使用", "use_environment", "use_environment", text("适合室外使用")))

    def test_unknown_subtype_and_unit_mismatch_still_fail(self):
        with self.assertRaises(ValidationError):
            norm(single("12伏输出", "mystery_voltage", "voltage", number(12, "V")))
        with self.assertRaisesRegex(ValueError, "unit mismatch"):
            norm(single("1800瓦", "power", "power", number(1800, "kWh")))

    def test_missing_nullable_only_does_not_fill_semantic_fields(self):
        r = single("金属材质", "material", "material", text("金属"))
        del r["specifications"][0]["price_basis"]
        self.assertIsNone(norm(r)[0]["specifications"][0]["price_basis"])
        del r["specifications"][0]["subject"]
        with self.assertRaises(ValidationError):
            norm(r)

    def test_condition_functions_route_but_new_condition_stays_condition(self):
        c, _ = norm(single("要求防水", "condition", "condition", text("防水")))
        self.assertEqual(c["functions"][0]["attribute"], "function")
        self.assertEqual(norm(single("原装全新", "condition", "condition", text("原装全新")))[0]["specifications"][0]["subtype"], "condition")

    def test_shipping_source_not_origin(self):
        c, _ = norm(single("从法国邮寄", "origin", "origin", text("从法国邮寄")))
        self.assertEqual(c["specifications"][0]["subtype"], "delivery")
        self.assertEqual(norm(single("法国生产", "origin", "origin", text("法国生产")))[0]["specifications"][0]["subtype"], "origin")

    def test_ordinary_price_recovery_preserves_budget_policy(self):
        for q, want in (("大概卖五十块钱出头", 54), ("售价在460元左右", 506)):
            c, _ = norm(single(q, "unit_price", "unit_price", text(q), "budget"))
            self.assertEqual(c["budget"][0]["subtype"], "total_price")
            self.assertEqual(c["budget"][0]["subject"], {"kind": "order", "name": "本次订单"})
            self.assertEqual(c["budget"][0]["expected"], number(want, "CNY"))
            self.assertEqual(c["budget"][0]["importance"], "required")

    def test_true_unit_price_and_invented_denominator_not_overridden(self):
        c, _ = norm(single("每米价格50元左右", "unit_price", "unit_price", text("每米价格50元左右"), "budget"))
        self.assertEqual(c["budget"][0]["price_basis"], {"amount": 1, "unit": "m"})
        raw = single("售价460元", "unit_price", "unit_price", text("售价460元"), "budget")
        raw["budget"][0]["price_basis"] = {"amount": 1, "unit": "bottle"}
        # A fabricated denominator is retained as a diagnostic failure, never
        # silently rewritten into a plausible ordinary price.
        with self.assertRaisesRegex(ValueError, "denominator|basis"):
            norm(raw)

    def test_free_shipping_zero_and_negated_or_total_price_not_accepted(self):
        raw = single("还要包邮费的", "shipping_fee", "shipping_fee", text("包邮费"), "budget", evidence=["包邮费"])
        c, _ = norm(raw)
        self.assertEqual(c["budget"][0]["expected"], number(0, "CNY"))
        with self.assertRaisesRegex(ValueError, "shipping_fee"):
            norm(single("包邮", "total_price", "total_price", text("包邮"), "budget"))
        c, _ = norm(single("不包邮", "shipping_fee", "shipping_fee", text("不包邮"), "budget"))
        self.assertEqual(c["budget"][0]["importance"], "uncertain")
        self.assertEqual(c["budget"][0]["expected"]["type"], "text")

    def test_book_count_alias_but_wrong_number_still_fails(self):
        c, _ = norm(single("买一本课本", "quantity", "purchase_count", number(1, "piece")))
        self.assertEqual(c["specifications"][0]["expected"]["unit"], "book")
        with self.assertRaisesRegex(ValueError, "purchase_quantity_coverage"):
            norm(single("买一本课本", "quantity", "purchase_count", number(2, "piece")))

    def test_price_and_free_shipping_same_clause_keep_distinct_roles(self):
        raw = single("预算20元包邮", "total_price", "total_price", text("预算20元"), "budget", evidence=["预算20元"])
        fee = single(raw["query"], "shipping_fee", "shipping_fee", text("包邮"), "budget", evidence=["包邮"])
        raw["budget"].extend(fee["budget"])
        c, _ = norm(raw)
        self.assertEqual([r["expected"]["value"] for r in c["budget"]], [20, 0])
        c, _ = norm(single("满50元包邮", "shipping_fee", "shipping_fee", text("满50元包邮"), "budget"))
        self.assertEqual(c["budget"][0]["importance"], "uncertain")

    def test_explicit_gifts_not_mixed_with_regular_components(self):
        value = {"type": "components", "items": [{"name": "练习册", "quantity": None, "unit": None}]}
        c, _ = norm(single("赠送练习册", "components", "included_components", value, operator="contains_all"))
        self.assertEqual(c["specifications"][0]["attribute"], "gift_presence")
        norm(single("带有练习册", "components", "included_components", value, operator="contains_all"))

    def test_pack_count_and_mass_are_not_order_counts(self):
        c, _ = norm(single("单支装", "quantity", "purchase_count", number(1, "piece")))
        self.assertEqual(c["specifications"][0]["attribute"], "pack_count")
        c, _ = norm(single("五斤装", "quantity", "purchase_count", number(5, "jin")))
        self.assertEqual(c["specifications"][0]["subtype"], "net_content")
        self.assertEqual(c["specifications"][0]["expected"], number(5, "jin"))

    def test_pack_conjunction_not_guessed(self):
        with self.assertRaisesRegex(ValueError, "pack_as_purchase"):
            norm(single("最好2个和10个装", "quantity", "purchase_count", {"type": "text_list", "values": ["2个装", "10个装"]}, operator="one_of", importance="preference"))

    def test_subject_evidence_can_be_separate_but_not_guessed(self):
        q = "镜头外壳为金属，颜色银白"
        subject = {"kind": "component", "name": "镜头外壳"}
        c, _ = norm(single(q, "color", "color", text("银白"), subject=subject, evidence=["镜头外壳", "颜色银白"]))
        self.assertEqual(c["specifications"][0]["subject"], subject)
        with self.assertRaisesRegex(ValueError, "subject_evidence"):
            norm(single(q, "color", "color", text("银白"), subject=subject, evidence=["颜色银白"]))
        with self.assertRaisesRegex(ValueError, "subject_evidence"):
            norm(single(q, "color", "color", text("银白"), subject={"kind": "component", "name": "金属镜头外壳"}))

    def test_quotes_not_fuzzily_fixed_and_approval_not_created(self):
        raw = single("小巧的机身，可放口袋", "size", "size", text("小巧"), evidence=["小巧，可放口袋"])
        with self.assertRaisesRegex(ValueError, "quote must occur once"):
            norm(raw)
        c, _ = norm(single("黑色", "color", "color", text("黑色")))
        with self.assertRaisesRegex(ValueError, "Unreviewed"):
            validate(c, source_query=c["query"], require_reviewed=True)

    def test_reaches_lower_bound_and_ac_relation_loss(self):
        c, _ = norm(single("储能达到1度电", "capacity", "energy_capacity", number(1, "kWh")))
        self.assertEqual(c["specifications"][0]["operator"], "at_least")
        with self.assertRaisesRegex(ValueError, "conversion_relation_coverage"):
            norm(single("兼容110伏与220伏交流电转换", "function", "voltage", {"type": "text_list", "values": ["110伏", "220伏"]}, "functions"))

    def test_category_scope_and_atomicity_are_blocked(self):
        for q, sub, attr, dim, code in (("超轻火箭筒", "product_category", "product_category", "category", "category_scope"),
                                       ("胡桃色实木", "material", "material", "specifications", "material_color_atomicity"),
                                       ("耐脏的蓝色", "color", "color", "specifications", "color_function_atomicity"),
                                       ("成分自然无添加", "composition", "composition", "specifications", "composition_atomicity")):
            with self.subTest(q=q), self.assertRaisesRegex(ValueError, code):
                norm(single(q, sub, attr, text(q), dim))

    def test_unresolved_roles_not_hidden_in_pending(self):
        with self.assertRaisesRegex(ValueError, "course_level_as_version"):
            norm(single("1级课本", "version", "version", text("1级"), "model"))
        with self.assertRaisesRegex(ValueError, "charge_as_energy"):
            norm(single("一万毫安", "capacity", "energy_capacity", text("一万毫安"), importance="uncertain"))
        with self.assertRaisesRegex(ValueError, "explicit_function_coverage"):
            norm(single("想买保暖外套", "product_category", "product_category", text("外套"), "category"))

    def test_clear_best_accessory_strength_flag_removed_without_approval(self):
        value = {"type": "components", "items": [{"name": "稀释液", "quantity": None, "unit": None}]}
        raw = single("最好配稀释液", "components", "included_components", value, operator="contains_all", importance="preference")
        raw["pending_mentions"] = [{"evidence": {"text": raw["query"]}, "importance": "uncertain", "reason": "ambiguous_strength"}]
        c, audit = norm(raw)
        self.assertEqual(c["pending_mentions"], [])
        self.assertEqual(c["specifications"][0]["importance"], "preference")
        self.assertEqual(c["specifications"][0]["review_status"], "pending")
        self.assertIn("redundant_explicit_preference_pending", [a["kind"] for a in audit])

    def test_drive_and_wheel_construction_not_graded_as_one_attribute(self):
        with self.assertRaisesRegex(ValueError, "drive_wheel_atomicity"):
            norm(single("双驱动气轮", "function", "drive_structure", text("双驱动气轮"), "functions"))
        norm(single("双驱动", "function", "drive_structure", text("双驱动"), "functions"))

    def test_requested_effect_and_flavor_carrier_are_not_lost(self):
        raw = single("用完皮肤不会红肿的", "product_category", "product_category", text("面膜"), "category")
        raw["pending_mentions"] = [{"evidence": {"text": raw["query"]}, "reason": "ambiguous_strength", "importance": "uncertain"}]
        with self.assertRaisesRegex(ValueError, "explicit_function_coverage"):
            norm(raw)
        with self.assertRaisesRegex(ValueError, "flavor_as_subject"):
            norm(single("薄荷香味", "flavor", "flavor", text("薄荷香味"), subject={"kind": "component", "name": "薄荷"}))

    def test_offline_probe_reports_new_guards_without_claiming_model_quality(self):
        from replay_semantic_v2_r3 import replay
        rows = []
        for tid, raw in enumerate((single("黑色", "color", "color", text("黑色")), single("超轻火箭筒", "product_category", "product_category", text("超轻火箭筒"), "category"))):
            raw["task_id"] = tid
            raw["schema_version"] = "shopping-semantic-requirements-v2-draft"
            rows.append({"task_id": tid, "query": raw["query"], "raw_selected": raw, "program_valid": True, "selected_attempt_line": tid+1})
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "source" / "review_inputs.jsonl"
            source.parent.mkdir()
            source.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            before = source.read_bytes()
            report = replay(source, Path(d) / "out")
            self.assertEqual(report["newly_blocked_r2_passes"], [1])
            self.assertEqual(report["r3_program_valid"], 1)
            self.assertEqual(report["api_calls"], 0)
            self.assertEqual(source.read_bytes(), before)
            self.assertIn("NOT new r3 LLM", report["scope"])
            with self.assertRaisesRegex(ValueError, "empty"):
                replay(source, Path(d) / "out")

    def test_r3_runner_schema_selection_and_both_modes_keep_original_queries(self):
        from extract_semantic_requirements import snapshot, contract_from_response
        from build_semantic_pilot50 import source_tasks
        tasks, requests, files, provenance = snapshot("50", "semantic-v2-r3")
        all_tasks, _, _, _ = snapshot("1000", "semantic-v2-r3")
        original = {t["task_id"]: t["query"] for t in source_tasks()[0]}
        self.assertEqual(len(tasks), 50)
        self.assertEqual(len(all_tasks), 1000)
        self.assertTrue(all(original[t["task_id"]] == t["query"] for t in all_tasks))
        self.assertIn("relation_rules_source.py", files)
        self.assertEqual(json.loads(files["output.schema.json"])["properties"]["schema_version"]["const"], VERSION)
        raw = raw_example()
        envelope = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(raw)}}]})
        c = contract_from_response(envelope, {"task_id": 0, "query": raw["query"]}, schema())
        self.assertEqual(c["schema_version"], VERSION)

    def test_local_school_preparation_no_network_or_key(self):
        import os
        import sys
        from extract_semantic_requirements import main
        with tempfile.TemporaryDirectory() as d:
            args = ["extract", "--mode", "50", "--profile", "semantic-v2-r3", "--output", d, "--prepare-only", "--base-url", "http://aigw.dlut.edu.cn/v1", "--model", "school-model"]
            with patch.object(sys, "argv", args), patch.dict(os.environ, {}, clear=True), patch("urllib.request.build_opener") as network:
                self.assertEqual(main(), 0)
                network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
