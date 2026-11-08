import copy
import json
from pathlib import Path
import tempfile
import unittest
import build_semantic_reference as ref
from semantic_contract_v2_r4 import normalize as normalize_runtime, VERSION


class ReferenceTest(unittest.TestCase):
    def test_identity_edits_and_stale_preconditions_are_rejected(self):
        c={"task_id":1,"query":"原文","category":[]}
        with self.assertRaisesRegex(ValueError,"identity"):
            ref.apply_edits(c,[{"path":"/query","before":"原文","after":"改文"}])
        with self.assertRaisesRegex(ValueError,"precondition"):
            ref.apply_edits(c,[{"path":"/category","before":[1],"after":[]}])

    def test_all_50_references_preserve_query_and_never_self_approve(self):
        root=Path(__file__).resolve().parents[1]
        src={x["task_id"]:x for x in (json.loads(l) for l in (root/"analysis/semantic_r4_offline_20261103/review_inputs.jsonl").read_text(encoding="utf-8").splitlines())}
        cs=[json.loads(l) for l in (root/"experiments/semantic_reference_v1/pilot50/reference_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(cs),50)
        self.assertEqual({c["task_id"] for c in cs},set(src))
        for c in cs:
            self.assertEqual(c["query"],src[c["task_id"]]["query"])
            ref.validate(c)
            self.assertTrue(all(r["review_status"]=="pending" for d in ref.DIMS for r in c[d]))

    def test_actual_shared_families_have_source_bound_corrections(self):
        root=Path(__file__).resolve().parents[1]
        cs={c["task_id"]:c for c in (json.loads(l) for l in (root/"experiments/semantic_reference_v1/pilot50/reference_candidates.jsonl").read_text(encoding="utf-8").splitlines())}
        self.assertEqual(cs[3723]["specifications"][0]["subtype"],"form_factor")
        self.assertTrue(any(r["subtype"]=="weight" for r in cs[4971]["specifications"]))
        self.assertTrue(any(r["attribute"]=="thickness" for r in cs[6893]["specifications"]))
        angle=next(r for r in cs[5598]["specifications"] if r["attribute"]=="angle")
        self.assertFalse(angle["expected"]["lower_inclusive"])
        gift=next(r for r in cs[2547]["specifications"] if r["attribute"]=="gift_presence")
        self.assertTrue(all(x["quantity"] is None for x in gift["expected"]["items"]))
        self.assertTrue(cs[6401]["pending_mentions"])  # No invented mAh.

    def test_general_physical_fields_work_without_task_specific_lookup(self):
        for sub,attr,v in (("weight","mass",{"type":"number","value":23,"unit":"g"}),
                           ("dimensions","thickness",{"type":"number","value":8,"unit":"mm"})):
            q="重量23克" if sub=="weight" else "厚度8毫米"
            r={"subtype":sub,"attribute":attr,"subject":{"kind":"primary_product","name":"主商品"},"importance":"required","operator":"eq","expected":v,"price_basis":None,"evidence":[{"text":q}],"review_status":"pending"}
            raw={"schema_version":VERSION,"task_id":987654,"query":q,**{d:[] for d in ref.DIMS},"pending_mentions":[],"specifications":[r]}
            c,_=normalize_runtime(raw,{"task_id":raw["task_id"],"query":q})
            self.assertEqual(c["specifications"][0]["expected"],v)
            raw["specifications"][0]["expected"]["unit"]="V"
            with self.assertRaisesRegex(ValueError,"unit mismatch"):
                normalize_runtime(raw,{"task_id":raw["task_id"],"query":q})

    def test_sources_and_reference_data_cannot_be_imported_as_extraction_rules(self):
        root=Path(__file__).resolve().parents[1]
        for name in ("semantic_contract_v2_r4.py","semantic_policy_v4.py","extract_semantic_requirements.py"):
            code=(root/"tools"/name).read_text(encoding="utf-8")
            self.assertNotIn("reference_edits.json",code)
            self.assertNotIn("author_edits",code)


if __name__=="__main__":unittest.main()
