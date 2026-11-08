import hashlib
import json
import unittest
from build_semantic_holdout import PACKAGE,verify,select
from build_semantic_pilot50 import TARGET,source_tasks


class HoldoutTest(unittest.TestCase):
    def test_sealed_tasks_are_train_only_distinct_and_excluded_from_pilot(self):
        chosen=verify()
        self.assertEqual(len(chosen),50)
        self.assertEqual(len({t["query"] for t in chosen}),50)
        pilot=[json.loads(l) for l in (TARGET/"tasks.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertFalse({t["task_id"] for t in chosen}&{t["task_id"] for t in pilot})
        self.assertFalse({t["query"] for t in chosen}&{t["query"] for t in pilot})
        ids={t["task_id"] for t in source_tasks()[0]}
        self.assertTrue(all(t["task_id"] in ids for t in chosen))

    def test_selection_is_reproducible_and_does_not_select_duplicate_queries(self):
        tasks=[{"task_id":i,"query":f"query{i//2}"} for i in range(20)]
        banned={hashlib.sha256(b"query0").hexdigest()}
        a=select(tasks,{4},banned,count=5)
        b=select(list(reversed(tasks)),{4},banned,count=5)
        self.assertEqual(a,b)
        self.assertNotIn(4,{t["task_id"] for t in a})
        self.assertNotIn("query0",{t["query"] for t in a})
        self.assertEqual(len({t["query"] for t in a}),5)

    def test_wrong_profile_rejected_and_mode_supported_without_inference(self):
        with self.assertRaisesRegex(ValueError,"profile"):
            verify("guided-v2")
        from extract_semantic_requirements import snapshot
        tasks,requests,files,p=snapshot("holdout50","semantic-v2-r4")
        self.assertEqual(len(tasks),50)
        self.assertEqual(p["mode"],"holdout50")
        self.assertEqual(json.loads((PACKAGE/"lock.json").read_text(encoding="utf-8"))["api_calls"],0)


if __name__=="__main__":unittest.main()
