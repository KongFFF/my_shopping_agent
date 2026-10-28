"""Regression checks for retained ranges after checkpoint rollback (no GPU)."""

import json
import tempfile
import unittest
from pathlib import Path

from analyze_grpo_diagnostics import analyze, parse_step_range


def step_events(step, branch):
    uid = f"{branch}-{step}"
    rollouts = [{
        "uid": uid, "task_id": step, "steps": 1,
        "actions": [{"tool": "search_products", "parameters": {"query": branch}}],
        "reward_type": "gold_purchase" if i == 0 else "repeat_loop",
        "reward_valid": True, "done": True, "infrastructure_invalid": False,
        "reward": {"terminal_utility": reward},
    } for i, reward in enumerate([1.0, -.65, -.65, -.65])]
    return [{
        "event": "generation_batch", "global_step": step,
        "groups": [{"uid": uid, "terminal_utilities": [1.0, -.65, -.65, -.65],
                    "kept": True, "accepted": True, "drop_reason": None}],
        "rollouts": rollouts,
    }, {
        "event": "optimizer_step", "global_step": step,
        "metrics": {"actor/lr": 1e-6, "actor/entropy": .1},
        "rollouts": rollouts,
    }]


def write_events(path, events):
    path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")


class RetainedChainTests(unittest.TestCase):
    def test_actual_rollback_boundaries_and_cost_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            old, new = Path(folder) / "old.jsonl", Path(folder) / "resume.jsonl"
            events = [event for step in range(1, 359) for event in step_events(step, "old")]
            events.append({"event": "skipped_update", "global_step": 359})
            write_events(old, events)
            write_events(new, [event for step in range(351, 501)
                               for event in step_events(step, "resume")])
            result, groups, updates, tasks = analyze(
                [old, new], 500, step_ranges=[(1, 350), (351, 500)])
            self.assertEqual(result["optimizer_steps"], list(range(1, 501)))
            self.assertEqual(result["counts"]["trained_rollouts"], 2000)
            self.assertEqual(result["counts"]["generated_rollouts"], 2000)
            self.assertEqual(result["sources"][0]["excluded_optimizer_steps"], list(range(351, 359)))
            self.assertEqual(result["sources"][0]["excluded_events"]["skipped_update"], 1)
            self.assertEqual(result["generation_cost_scope"]["excluded_generated_rollouts"], 32)
            self.assertEqual(result["generation_cost_scope"]["all_input_generated_rollouts"], 2032)
            self.assertEqual(len(groups), 500)
            self.assertEqual(len(updates), 500)
            self.assertEqual(len(tasks), 500)
            self.assertTrue(all(g["trained"] for g in groups))
            self.assertTrue(all("resume" in g["uid"] for g in groups if g["step"] >= 351))
            # Without a lineage selection, a real conflict must still fail.
            with self.assertRaisesRegex(ValueError, "duplicate optimizer step 351"):
                analyze([old, new], 500)

    def test_duplicate_inside_selected_range_still_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "duplicate.jsonl"
            write_events(path, step_events(1, "a") + step_events(1, "b"))
            with self.assertRaisesRegex(ValueError, "duplicate optimizer step 1"):
                analyze([path], 1, step_ranges=[(1, 1)])

    def test_missing_retained_update_still_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing.jsonl"
            write_events(path, step_events(1, "a"))
            with self.assertRaisesRegex(ValueError, "optimizer coverage mismatch"):
                analyze([path], 2, step_ranges=[(1, 2)])

    def test_range_validation(self):
        self.assertEqual(parse_step_range("351:500"), (351, 500))
        with self.assertRaisesRegex(ValueError, "one --step-ranges"):
            analyze([Path("unused")], step_ranges=[])
        with self.assertRaisesRegex(ValueError, "ranges overlap"):
            analyze([Path("unused1"), Path("unused2")], step_ranges=[(1, 351), (351, 500)])


if __name__ == "__main__":
    unittest.main()
