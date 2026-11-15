"""Verify retained-update accounting with synthetic logs; never launch services."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import diagnose_grpo651_to800 as tool
from prepare_bounded_grpo_resume import RUN, digest, write
from test_analyze_grpo_diagnostics import step_events


class DiagnosisTests(unittest.TestCase):
    def fixture(self, project):
        run = project / RUN
        prep = run / "preparation/formal_resume650_to800_v1"
        prep.mkdir(parents=True)
        write(prep / "launch_spec.json", {"resume_step": 650, "total_updates": 800})
        write(prep / "exit_status.json", {"exit_code": 0, "elapsed_seconds": 18280})
        folder = run / "formal_resume650_to800_v1"
        (folder / "validation").mkdir(parents=True)
        events = []
        for step in range(651, 801):
            gen, opt = step_events(step, "a")
            gen2, opt2 = step_events(step, "b")
            opt = copy.deepcopy(opt)
            gen["rollouts"] += gen2["rollouts"]
            gen["groups"] += gen2["groups"]
            opt["rollouts"] += copy.deepcopy(opt2["rollouts"])
            opt["metrics"].update({"training/optimizer_updated": 1, "actor/grad_norm": .2,
                "actor/pg_clipfrac": .01, "timing_s/step": 100 if step <= 750 else 120})
            events.extend([gen, opt])
        # An accepted group can be discarded on a skipped attempt at the same step.
        gen, _ = step_events(751, "discarded")
        events.insert(200, gen)
        events.insert(201, {"event": "skipped_update", "global_step": 751})
        # Constant/no-gold groups still cost generation, but contribute no update.
        gen, _ = step_events(651, "constant")
        gen["groups"][0].update(terminal_utilities=[0.0]*4, kept=False, accepted=False,
                                  drop_reason="constant_reward")
        for rollout in gen["rollouts"]:
            rollout.update(reward_type="repeat_loop", reward={"terminal_utility": 0.0})
        events.insert(0, gen)
        path = folder / "training_diagnostics.jsonl"
        self.save(path, events)
        for step, gold in ((700, 28), (750, 30), (800, 28)):
            (folder / "validation" / f"{step}.jsonl").write_text("".join(
                json.dumps({"step": step, "score": float(i < gold)})+"\n" for i in range(50)), encoding="utf-8")
        return path, events

    def save(self, path, events):
        path.write_text("".join(json.dumps(event)+"\n" for event in events), encoding="utf-8")

    def test_counts_boundaries_discarded_cost_and_missing_metric(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td)
            path, _ = self.fixture(project)
            original = digest(path)
            result = tool.diagnose(project)
            first, middle, last = result["windows"]
            self.assertEqual([w["counts"]["updates"] for w in result["windows"]], [50]*3)
            self.assertEqual(first["counts"]["generated_groups"], 101)
            self.assertEqual(first["counts"]["constant_reward_groups"], 1)
            self.assertEqual(middle["counts"]["generated_groups"], 100)
            self.assertEqual(last["counts"]["generated_groups"], 101)
            self.assertEqual(last["counts"]["trained_groups"], 100)
            self.assertEqual(last["counts"]["accepted_not_trained"], 1)
            self.assertEqual(last["counts"]["skipped_updates"], 1)
            self.assertEqual(last["rates"]["generated_rollouts_per_update"], 404/50)
            self.assertEqual(last["metrics"]["timing_s/step"]["mean"], 120)
            self.assertEqual(result["training_dev"][750]["gold"], 30)
            self.assertNotIn("actor/ppo_kl", last["metrics"])
            self.assertIn("未记录", tool.report(result))
            self.assertEqual(digest(path), original)

    def test_missing_update_duplicate_or_nonfinite_fails(self):
        for case in ("missing", "duplicate", "nonfinite", "unconfirmed", "uid_mismatch"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as td:
                project = Path(td)
                path, events = self.fixture(project)
                index = next(i for i,e in enumerate(events) if e["event"] == "optimizer_step")
                if case == "missing":
                    events.pop(index)
                elif case == "duplicate":
                    events.append(copy.deepcopy(events[index]))
                elif case == "nonfinite":
                    events[index]["metrics"]["actor/grad_norm"] = float("nan")
                elif case == "unconfirmed":
                    events[index]["metrics"]["training/optimizer_updated"] = 0
                else:
                    for rollout in events[index]["rollouts"][:4]:
                        rollout["uid"] = "unmatched"
                self.save(path, events)
                expected = {"missing": "exactly651", "duplicate": "duplicate optimizer",
                    "nonfinite": "non-finite", "unconfirmed": "confirmed optimizer",
                    "uid_mismatch": "no matching accepted generation"}[case]
                with self.assertRaisesRegex(ValueError, expected):
                    tool.diagnose(project)

    def test_incomplete_dev_and_parent_failure_fail(self):
        for case in ("dev", "parent"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as td:
                project = Path(td)
                path, _ = self.fixture(project)
                if case == "dev":
                    val = path.parent / "validation/800.jsonl"
                    val.write_text("\n".join(val.read_text().splitlines()[:-1]), encoding="utf-8")
                else:
                    write(project / RUN / "preparation/formal_resume650_to800_v1/exit_status.json", {"exit_code": 1})
                with self.assertRaises(ValueError):
                    tool.diagnose(project)


if __name__ == "__main__":
    unittest.main()
