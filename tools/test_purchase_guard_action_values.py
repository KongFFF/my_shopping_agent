"""Action strings must survive rendering unchanged, including repeated spaces."""

import asyncio
import importlib.util
import json
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PURCHASE_GUARD_BUNDLE", str(ROOT / "outputs/purchase_guard_v1_1"))
import test_purchase_guard_v1 as base
from shopping_grpo.environment.actions import action_reject_reason, clickable_buttons
from shopping_grpo.environment.projection import project_observation

EXACT = "bykski cpu冷头  黑色款"


def product(value=EXACT):
    p = base.product()
    p["options"]["颜色分类"][0] = value
    p["customization_options"]["颜色分类"][0]["value"] = value
    p["option_to_price"] = {value: 10, "白色": 20}
    return p


class ActionValueTest(unittest.TestCase):
    def test_footer_preserves_original_executable_strings(self):
        state = base.public_state(base.fixture(product()))
        text = base.render_structured_observation(state)
        self.assertEqual(clickable_buttons(text), state["actions"])
        self.assertIn(EXACT, json.loads(next(line.split(": ", 1)[1] for line in text.splitlines()
                                             if line.startswith("available_options:")))["颜色分类"])
        self.assertIsNone(action_reject_reason("select_option", {"value": EXACT}, text))

    def test_old_renderer_reproduces_false_rejection(self):
        path = ROOT / "experiments/purchase_guard_v1/enabled/agent/shopping_grpo/environment/observation.py"
        spec = importlib.util.spec_from_file_location("original_guard_renderer", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        text = module.render_structured_observation(base.public_state(base.fixture(product())))
        self.assertEqual(action_reject_reason("select_option", {"value": EXACT}, text),
                         "click_not_in_previous_observation")

    def test_real_html_step_selects_double_space_value_after_purchase_hint(self):
        env = base.EnvironmentStepTest().make_env(p=product())
        self.assertFalse(env.step("click[Buy Now]")[1]["done"])
        state = env.structured_observation()
        text = base.render_structured_observation(state)
        self.assertIsNone(action_reject_reason("select_option", {"value": EXACT}, text))
        self.assertFalse(env.step(f"click[{EXACT}]")[1]["done"])
        session = env.server.user_sessions["s"]
        self.assertEqual(session["options"]["颜色分类"], EXACT)
        self.assertEqual(session["selected_price"], 10)
        self.assertEqual(session["actions"]["options"], 1)
        self.assertNotIn("purchase_guard", env.structured_observation())
        self.assertTrue(env.step("click[Buy Now]")[1]["reward_valid"])

    def test_missing_space_is_still_rejected_without_guessing(self):
        exact = "3包不锈钢料箱- 全自动高速+无料停机（质保一年"
        text = base.render_structured_observation(base.public_state(base.fixture(product(exact))))
        self.assertIsNone(action_reject_reason("select_option", {"value": exact}, text))
        wrong = exact.replace("- ", "-")
        self.assertEqual(action_reject_reason("select_option", {"value": wrong}, text),
                         "click_not_in_previous_observation")

    def test_similar_action_values_are_not_merged(self):
        state = base.public_state(base.fixture(product()))
        state["actions"] += [EXACT.replace("  ", " "), " 无默认规格 ", "a\tb"]
        text = base.render_structured_observation(state)
        self.assertEqual(clickable_buttons(text), state["actions"])

    def test_projection_keeps_action_values_and_hint(self):
        server = base.fixture(product())
        base.buy(server)
        state = base.public_state(server)
        state["product"]["title"] = "很长的商品介绍" * 3000
        raw = base.render_structured_observation(state)
        text, meta = project_observation("buy_now", raw, count_tokens=len,
                                         token_budget=1536, detail_token_budget=4096,
                                         generic_token_budget=768)
        self.assertTrue(meta.truncated)
        self.assertEqual(clickable_buttons(text), state["actions"])
        self.assertIn(state["purchase_guard"]["message"], text)

    def test_training_adapter_can_execute_exact_value(self):
        class Client:
            def __init__(self):
                self.server = base.fixture(product())

            def step(self, action):
                if action == "click[Buy Now]":
                    status = base.buy(self.server)[2]
                elif action == f"click[{EXACT}]":
                    status = base.select(self.server, EXACT)[2]
                else:
                    raise AssertionError(action)
                return {**status, "over": status["done"], "observation_state": base.public_state(self.server)}

        client = Client()
        state = base.make_runtime_state(1, 30)
        state["latest_observation"] = base.render_structured_observation(base.public_state(client.server))
        env_token = base.current_environment.set(client)
        state_token = base.current_runtime_state.set(state)
        try:
            for name, arguments in (("buy_now", {}), ("select_option", {"value": EXACT}), ("buy_now", {})):
                schema = {"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {}}}}
                asyncio.run(base.ShopSimulatorTool({}, schema).execute("test", arguments))
            self.assertEqual([step["tool"] for step in state["steps"]], ["buy_now", "select_option", "buy_now"])
            self.assertEqual(client.server.user_sessions["s"]["options"]["颜色分类"], EXACT)
            self.assertFalse(base.reward_breakdown(state)["sampling_invalid"])
            self.assertNotEqual(state["reward_type"], "reward_unverifiable")
        finally:
            base.current_environment.reset(env_token)
            base.current_runtime_state.reset(state_token)


if __name__ == "__main__":
    unittest.main()
