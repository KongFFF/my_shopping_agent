"""CPU tests: actual receive/render/adapter paths, no model or HTTP service.

Gym's Env base is stubbed only when Gym is unavailable; server initialization,
catalog loading and NLP models are bypassed using a tiny in-memory fixture.
"""

import asyncio
import copy
import importlib.machinery
import importlib.util
import json
import os
import sys
import types
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = Path(os.environ.get("PURCHASE_GUARD_BUNDLE", ROOT / "outputs/purchase_guard_v1"))
sys.path[:0] = [str(BUNDLE / "agent"), str(BUNDLE / "environment"), str(ROOT / "tools")]
if importlib.util.find_spec("gym") is None:
    gym = types.ModuleType("gym")
    gym.Env = object
    gym.__spec__ = importlib.machinery.ModuleSpec("gym", None)
    gym.__path__ = []
    sys.modules["gym"] = gym
    envs = types.ModuleType("gym.envs")
    envs.__path__ = []
    registration = types.ModuleType("gym.envs.registration")
    registration.register = lambda **kwargs: None
    sys.modules["gym.envs"] = envs
    sys.modules["gym.envs.registration"] = registration

from web_agent_site.envs import web_agent_text_env as web
from web_agent_site.engine.purchase_guard import guard_purchase, inspect_purchase
from web_agent_site.engine.reward_features import compile_reward_features
from web_agent_site.engine.variant_price import resolve_variant_price
from shopping_grpo.environment.observation import render_structured_observation
from shopping_grpo.training.grpo.adapter.runtime import current_environment, current_runtime_state, make_runtime_state, reward_breakdown
from shopping_grpo.training.grpo.adapter.tools import ShopSimulatorTool
from shopping_grpo.training.grpo.dynamic_sampling import select_reward_varying_groups

Server = next(value for value in vars(web).values() if isinstance(value, type) and hasattr(value, "receive") and hasattr(value, "_terminal_status"))


def product():
    return {"asin": "100000000001", "title": "照明灯", "category": "家居›灯具", "query": "灯",
            "product_category": "灯具", "instruction_text": "照明灯", "attribute": ["照明"], "Attributes": ["照明"],
            "brand": "", "pricing": [10, 20], "Price": [10, 20], "images": [],
            "option_to_price": {"黑色": 10, "白色": 20}, "option_to_image": {},
            "description": "照明", "Description": "照明", "BulletPoints": [], "Reviews": [],
            "options": {"颜色分类": ["黑色", "白色"]},
            "customization_options": {"颜色分类": [{"value": "黑色", "price": 10}, {"value": "白色", "price": 20}]}}


def fixture(p=None):
    p = p or product()
    server = Server.__new__(Server)
    server.base_url = "http://fixture"
    server.assigned_instruction_text = None
    server.show_attrs = False
    server.environment_config = json.loads((BUNDLE / "environment/configs/environment.json").read_text(encoding="utf-8"))
    server.max_steps = server.environment_config["termination"]["max_steps"]
    server.product_item_dict = {p["asin"]: p}
    server.product_prices = {p["asin"]: 10}
    instruction = {"instruction": "黑色照明灯，预算30元以内", "attributes": ["照明"], "instruction_options": ["黑色"]}
    goal = {"asin": p["asin"], "price_upper": 30, "instruction_text": instruction["instruction"], **compile_reward_features(instruction, p)}
    server.user_sessions = {"s": {"asin": p["asin"], "options": {}, "goal": goal, "keywords": [], "page": 1,
                                 "actions": defaultdict(int), "selected_price": None,
                                 "price_resolution": resolve_variant_price(p, {}), "done": False}}
    return server


def buy(server):
    with web.app.app_context():
        return server.receive("s", "http://fixture/item_page/s", clickable_name="Buy Now")


def select(server, value="黑色"):
    from bs4 import BeautifulSoup
    node = BeautifulSoup('<label name="颜色分类"></label>', "html.parser").label
    return server.receive("s", "http://fixture/item_page/s", clickable_name=value,
                          text_to_clickable={value: node})


def public_state(server):
    session = server.user_sessions["s"]
    return web.build_observation_state(page_type="terminal" if session["done"] else "product_detail", session=session,
                                       product_item_dict=server.product_item_dict,
                                       available_actions={"has_search_bar": False, "clickables": ["Buy Now", "黑色", "白色", "Back to Search"]})


class GuardBehaviorTest(unittest.TestCase):
    def test_rejects_without_ending_or_purchase_side_effect(self):
        server = fixture()
        _, url, status = buy(server)
        session = server.user_sessions["s"]
        self.assertFalse(status["done"])
        self.assertFalse(session["done"])
        self.assertEqual(session["options"], {})
        self.assertEqual(session["actions"]["purchase"], 0)
        self.assertIn("item_page", url)
        self.assertEqual(session["purchase_guard_feedback"]["missing_axes"], ["颜色分类"])

    def test_selection_can_recover_to_original_gold(self):
        server = fixture()
        buy(server)
        select(server)
        _, _, status = buy(server)
        self.assertTrue(status["done"])
        self.assertEqual(status["reward_detail"]["reward_type"], "gold_purchase")
        self.assertEqual(status["reward"], 1)
        self.assertEqual(server.user_sessions["s"]["actions"]["purchase"], 1)

    def test_third_attempt_is_valid_failure_not_actual_purchase(self):
        server = fixture()
        for count in (1, 2):
            self.assertFalse(buy(server)[2]["done"])
        status = buy(server)[2]
        self.assertTrue(status["done"])
        self.assertEqual(status["reward"], -0.85)
        self.assertEqual(status["reward_detail"]["reward_type"], "wrong_purchase")
        self.assertTrue(status["reward_valid"])
        self.assertFalse(status["reward_detail"]["sampling_invalid"])
        self.assertFalse(status["reward_detail"]["purchase_success"])
        self.assertEqual(status["purchase"], {})
        self.assertEqual(server.user_sessions["s"]["actions"]["purchase"], 0)

    def test_counter_does_not_reset_on_unrelated_progress_or_product_change(self):
        server = fixture()
        buy(server)
        session = server.user_sessions["s"]
        session.pop("purchase_guard_feedback", None)
        session["actions"]["search"] += 1
        other = copy.deepcopy(product())
        other["asin"] = "100000000002"
        server.product_item_dict[other["asin"]] = other
        session["asin"] = other["asin"]
        buy(server)
        self.assertTrue(buy(server)[2]["done"])

    def test_reset_clears_attempts_history_and_feedback(self):
        server = fixture()
        buy(server)
        server.receive("s", "http://fixture/item_page/s")
        session = server.user_sessions["s"]
        self.assertEqual(session["purchase_guard_incomplete_attempts"], 0)
        self.assertEqual(session["purchase_guard_history"], [])
        self.assertNotIn("purchase_guard_feedback", session)
        self.assertIsNone(session["asin"])

    def test_real_selection_clears_message_but_preserves_attempt_count(self):
        server = fixture()
        buy(server)
        status = select(server)[2]
        self.assertFalse(status["done"])
        self.assertEqual(server.user_sessions["s"]["options"], {"颜色分类": "黑色"})
        self.assertEqual(server.user_sessions["s"]["selected_price"], 10)
        self.assertNotIn("purchase_guard_feedback", public_state(server))
        self.assertNotIn("purchase_guard", public_state(server))
        self.assertEqual(server.user_sessions["s"]["purchase_guard_incomplete_attempts"], 1)

    def test_metadata_missing_is_not_misreported_as_missing_selection(self):
        p = product()
        p["customization_options"]["尺寸"] = [{"value": "小", "price": 30}, {"value": "大", "price": 40}]
        server = fixture(p)
        session = server.user_sessions["s"]
        session["options"] = {"颜色分类": "黑色", "尺寸": "小"}
        session["price_resolution"] = resolve_variant_price(p, session["options"])
        self.assertEqual(inspect_purchase(p, session["options"])["status"], "metadata_unverifiable")
        result = buy(server)[2]
        self.assertEqual(result["reward_detail"]["reward_type"], "reward_unverifiable")
        self.assertFalse(result["reward_valid"])
        self.assertNotIn("purchase_guard_incomplete_attempts", session)

    def test_constant_price_axes_not_artificially_required(self):
        p = product()
        for entry in p["customization_options"]["颜色分类"]:
            entry["price"] = 10
        server = fixture(p)
        status = buy(server)[2]
        self.assertTrue(status["done"])
        self.assertEqual(status["reward_detail"]["reward_type"], "partial_alternative_purchase")
        self.assertNotIn("purchase_guard_incomplete_attempts", server.user_sessions["s"])

    def test_no_options_purchase_is_unchanged(self):
        p = product()
        p["customization_options"] = {}
        p["pricing"] = [10]
        self.assertEqual(inspect_purchase(p, {})["status"], "ready")

    def test_disabled_guard_has_original_unverifiable_behavior(self):
        server = fixture()
        self.assertIsNone(guard_purchase(server, "s", enabled=False))
        self.assertNotIn("purchase_guard_incomplete_attempts", server.user_sessions["s"])

    def test_does_not_read_private_goal_or_propose_correct_value(self):
        server = fixture()
        class Forbidden(dict):
            def __getitem__(self, key):
                raise AssertionError("Private goal read")
        server.user_sessions["s"]["goal"] = Forbidden()
        result = guard_purchase(server, "s")
        text = json.dumps(result["feedback"], ensure_ascii=False)
        self.assertNotIn("黑色", text)
        self.assertNotIn("白色", text)
        self.assertNotIn("100000000001", text)

    def test_feedback_is_visible_to_shared_renderer(self):
        server = fixture()
        buy(server)
        text = render_structured_observation(public_state(server))
        self.assertIn("购买未执行", text)
        self.assertIn("颜色分类", text)
        self.assertTrue(text.endswith('可点击的按钮: ["Buy Now", "黑色", "白色", "Back to Search"]'))

    def test_limit_validation(self):
        for value in (0, 11, True, 1.5):
            with self.assertRaises(ValueError):
                guard_purchase(fixture(), "s", max_attempts=value)


class FakeServiceClient:
    def __init__(self):
        self.server = fixture()

    def step(self, action):
        if action == "click[Buy Now]":
            status = buy(self.server)[2]
        elif action == "click[黑色]":
            select(self.server)
            status = {"done": False, "reward": 0}
        else:
            raise ValueError(action)
        return {**status, "over": bool(status["done"]), "observation_state": public_state(self.server)}


class TrainingAdapterTest(unittest.TestCase):
    def execute(self, actions):
        client = FakeServiceClient()
        state = make_runtime_state(1, 30)
        state["latest_observation"] = render_structured_observation(public_state(client.server))
        env_token = current_environment.set(client)
        state_token = current_runtime_state.set(state)
        responses = []
        try:
            for name in actions:
                schema = {"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {}}}}
                tool = ShopSimulatorTool({}, schema)
                responses.append(asyncio.run(tool.execute("test", {"value": "黑色"} if name == "select_option" else {})))
            return state, responses
        finally:
            current_environment.reset(env_token)
            current_runtime_state.reset(state_token)

    def test_prompt_then_selection_then_gold(self):
        state, responses = self.execute(["buy_now", "select_option", "buy_now"])
        self.assertIn("购买未执行", responses[0][0].text)
        self.assertEqual(state["steps"][0]["purchase_guard"]["status"], "rejected")
        self.assertEqual(state["reward_type"], "gold_purchase")
        self.assertEqual(reward_breakdown(state)["total"], 1)
        self.assertFalse(reward_breakdown(state)["sampling_invalid"])

    def test_exhaustion_survives_adapter_and_group_filter(self):
        state, _ = self.execute(["buy_now", "buy_now", "buy_now"])
        result = reward_breakdown(state)
        self.assertEqual(result["total"], -0.85)
        self.assertFalse(result["sampling_invalid"])
        self.assertFalse(result["infrastructure_invalid"])
        kept, stats = select_reward_varying_groups(["g"] * 4, [1, -0.85, -0.85, 1],
                                                   terminal_utilities=[1, -0.85, -0.85, 1],
                                                   sampling_invalid=[False, result["sampling_invalid"], False, False])
        self.assertEqual(len(kept), 4)


class EvaluationAdapterTest(unittest.TestCase):
    def test_same_feedback_and_recovery_on_evaluation_path(self):
        from shopping_grpo.evaluation.rollout import _execute_tool_call
        client = FakeServiceClient()
        def execute(name, index, arguments=None):
            return _execute_tool_call(client, {"id": str(index), "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments or {})}}, index)
        first = execute("buy_now", 1)
        self.assertFalse(first["done"])
        self.assertIn("购买未执行", first["observation"])
        execute("select_option", 2, {"value": "黑色"})
        final = execute("buy_now", 3)
        self.assertTrue(final["done"])
        self.assertEqual(final["result"]["reward_detail"]["reward_type"], "gold_purchase")


class EnvironmentStepTest(unittest.TestCase):
    def make_env(self, max_steps=None):
        server = fixture()
        if max_steps is not None:
            server.max_steps = max_steps
        server.receive("s", "http://fixture/item_page/s")
        session = server.user_sessions["s"]
        session.update(asin="100000000001", keywords=[], page=1)
        env = web.WebAgentTextEnv(server=server, session="s", observation_mode="html")
        env.instruction_text = session["goal"]["instruction_text"]
        env.browser.session_id = "s"
        env.browser.current_url = "http://fixture/item_page/s"
        with web.app.app_context(), web.app.test_request_context():
            env.browser.page_source = web.map_action_to_html(
                "click", session_id="s", product_info=server.product_item_dict[session["asin"]],
                keywords=[], page=1, asin=session["asin"], options={},
                instruction_text=env.instruction_text, show_attrs=False, selected_price=None)
        return env

    def test_rejections_count_as_steps_and_third_failure_is_preserved(self):
        env = self.make_env()
        self.assertFalse(env.step("click[Buy Now]")[1]["done"])
        self.assertFalse(env.step("click[Buy Now]")[1]["done"])
        self.assertEqual(env.server.user_sessions["s"]["progress_tracker"].steps, 2)
        status = env.step("click[Buy Now]")[1]
        self.assertEqual(status["termination_reason"], "wrong_purchase")
        self.assertEqual(env.structured_observation()["purchase_guard"]["status"], "exhausted")

    def test_global_step_limit_still_applies(self):
        env = self.make_env(max_steps=1)
        status = env.step("click[Buy Now]")[1]
        self.assertTrue(status["done"])
        self.assertEqual(status["termination_reason"], "max_steps")

    def test_disabled_bundle_executes_original_incomplete_purchase(self):
        path = ROOT / "outputs/purchase_guard_v1_disabled/environment/web_agent_site/envs/web_agent_text_env.py"
        self.assertTrue(path.exists(), "Build the disabled control bundle before running this suite")
        spec = importlib.util.spec_from_file_location("purchase_guard_disabled_server", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        server = fixture()
        server.__class__ = module.SimServer
        result = buy(server)[2]
        self.assertTrue(result["done"])
        self.assertEqual(result["reward_detail"]["reward_type"], "reward_unverifiable")
        self.assertFalse(result["reward_valid"])
        self.assertNotIn("purchase_guard_incomplete_attempts", server.user_sessions["s"])


class OriginalDataTest(unittest.TestCase):
    def test_original_files_and_scoring_are_unchanged(self):
        import hashlib
        manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["train_count"], 1000)
        self.assertEqual(manifest["base"], "original-upstream-not-task-contract-v2")
        for name, digest in manifest["dataset_unchanged_sha256"].items():
            self.assertEqual(hashlib.sha256((BUNDLE / "data" / name).read_bytes()).hexdigest(), digest)
            self.assertEqual((BUNDLE / "data" / name).read_bytes(), (ROOT / "upstream/shopping-grpo-longhorizon/data/grpo" / name).read_bytes())
        for name in ("constraints.py", "reward.py", "reward_features.py", "variant_price.py"):
            self.assertEqual((BUNDLE / "environment/web_agent_site/engine" / name).read_bytes(),
                             (ROOT / "upstream/shopping-grpo-longhorizon/environments/ShopSimulator/shop_env/web_agent_site/engine" / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
