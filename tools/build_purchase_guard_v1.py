"""Build an isolated guard experiment using ORIGINAL 1000-task data and scoring."""

import argparse
import ast
import difflib
import gzip
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "upstream/shopping-grpo-longhorizon"
SHOP = UP / "environments/ShopSimulator/shop_env"
PACKAGE = ROOT / "experiments/purchase_guard_v1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def sources(enabled=True, max_attempts=3, preserve_action_values=False):
    server_path = "web_agent_site/envs/web_agent_text_env.py"
    original = (SHOP / server_path).read_text(encoding="utf-8")
    source = original
    start = source.index('                    # If "buy now" clicked, calculate reward')
    end = source.index("                elif clickable_name == BACK_TO_SEARCH.lower():", start)
    old_block = source[start:end]
    guard_block = f'''                    from web_agent_site.engine.purchase_guard import guard_purchase
                    guard = guard_purchase(self, session_id, enabled={enabled!r}, max_attempts={max_attempts})
                    if guard is not None:
                        product = self.product_item_dict[session["asin"]]
                        if guard["result"] is None:
                            html = map_action_to_html(
                                'click', session_id=session_id, product_info=product,
                                keywords=session.get("keywords", []), page=session.get("page"),
                                asin=session["asin"], options=session["options"],
                                instruction_text=session["goal"]["instruction_text"],
                                show_attrs=self.show_attrs, selected_price=session.get("selected_price"))
                            url = current_url
                        else:
                            status.update(self._terminal_status(session_id, guard["result"]))
                            html = map_action_to_html(
                                f'click[{{END_BUTTON}}]', session_id=session_id,
                                reward=guard["result"].reward, asin=session["asin"],
                                options=session["options"], instruction_text=session["goal"]["instruction_text"])
                            url = f'{{self.base_url}}/done/{{session_id}}/{{session["asin"]}}/{{session["options"]}}'
                    else:
'''
    source = source[:start] + guard_block + "".join("    " + line if line.strip() else line for line in old_block.splitlines(keepends=True)) + source[end:]
    needle = '        status = dict(reward=0.0, done=False)\n'
    assert source.count(needle) == 1
    source = source.replace(needle, needle + '''        # Clear the last message, not the trajectory-wide attempt budget.
        if session_id in self.user_sessions:
            self.user_sessions[session_id].pop("purchase_guard_feedback", None)
''')
    needle = "                        'options': dict(),\n"
    assert source.count(needle) == 1
    source = source.replace(needle, needle + "                        'purchase_guard_incomplete_attempts': 0,\n                        'purchase_guard_history': [],\n")
    changes = {"environment/" + server_path: (original, source)}
    name = "web_agent_site/engine/observation.py"
    original = (SHOP / name).read_text(encoding="utf-8")
    needle = '    if page_type == "search_results":\n'
    assert original.count(needle) == 1
    source = original.replace(needle, '''    if session.get("purchase_guard_feedback"):
        state["purchase_guard"] = dict(session["purchase_guard_feedback"])
    if page_type == "search_results":
''')
    changes["environment/" + name] = (original, source)
    name = "shopping_grpo/environment/observation.py"
    original = (UP / "src" / name).read_text(encoding="utf-8")
    needle = '    return "\\n".join(lines) + "\\n\\n" + "\\n".join(_footer(state))\n'
    assert original.count(needle) == 1
    source = original.replace(needle, '''    feedback = state.get("purchase_guard")
    if isinstance(feedback, Mapping):
        # Whitelist public diagnostic fields, never echo an arbitrary payload.
        public = {key: feedback[key] for key in ("version", "status", "missing_axes", "attempt", "max_attempts", "reason", "message") if key in feedback}
        lines.append("purchase_guard: " + json.dumps(public, ensure_ascii=False))
    return "\\n".join(lines) + "\\n\\n" + "\\n".join(_footer(state))
''')
    if preserve_action_values:
        needle = '    actions = _list(state.get("actions"))\n'
        assert source.count(needle) == 1
        source = source.replace(needle, '''    # Action strings are executable identifiers, not display prose.
    # Preserve whitespace so the footer, option list and server agree.
    raw_actions = state.get("actions")
    actions = [value for value in raw_actions if isinstance(value, str) and value.strip()] if isinstance(raw_actions, list) else []
''')
    changes["agent/" + name] = (original, source)
    name = "shopping_grpo/training/grpo/adapter/tools.py"
    original = (UP / "src" / name).read_text(encoding="utf-8")
    needle = '        state["consecutive_guard_rejections"] = 0\n'
    assert original.count(needle) == 1
    source = original.replace(needle, '''        feedback = (result.get("observation_state") or {}).get("purchase_guard")
        if isinstance(feedback, dict):
            step["purchase_guard"] = {key: feedback[key] for key in ("version", "status", "missing_axes", "attempt", "max_attempts", "reason") if key in feedback}
        state["consecutive_guard_rejections"] = 0
''')
    changes["agent/" + name] = (original, source)
    name = "shopping_grpo/training/grpo/adapter/agent_loop.py"
    original = (UP / "src" / name).read_text(encoding="utf-8")
    needle = '                "steps": len(state["steps"]),\n'
    assert original.count(needle) == 1
    source = original.replace(needle, needle + '''                "purchase_guard_events": [step["purchase_guard"] for step in state["steps"] if "purchase_guard" in step],
''')
    changes["agent/" + name] = (original, source)
    helper = (ROOT / "tools/purchase_guard_v1.py").read_text(encoding="utf-8")
    changes["environment/web_agent_site/engine/purchase_guard.py"] = ("", helper)
    for name, (_, text) in changes.items():
        ast.parse(text, filename=name)
    return changes


def build(target, *, enabled=True, max_attempts=3, preserve_action_values=False):
    if not 1 <= max_attempts <= 10:
        raise ValueError("max_attempts must be in [1,10]")
    target = target.resolve()
    if target.exists():
        raise ValueError(f"Refusing to overwrite {target}")
    changes = sources(enabled, max_attempts, preserve_action_values)
    target.mkdir(parents=True)
    ignore = shutil.ignore_patterns("data", "search_engine", "__pycache__", ".venv*", ".git", "*.log", ".ipynb_checkpoints", "outputs")
    shutil.copytree(SHOP, target / "environment", ignore=ignore)
    shutil.copytree(UP / "src/shopping_grpo", target / "agent/shopping_grpo", ignore=ignore)
    for name, (_, text) in changes.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    data = target / "data"
    data.mkdir()
    unchanged = {}
    for name in ("train.jsonl", "train.parquet", "validation.jsonl", "validation.parquet"):
        content = (UP / "data/grpo" / name).read_bytes()
        (data / name).write_bytes(content)
        unchanged[name] = sha(content)
    train = [json.loads(line)["task_id"] for line in (data / "train.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    val = [json.loads(line)["task_id"] for line in (data / "validation.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    final = [json.loads(line)["task_id"] for line in (UP / "data/evaluation/tasks.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    assert len(train) == len(set(train)) == 1000 and len(val) == 50 and not set(train) & (set(val) | set(final))
    with gzip.open(SHOP / "data/fine_items_eval_train_all.json.gz", "rb") as handle:
        catalog_bytes = handle.read()
    (target / "environment/data").mkdir()
    (target / "environment/data/items_eval_train.json").write_bytes(catalog_bytes)
    # Build the original product search index with the original catalog bytes.
    import sys
    sys.path.insert(0, str(SHOP))
    from web_agent_site.engine.search import build_index
    index = build_index(json.loads(catalog_bytes), target / "environment/search_engine/products.sqlite3", product_data_sha256=sha(catalog_bytes))
    (target / "index_manifest.json").write_bytes(json_bytes(index))
    manifest = {"variant": "purchase-guard-v1", "base": "original-upstream-not-task-contract-v2",
                "enabled": enabled, "max_incomplete_attempts": max_attempts, "train_count": 1000, "validation_count": 50,
                "dataset_unchanged_sha256": unchanged, "original_catalog_sha256": sha(catalog_bytes),
                "changed_files_sha256": {name: sha(text.encode("utf-8")) for name, (_, text) in changes.items()},
                "base_sources_normalized_sha256": {name: sha(old.encode("utf-8")) for name, (old, _) in changes.items() if old},
                "limits": ["Only demonstrably unselected effective price axes trigger rejection.",
                           "No gold, annotation or query is used to choose missing axes or values.",
                           "Metadata-unverifiable purchases retain original behavior.",
                           "Exhaustion is valid wrong_purchase at the existing configured failure reward; no actual purchase recorded.",
                           "Model and HTTP service not started by this builder."]}
    if preserve_action_values:
        manifest["variant"] = "purchase-guard-v1.1"
        manifest["preserve_action_values"] = True
    env = json.loads((UP / "data/environment.json").read_text(encoding="utf-8"))
    env["product_data_sha256"] = sha(catalog_bytes)
    paths = {"observation.py": "web_agent_site/engine/observation.py", "pack_api.py": "shop_env/pack_api.py",
             "reward.py": "web_agent_site/engine/reward.py", "slot_lease_pool.py": "shop_env/slot_lease_pool.py",
             "web_agent_text_env.py": "web_agent_site/envs/web_agent_text_env.py"}
    env["runtime_files_sha256"] = {name: sha((target / "environment" / path).read_bytes()) for name, path in paths.items()}
    env["purchase_guard_experiment"] = {key: manifest[key] for key in ("variant", "enabled", "max_incomplete_attempts", "changed_files_sha256")}
    (target / "environment.json").write_bytes(json_bytes(env))
    (target / "manifest.json").write_bytes(json_bytes(manifest))
    # Publish small, reviewable source overlays. Separate on/off receipts prevent
    # accidental mixing; no files under upstream or frozen repairs are edited.
    package_root = PACKAGE.with_name("purchase_guard_v1_1") if preserve_action_values else PACKAGE
    package = package_root / ("enabled" if enabled else "disabled")
    package.mkdir(parents=True, exist_ok=True)
    patch = []
    for name, (old, text) in changes.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
        patch.extend(difflib.unified_diff(old.splitlines(keepends=True), text.splitlines(keepends=True), fromfile="a/" + name, tofile="b/" + name))
    (package / "runtime.patch").write_bytes("".join(patch).encode("utf-8"))
    (package / "manifest.json").write_bytes(json_bytes(manifest))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialize", required=True, type=Path)
    parser.add_argument("--disabled", action="store_true")
    parser.add_argument("--max-attempts", default=3, type=int)
    parser.add_argument("--preserve-action-values", action="store_true",
                        help="Build isolated v1.1: preserve exact action strings in the observation footer")
    args = parser.parse_args()
    print(json.dumps(build(args.materialize, enabled=not args.disabled, max_attempts=args.max_attempts,
                           preserve_action_values=args.preserve_action_values), ensure_ascii=False, indent=2))
