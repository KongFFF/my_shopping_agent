"""Verify an isolated original-1000 purchase-guard bundle without launching it."""

import argparse
import gzip
import json
from pathlib import Path

from build_purchase_guard_v1 import PACKAGE, SHOP, UP, sha, sources


def verify(bundle):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest["base"] != "original-upstream-not-task-contract-v2":
        raise ValueError("Wrong experimental base")
    changes = sources(manifest["enabled"], manifest["max_incomplete_attempts"])
    expected = {name: sha(text.encode("utf-8")) for name, (_, text) in changes.items()}
    if manifest["changed_files_sha256"] != expected:
        raise ValueError("Source receipt differs from current builder")
    package = PACKAGE / ("enabled" if manifest["enabled"] else "disabled")
    for name, digest in expected.items():
        if sha((bundle / name).read_bytes()) != digest or sha((package / name).read_bytes()) != digest:
            raise ValueError(f"Changed overlay: {name}")
    # Compare every other runtime Python source to the original checkout.
    checked = 0
    for prefix, original_root in (("environment", SHOP), ("agent/shopping_grpo", UP / "src/shopping_grpo")):
        for path in (bundle / prefix).rglob("*.py"):
            relative = path.relative_to(bundle).as_posix()
            if relative not in expected:
                original = original_root / path.relative_to(bundle / prefix)
                if path.read_text(encoding="utf-8") != original.read_text(encoding="utf-8"):
                    raise ValueError(f"Unexpected runtime change: {relative}")
            checked += 1
    for name, digest in manifest["dataset_unchanged_sha256"].items():
        data = (bundle / "data" / name).read_bytes()
        if sha(data) != digest or data != (UP / "data/grpo" / name).read_bytes():
            raise ValueError(f"Dataset changed: {name}")
    counts = {name: len((bundle / "data" / name).read_text(encoding="utf-8-sig").splitlines())
              for name in ("train.jsonl", "validation.jsonl")}
    if counts != {"train.jsonl": 1000, "validation.jsonl": 50}:
        raise ValueError(f"Wrong task counts: {counts}")
    catalog = (bundle / "environment/data/items_eval_train.json").read_bytes()
    with gzip.open(SHOP / "data/fine_items_eval_train_all.json.gz", "rb") as handle:
        if catalog != handle.read():
            raise ValueError("Catalog changed")
    digest = sha(catalog)
    if digest != manifest["original_catalog_sha256"]:
        raise ValueError("Catalog receipt differs")
    index = json.loads((bundle / "index_manifest.json").read_text(encoding="utf-8"))
    if index["product_data_sha256"] != digest or index["index_sha256"] != sha((bundle / "environment/search_engine/products.sqlite3").read_bytes()):
        raise ValueError("Search index binding differs")
    env = json.loads((bundle / "environment.json").read_text(encoding="utf-8"))
    base_env = json.loads((UP / "data/environment.json").read_text(encoding="utf-8"))
    for key, value in base_env.items():
        if key not in ("runtime_files_sha256", "product_data_sha256") and env[key] != value:
            raise ValueError(f"Original environment setting changed: {key}")
    if env["product_data_sha256"] != digest or env["purchase_guard_experiment"]["changed_files_sha256"] != expected:
        raise ValueError("Environment manifest differs")
    for name, digest in env["runtime_files_sha256"].items():
        paths = list((bundle / "environment").rglob(name))
        if len(paths) != 1 or sha(paths[0].read_bytes()) != digest:
            raise ValueError(f"Environment runtime binding differs: {name}")
    return {"status": "PASS", "enabled": manifest["enabled"], "train_count": 1000,
            "validation_count": 50, "runtime_python_files_checked": checked,
            "original_scoring_and_catalog": "unchanged"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    print(json.dumps(verify(parser.parse_args().bundle), ensure_ascii=False, indent=2))
