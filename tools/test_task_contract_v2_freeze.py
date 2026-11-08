"""Check that freeze guards reject drift without modifying frozen files."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import freeze_task_contract_v2 as freeze


class FrozenContractGuardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.package = Path(self.temp.name) / "package"
        self.package.mkdir()
        train = list(range(928))
        quarantined = list(range(928, 1000))
        self.artifacts = {
            "train.jsonl": "".join(json.dumps({"task_id": tid}) + "\n" for tid in train).encode(),
            "quarantine.jsonl": "".join(json.dumps({"task_id": tid}) + "\n" for tid in quarantined).encode(),
            "validation.jsonl": "".join(json.dumps({"task_id": tid}) + "\n" for tid in range(1000, 1050)).encode(),
            "manifest.json": json.dumps({"retained_count": 928, "quarantined_count": 72,
                                          "retained_task_ids": train, "quarantined_task_ids": quarantined}).encode(),
        }
        for name, content in self.artifacts.items():
            (self.package / name).write_bytes(content)
        self.lock = {"freeze_id": freeze.FREEZE_ID, "status": "FROZEN", "train_count": 928,
                     "quarantine_count": 72, "validation_count": 50,
                     "contract_files_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in self.artifacts.items()},
                     "rebuild_inputs_sha256": {}, "normalized_sources_sha256": {}}
        (self.package / freeze.LOCK_NAME).write_text(json.dumps(self.lock), encoding="utf-8")

    def test_exact_rebuild_and_lock_check_are_read_only(self):
        before = {p.name: p.read_bytes() for p in self.package.iterdir()}
        freeze.verify_lock(self.package)
        freeze.check_candidate(self.artifacts, self.lock)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.package.iterdir()})

    def test_changed_rebuild_is_rejected_before_write(self):
        changed = {**self.artifacts, "train.jsonl": self.artifacts["train.jsonl"] + b'{}\n'}
        before = (self.package / "train.jsonl").read_bytes()
        with self.assertRaisesRegex(ValueError, "would replace frozen"):
            freeze.check_candidate(changed, self.lock)
        self.assertEqual((self.package / "train.jsonl").read_bytes(), before)

    def test_modified_frozen_artifact_is_detected(self):
        (self.package / "validation.jsonl").write_bytes(b'{}\n')
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            freeze.verify_lock(self.package)

    def test_added_artifact_cannot_silently_extend_contract(self):
        with self.assertRaisesRegex(ValueError, "artifact set"):
            freeze.check_candidate({**self.artifacts, "extra.json": b'{}'}, self.lock)

    def test_indirect_price_dependency_drift_is_detected(self):
        bundle = Path(self.temp.name) / "bundle"
        name = "web_agent_site/engine/variant_price.py"
        source = bundle / "runtime" / name
        source.parent.mkdir(parents=True)
        source.write_bytes(b'PRICE = "original"\n')
        lock = {"runtime_files_normalized_sha256": {name: freeze.digest(source, normalized=True)}}
        freeze.verify_bundle_lock(bundle, lock)
        source.write_bytes(b'PRICE = "lowest_axis"\n')
        with self.assertRaisesRegex(ValueError, "dependency changed"):
            freeze.verify_bundle_lock(bundle, lock)

    def test_hash_paths_must_stay_inside_expected_root(self):
        with self.assertRaises(ValueError):
            freeze.local_path(self.package, "../outside")


if __name__ == "__main__":
    unittest.main()
