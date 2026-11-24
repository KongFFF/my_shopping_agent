"""Run all nine original protocol guards without external historical artifacts.

Only the unit-test pipeline's A500 summary is a fixture. Real evaluation still
loads and validates the original A650 summaries through c650_evaluation.prepare.
"""
from pathlib import Path
from unittest.mock import patch
import unittest
import test_b500_dev50 as shared


class IndependentProtocolGuards(shared.IndependentDev):
    def setUp(self):
        super().setUp()
        original=shared.tool.read
        summary=shared.tool.ROOT/'analysis/grpo_baseline_20261022_202816/grpo_dev50/summary.json'
        def fixture_read(path):
            if Path(path)==summary:
                return {'strict_successes':27,'expected_tasks':50,'strict_success_task_ids':list(range(27))}
            return original(path)
        self.fixture_patch=patch.object(shared.tool,'read',side_effect=fixture_read)
        self.fixture_patch.start()
    def tearDown(self):
        try:self.fixture_patch.stop()
        finally:super().tearDown()


if __name__=='__main__':unittest.main()
