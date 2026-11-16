"""Recovery eligibility uses real returned failure shape; no actor/model loads."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from decision_probe_common import ROOT,save
from recover_decision_probe_startup import validate


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'outputs').mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'outputs');self.old=Path(self.tmp.name)
        save(self.old/'budget.json',{'started_at':100})
        save(self.old/'services_stopped.json',{'at':102.4,'budget_expired':False})
        save(self.old/'cpu_check.json',{'passed':True,'protocol_sha256':'frozen'})
        (self.old/'supervisor.log').write_text('file_hash\nKeyboardInterrupt',encoding='utf-8')
        self.mock=patch('recover_decision_probe_startup.checked_run',return_value={'protocol_sha256':'frozen'});self.mock.start()
    def tearDown(self):self.mock.stop();self.tmp.cleanup()
    def test_empty_startup_failure_allowed(self):
        self.assertAlmostEqual(validate(self.old)[1],2.4)
    def test_any_generation_or_service_artifact_blocks(self):
        for name in ('usage.jsonl','events.jsonl','trajectories.jsonl','ready.json','service_pids.json','model.log','driver.lock'):
            p=self.old/name;p.touch()
            with self.assertRaisesRegex(ValueError,'found'):validate(self.old)
            p.unlink()
    def test_unexpected_error_blocks(self):
        (self.old/'supervisor.log').write_text('OOM',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'differs'):validate(self.old)
    def test_long_or_expired_attempt_blocks(self):
        save(self.old/'services_stopped.json',{'at':7300,'budget_expired':True})
        with self.assertRaisesRegex(ValueError,'duration'):validate(self.old)
    def test_cpu_protocol_drift_blocks(self):
        save(self.old/'cpu_check.json',{'passed':True,'protocol_sha256':'changed'})
        with self.assertRaisesRegex(ValueError,'CPU'):validate(self.old)


if __name__=='__main__':unittest.main()
