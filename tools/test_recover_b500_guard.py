from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from b500_run import write,read,PACKAGE,digest,accepted_cpu_receipt
from recover_b500_guard import OLD_IMPL,OLD_ERROR,zero_update_proof,prepare_recovery


class Recovery(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.old=Path(self.temp.name)/'old';self.old.mkdir()
        (self.old/'train').mkdir();(self.old/'configs').mkdir();(self.old/'validation').mkdir()
        self.spec={'protocol_sha256':digest(PACKAGE/'protocol.json'),'implementation_manifest_sha256':OLD_IMPL,
            'project':'/asset','training_started':True,'launch_attempted':True,'started_at':time.time()-600,'deadline':time.time()+71400}
        write(self.old/'run.json',self.spec);write(self.old/'stopped.json',{'error':'test failure'})
        write(self.old/'generation_budget.json',{'train':16,'validation_physical':200})
        (self.old/'train_to_350.log').write_text(OLD_ERROR)
    def tearDown(self):self.temp.cleanup()
    def test_zero_update_and_budget_proof(self):
        p=zero_update_proof(self.old);self.assertTrue(p['zero_updates'])
        self.assertEqual(p['carried_budget'],{'train':16,'validation_physical':200})
        self.assertEqual(p['inherited_deadline'],self.spec['deadline'])
    def test_actor_started_or_optimizer_or_checkpoint_forbids_retry(self):
        for name,value in [('weights.jsonl',{'event':'actor_update_started'}),
            ('training_diagnostics.jsonl',{'event':'optimizer_step','global_step':1})]:
            write(self.old/name,value)
            with self.assertRaises(ValueError):zero_update_proof(self.old)
            (self.old/name).unlink()
        (self.old/'train/global_step_1').mkdir()
        with self.assertRaises(ValueError):zero_update_proof(self.old)
    def test_unknown_failure_and_missing_counter_forbid_retry(self):
        (self.old/'train_to_350.log').write_text('unrelated failure')
        with self.assertRaises(ValueError):zero_update_proof(self.old)
        (self.old/'train_to_350.log').write_text(OLD_ERROR)
        write(self.old/'generation_budget.json',{'train':16})
        with self.assertRaises(ValueError):zero_update_proof(self.old)
    def test_expired_clock_or_extra_validation_forbid_retry(self):
        self.spec.update(started_at=time.time()-72000,deadline=time.time()-1);write(self.old/'run.json',self.spec)
        with self.assertRaises(ValueError):zero_update_proof(self.old)
        self.spec.update(started_at=time.time()-600,deadline=time.time()+71400);write(self.old/'run.json',self.spec)
        write(self.old/'generation_budget.json',{'train':16,'validation_physical':208})
        with self.assertRaises(ValueError):zero_update_proof(self.old)
    def test_recovery_carries_cost_preserves_source_and_keeps_initial_Dev(self):
        out=Path(self.temp.name)/'new';receipt=Path(self.temp.name)/'cpu.json';write(receipt,{'passed':True})
        (self.old/'generation_tokens.jsonl').write_text('{"validation":false}\n')
        (self.old/'configs/phase350.yaml').write_text('old config')
        (self.old/'validation/0.jsonl').write_text('old Dev')
        before={str(p.relative_to(self.old)):p.read_bytes()for p in self.old.rglob('*')if p.is_file()}
        def fake_prepare(asset,target,cpu_receipt):
            self.assertEqual(asset,Path('/asset'));self.assertEqual(cpu_receipt,receipt)
            target.mkdir();(target/'configs').mkdir();write(target/'run.json',{'training_started':False})
            (target/'configs/phase350.yaml').write_text('val_before_train: true\n')
        with patch('recover_b500_guard.prepare',side_effect=fake_prepare):
            prepare_recovery(self.old,out,receipt,zero_update_proof(self.old))
        self.assertEqual(before,{str(p.relative_to(self.old)):p.read_bytes()for p in self.old.rglob('*')if p.is_file()})
        self.assertEqual(read(out/'generation_budget.json'),{'train':16,'validation_physical':200})
        self.assertEqual((out/'generation_tokens.jsonl').read_bytes(),(self.old/'generation_tokens.jsonl').read_bytes())
        self.assertEqual(read(out/'run.json')['inherited_deadline'],self.spec['deadline'])
        self.assertFalse(read(out/'run.json')['training_started'])
        self.assertEqual((out/'configs/phase350.yaml').read_text(),'val_before_train: true\n')
        self.assertTrue((out/'prior_attempt__Dev_0.jsonl').exists())
    def test_old_or_gpu_initialized_CPU_receipt_rejected(self):
        receipt=Path(self.temp.name)/'receipt.json';r={'passed':True,'gpu_initialized':False,
            'model_weights_loaded':False,'training_started':False,'protocol_sha256':digest(PACKAGE/'protocol.json'),
            'implementation_manifest_sha256':digest(PACKAGE/'implementation_manifest.json')}
        write(receipt,r);accepted_cpu_receipt(receipt)
        r['gpu_initialized']=True;write(receipt,r)
        with self.assertRaises(ValueError):accepted_cpu_receipt(receipt)
        r.update(gpu_initialized=False,implementation_manifest_sha256=OLD_IMPL);write(receipt,r)
        # After the repair, the old accepted code receipt no longer qualifies.
        if r['implementation_manifest_sha256']!=digest(PACKAGE/'implementation_manifest.json'):
            with self.assertRaises(ValueError):accepted_cpu_receipt(receipt)


if __name__=='__main__':unittest.main()
