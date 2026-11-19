from pathlib import Path
import json
import sys
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

from b500_run import ROOT,PACKAGE,read,write,digest
from prepare_b500_resume150 import evidence,stage
from b500_run_control import REQUIRED,reserve


class Resume150(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'outputs',prefix='b500-resume-test-')
        self.base=Path(self.temp.name);self.old=self.base/'old';self.out=self.base/'new'
        (self.old/'configs').mkdir(parents=True)
        hashes={}
        for n in (350,500):
            p=self.old/'configs'/f'phase{n}.yaml';p.write_text('original config');hashes[str(n)]=digest(p)
        for name in REQUIRED:
            p=self.old/'train/global_step_150'/name;p.parent.mkdir(parents=True,exist_ok=True)
            if p.suffix=='.pt':
                with zipfile.ZipFile(p,'w')as z:z.writestr('data.pkl',b'synthetic fixture')
            else:p.write_text('{}')
        (self.old/'train/latest_checkpointed_iteration.txt').write_text('150')
        write(self.old/'run.json',{'project':'/asset','protocol_sha256':digest(PACKAGE/'protocol.json'),
            'implementation_manifest_sha256':digest(PACKAGE/'implementation_manifest.json'),
            'config_hashes':hashes,'training_started':True})
        write(self.old/'stopped.json',{'error':'disk full'})
        write(self.old/'generation_budget.json',{'train':6392,'validation_physical':1000})
        (self.old/'training_diagnostics.jsonl').write_text(''.join(json.dumps({'event':'optimizer_step','global_step':n})+'\n'for n in range(1,200)))
        (self.old/'weights.jsonl').write_text('{"event":"old weights"}\n')
        self.receipt=self.base/'cpu.json';write(self.receipt,{'passed':True,
            'implementation_manifest_sha256':digest(PACKAGE/'implementation_manifest.json'),
            'protocol_sha256':digest(PACKAGE/'protocol.json'),'gpu_initialized':False,'model_weights_loaded':False,'training_started':False})
    def tearDown(self):
        self.base.resolve().relative_to((ROOT/'outputs').resolve());self.temp.cleanup()
    def test_evidence_chain_and_no_cost_reset(self):
        _,events,budget=evidence(self.old);self.assertEqual(len(events),199);self.assertEqual(budget['train'],6392)
    def test_missing_data_refuses_recovery(self):
        (self.old/'train/global_step_150/data.pt').unlink()
        with self.assertRaises(RuntimeError):evidence(self.old)
    def test_broken_original_chain_refuses_recovery(self):
        (self.old/'training_diagnostics.jsonl').write_text('{"event":"optimizer_step","global_step":150}\n')
        with self.assertRaises(ValueError):evidence(self.old)
    def test_stage_full_resume_without_copy_and_trim_rolled_back_updates(self):
        import yaml
        before={str(p.relative_to(self.old)):p.read_bytes()for p in self.old.rglob('*')if p.is_file()}
        with patch('b500_run.call'),patch('b500_run.verify_runtime_snapshot'),patch('prepare_b500_resume150.shutil.disk_usage',return_value=SimpleNamespace(free=2**50)):
            stage(self.old,self.out,self.receipt)
        cfg=yaml.safe_load((self.out/'configs/phase350.yaml').read_text());later=yaml.safe_load((self.out/'configs/phase500.yaml').read_text())
        self.assertEqual(cfg['trainer']['resume_from_path'],str(self.old/'train/global_step_150'))
        self.assertEqual(cfg['trainer']['resume_mode'],'resume_path');self.assertFalse(cfg['trainer']['val_before_train'])
        self.assertFalse(cfg['trainer']['del_local_ckpt_after_load'])
        self.assertEqual(cfg['trainer']['total_training_steps'],500)
        self.assertEqual(later['trainer']['resume_from_path'],str(self.out/'train/global_step_350'))
        self.assertEqual(len((self.out/'training_diagnostics.jsonl').read_text().splitlines()),150)
        self.assertEqual(read(self.out/'generation_budget.json')['train'],6392)
        self.assertTrue(read(self.out/'run.json')['budget_record_only'])
        self.assertEqual(list((self.out/'train').iterdir()),[])
        self.assertTrue((self.out/'weights_before_rollback.jsonl').exists())
        self.assertEqual(before,{str(p.relative_to(self.old)):p.read_bytes()for p in self.old.rglob('*')if p.is_file()})
    def test_disk_shortage_fails_before_new_run(self):
        with patch('prepare_b500_resume150.shutil.disk_usage',return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(ValueError,'Insufficient free space'):stage(self.old,self.out,self.receipt)
        self.assertFalse(self.out.exists())
    def test_budget_record_only_counts_beyond_old_caps_and_ignores_clock(self):
        p=self.base/'budget.json';write(p,{'train':24000,'validation_physical':2400})
        reserve(p,8,False,None,record_only=True);reserve(p,8,True,None,record_only=True)
        self.assertEqual(read(p),{'train':24008,'validation_physical':2408})


if __name__=='__main__':unittest.main()
