import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import prepare_b800_finish750 as tool
from b500_run import ROOT,PACKAGE,read,write,digest
from b500_run_control import REQUIRED,install


class Finish750(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'outputs',prefix='finish750-test-');self.base=Path(self.temp.name).resolve();self.old=self.base/'old'
        (self.old/'configs').mkdir(parents=True)
        self.cfg=read(PACKAGE/'historical_a_composed_config.json');self.cfg['trainer']['total_training_steps']=800
        self.spec={'project':'/asset','runtime':'/runtime','prior_B650':'/parent650','env':{'OMP_NUM_THREADS':'1'},
            'protocol_sha256':digest(PACKAGE/'protocol.json'),'implementation_manifest_sha256':digest(PACKAGE/'implementation_manifest.json')}
        self.cfg['ray_kwargs']['ray_init']['runtime_env']['env_vars']=self.spec['env']
        import yaml
        (self.old/'configs/phase800.yaml').write_text(yaml.safe_dump(self.cfg,sort_keys=False));self.spec['config_hashes']={'800':digest(self.old/'configs/phase800.yaml')};write(self.old/'run.json',self.spec)
        self.events=[{'event':'optimizer_step','global_step':n}for n in range(1,788)]+[{'event':'skipped_update','global_step':788}]
        self.weights=[{'event':event,'driver_global_step_before_update':n}for n in range(650,787)for event in ['actor_update_started','actor_update_returned']]
        for name,rs in [('training_diagnostics.jsonl',self.events),('weights.jsonl',self.weights)]:
            (self.old/name).write_text(''.join(json.dumps(r)+'\n'for r in rs))
        self.budget={'train':30144,'validation_physical':3400};write(self.old/'generation_budget.json',self.budget);write(self.old/'stopped.json',{'error':'failed'})
        (self.old/'train_to_800.log').write_text('RuntimeError: '+tool.FAILURE);(self.old/'generation_tokens.jsonl').write_text('physical-history\n')
        for n in [700,750]:
            for name in REQUIRED:
                p=self.old/f'train/global_step_{n}'/name;p.parent.mkdir(parents=True,exist_ok=True)
                if p.suffix=='.pt':
                    with zipfile.ZipFile(p,'w')as z:z.writestr('data.pkl',b'fixture750')
                else:p.write_text('{}')
        (self.old/'train/latest_checkpointed_iteration.txt').write_text('750')
        (self.old/'train/checkpoint_retention.jsonl').write_text(json.dumps({'event':'retention_plan','new_step':750,'keep':[700,750]})+'\n')
        (self.old/'validation').mkdir();(self.old/'validation/750.jsonl').write_text('native-validation\n')
        self.audit=self.base/'audit.json';expected={'audit_passed':True,'source_run':str(self.old),'cumulative_generation':self.budget}
        for rel,k in [('run.json','source_run_sha256'),('training_diagnostics.jsonl','source_diagnostics_sha256'),('weights.jsonl','source_weights_sha256'),('configs/phase800.yaml','source_config_sha256')]:expected[k]=digest(self.old/rel)
        write(self.audit,expected);self.audit_patch=patch.object(tool,'AUDIT',self.audit);self.audit_patch.start()
        self.cpu=self.base/'cpu.json';write(self.cpu,{'passed':True,'protocol_sha256':digest(PACKAGE/'protocol.json'),
            'implementation_manifest_sha256':digest(PACKAGE/'implementation_manifest.json'),'gpu_initialized':False,'model_weights_loaded':False,'training_started':False})
    def tearDown(self):
        self.audit_patch.stop();self.base.relative_to((ROOT/'outputs').resolve());self.temp.cleanup()
    def stage(self,out,free=2**50):
        with patch.object(tool,'verify_runtime_snapshot'),patch.object(tool.subprocess,'run'),patch.object(tool.shutil,'disk_usage',return_value=SimpleNamespace(free=free)):
            tool.stage(self.old,out,self.cpu)
    def test_only_stopping_parameter_changes_in_learning_recipe(self):
        before=copy.deepcopy(self.cfg);cfg,env=tool.derive_config(self.cfg,self.spec,self.old,self.base/'new')
        self.assertEqual(self.cfg,before)
        expected=copy.deepcopy(before['shopping_dynamic_sampling']);expected['max_consecutive_skipped_updates']=50
        self.assertEqual(cfg['shopping_dynamic_sampling'],expected)
        for key in ['algorithm','data']:self.assertEqual(cfg[key],before[key])
        self.assertEqual(cfg['actor_rollout_ref']['actor'],before['actor_rollout_ref']['actor']);self.assertEqual(cfg['actor_rollout_ref']['model'],before['actor_rollout_ref']['model'])
        self.assertEqual(cfg['trainer']['resume_from_path'],str(self.old/'train/global_step_750'));self.assertFalse(cfg['trainer']['val_before_train'])
        self.assertEqual(cfg['trainer']['total_training_steps'],800);self.assertEqual(cfg['actor_rollout_ref']['actor']['optim']['lr'],1e-6)
        self.assertEqual(cfg['ray_kwargs']['ray_init']['runtime_env']['env_vars'],env)
    def test_restore_prefix750_and_preserve_all_physical_cost_and_failed_state(self):
        out=self.base/'new';before={str(p.relative_to(self.old)):p.read_bytes()for p in self.old.rglob('*')if p.is_file()};self.stage(out)
        self.assertEqual([e['global_step']for e in tool.rows(out/'training_diagnostics.jsonl')],list(range(1,751)))
        self.assertEqual(len(tool.rows(out/'weights.jsonl')),200);self.assertLess(max(w['driver_global_step_before_update']for w in tool.rows(out/'weights.jsonl')),750)
        self.assertEqual(read(out/'generation_budget.json'),self.budget);self.assertEqual((out/'generation_tokens.jsonl').read_bytes(),(self.old/'generation_tokens.jsonl').read_bytes())
        self.assertEqual(read(out/'finish750_preparation.json')['rolled_back_updates'],list(range(751,788)))
        self.assertEqual(list((out/'train').iterdir()),[]);self.assertEqual(read(out/'run.json')['resume_step'],750)
        self.assertEqual((out/'interrupted__training_diagnostics.jsonl').read_bytes(),(self.old/'training_diagnostics.jsonl').read_bytes())
        self.assertEqual(before,{str(p.relative_to(self.old)):p.read_bytes()for p in self.old.rglob('*')if p.is_file()})
    def test_missing_optimizer_blocks_before_output(self):
        (self.old/'train/global_step_750/actor/optim_world_size_1_rank_0.pt').unlink()
        with self.assertRaisesRegex(RuntimeError,'Incomplete checkpoint'):self.stage(self.base/'new')
        self.assertFalse((self.base/'new').exists())
    def test_bad_checkpoint_crc_blocks_before_output(self):
        (self.old/'train/global_step_750/data.pt').write_bytes(b'truncated')
        with self.assertRaises(zipfile.BadZipFile):self.stage(self.base/'new')
        self.assertFalse((self.base/'new').exists())
    def test_low_disk_preserves_source_and_creates_no_output(self):
        with self.assertRaisesRegex(ValueError,'Insufficient disk'):self.stage(self.base/'new',free=0)
        self.assertTrue((self.old/'train/global_step_750/data.pt').exists());self.assertFalse((self.base/'new').exists())
    def test_diagnostics_changed_since_return_blocks(self):
        with(self.old/'training_diagnostics.jsonl').open('a')as f:f.write('{}\n')
        with self.assertRaisesRegex(ValueError,'Audited source changed'):tool.evidence(self.old)
    def test_unaccounted_generation_change_blocks(self):
        write(self.old/'generation_budget.json',{'train':30145,'validation_physical':3400})
        with self.assertRaisesRegex(ValueError,'ledger changed'):tool.evidence(self.old)
    def test_tracker_mismatch_blocks(self):
        (self.old/'train/latest_checkpointed_iteration.txt').write_text('700')
        with self.assertRaisesRegex(RuntimeError,'tracker mismatch'):tool.evidence(self.old)
    def test_first751_and_final800_receipts(self):
        out=self.base/'live';out.mkdir();train=out/'train'
        for name in REQUIRED:
            p=train/'global_step_800'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'x')
        (train/'latest_checkpointed_iteration.txt').write_text('800');module=SimpleNamespace(append_training_diagnostic=lambda *a,**k:None)
        class T:
            def _save_checkpoint(self):pass
            def __init__(self):self.async_rollout_manager=SimpleNamespace(generate_sequences=lambda b:None)
            def fit(self):
                for n in (751,800):module.append_training_diagnostic('file','optimizer_step',n,metrics={'actor/lr':1e-6,'actor/loss':0.,'actor/grad_norm':1.,'training/optimizer_updated':1,'b500/actor_tokens':8})
        install(T,module,{'output':str(out),'train_dir':str(train),'stop_at':800,'final_step':800,'first_new_step':751,'deadline':None,'budget_record_only':True})
        T().fit();self.assertTrue((out/'accepted_step_751.json').exists());self.assertTrue((out/'accepted_step_800.json').exists())


if __name__=='__main__':unittest.main()
