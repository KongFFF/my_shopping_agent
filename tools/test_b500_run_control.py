from pathlib import Path
import json
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/group_weight_b500_v1/overlay'))
from b500_run_control import reserve,BudgetStop,checkpoint,REQUIRED,install,retain_checkpoints,install_checkpoint_retention
from b500_run import remap,prepare,belongs_to_run,digest,PACKAGE


class Controls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='b500-control-',dir=ROOT/'outputs');self.out=Path(self.temp.name)
    def tearDown(self):
        self.out.resolve().relative_to((ROOT/'outputs').resolve());self.temp.cleanup()
    def test_generation_reserved_before_call_and_persistent(self):
        p=self.out/'budget.json';reserve(p,8,False,time.time()+3600);reserve(p,8,False,time.time()+3600)
        self.assertEqual(json.loads(p.read_text())['train'],16)
        reserve(p,8,True,time.time()+3600);self.assertEqual(json.loads(p.read_text())['train'],16)
    def test_cap_refuses_next_full_batch_without_increment(self):
        p=self.out/'budget.json';p.write_text(json.dumps({'train':23996,'validation_physical':0}))
        with self.assertRaises(BudgetStop):reserve(p,8,False,time.time()+3600)
        self.assertEqual(json.loads(p.read_text())['train'],23996)
    def test_deadline_blocks_generation(self):
        with self.assertRaises(BudgetStop):reserve(self.out/'budget.json',8,False,time.time()+60)
        self.assertFalse((self.out/'budget.json').exists())
    def make_checkpoint(self,n):
        train=self.out/'train'
        for f in REQUIRED:
            p=train/f'global_step_{n}'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'x')
        (train/'latest_checkpointed_iteration.txt').write_text(str(n));return train
    def test_checkpoint_all_components_and_tracker(self):
        train=self.make_checkpoint(350);checkpoint(train,350)
        (train/'latest_checkpointed_iteration.txt').write_text('300')
        with self.assertRaises(RuntimeError):checkpoint(train,350)
    def test_control_stops_after_checkpoint_and_diagnostics(self):
        train=self.make_checkpoint(350);events=[]
        module=SimpleNamespace(append_training_diagnostic=lambda *a,**k:events.append(a))
        class Trainer:
            def _save_checkpoint(self):pass
            def __init__(self):self.async_rollout_manager=SimpleNamespace(generate_sequences=lambda b:'ok')
            def _shutdown_dump_executor(self):self.closed=True
            def fit(self):
                module.append_training_diagnostic('file','optimizer_step',350,metrics={'actor/lr':1e-6,'actor/loss':0.,
                    'actor/grad_norm':1.,'training/optimizer_updated':1,'b500/actor_tokens':100})
                raise AssertionError('Must not generate step351 in phase350')
        install(Trainer,module,{'output':str(self.out),'train_dir':str(train),'stop_at':350,'deadline':time.time()+3600})
        t=Trainer();t.fit();self.assertTrue(t.closed);self.assertEqual(len(events),1)
        self.assertTrue((self.out/'phase350_complete.json').is_file())
    def test_path_remap_preserves_shared_SFT_and_data(self):
        a=Path('/asset');r=Path('/runtime')
        self.assertEqual(remap('/asset/configs/grpo.yaml',a,r),'/runtime/configs/grpo.yaml')
        self.assertEqual(remap('/asset/outputs/sft/merged',a,r),'/asset/outputs/sft/merged')
        self.assertEqual(remap('/asset/data/train.parquet',a,r),'/asset/data/train.parquet')
    def test_cleanup_only_exact_run_and_trainer_role(self):
        self.assertTrue(belongs_to_run(b'B500_RUN_DIR=/run/a\0B500_PROCESS_ROLE=trainer\0',Path('/run/a')))
        self.assertFalse(belongs_to_run(b'B500_RUN_DIR=/run/ab\0B500_PROCESS_ROLE=trainer\0',Path('/run/a')))
        self.assertFalse(belongs_to_run(b'B500_RUN_DIR=/run/a\0B500_PROCESS_ROLE=environment\0',Path('/run/a')))
    def test_prepare_preserves_training_500_warmup_and_resume(self):
        import yaml
        out=self.out/'new-run';asset=Path('/root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon')
        receipt=self.out/'native_cpu.json';receipt.write_text(json.dumps({'passed':True,
            'implementation_manifest_sha256':digest(PACKAGE/'implementation_manifest.json'),
            'protocol_sha256':digest(PACKAGE/'protocol.json'),'gpu_initialized':False,
            'model_weights_loaded':False,'training_started':False}))
        with patch('b500_run.call')as invoke,patch('b500_run.verify_runtime_snapshot'):
            prepare(asset,out,receipt)
            self.assertEqual(invoke.call_count,1)
        a=yaml.safe_load((out/'configs/phase350.yaml').read_text());b=yaml.safe_load((out/'configs/phase500.yaml').read_text())
        self.assertEqual(a['trainer']['total_training_steps'],500)
        self.assertEqual(a['actor_rollout_ref']['actor']['optim']['lr_warmup_steps_ratio'],.03)
        self.assertEqual(a['trainer']['resume_mode'],'disable')
        self.assertEqual(b['trainer']['resume_mode'],'resume_path');self.assertFalse(b['trainer']['val_before_train'])
        self.assertTrue(b['trainer']['resume_from_path'].endswith('global_step_350'))
        self.assertEqual(a['ray_kwargs']['ray_init']['runtime_env']['worker_process_setup_hook'],'b500_run_control.setup_worker')
        with self.assertRaises(ValueError):prepare(asset,out)


class Retention(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='b500-retention-',dir=ROOT/'outputs');self.root=Path(self.temp.name).resolve()
    def tearDown(self):
        self.root.relative_to((ROOT/'outputs').resolve());self.temp.cleanup()
    def save(self,n):
        for name in REQUIRED:
            p=self.root/f'global_step_{n}'/name;p.parent.mkdir(parents=True,exist_ok=True)
            if p.suffix=='.pt':
                with zipfile.ZipFile(p,'w')as z:z.writestr('data.pkl',b'synthetic test state')
            else:p.write_text('{}')
        (self.root/'latest_checkpointed_iteration.txt').write_text(str(n))
    def existing(self):return {int(p.name.rsplit('_',1)[1])for p in self.root.glob('global_step_*')}
    def trainer(self,fn):
        from test_b500_weighting import attrs
        cfg=attrs({'trainer':{'default_local_dir':str(self.root),'max_actor_ckpt_to_keep':10},
            'actor_rollout_ref':{'actor':{'checkpoint':{'async_save':False}}}})
        class T:
            config=cfg;global_steps=200
            def _save_checkpoint(self):
                if self.config.trainer.max_actor_ckpt_to_keep is not None:raise AssertionError('Native pre-delete not disabled')
                return fn(self.global_steps)
        install_checkpoint_retention(T);install_checkpoint_retention(T);return T()
    def test_scan_existing_and_keep_latest_two(self):
        for n in (1,10,25,50,100,150,200):self.save(n)
        retain_checkpoints(self.root,200);self.assertEqual(self.existing(),{150,200})
    def test_350_pinned_across_process_restart_and_later_saves(self):
        for n in (300,350,400):self.save(n)
        retain_checkpoints(self.root,400);self.assertEqual(self.existing(),{350,400})
        self.save(450);retain_checkpoints(self.root,450);self.assertEqual(self.existing(),{350,400,450})
        self.save(500);retain_checkpoints(self.root,500);self.assertEqual(self.existing(),{350,450,500})
    def test_failed_save_restores_config_without_deleting(self):
        self.save(100);self.save(150)
        def fail(n):raise OSError('disk full')
        t=self.trainer(fail)
        with self.assertRaises(OSError):t._save_checkpoint()
        self.assertEqual(self.existing(),{100,150});self.assertEqual(t.config.trainer.max_actor_ckpt_to_keep,10)
    def test_new_truncated_file_prevents_any_cleanup(self):
        for n in (50,100,150,200):self.save(n)
        (self.root/'global_step_200/data.pt').write_bytes(b'truncated')
        with self.assertRaises((zipfile.BadZipFile,RuntimeError)):retain_checkpoints(self.root,200)
        self.assertEqual(self.existing(),{50,100,150,200})
    def test_wrong_tracker_prevents_any_cleanup(self):
        for n in (100,150,200):self.save(n)
        (self.root/'latest_checkpointed_iteration.txt').write_text('150')
        with self.assertRaises(RuntimeError):retain_checkpoints(self.root,200)
        self.assertEqual(self.existing(),{100,150,200})
    def test_unfinished_old_directory_preserved(self):
        for n in (50,100,150,200):self.save(n)
        (self.root/'global_step_50/data.pt').unlink()
        retain_checkpoints(self.root,200);self.assertEqual(self.existing(),{50,150,200})
    def test_CRC_corruption_in_retained_fallback_stops_cleanup(self):
        for n in (50,100,150,200):self.save(n)
        p=self.root/'global_step_150/data.pt';raw=p.read_bytes().replace(b'synthetic test state',b'corrupted test state');p.write_bytes(raw)
        with self.assertRaises((zipfile.BadZipFile,RuntimeError)):retain_checkpoints(self.root,200)
        self.assertEqual(self.existing(),{50,100,150,200})
    def test_successful_save_prunes_after_original_returns(self):
        self.save(50);self.save(100);self.save(150)
        def original(n):
            self.assertEqual(self.existing(),{50,100,150});self.save(n);return 'saved'
        t=self.trainer(original);self.assertEqual(t._save_checkpoint(),'saved')
        self.assertEqual(self.existing(),{150,200});self.assertEqual(t.config.trainer.max_actor_ckpt_to_keep,10)
    def test_low_space_refuses_before_original_and_before_deletion(self):
        self.save(100);self.save(150);t=self.trainer(self.save)
        with patch('b500_run_control.shutil.disk_usage',return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(RuntimeError,'Insufficient space'):t._save_checkpoint()
        self.assertEqual(self.existing(),{100,150})
    def test_future_checkpoint_prevents_rollback_cleanup(self):
        self.save(150);self.save(200)
        (self.root/'latest_checkpointed_iteration.txt').write_text('150')
        with self.assertRaisesRegex(RuntimeError,'Future checkpoint'):retain_checkpoints(self.root,150)
        self.assertEqual(self.existing(),{150,200})


if __name__=='__main__':unittest.main()
