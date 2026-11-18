from pathlib import Path
import json
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/group_weight_b500_v1/overlay'))
from b500_run_control import reserve,BudgetStop,checkpoint,REQUIRED,install
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


if __name__=='__main__':unittest.main()
