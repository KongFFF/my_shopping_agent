import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

import c800_training as tool
from c800_run import ROOT,PACKAGE,write,read,digest
from c500_run_control import REQUIRED,install


class Continue800(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'outputs',prefix='c800-test-');self.base=Path(self.temp.name).resolve();self.parent=self.base/'C650'
        (self.parent/'configs').mkdir(parents=True)
        import yaml
        self.cfg=read(PACKAGE/'historical_a_composed_config.json');(self.parent/'configs/phase650.yaml').write_text(yaml.safe_dump(self.cfg))
        for name in REQUIRED:
            p=self.parent/'train/global_step_650'/name;p.parent.mkdir(parents=True,exist_ok=True)
            if p.suffix=='.pt':
                with zipfile.ZipFile(p,'w')as z:z.writestr('data.pkl',b'fixture')
            else:p.write_text('{}')
        (self.parent/'train/latest_checkpointed_iteration.txt').write_text('650')
        self.cfg['trainer']['total_training_steps']=650
        self.cfg['ray_kwargs']['ray_init']['runtime_env']['worker_process_setup_hook']='c500_run_control.setup_worker'
        (self.parent/'configs/phase650.yaml').write_text(yaml.safe_dump(self.cfg))
        self.spec={'experiment':'C650','gamma':4.0,'project':'/asset','runtime':'/runtime','repo':'/oldcode','protocol_sha256':digest(PACKAGE/'protocol.json'),
            'implementation_manifest_sha256':digest(PACKAGE/'implementation_manifest.json'),'config_hashes':{'650':digest(self.parent/'configs/phase650.yaml')},'env':{'OMP_NUM_THREADS':'1'}}
        self.spec.update(resume_step=500,target_step=650,phase_stops=[650],continuation_protocol_sha256=digest(ROOT/'experiments/group_weight_c650_v1/protocol.json'))
        write(self.parent/'accepted_step_650.json',{'step':650,'checkpoint':str(self.parent/'train/global_step_650')})
        self.budget={'train':22920,'validation_physical':2800};write(self.parent/'run.json',self.spec)
        write(self.parent/'training_complete.json',{'experiment':'C650','gamma':4.0,'completed_updates':650,'generation_budget':self.budget,'protocol_sha256':self.spec['protocol_sha256'],'continuation_protocol_sha256':self.spec['continuation_protocol_sha256']})
        write(self.parent/'generation_budget.json',self.budget)
        (self.parent/'training_diagnostics.jsonl').write_text(''.join(json.dumps({'event':'optimizer_step','global_step':n})+'\n'for n in range(1,651)))
        self.cpu=self.base/'cpu.json';write(self.cpu,{'passed':True,'implementation_manifest_sha256':digest(PACKAGE/'implementation_manifest.json'),
            'protocol_sha256':digest(PACKAGE/'protocol.json'),'gpu_initialized':False,'model_weights_loaded':False,'training_started':False})
    def tearDown(self):
        self.base.relative_to((ROOT/'outputs').resolve());self.temp.cleanup()
    def test_derive_keeps_learning_and_sampling_fields(self):
        cfg,env=tool.derive_config(self.cfg,self.spec,self.parent,self.base/'new')
        self.assertEqual(cfg['trainer']['total_training_steps'],800);self.assertFalse(cfg['trainer']['val_before_train'])
        self.assertEqual(cfg['trainer']['resume_from_path'],str(self.parent/'train/global_step_650'))
        for key in ['algorithm','data','shopping_dynamic_sampling']:self.assertEqual(cfg[key],self.cfg[key])
        self.assertEqual(cfg['actor_rollout_ref']['actor'],self.cfg['actor_rollout_ref']['actor'])
        self.assertEqual(env['C500_RUN_DIR'],str(self.base/'new'));self.assertNotIn('B500_RUN_DIR',env);self.assertNotIn('/oldcode',env['PYTHONPATH'])
    def test_complete_stage_no_checkpoint_copy_no_parent_mutation(self):
        out=self.base/'new';sha=digest(self.parent/'train/global_step_650/actor/model_world_size_1_rank_0.pt')
        before={str(p.relative_to(self.parent)):p.read_bytes()for p in self.parent.rglob('*')if p.is_file()}
        def history(cmd,**kw):write(Path(cmd[cmd.index('--output')+1]),{'passed':True})
        with patch.object(tool,'MODEL_SHA',sha),patch.object(tool,'verify_runtime_snapshot'),patch.object(tool.subprocess,'run',side_effect=history),patch.object(tool.shutil,'disk_usage',return_value=SimpleNamespace(free=2**50)):
            tool.stage(self.parent,out,self.cpu)
        new=read(out/'run.json');self.assertEqual(new['phase_stops'],[800]);self.assertEqual(new['target_step'],800)
        self.assertEqual(new['experiment'],'C800');self.assertEqual(new['gamma'],4.);self.assertTrue(new['budget_record_only'])
        self.assertEqual(read(out/'generation_budget.json'),self.budget);self.assertEqual(list((out/'train').iterdir()),[])
        self.assertEqual(len((out/'training_diagnostics.jsonl').read_text().splitlines()),650)
        self.assertEqual(before,{str(p.relative_to(self.parent)):p.read_bytes()for p in self.parent.rglob('*')if p.is_file()})
    def test_wrong_model_rejected(self):
        with patch.object(tool,'verify_runtime_snapshot'):
            with self.assertRaisesRegex(ValueError,'not the C650'):tool.stage(self.parent,self.base/'new',self.cpu)
    def test_missing_optimizer_rejected(self):
        (self.parent/'train/global_step_650/actor/optim_world_size_1_rank_0.pt').unlink()
        with patch.object(tool,'verify_runtime_snapshot'):
            with self.assertRaises(RuntimeError):tool.stage(self.parent,self.base/'new',self.cpu)
    def test_low_disk_blocks_before_new_output(self):
        sha=digest(self.parent/'train/global_step_650/actor/model_world_size_1_rank_0.pt')
        with patch.object(tool,'MODEL_SHA',sha),patch.object(tool,'verify_runtime_snapshot'),patch.object(tool.shutil,'disk_usage',return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(ValueError,'Insufficient disk'):tool.stage(self.parent,self.base/'new',self.cpu)
        self.assertFalse((self.base/'new').exists())
    def test_wrong_gamma_parent_rejected(self):
        spec=read(self.parent/'run.json');spec['gamma']=2;write(self.parent/'run.json',spec)
        with self.assertRaisesRegex(ValueError,'gamma4'):tool.source_evidence(self.parent)
    def test_parent_continuation_drift_rejected(self):
        spec=read(self.parent/'run.json');spec['continuation_protocol_sha256']='wrong';write(self.parent/'run.json',spec)
        with self.assertRaisesRegex(ValueError,'continuation protocol drift'):tool.source_evidence(self.parent)
    def test_failed_parent_rejected(self):
        write(self.parent/'stopped.json',{'error':'incomplete'})
        with self.assertRaisesRegex(ValueError,'completed C650'):tool.source_evidence(self.parent)
    def test_noncontinuous_parent_chain_rejected(self):
        p=self.parent/'training_diagnostics.jsonl';p.write_text(p.read_text().replace(json.dumps({'event':'optimizer_step','global_step':649})+'\n',''))
        with self.assertRaisesRegex(ValueError,'exactly1..650'):tool.source_evidence(self.parent)
    def test_wrong_checkpoint_receipt_rejected(self):
        write(self.parent/'accepted_step_650.json',{'step':650,'checkpoint':'/other/global_step_650'})
        with self.assertRaisesRegex(ValueError,'checkpoint receipt'):tool.source_evidence(self.parent)
    def test_record_only_generation_can_exceed_old_cap(self):
        from c500_run_control import reserve
        path=self.base/'budget.json';write(path,{'train':30000,'validation_physical':3000})
        reserve(path,8,False,None,record_only=True)
        self.assertEqual(read(path)['train'],30008)

    def test_651_and_800_receipts_no_750_restart(self):
        out=self.base/'live';out.mkdir();train=out/'train'
        for name in REQUIRED:
            p=train/'global_step_800'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'x')
        (train/'latest_checkpointed_iteration.txt').write_text('800')
        module=SimpleNamespace(append_training_diagnostic=lambda *a,**k:None)
        class T:
            def _save_checkpoint(self):pass
            def __init__(self):self.async_rollout_manager=SimpleNamespace(generate_sequences=lambda b:None)
            def fit(self):
                for n in (651,750,800):module.append_training_diagnostic('file','optimizer_step',n,metrics={'actor/lr':1e-6,'actor/loss':0.,'actor/grad_norm':1.,'training/optimizer_updated':1,'c500/actor_tokens':8})
        install(T,module,{'output':str(out),'train_dir':str(train),'stop_at':800,'final_step':800,'first_new_step':651,'deadline':None,'budget_record_only':True})
        T().fit();self.assertTrue((out/'accepted_step_651.json').exists());self.assertTrue((out/'accepted_step_800.json').exists());self.assertFalse((out/'phase350_complete.json').exists())


if __name__=='__main__':unittest.main()
