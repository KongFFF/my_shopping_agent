import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import retry_b800_training as tool
from b500_run import ROOT,PACKAGE,write,read,digest


class Retry800(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'outputs',prefix='b800-retry-test-');self.base=Path(self.temp.name).resolve()
        self.old=self.base/'old';self.parent=self.base/'B650';(self.old/'configs').mkdir(parents=True);(self.old/'train').mkdir()
        self.prefix=[{'event':'optimizer_step','global_step':i}for i in range(1,651)]
        self.parent_budget={'train':24008,'validation_physical':3000};self.budget={'train':24352,'validation_physical':3000}
        self.cfg=read(PACKAGE/'historical_a_composed_config.json');self.cfg['trainer']['total_training_steps']=800
        write(self.old/'configs/phase800.yaml',self.cfg)
        self.spec={'training_started':True,'resume_step':650,'target_step':800,'prior_B650':str(self.parent),
            'source_checkpoint':str(self.parent/'train/global_step_650'),'source_model_sha256':tool.continuation.MODEL_SHA,
            'protocol_sha256':digest(PACKAGE/'protocol.json'),'implementation_manifest_sha256':digest(PACKAGE/'implementation_manifest.json'),
            'config_hashes':{'800':digest(self.old/'configs/phase800.yaml')}}
        write(self.old/'run.json',self.spec);write(self.old/'stopped.json',{'error':'Phase800 failed/truncated; do not restart blindly'})
        write(self.old/'generation_budget.json',self.budget)
        (self.old/'train_to_800.log').write_text('RuntimeError: '+tool.SKIP_ERROR)
        (self.old/'training_diagnostics.jsonl').write_text(''.join(json.dumps(e)+'\n'for e in self.prefix+[{'event':'optimizer_step','global_step':i}for i in (651,652,653)]))
        (self.old/'generation_tokens.jsonl').write_text('physical-cost-history\n')
        self.mock=patch.object(tool.continuation,'source_evidence',return_value=({},self.prefix,self.parent_budget));self.mock.start()
    def tearDown(self):
        self.mock.stop();self.base.relative_to((ROOT/'outputs').resolve());self.temp.cleanup()
    def test_supported_failed_attempt_identified(self):
        _,parent,budget=tool.evidence(self.old);self.assertEqual(parent,self.parent);self.assertEqual(budget,self.budget)
    def test_retry_preserves_failed_files_and_sampling_cost_but_rolls_back_updates(self):
        before={str(p.relative_to(self.old)):p.read_bytes()for p in self.old.rglob('*')if p.is_file()};out=self.base/'new'
        def prepare(parent,dest):
            self.assertEqual(parent,self.parent);(dest/'configs').mkdir(parents=True)
            write(dest/'configs/phase800.yaml',self.cfg);write(dest/'generation_budget.json',self.parent_budget)
            write(dest/'run.json',{'source_checkpoint':str(parent/'train/global_step_650'),'code_branch':'codex/b800-training'})
            (dest/'training_diagnostics.jsonl').write_text(''.join(json.dumps(e)+'\n'for e in self.prefix))
        with patch.object(tool.continuation,'prepare',side_effect=prepare)as native:receipt=tool.stage(self.old,out)
        native.assert_called_once_with(self.parent,out)
        self.assertEqual(read(out/'generation_budget.json'),self.budget)
        self.assertEqual(read(out/'configs/phase800.yaml'),self.cfg)
        self.assertEqual(tool.rows(out/'training_diagnostics.jsonl'),self.prefix)
        self.assertEqual((out/'generation_tokens.jsonl').read_bytes(),(self.old/'generation_tokens.jsonl').read_bytes())
        self.assertEqual(read(out/'run.json')['code_branch'],'codex/b800-retry650')
        self.assertEqual(receipt['rolled_back_updates'],[651,652,653]);self.assertFalse(receipt['seed_changed']);self.assertFalse(receipt['skip_limit_changed'])
        self.assertEqual(before,{str(p.relative_to(self.old)):p.read_bytes()for p in self.old.rglob('*')if p.is_file()})
        self.assertEqual((out/'prior_attempt__training_diagnostics.jsonl').read_bytes(),(self.old/'training_diagnostics.jsonl').read_bytes())
    def test_other_failure_blocks_before_preparation(self):
        (self.old/'train_to_800.log').write_text('CUDA out of memory')
        with patch.object(tool.continuation,'prepare')as native:
            with self.assertRaisesRegex(ValueError,'does not confirm'):tool.stage(self.old,self.base/'new')
            native.assert_not_called()
    def test_even_partial_new_checkpoint_requires_inspection(self):
        (self.old/'train/global_step_700').mkdir()
        with self.assertRaisesRegex(ValueError,'checkpoint evidence'):tool.evidence(self.old)
    def test_unexpected_later_updates_block(self):
        with(self.old/'training_diagnostics.jsonl').open('a')as f:f.write(json.dumps({'event':'optimizer_step','global_step':654})+'\n')
        with self.assertRaisesRegex(ValueError,'chain1..653'):tool.evidence(self.old)
    def test_decreasing_cost_rejected(self):
        write(self.old/'generation_budget.json',{'train':24000,'validation_physical':3000})
        with self.assertRaisesRegex(ValueError,'ledger'):tool.evidence(self.old)
    def test_second_automatic_retry_forbidden(self):
        write(self.old/'run.json',dict(self.spec,retry_of='previous'))
        with self.assertRaisesRegex(ValueError,'only one retry'):tool.evidence(self.old)


if __name__=='__main__':unittest.main()
