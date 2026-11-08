"""Regression for stale environment metadata and safe cloud synchronization."""
import copy
import hashlib
import json
import tempfile
import unittest
import importlib.util
import sys
from pathlib import Path

from install_evidence_training_runtime import ROOT,UPSTREAM,PACKAGE,FILES,install
from semantic_environment_preflight import check_environment,RUNTIME_FILES


def sha(path):return hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()


class EnvironmentSyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'outputs',prefix='environment_fix_')
        self.root=Path(self.tmp.name).resolve()
        assert self.root.is_relative_to((ROOT/'outputs').resolve())
        for rel in FILES:
            p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((PACKAGE/'files'/rel).read_bytes())
        for rel in RUNTIME_FILES.values():
            p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((UPSTREAM/rel).read_bytes().replace(b'\r\n',b'\n'))
        self.env=self.root/'data/environment.json'
    def tearDown(self):self.tmp.cleanup()
    def write(self,m):self.env.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

    def test_stale_hash_metadata_rejected_even_if_file_itself_was_frozen(self):
        m=json.loads(self.env.read_text(encoding='utf-8'))
        m['runtime_files_sha256']['web_agent_text_env.py']='448f5ec31fe5a71d1e43376d42cd84787a0e48fe96a8032fc0383f65e5928ea2'
        self.write(m)
        with self.assertRaisesRegex(ValueError,'disagrees with actual runtime'):
            check_environment(self.env,self.root,sha)

    def test_confirmed_metadata_passes_all_five_real_source_checks(self):
        self.assertEqual(check_environment(self.env,self.root,sha)['runtime_files_verified'],5)

    def test_linux_exact_byte_runtime_preflight_also_accepts_confirmed_fix(self):
        sys.path.insert(0,str(UPSTREAM/'src'))
        spec=importlib.util.spec_from_file_location('environment_sync_runtime_check',UPSTREAM/'scripts/check_grpo_runtime.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.validate_reward_runtime_files(json.loads(self.env.read_text(encoding='utf-8')),self.root)

    def test_actual_changed_source_still_rejected(self):
        p=self.root/RUNTIME_FILES['web_agent_text_env.py'];p.write_bytes(p.read_bytes()+b'\n# real drift\n')
        with self.assertRaisesRegex(ValueError,'web_agent_text_env.py'):
            check_environment(self.env,self.root,sha)

    def test_known_old_metadata_is_backed_up_then_fixed(self):
        m=json.loads(self.env.read_text(encoding='utf-8'))
        m['runtime_files_sha256']['web_agent_text_env.py']='448f5ec31fe5a71d1e43376d42cd84787a0e48fe96a8032fc0383f65e5928ea2'
        self.write(m);before=self.env.read_bytes()
        result=install(self.root,backup_existing=True,backup_root=self.root/'backups')
        self.assertEqual(result['changed_files'],['data/environment.json'])
        self.assertEqual((Path(result['backup'])/'data/environment.json').read_bytes(),before)
        check_environment(self.env,self.root,sha)

    def test_semantically_equal_json_formatting_is_canonicalized_with_backup(self):
        m=json.loads(self.env.read_text(encoding='utf-8'));self.env.write_text(json.dumps(m),encoding='utf-8')
        before=self.env.read_bytes()
        result=install(self.root,backup_existing=True,backup_root=self.root/'backups')
        self.assertEqual((Path(result['backup'])/'data/environment.json').read_bytes(),before)
        self.assertEqual(self.env.read_bytes(),(PACKAGE/'files/data/environment.json').read_bytes())

    def test_unknown_environment_policy_is_never_overwritten_even_with_backup_flag(self):
        m=json.loads(self.env.read_text(encoding='utf-8'));m['reward']['gold_purchase']=9;self.write(m)
        code=self.root/'scripts/train_grpo.py';code.write_bytes(b'local source must remain')
        before=self.env.read_bytes()
        with self.assertRaisesRegex(ValueError,'/reward/gold_purchase'):
            install(self.root,backup_existing=True,backup_root=self.root/'backups')
        self.assertEqual(self.env.read_bytes(),before);self.assertEqual(code.read_bytes(),b'local source must remain')


if __name__=='__main__':unittest.main()
