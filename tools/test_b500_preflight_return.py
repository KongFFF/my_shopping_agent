import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import run_b500_preflight_return as runner

ROOT=Path(__file__).resolve().parents[1]


class FlowTests(unittest.TestCase):
    def exercise(self,mode):
        temporary=tempfile.TemporaryDirectory(prefix='b500-flow-test-',dir=ROOT/'outputs')
        repo=Path(temporary.name);calls=[];published=[]
        protocol=b'{}\n';h=hashlib.sha256(protocol).hexdigest()
        def git(args,cwd,**kw):
            calls.append(args)
            if args[0]=='remote':return 'test-origin'
            if args[0]=='rev-parse':return 'a'*40
            if args[0]=='config'and '--get'in args:return ''
            if args[0]=='clone'and mode!='clone_failure':
                clone=Path(args[-1]);(clone/'tools').mkdir()
                package=clone/'experiments/group_weight_b500_v1';package.mkdir(parents=True)
                (package/'protocol.json').write_bytes(protocol)
                if mode!='missing_tool':
                    for name in ['check_b500_history.py','run_b500_cpu_acceptance.py']:(clone/'tools'/name).write_bytes(b'test-tool')
                (package/'manifest.json').write_text(json.dumps({'files_sha256':{'protocol.json':h},
                    'tools_sha256':{'check_b500_history.py':hashlib.sha256(b'test-tool').hexdigest()}}))
            if args[0]=='clone'and mode=='clone_failure':raise RuntimeError('clone failed')
            if args[0]=='commit'and mode=='commit_failure':raise RuntimeError('commit failed')
            return 0
        def process(args,**kw):
            if mode=='no_receipt':return SimpleNamespace(returncode=2)
            r={'version':'b500-history-preflight-v1','protocol_sha256':h,'training_started':False,
               'tensor_loading':False,'passed':mode!='failed_receipt'}
            if mode=='cpu_success':r.update(version='b500-cpu-acceptance-v1',model_weights_loaded=False,gpu_initialized=False)
            if mode=='wrong_receipt':r['protocol_sha256']='wrong'
            p=Path(args[-1]);p.parent.mkdir(parents=True);p.write_text(json.dumps(r))
            return SimpleNamespace(returncode=0 if r['passed']else 1)
        def publish(*args):
            published.append(args)
            if mode=='push_failure':raise RuntimeError('push failed')
        try:
            output=io.StringIO()
            with patch.object(runner,'git',side_effect=git),patch.object(runner.subprocess,'run',side_effect=process),patch.object(runner,'publish',side_effect=publish),contextlib.redirect_stdout(output),contextlib.redirect_stderr(io.StringIO()):
                if mode in ('success','failed_receipt','cpu_success'):
                    state=runner.execute(repo,Path(sys.executable),repo/'assets',repo/'runtime','cpu'if mode=='cpu_success'else 'identity')
                    self.assertTrue(state['uploaded']);self.assertEqual(state['preflight_passed'],mode!='failed_receipt')
                    expected='CPU acceptance passed.'if mode=='cpu_success'else 'Identity preflight passed.'if mode=='success'else 'Failed identity receipt returned'
                    self.assertIn(expected,output.getvalue())
                else:
                    with self.assertRaises(RuntimeError):runner.execute(repo,Path(sys.executable),repo/'assets',repo/'runtime')
            if mode in ('clone_failure','missing_tool','no_receipt','wrong_receipt'):
                self.assertFalse(any(c[0]=='commit'for c in calls));self.assertFalse(published)
            if mode=='commit_failure':self.assertFalse(published)
            if mode=='push_failure':
                state=json.loads(next((repo/'outputs').glob('*-state.json')).read_text())
                self.assertEqual(state['commit'],'a'*40);self.assertFalse(state['uploaded'])
            self.assertFalse(any('--global'in c for c in calls))
            if mode=='success':self.assertIn(['config','--local','user.email','b500-evidence@localhost'],calls)
        finally:
            # Verify the exact recursive-cleanup target remains inside workspace outputs.
            repo.resolve().relative_to((ROOT/'outputs').resolve())
            temporary.cleanup()

    def test_clone_failure_blocks_everything(self):self.exercise('clone_failure')
    def test_missing_script_blocks_execution(self):self.exercise('missing_tool')
    def test_missing_receipt_blocks_commit(self):self.exercise('no_receipt')
    def test_wrong_receipt_blocks_commit(self):self.exercise('wrong_receipt')
    def test_valid_failed_preflight_is_returned(self):self.exercise('failed_receipt')
    def test_missing_author_uses_clone_local_identity(self):self.exercise('success')
    def test_commit_failure_blocks_push(self):self.exercise('commit_failure')
    def test_push_failure_preserves_commit(self):self.exercise('push_failure')
    def test_cpu_acceptance_return_flow(self):self.exercise('cpu_success')


if __name__=='__main__':unittest.main()
