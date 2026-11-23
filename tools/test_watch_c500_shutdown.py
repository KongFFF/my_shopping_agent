import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import watch_c500_shutdown as w


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run = Path(self.temp.name)
        self.sync = patch.object(w.os,'sync',create=True);self.sync.start()
    def tearDown(self):
        self.sync.stop();self.temp.cleanup()
    def events(self):
        return [json.loads(x)for x in (self.run/'auto_shutdown_events.jsonl').read_text().splitlines()]
    def check_exit(self, marker, expected):
        if marker:(self.run/marker).write_text('{}')
        calls=[];states=iter([True,False])
        w.finish(self.run,{},10,sleep=lambda n:None,alive=lambda _:next(states),shutdown=lambda:calls.append(1))
        self.assertEqual(calls,[1]);self.assertEqual(self.events()[-1]['reason'],expected)
    def test_success(self):self.check_exit('training_complete.json','training_complete')
    def test_error(self):self.check_exit('stopped.json','training_error')
    def test_crash_without_receipt(self):self.check_exit(None,'supervisor_exited_without_receipt')
    def test_350_not_completion(self):
        (self.run/'phase350_complete.json').write_text('{}');calls=[];states=iter([True,True,False]);ticks=[]
        w.finish(self.run,{},0,sleep=ticks.append,alive=lambda _:next(states),shutdown=lambda:calls.append(1))
        self.assertEqual(len(ticks),2);self.assertEqual(calls,[1])
    def test_cancel_while_training(self):
        (self.run/'auto_shutdown.cancel').touch();calls=[]
        w.finish(self.run,{},60,alive=lambda _:True,shutdown=lambda:calls.append(1))
        self.assertFalse(calls);self.assertEqual(self.events()[-1]['event'],'cancelled')
    def test_cancel_in_grace(self):
        calls=[]
        w.finish(self.run,{},10,sleep=lambda n:(self.run/'auto_shutdown.cancel').touch(),alive=lambda _:False,shutdown=lambda:calls.append(1))
        self.assertFalse(calls);self.assertEqual(self.events()[-1]['event'],'cancelled')
    def test_shutdown_failure_propagates_no_retry(self):
        calls=[]
        def fail():calls.append(1);raise RuntimeError('shutdown denied')
        with self.assertRaises(RuntimeError):w.finish(self.run,{},0,alive=lambda _:False,shutdown=fail)
        self.assertEqual(calls,[1])
    def test_reused_PID_or_boot_is_not_same_process(self):
        old={'pid':123,'start_ticks':'10','boot_id':'boot1','argv':[b'python']}
        for field,value in [('start_ticks','11'),('boot_id','boot2'),('argv',[b'other'])]:
            new={**old,field:value}
            with patch.object(w,'snapshot',return_value=new):self.assertFalse(w.same_process(old))
        with patch.object(w,'snapshot',return_value=old):self.assertTrue(w.same_process(old))
    def test_binding_rejects_other_run_or_dead_process(self):
        (self.run/'run.json').write_text(json.dumps({'experiment':'C500','gamma':4,'repo':'/code'}))
        (self.run/'supervisor.pid').write_text('123')
        with patch.object(w,'snapshot',return_value=None):
            with self.assertRaises(ValueError):w.bind_supervisor(self.run)
        identity={'pid':123,'start_ticks':'1','boot_id':'b','argv':[b'python',str(Path('/code')/'tools/c500_run.py').encode(),b'run',b'--run-dir',b'/other']}
        with patch.object(w,'snapshot',return_value=identity):
            with self.assertRaises(ValueError):w.bind_supervisor(self.run)
        identity['argv'][-1]=str(self.run).encode()
        with patch.object(w,'snapshot',return_value=identity):self.assertEqual(w.bind_supervisor(self.run),identity)
    def test_zombie_not_live(self):
        stat='123 (process name) Z '+' '.join(['0']*20)
        (self.run/'123').mkdir();(self.run/'123/stat').write_text(stat)
        with patch.object(w,'PROC',self.run):self.assertIsNone(w.snapshot(123))


if __name__=='__main__':unittest.main()
