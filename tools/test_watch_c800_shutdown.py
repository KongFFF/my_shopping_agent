import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import watch_c500_shutdown as base
import watch_c800_shutdown as stage


class C800Guard(unittest.TestCase):
    def test_wrapper_binds_C800_supervisor_only(self):
        with patch.object(base,'bind_supervisor',return_value='bound') as bind:
            original=base.bind_supervisor
            with patch.object(base,'main',side_effect=lambda:self.assertEqual(base.bind_supervisor(Path('/run')),'bound')):
                stage.main()
            self.assertIs(base.bind_supervisor,original)
            bind.assert_called_once_with(Path('/run'),experiment='C800',supervisor_script='c800_run.py')
    def test_binding_rejects_C650_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            (run/'run.json').write_text(json.dumps({'experiment':'C800','gamma':4,'repo':'/code'}))
            (run/'supervisor.pid').write_text('123')
            identity={'pid':123,'start_ticks':'1','boot_id':'b','argv':[b'python',str(Path('/code')/'tools/c650_run.py').encode(),b'run',b'--run-dir',str(run).encode()]}
            with patch.object(base,'snapshot',return_value=identity):
                with self.assertRaises(ValueError):base.bind_supervisor(run,experiment='C800',supervisor_script='c800_run.py')
                identity['argv'][1]=str(Path('/code')/'tools/c800_run.py').encode()
                self.assertEqual(base.bind_supervisor(run,experiment='C800',supervisor_script='c800_run.py'),identity)


if __name__=='__main__':unittest.main()
