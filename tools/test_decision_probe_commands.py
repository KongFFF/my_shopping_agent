"""Exercise shell failure handling with a fake interpreter; no model/services."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get('DECISION_TEST_BASH') or shutil.which('bash')


@unittest.skipUnless(BASH, 'Bash is required')
class CommandTests(unittest.TestCase):
    def test_failure_stops_child_keeps_parent_and_logs_stderr(self):
        (ROOT/'outputs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs', prefix='probe_shell_test_') as tmp:
            project = Path(tmp)
            (project/'tools').mkdir()
            script = project/'tools/decision_probe_commands.sh'
            shutil.copy2(ROOT/'tools/decision_probe_commands.sh', script)
            fake = project/'upstream/shopping-grpo-longhorizon/.venv/bin/python'
            fake.parent.mkdir(parents=True)
            fake.write_bytes(b'#!/usr/bin/env bash\necho injected-failure >&2\nexit 23\n')
            fake.chmod(0o755)
            command = 'set -euo pipefail; status=0; bash "$1" prepare || status=$?; printf "PARENT_ALIVE=%s\\n" "$status"'
            proc = subprocess.run([BASH, '-c', command, 'test', script.as_posix()], text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn('PARENT_ALIVE=23', proc.stdout)
            self.assertIn('EXIT_CODE=23', proc.stdout)
            logs = list((project/'outputs/decision_probe_v1/command_logs').glob('*.log'))
            self.assertEqual(len(logs), 1)
            self.assertIn('injected-failure', logs[0].read_text())
            self.assertIn('FAILED:', logs[0].read_text())
            self.assertFalse((project/'outputs/decision_probe_v1/run800_v1').exists())


if __name__ == '__main__': unittest.main()
