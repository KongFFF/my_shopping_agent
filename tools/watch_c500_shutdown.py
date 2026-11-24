"""Attach to an existing C500 supervisor; shut down AutoDL only after its exit."""
import argparse
import errno
import json
import os
from pathlib import Path
import subprocess
import time

PROC = Path('/proc')
SHUTDOWN = Path('/usr/bin/shutdown')


def read_json(path):
    return json.loads(path.read_text())


def snapshot(pid):
    """PID + Linux start tick + boot ID avoid mistaking a recycled PID for this run."""
    try:
        text = (PROC/str(pid)/'stat').read_text()
        fields = text[text.rindex(')')+2:].split()
        if fields[0] in ('Z', 'X'):
            return None
        return {'pid':pid, 'start_ticks':fields[19],
                'boot_id':(PROC/'sys/kernel/random/boot_id').read_text().strip(),
                'argv':(PROC/str(pid)/'cmdline').read_bytes().split(b'\0')}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(expected):
    actual = snapshot(expected['pid'])
    return actual is not None and all(actual[k] == expected[k] for k in ('pid','start_ticks','boot_id','argv'))


def bind_supervisor(run, experiment="C500", supervisor_script="c500_run.py"):
    spec = read_json(run/'run.json')
    if spec.get('experiment') != experiment or spec.get('gamma') != 4.0:
        raise ValueError('Not the current C500 gamma4 run')
    pid = int((run/'supervisor.pid').read_text().strip())
    if pid <= 1:
        raise ValueError('Invalid supervisor PID')
    identity = snapshot(pid)
    if identity is None:
        raise ValueError('Supervisor already exited; guard NOT armed, no shutdown')
    argv = [x.decode() for x in identity['argv'] if x]
    code = str(Path(spec['repo'])/'tools'/supervisor_script)
    if code not in argv or 'run' not in argv or '--run-dir' not in argv:
        raise ValueError('PID is not the expected C500 supervisor')
    i = argv.index('--run-dir')
    if i+1 == len(argv) or Path(argv[i+1]).resolve() != run:
        raise ValueError('Supervisor run path differs')
    return identity


def write_state(run, event, **fields):
    record = {'event':event, 'at':time.time(), **fields}
    with (run/'auto_shutdown_events.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(record), flush=True)


def call_shutdown():
    try:
        subprocess.run([str(SHUTDOWN)],check=True,timeout=30)
    except OSError as error:
        if error.errno != errno.ENOEXEC:
            raise
        # Some AutoDL images expose a shell script without a shebang.
        # A shell command supports this; direct execve does not.
        subprocess.run(['/bin/bash',str(SHUTDOWN)],check=True,timeout=30)


def finish(run, identity, grace, poll=5, sleep=time.sleep, alive=same_process,
           shutdown=None):
    """Cancellation is checked throughout; phase350 receipt alone has no effect."""
    cancel = run/'auto_shutdown.cancel'
    while alive(identity):
        if cancel.exists():
            write_state(run, 'cancelled');return
        sleep(poll)
    reason = ('training_complete' if (run/'training_complete.json').exists() else
              'training_error' if (run/'stopped.json').exists() else 'supervisor_exited_without_receipt')
    write_state(run, 'supervisor_exited', reason=reason, grace_seconds=grace)
    remaining = grace
    while remaining > 0:
        if cancel.exists():
            write_state(run, 'cancelled');return
        step = min(poll, remaining);sleep(step);remaining -= step
    if cancel.exists():
        write_state(run, 'cancelled');return
    write_state(run, 'shutdown_requested', reason=reason)
    os.sync()
    # AutoDL documents this exact command, rather than systemctl/poweroff.
    (shutdown or call_shutdown)()


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--grace-seconds',type=int,default=60)
    args = parser.parse_args();run = args.run_dir.resolve()
    if not 0 <= args.grace_seconds <= 3600:
        raise ValueError('Invalid grace duration')
    if not SHUTDOWN.is_file() or not os.access(SHUTDOWN,os.X_OK):
        raise ValueError('AutoDL /usr/bin/shutdown missing or not executable; guard NOT armed')
    with (run/'auto_shutdown.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (run/'auto_shutdown.cancel').exists():
            raise ValueError('Cancellation marker already exists; guard NOT armed')
        identity = bind_supervisor(run)
        write_state(run,'armed',supervisor_pid=identity['pid'],start_ticks=identity['start_ticks'],
                    boot_id=identity['boot_id'],guard_pid=os.getpid(),grace_seconds=args.grace_seconds)
        try:
            finish(run,identity,args.grace_seconds)
        except Exception as error:
            write_state(run,'guard_failed',error=str(error));raise


if __name__ == '__main__':
    main()
