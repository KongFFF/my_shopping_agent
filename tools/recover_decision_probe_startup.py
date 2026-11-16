"""Recover only the audited zero-generation startup failure; preserve its files."""
import argparse
from datetime import datetime, timezone
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from decision_probe_common import ROOT, PACKAGE, file_hash, load, save, runtime_receipt
from run_decision_probe import checked_run, init_run

OLD = ROOT/'outputs/decision_probe_v1/run800_v1'
NEW = ROOT/'outputs/decision_probe_v1/run800_startup_recovery_v1'


def validate(old):
    cfg = checked_run(old)
    for name in ('usage.jsonl','events.jsonl','trajectories.jsonl','restorations.jsonl',
                 'ready.json','service_pids.json','model.log','environment.log','driver.lock'):
        if (old/name).exists():
            raise ValueError('Recovery is restricted to a pre-service failure; found '+name)
    log = (old/'supervisor.log').read_text(encoding='utf-8')
    if 'KeyboardInterrupt' not in log or 'file_hash' not in log:
        raise ValueError('Failure differs from the audited interrupted hash check')
    budget, stopped = load(old/'budget.json'), load(old/'services_stopped.json')
    elapsed = stopped['at']-budget['started_at']
    if not 0 <= elapsed <= 60 or stopped['budget_expired']:
        raise ValueError('Unexpected startup duration; do not automatically recover')
    cpu = load(old/'cpu_check.json')
    if not cpu['passed'] or cpu['protocol_sha256'] != cfg['protocol_sha256']:
        raise ValueError('Original CPU validation/protocol differs')
    return cfg, elapsed


def prepare(old=OLD, new=NEW):
    cfg, elapsed = validate(old)
    if new.exists(): raise ValueError('Recovery directory already exists; preserve it, do not overwrite')
    for port in (5714,8014):
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1',port)) == 0:
                raise ValueError('Probe port occupied: '+str(port))
    # Catch a previous pipeline/serve still running even before it binds a port.
    proc = Path('/proc')
    if not proc.is_dir(): raise ValueError('Recovery process check requires Linux /proc')
    for p in proc.iterdir():
        if not p.name.isdigit(): continue
        try: args = (p/'cmdline').read_bytes().split(b'\0')
        except OSError: continue
        if any(Path(a.decode(errors='replace')).name == 'run_decision_probe.py' for a in args if a):
            raise ValueError('A probe process is still running: PID '+p.name)
    bundle, model = Path(cfg['bundle']), Path(cfg['model_dir'])
    if runtime_receipt(bundle) != load(PACKAGE/'runtime_receipt.json'):
        raise ValueError('Native runtime differs')
    print('Rechecking original800 weight/export hashes on CPU; this may take a while.',flush=True)
    init_run(new,bundle,model)
    shutil.copy2(old/'cpu_check.json',new/'cpu_check.json')
    commit = subprocess.check_output(['git','rev-parse','refs/remotes/origin/codex/decision-probe800-v1'],cwd=ROOT,text=True).strip()
    (new/'code_ref.txt').write_text(commit+'\n',encoding='utf-8')
    receipt = {'schema':'zero-generation-startup-recovery-v1',
        'parent':str(old.resolve()),'original_service_window_seconds':elapsed,
        'prior_generated_requests':0,'prior_tasks_attempted':0,
        'old_files_sha256':{p.name:file_hash(p) for p in old.iterdir() if p.is_file()},
        'at':datetime.now(timezone.utc).isoformat(),
        'recovery_tool_sha256':file_hash(Path(__file__)),
        'budget_note':'One fresh <=7200s probe attempt; prior interrupted startup recorded separately, not erased.'}
    save(new/'startup_recovery.json',receipt)
    fresh=load(new/'run.json');fresh['startup_recovery']=receipt;save(new/'run.json',fresh)
    print('PASS: recovery prepared; original run and clock preserved. No actor inference performed.',flush=True)


def sample(new=NEW):
    receipt = load(new/'startup_recovery.json')
    if receipt['prior_generated_requests'] != 0: raise ValueError('Not a zero-generation recovery')
    if (new/'budget.json').exists(): raise ValueError('Recovery attempt already started; return logs instead of rerunning')
    print('Starting one recovery attempt. The original interrupted startup remains recorded separately.',flush=True)
    print('Weight hashes are checked again before services launch. Startup progress prints every 15 seconds.',flush=True)
    child = subprocess.Popen([sys.executable,str(ROOT/'tools/run_decision_probe.py'),'pipeline','--run-dir',str(new)],cwd=ROOT,start_new_session=True)
    try:
        while True:
            try: status=child.wait(timeout=15); break
            except subprocess.TimeoutExpired:
                if (new/'services_stopped.json').exists() and not (new/'ready.json').exists():
                    print('Supervisor stopped before readiness. Stopping the waiting pipeline; startup logs follow.',flush=True)
                    child.send_signal(__import__('signal').SIGINT)
                    status=child.wait(timeout=45)
                    if status == 0: status=1
                    break
                if (new/'ready.json').exists(): stage='sampling (see phase log)'
                elif (new/'model.log').exists(): stage='loading model/environment services'
                else: stage='pre-service weight hash verification'
                print('Probe alive: '+stage,flush=True)
        if status:
            for name in ('supervisor.log','model.log','environment.log'):
                p=new/name
                if p.exists():
                    print('\n--- '+name+' (last 60 lines) ---',flush=True)
                    print('\n'.join(p.read_text(encoding='utf-8',errors='replace').splitlines()[-60:]),flush=True)
            raise SystemExit(status)
    except KeyboardInterrupt:
        child.send_signal(__import__('signal').SIGINT)
        child.wait()
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','sample'])
    a=p.parse_args();prepare() if a.phase=='prepare' else sample()
