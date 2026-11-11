"""One resumable inference pipeline: smoke both models, evaluate both, report."""
import argparse
import os
import signal
import subprocess
import sys
import time
from checkpoint_pair_common import ROOT,load,save
from run_checkpoint_pair import checked


def pipeline(run):
    checked(run);lock=run/'pipeline.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    child=None;code=1;error=None
    def interrupted(*_):
        if child is not None and child.poll() is None:child.terminate()
        raise KeyboardInterrupt
    old=signal.signal(signal.SIGTERM,interrupted)
    try:
        for phase in ('smoke','full'):
            for model in ('sft','grpo'):
                cmd=[sys.executable,'-u',str(ROOT/'tools/run_checkpoint_pair.py'),'evaluate',
                     '--run-dir',str(run),'--model',model,'--phase',phase]
                print('PHASE',phase,model,flush=True)
                child=subprocess.Popen(cmd,cwd=ROOT)
                result=child.wait()
                if result:raise RuntimeError(f'{phase}/{model} exited {result}; inspect saved slot and phase status')
                child=None
        output=run/'report_final'
        if output.exists():
            result=load(output/'summary.json')
            if not result['complete']:raise ValueError('Existing final report is incomplete; preserve it and inspect')
        else:
            subprocess.run([sys.executable,str(ROOT/'tools/report_checkpoint_pair.py'),'--run-dir',str(run),
                            '--output',str(output)],cwd=ROOT,check=True)
        code=0;print('PAIR_EVALUATION_COMPLETE: report_final/summary.json',flush=True)
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}';print(error,flush=True)
        code=130 if isinstance(exc,KeyboardInterrupt) else 1
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=35)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        save(run/'pipeline_exit.json',{'exit_code':code,'error':error,'at':time.time()})
        signal.signal(signal.SIGTERM,old);lock.unlink()
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=__import__('pathlib').Path,required=True)
    a=p.parse_args();raise SystemExit(pipeline(a.run_dir.resolve()))
