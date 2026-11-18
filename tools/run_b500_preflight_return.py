"""CPU preflight + Git return with isolated checkout and fail-closed publishing."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

BRANCH='codex/b500-identity-accepted'


def git(args,cwd,*,capture=False,check=True):
    r=subprocess.run(['git','-c','http.version=HTTP/1.1',*args],cwd=cwd,text=True,
                     stdout=subprocess.PIPE if capture else None)
    if check and r.returncode:raise RuntimeError('Git operation failed: '+args[0]+f' (exit {r.returncode})')
    return (r.stdout or '').strip() if capture else r.returncode


def file_hash(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_state(path,state):path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def validate_receipt(path,expected_hash):
    if not path.is_file():raise RuntimeError('Preflight produced no receipt; no commit or upload will run.')
    r=json.loads(path.read_text(encoding='utf-8'))
    if (r.get('version')!='b500-history-preflight-v1' or r.get('protocol_sha256')!=expected_hash
            or r.get('training_started')is not False or r.get('tensor_loading')is not False
            or not isinstance(r.get('passed'),bool)):
        raise RuntimeError('Receipt identity/schema invalid; refusing upload.')
    return r


def configure_author(clone,repo):
    # Set only this new clone. No global identity changes or inferred user email.
    for field,fallback in [('user.name','B500 Evidence Export'),('user.email','b500-evidence@localhost')]:
        value=git(['config','--get',field],repo,capture=True,check=False) or fallback
        git(['config','--local',field,value],clone)


def publish(clone,branch,commit):
    pushed=git(['push','-u','origin',branch],clone,check=False)
    # A transport error can occur after the remote accepted the commit.
    remote=git(['ls-remote','--heads','origin','refs/heads/'+branch],clone,capture=True)
    if not remote or remote.split()[0]!=commit:
        raise RuntimeError(f'Remote commit not verified (push exit {pushed}); preserve this checkout and retry push.')


def execute(repo,python,asset,runtime):
    repo=repo.resolve();out=repo/'outputs';out.mkdir(exist_ok=True)
    origin=git(['remote','get-url','origin'],repo,capture=True)
    if not origin:raise RuntimeError('Main repo has no origin.')
    # mkdtemp creates a unique empty directory: no reuse/deletion of broken checkout.
    clone=Path(tempfile.mkdtemp(prefix='b500-review-',dir=out))
    state_path=out/(clone.name+'-state.json')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    branch='codex/b500-preflight-results-'+stamp+'-'+clone.name.rsplit('-',1)[-1]
    relative=Path('analysis/b500_matching_20261117/cloud_preflight')/(stamp+'.json')
    state={'checkout':str(clone),'branch':branch,'receipt':relative.as_posix(),'phase':'clone','uploaded':False}
    write_state(state_path,state);print('STATE='+str(state_path),flush=True)
    try:
        git(['clone','--single-branch','--branch',BRANCH,origin,str(clone)],repo)
        git(['rev-parse','--verify','HEAD'],clone,capture=True)
        tool=clone/'tools/check_b500_history.py';package=clone/'experiments/group_weight_b500_v1'
        if not tool.is_file():raise RuntimeError('Cloned branch lacks preflight tool; stopped before execution.')
        m=json.loads((package/'manifest.json').read_text(encoding='utf-8'))
        for n,h in m['files_sha256'].items():
            if file_hash(package/n)!=h:raise RuntimeError('Frozen file hash mismatch: '+n)
        for n,h in m['tools_sha256'].items():
            if file_hash(clone/'tools'/n)!=h:raise RuntimeError('Frozen tool hash mismatch: '+n)
        configure_author(clone,repo)
        state['phase']='preflight';write_state(state_path,state)
        print('Running CPU identity checks; hashing model files may take a few minutes. No GPU/model loading.',flush=True)
        code=subprocess.run([str(python),str(tool),'--asset-project',str(asset),
            '--runtime-project',str(runtime),'--output',str(clone/relative)],cwd=clone).returncode
        receipt=validate_receipt(clone/relative,file_hash(package/'protocol.json'))
        if code not in (0,1)or (code==0)!=receipt['passed']:
            raise RuntimeError('Preflight exit status and receipt disagree; no upload.')
        state.update(phase='commit',preflight_exit=code,preflight_passed=receipt['passed'])
        write_state(state_path,state)
        git(['switch','-c',branch],clone)
        git(['add','--',relative.as_posix()],clone)
        git(['commit','-m','Return B500 CPU identity preflight (pass or fail)'],clone)
        commit=git(['rev-parse','--verify','HEAD'],clone,capture=True)
        state.update(phase='push',commit=commit);write_state(state_path,state)
        publish(clone,branch,commit)
        state.update(phase='complete',uploaded=True);write_state(state_path,state)
        print('PREFLIGHT_PASSED='+str(receipt['passed']).lower())
        print('RETURN_BRANCH='+branch+'\nRETURN_COMMIT='+commit)
        print('No training started. Identity preflight passed.' if receipt['passed']
              else 'No training started. Failed identity receipt returned for diagnosis.')
        return state
    except Exception as e:
        state['error']=str(e);write_state(state_path,state)
        print('STOPPED: '+str(e)+'\nPreserved checkout: '+str(clone)+'\nState: '+str(state_path),file=sys.stderr)
        if state.get('commit'):print('Retry upload only: python '+str(Path(__file__))+' --retry-push '+str(state_path),file=sys.stderr)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path);p.add_argument('--python',type=Path)
    p.add_argument('--asset-project',type=Path);p.add_argument('--runtime-project',type=Path)
    p.add_argument('--retry-push',type=Path);a=p.parse_args()
    try:
        if a.retry_push:
            state=json.loads(a.retry_push.read_text(encoding='utf-8'));clone=Path(state['checkout'])
            commit=git(['rev-parse','--verify','HEAD'],clone,capture=True)
            if commit!=state.get('commit'):raise RuntimeError('Checkout HEAD differs from saved receipt commit.')
            publish(clone,state['branch'],commit);state.update(phase='complete',uploaded=True)
            write_state(a.retry_push,state);print('RETURN_BRANCH='+state['branch']+'\nRETURN_COMMIT='+commit)
        else:
            if not a.repo:p.error('--repo is required')
            asset=a.asset_project or a.repo/'upstream/shopping-grpo-longhorizon'
            python=a.python or asset/'.venv/bin/python'
            runtime=a.runtime_project or asset/'outputs/bounded_resume500_to650/runtime'
            if not python.is_file():raise RuntimeError('Project Python missing: '+str(python))
            execute(a.repo,python,asset,runtime)
        return 0
    except Exception as e:
        print('B500 bootstrap failed: '+str(e),file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())
