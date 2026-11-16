"""Return only this probe's bounded artifacts via an isolated Git clone."""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile
from decision_probe_common import ROOT, PACKAGE, file_hash, load, dump


def export(run, code_ref, upload):
    cfg = load(run/'run.json')
    if cfg['version'] != 'decision-probe-v1': raise ValueError('Wrong run')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = ROOT/'outputs'/('decision_probe_'+stamp+'.zip')
    names = ['run.json', 'original_export_receipt.json', 'cpu_check.json', 'budget.json', 'ready.json',
             'smoke.json', 'sampling_stop.json', 'services_stopped.json', 'events.jsonl', 'usage.jsonl',
             'restorations.jsonl', 'environment.log', 'model.log', 'supervisor.log', 'trajectories.jsonl']
    paths = [run/n for n in names if (run/n).is_file()]
    paths += sorted(p for p in (run/'report').glob('*.json') if p.is_file())
    code_commit = subprocess.check_output(['git','rev-parse','--verify',code_ref+'^{commit}'],cwd=ROOT,text=True).strip()
    manifest = {'schema': 'decision-probe-return-v1', 'code_commit': code_commit,
                'files': {p.relative_to(run).as_posix(): file_hash(p) for p in paths}}
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths: z.write(p, 'run/'+p.relative_to(run).as_posix())
        z.writestr('manifest.json', dump(manifest)); z.write(PACKAGE/'manifest.json','protocol.json')
    if archive.stat().st_size >= 90*1024*1024: raise ValueError('Return exceeds Git limit; preserve archive')
    print('ARCHIVE:', archive); print('SHA256:', file_hash(archive))
    if not upload: return
    def git(*args, cwd=ROOT):
        return subprocess.check_output(['git', *args],cwd=cwd,text=True).strip()
    commit = git('rev-parse','--verify',code_ref+'^{commit}')
    remote = git('remote','get-url','origin')
    clone = Path(tempfile.mkdtemp(prefix='decision-return-',dir=ROOT/'outputs'))
    git('clone','--shared','--no-checkout',str(ROOT),str(clone))
    git('remote','set-url','origin',remote,cwd=clone)
    branch = 'codex/decision-probe-results-'+stamp
    git('switch','-c',branch,commit,cwd=clone)
    for key in ('user.name','user.email'):
        r = subprocess.run(['git','config','--get',key],cwd=ROOT,text=True,capture_output=True)
        if r.returncode == 0: git('config',key,r.stdout.strip(),cwd=clone)
    target = clone/'analysis/decision_probe_return_20261116/evidence.zip'
    target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(archive.read_bytes())
    git('add','--',target.relative_to(clone).as_posix(),cwd=clone)
    git('commit','-m','data: return fixed800 public decision probe',cwd=clone)
    # One-off HTTP/1.1 addresses the observed AutoDL HTTP/2 push failure.
    try: git('-c','http.version=HTTP/1.1','push','-u','origin',branch,cwd=clone)
    except subprocess.CalledProcessError:
        print('RETRY_DIR='+str(clone)); print('RETURN_BRANCH='+branch)
        print('RETURN_COMMIT='+git('rev-parse','HEAD',cwd=clone)); raise
    print('RETURN_BRANCH='+branch); print('RETURN_COMMIT='+git('rev-parse','HEAD',cwd=clone))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True,type=Path); p.add_argument('--code-ref',default='HEAD')
    p.add_argument('--git-upload',action='store_true'); a=p.parse_args(); export(a.run_dir,a.code_ref,a.git_upload)
