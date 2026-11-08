"""Select reproducibility assets for Git; never include model/API run payloads."""
import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/semantic_batch_acceptance_20261108'
ARCHIVE_ONLY=(
 'analysis/semantic_contract_v2_r3_replay_20261102/',
 'analysis/semantic_contract_v2_r3_replay_20261102_current/',
 'analysis/semantic_contract_v2_r3_replay_20261102_draft2/',
 'analysis/semantic_r4_offline_20261103/',
 'analysis/semantic_r6_replay_20261106/',
 'analysis/semantic_r7_replay_20261106/',
 'analysis/semantic_train1000_draft_20261104/')
LOCAL_DUPLICATE={'analysis/semantic_review_four_steps_20261106/candidate_drafts1000.jsonl'}
DENIED={'.safetensors','.ckpt','.pt','.pth','.bin','.log','.pyc','.pyo'}


def selected_files():
    selected={'.gitignore','.gitattributes','.gitmodules'}
    selected.update(p.relative_to(ROOT).as_posix() for p in ROOT.glob('*.md'))
    selected.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'tools').glob('*.py'))
    for directory in ('analysis','experiments','repairs'):
        for p in (ROOT/directory).rglob('*'):
            if not p.is_file() or p.suffix in DENIED or any(x in p.parts for x in ('__pycache__','.ipynb_checkpoints')):continue
            rel=p.relative_to(ROOT).as_posix()
            if rel in LOCAL_DUPLICATE:continue
            if any(rel.startswith(prefix) for prefix in ARCHIVE_ONLY):
                if p.suffix!='.md' and p.name not in ('summary.json','report.json','change_summary.json','comparison.json'):continue
            if directory=='analysis' and p.suffix=='.py':continue  # temporary inspection helpers
            if p.suffix in ('.py','.json','.jsonl','.md','.txt','.patch','.yaml','.toml','.csv','.png'):
                selected.add(rel)
    return sorted(p for p in selected if (ROOT/p).is_file())


def verify(files):
    secrets=[];large=[]
    # Report locations only. Never echo a suspected credential to the console.
    token=re.compile(rb'(?<![A-Za-z0-9_-])(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{35,}|sk-[A-Za-z0-9_-]{24,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
    for rel in files:
        p=ROOT/rel
        if p.stat().st_size>40*1024*1024:large.append(rel)
        if p.suffix not in ('.png',):
            data=p.read_bytes()
            for hit in token.finditer(data):
                if any(word in hit.group().lower() for word in (b'example',b'placeholder',b'fake',b'dummy',b'test')):continue
                secrets.append({'path':rel,'line':data[:hit.start()].count(b'\n')+1})
    if secrets or large:
        raise ValueError(json.dumps({'possible_secrets':secrets,'oversize_files':large},ensure_ascii=False))
    manifest=json.loads((OUT/'training_bundle/manifest.json').read_text(encoding='utf-8'))
    needed={*(a['path'] for a in manifest['assets'].values()),*manifest['dependencies']}
    snapshot=json.loads((ROOT/'experiments/evidence_training_runtime_v1/manifest.json').read_text(encoding='utf-8'))
    missing=[]
    for rel in needed:
        if rel.startswith('upstream/shopping-grpo-longhorizon/'):
            sub=rel.removeprefix('upstream/shopping-grpo-longhorizon/')
            if sub not in snapshot['files']:
                result=subprocess.run(['git','-C',str(ROOT/'upstream/shopping-grpo-longhorizon'),'cat-file','-e',snapshot['base_revision']+':'+sub],capture_output=True)
                if result.returncode:missing.append(rel)
        elif rel not in files:missing.append(rel)
    if missing:raise ValueError('Frozen dependency missing from publication: '+str(missing))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',action='store_true',help='Stage only reviewed publication paths; never commit/push')
    args=parser.parse_args()
    files=selected_files()
    # Include the selection inventory itself, which is not a runtime dependency.
    inventory=OUT/'git_publication.json'
    rel_inventory=inventory.relative_to(ROOT).as_posix()
    files=sorted(set(files)|{rel_inventory})
    paths=[p for p in files if p!=rel_inventory]
    verify(paths)
    local_only=sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/'analysis').rglob('*')
                      if p.is_file() and p.relative_to(ROOT).as_posix() not in files and '__pycache__' not in p.parts)
    inventory.write_text(json.dumps({'version':'semantic-git-publication-v1','selected_files':files,
         'source_snapshot':'experiments/evidence_training_runtime_v1/manifest.json',
         'raw_bytes_without_inventory':sum((ROOT/p).stat().st_size for p in paths),
         'excluded_analysis_intermediates':local_only,'excluded_categories':['weights','virtualenvs','outputs/API attempts','logs','caches','notebook checkpoints','local download folder'],
         'frozen_runtime_dependencies_covered':True,'high_confidence_secret_scan':'passed'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.stage:
        existing=subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0')
        unrelated=[p for p in existing if p and p not in files]
        if unrelated:raise ValueError('Unrelated staged changes; leave untouched: '+str(unrelated))
        pathspec=ROOT/'outputs/git_publication_paths.nul';pathspec.parent.mkdir(exist_ok=True)
        pathspec.write_bytes(b''.join(p.encode()+b'\0' for p in files))
        staged=subprocess.run(['git','add','--pathspec-from-file='+str(pathspec),'--pathspec-file-nul'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
        if staged.returncode:raise ValueError('git add failed: '+staged.stderr[:4000])
    print(json.dumps({'selected_files':len(files),'raw_mib_without_inventory':round(sum((ROOT/p).stat().st_size for p in paths)/1048576,2),
                      'excluded_analysis_intermediates':len(local_only),'staged':args.stage},ensure_ascii=False))


if __name__=='__main__':main()
