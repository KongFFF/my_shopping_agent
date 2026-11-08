"""Install the main-repo's frozen code snapshot over the pinned submodule.

Only the eight named source/config files may be written. Unknown local edits
require --backup-existing and are preserved byte-for-byte under outputs/.
No training, dataset, environment manifest or model file is changed.
"""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'experiments/evidence_training_runtime_v1'
UPSTREAM=ROOT/'upstream/shopping-grpo-longhorizon'
FILES={
 'configs/agent_loop.yaml','configs/grpo.yaml','pyproject.toml',
 'scripts/check_grpo_runtime.py','scripts/train_grpo.py',
 'src/shopping_grpo/training/grpo/adapter/agent_loop.py',
 'src/shopping_grpo/training/grpo/adapter/tools.py',
 'src/shopping_grpo/training/grpo/adapter/evidence.py'}


def sha(data):return hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()


def inspect(target,package=PACKAGE):
    target=Path(target).resolve();package=Path(package).resolve()
    manifest=json.loads((package/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('version')!='evidence-training-runtime-v1' or set(manifest['files'])!=FILES:
        raise ValueError('Unexpected runtime snapshot roster')
    plan=[]
    for rel,entry in manifest['files'].items():
        source=(package/'files'/rel).resolve();destination=(target/rel).resolve()
        if not source.is_relative_to(package/'files') or not destination.is_relative_to(target):
            raise ValueError('Snapshot path escapes intended directory')
        data=source.read_bytes()
        if sha(data)!=entry['sha256_lf']:raise ValueError('Snapshot fingerprint mismatch: '+rel)
        before=destination.read_bytes() if destination.exists() else None
        value=sha(before) if before is not None else None
        state='installed' if value==entry['sha256_lf'] else 'base' if value==entry['base_sha256_lf'] else 'local_changes'
        plan.append({'path':rel,'destination':destination,'data':data,'before':before,'state':state})
    return manifest,plan


def install(target=UPSTREAM,package=PACKAGE,*,backup_existing=False,backup_root=None):
    manifest,plan=inspect(target,package)
    pending=[p for p in plan if p['state']!='installed']
    unknown=[p['path'] for p in pending if p['state']=='local_changes']
    if unknown and not backup_existing:
        raise ValueError('Local changes require --backup-existing; nothing written: '+', '.join(unknown))
    backup=None
    if pending:
        parent=Path(backup_root) if backup_root else ROOT/'outputs/evidence_runtime_backups'
        backup=parent/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup.mkdir(parents=True,exist_ok=False)
        # Back up all touched paths before writing even the first source file.
        for p in pending:
            if p['before'] is not None:
                saved=backup/p['path'];saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(p['before'])
        (backup/'receipt.json').write_text(json.dumps({'base_revision':manifest['base_revision'],
             'paths':[{'path':p['path'],'existed':p['before'] is not None,'state':p['state']} for p in pending]},indent=2)+'\n',encoding='utf-8')
        try:
            for p in pending:
                p['destination'].parent.mkdir(parents=True,exist_ok=True)
                p['destination'].write_bytes(p['data'])
            _,after=inspect(target,package)
            if any(p['state']!='installed' for p in after):raise ValueError('Runtime installation verification failed')
        except BaseException:
            for p in pending:
                if p['before'] is not None:p['destination'].write_bytes(p['before'])
                elif p['destination'].exists():p['destination'].unlink()
            raise
    return {'installed':True,'changed_files':[p['path'] for p in pending],
            'backup':str(backup) if backup else None,'data_or_models_changed':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--check',action='store_true');choice.add_argument('--apply',action='store_true')
    parser.add_argument('--backup-existing',action='store_true')
    args=parser.parse_args()
    manifest,plan=inspect(UPSTREAM)
    revision=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()
    if revision!=manifest['base_revision']:
        raise SystemExit('Wrong submodule revision; run git submodule update --init --recursive first')
    if args.check:
        ready=all(p['state']=='installed' for p in plan)
        print(json.dumps({'installed':ready,'files':{p['path']:p['state'] for p in plan}},ensure_ascii=False))
        if not ready:raise SystemExit(1)
    else:print(json.dumps(install(backup_existing=args.backup_existing),ensure_ascii=False))


if __name__=='__main__':main()
