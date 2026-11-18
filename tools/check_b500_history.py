"""Read-only AutoDL CPU identity gate for frozen B500. Never load tensors or train."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'experiments/group_weight_b500_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def resolve_source_path(original,old_project,asset,runtime):
    relative=original.relative_to(old_project)
    shared=relative.parts[0]in ('.venv','data','environments') or relative.as_posix()=='uv.lock'
    return (asset if shared else runtime)/relative


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-project',type=Path,required=True)
    parser.add_argument('--runtime-project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();manifest=read(PACKAGE/'manifest.json')
    if digest(Path(__file__))!=manifest['tools_sha256'][Path(__file__).name]:raise ValueError('Preflight tool changed')
    for n,h in manifest['files_sha256'].items():
        if digest(PACKAGE/n)!=h:raise ValueError('Frozen file changed: '+n)
    spec=read(PACKAGE/'protocol.json');asset=a.asset_project.resolve();runtime=a.runtime_project.resolve()
    report={'version':'b500-history-preflight-v1','at':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':digest(PACKAGE/'protocol.json'),'tensor_loading':False,'training_started':False,
        'checks':[],'errors':[],'packages':{},'passed':False,'scope':'identity gate only; not B implementation or launch approval'}
    def fail(message):report['errors'].append(message)
    def check(p,h,kind):
        try:actual=digest(p)
        except OSError as e:fail(f'{kind}: {p}: {e}');return
        report['checks'].append({'category':kind,'path':str(p),'sha256':actual,'matched':actual==h})
        if actual!=h:fail(f'{kind} hash mismatch: {p}')
    run=asset/'outputs/runs/grpo_baseline_20261022_202816'
    plan_path=run/'preparation/baseline_plan.json'
    try:plan=read(plan_path)
    except (OSError,ValueError)as e:plan={};fail('Cannot read original baseline_plan: '+str(e))
    expected_sft=spec['baseline']['sft_files'];model=Path(spec['baseline']['actor_init'])
    if plan:
        if Path(plan.get('model','')).resolve()!=model.resolve():fail('Original A model path differs from frozen SFT')
        original=plan.get('model_sha256',{})
        if 'model.safetensors'not in original:fail('Original A lacks model.safetensors hash')
        for n,h in original.items():
            if n in expected_sft and expected_sft[n]!=h:fail('Original A SFT hash differs from later identity receipt: '+n)
            check(model/n,h,'original_A_SFT')
        for n,h in plan.get('data_sha256',{}).items():
            if n in spec['baseline']['data_sha256']and spec['baseline']['data_sha256'][n]!=h:fail('Original A data binding differs: '+n)
    for n,h in expected_sft.items():check(model/n,h,'frozen_SFT')
    for n,h in spec['baseline']['data_sha256'].items():check(asset/n,h,'data')
    binding=spec['baseline']['reference_launch_binding'];old_project=Path(binding['project'])
    for n,h in binding['source_sha256'].items():
        old=Path(n)
        try:relative=old.relative_to(old_project)
        except ValueError:fail('Unexpected historical source outside project: '+n);continue
        # Preserve shared package/data assets; only project code may be isolated.
        target=resolve_source_path(old,old_project,asset,runtime)
        check(target,h,'baseline_runtime')
    for distribution in ('verl','vllm','torch','transformers','peft'):
        try:report['packages'][distribution]=metadata.version(distribution)
        except metadata.PackageNotFoundError:fail('Missing distribution: '+distribution)
    if report['packages'].get('verl')!='0.8.0':fail('Expected historical verl0.8.0')
    if report['packages'].get('vllm')!='0.25.1':fail('Expected historical vllm0.25.1')
    report['passed']=not report['errors'];a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':report['passed'],'checks':len(report['checks']),'errors':report['errors'],
        'receipt':str(a.output),'training_started':False},ensure_ascii=False,indent=2))
    return 0 if report['passed']else 1


if __name__=='__main__':sys.exit(main())
