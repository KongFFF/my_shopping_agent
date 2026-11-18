"""Run synthetic CPU acceptance; never load models, start services or train B."""
import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--asset-project',type=Path);p.add_argument('--runtime-project',type=Path);a=p.parse_args()
    package=ROOT/'experiments/group_weight_b500_v1'
    m=json.loads((package/'implementation_manifest.json').read_text(encoding='utf-8'))
    errors=[];checks=[]
    for n,h in m['files_sha256'].items():
        actual=digest(ROOT/n);checks.append({'path':n,'matched':actual==h})
        if actual!=h:errors.append('Implementation hash mismatch: '+n)
    if digest(package/'protocol.json')!=m['protocol_sha256']:errors.append('Research protocol changed')
    if a.asset_project:
        spec=json.loads((package/'protocol.json').read_text(encoding='utf-8'))
        binding=spec['baseline']['reference_launch_binding'];old_project=Path(binding['project'])
        # Source identity only, no veRL/Ray import or GPU initialization.
        for n,h in binding['source_sha256'].items():
            relative=Path(n).relative_to(old_project)
            if relative.parts[0]!='.venv':continue
            target=a.asset_project/relative
            matched=target.is_file()and digest(target)==h
            checks.append({'path':str(target),'matched':matched})
            if not matched:errors.append('Installed historical engine source differs: '+str(target))
        wheel=ROOT/'analysis/grpo_mechanism_audit_20261116/package_reference'
        wm=json.loads((wheel/'manifest.json').read_text(encoding='utf-8'))['files_sha256']
        for name in ['verl/trainer/ppo/core_algos.py','verl/workers/utils/padding.py']:
            target=a.asset_project/'.venv/lib/python3.12/site-packages'/name
            matched=target.is_file()and digest(target)==wm[name]
            checks.append({'path':str(target),'matched':matched})
            if not matched:errors.append('Installed loss/padding differs from CPU-tested reference: '+name)
        target=a.asset_project/'.venv/lib/python3.12/site-packages/verl/trainer/ppo/rollout_corr_helper.py'
        matched=target.is_file()and digest(target)==digest(package/'reference/rollout_corr_helper.py')
        checks.append({'path':str(target),'matched':matched})
        if not matched:errors.append('Installed rollout bypass helper differs from locked reference')
    report={'version':'b500-cpu-acceptance-v1','at':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':m['protocol_sha256'],'implementation_manifest_sha256':digest(package/'implementation_manifest.json'),
        'training_started':False,'model_weights_loaded':False,'checks':checks,'errors':errors,'passed':False}
    if not errors:
        import torch
        if torch.cuda.is_initialized():raise RuntimeError('Unexpected CUDA initialization before tests')
        # The repair must qualify both method compatibility and zero-update
        # recovery/control gates in the native environment, without CUDA.
        stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(ROOT/'tools'),pattern='*b500*.py')
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        report.update(tests_run=result.testsRun,test_log=stream.getvalue(),torch_version=torch.__version__,
            gpu_initialized=torch.cuda.is_initialized(),passed=result.wasSuccessful()and not torch.cuda.is_initialized())
        if not result.wasSuccessful():errors.append('CPU acceptance tests failed')
        if torch.cuda.is_initialized():errors.append('Unexpected CUDA initialization')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'passed':report['passed'],'tests_run':report.get('tests_run',0),'errors':errors,
        'receipt':str(a.output),'training_started':False},ensure_ascii=False,indent=2))
    return 0 if report['passed']else 1


if __name__=='__main__':sys.exit(main())
