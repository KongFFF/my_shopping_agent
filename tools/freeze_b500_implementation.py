"""Freeze code hashes for B500 overlay independently of immutable research protocol."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'experiments/group_weight_b500_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


def main():
    expected='390201e5e1a75b4be136ec2e0a89f09f38855b19f3a6c372bb0f64d5c122c2e8'
    if sha(P/'protocol.json')!=expected:raise ValueError('Research protocol changed')
    files=['experiments/group_weight_b500_v1/overlay/b500_weighting.py',
        'experiments/group_weight_b500_v1/overlay/b500_hook.py','tools/test_b500_weighting.py',
        'tools/run_b500_cpu_acceptance.py','tools/run_b500_preflight_return.py',
        'tools/test_b500_preflight_return.py','tools/freeze_b500_implementation.py',
        'experiments/group_weight_b500_v1/historical_a_composed_config.json',
        'analysis/grpo_mechanism_audit_20261116/package_reference/verl/trainer/ppo/core_algos.py',
        'analysis/grpo_mechanism_audit_20261116/package_reference/manifest.json']
    impl={'version':'b500-overlay-v1','protocol_sha256':expected,
          'files_sha256':{n:sha(ROOT/n)for n in files},
          'state':'local_CPU_accepted_cloud_CPU_and_GPU_integration_pending'}
    save(P/'implementation_manifest.json',impl)
    m=json.loads((P/'manifest.json').read_text(encoding='utf-8'))
    for n in ['README.md','implementation_manifest.json']:m['files_sha256'][n]=sha(P/n)
    for n in ['run_b500_preflight_return.py','run_b500_cpu_acceptance.py','freeze_b500_implementation.py']:
        m['tools_sha256'][n]=sha(ROOT/'tools'/n)
    m['workflow_revision']='2026-11-18-overlay-CPU-v1';save(P/'manifest.json',m)
    print('Frozen B500 implementation; research protocol unchanged.')


if __name__=='__main__':main()
