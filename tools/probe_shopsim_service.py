"""CPU-only ShopSimulator TCP and reset/release probe; no shopping actions."""
import argparse
import json
import socket
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT/'upstream/shopping-grpo-longhorizon'
sys.path.insert(0,str(UPSTREAM/'src'))
from shopping_grpo.environment.client import ShopAgentEnv


def probe_slots(base_url,task_id,slots=8,timeout=35,env_factory=ShopAgentEnv):
    if not 1<=slots<=8:raise ValueError('Probe slots must be between1 and8')
    indices=[]
    with ExitStack() as stack:
        for _ in range(slots):
            env=stack.enter_context(env_factory(base_url=base_url,timeout=timeout))
            result=env.reset(task_id)
            if result.get('environment_version')!='shopsimulator-environment-v2.1':
                raise ValueError('Unexpected environment version: '+str(result.get('environment_version')))
            indices.append(env.env_idx)
        if len(set(indices))!=slots:raise ValueError('Environment leases are not unique')
    return {'environment_service_probe_passed':True,'base_url':base_url,'task_id':task_id,
            'slots_verified':slots,'leases_released':True,'environment_version':'shopsimulator-environment-v2.1',
            'shopping_actions_executed':0,'model_loaded':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:5700')
    parser.add_argument('--wait-seconds',type=float,default=0)
    parser.add_argument('--slots',type=int,default=8)
    parser.add_argument('--timeout',type=float,default=35)
    args=parser.parse_args()
    try:
        endpoint=urlsplit(args.base_url)
        if endpoint.scheme not in ('http','https') or not endpoint.hostname:raise ValueError('Invalid environment URL')
        deadline=time.monotonic()+args.wait_seconds
        while True:
            try:
                with socket.create_connection((endpoint.hostname,endpoint.port or (443 if endpoint.scheme=='https' else 80)),timeout=2):pass
                break
            except OSError:
                if time.monotonic()>=deadline:raise
                print('Waiting for ShopSimulator at '+args.base_url,flush=True);time.sleep(2)
        task=int(json.loads((UPSTREAM/'data/grpo/train.jsonl').read_text(encoding='utf-8').splitlines()[0])['task_id'])
        print(json.dumps(probe_slots(args.base_url,task,args.slots,args.timeout),ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({'environment_service_probe_passed':False,'base_url':args.base_url,
                          'error_type':type(exc).__name__,'error':str(exc)},ensure_ascii=False),file=sys.stderr)
        raise SystemExit(1)


if __name__=='__main__':main()
