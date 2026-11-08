"""CPU-only nested Ray worker probe before allocating/loading any model.

Run on AutoDL with the existing training venv. Does not import veRL/vLLM,
generate trajectories, update weights or stop any existing Ray services.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def probe_worker(manifest,config,src):
    sys.path.insert(0,src)
    from shopping_grpo.training.grpo.adapter.evidence import load_backend
    bindings=config['ray_kwargs']['ray_init']['runtime_env']['env_vars']
    matches={key:os.environ.get(key)==value for key,value in bindings.items()}
    matches['PYTHONPATH']=set(bindings['PYTHONPATH'].split(os.pathsep))<=set(os.environ.get('PYTHONPATH','').split(os.pathsep))
    if not all(matches.values()):raise ValueError('Ray worker environment differs: '+json.dumps(matches))
    # Also prove the explicit trainer config survives when worker env is absent.
    os.environ.pop('SHOPPING_REWARD_TOOLS_DIR',None)
    os.environ.pop('SHOPPING_EVIDENCE_MANIFEST',None)
    backend=load_backend(manifest,trainer_config=config)
    return {'ray_environment_matches':True,'backend_loaded_with_environment_removed':True,
            'released_tasks':len(backend.released_ids),'held_tasks':len(backend.held_ids),
            'runtime_files_verified':backend.environment_check['runtime_files_verified'],'model_loaded':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--reward-tools-dir',type=Path,default=ROOT/'tools')
    args=parser.parse_args()
    manifest=str(args.manifest.resolve());tools=str(args.reward_tools_dir.resolve())
    src=str(ROOT/'upstream/shopping-grpo-longhorizon/src')
    bindings={'SHOPPING_EVIDENCE_MANIFEST':manifest,'SHOPPING_REWARD_TOOLS_DIR':tools,
              'PYTHONPATH':os.pathsep.join((src,tools))}
    config={'trainer':{'total_training_steps':20,'resume_mode':'disable'},'data':{'train_batch_size':2},
            'actor_rollout_ref':{'rollout':{'n':4}},'shopping_dynamic_sampling':{'enable':True,
                'max_num_gen_batches':3,'max_consecutive_skipped_updates':3},
            'ray_kwargs':{'ray_init':{'runtime_env':{'env_vars':bindings}}}}
    import ray
    # An isolated local CPU cluster; do not reuse or kill a previous GPU job.
    ray.init(address='local',num_cpus=2,num_gpus=0,include_dashboard=False,object_store_memory=128*1024*1024,
             runtime_env={'env_vars':bindings},log_to_driver=True)
    try:
        inner=ray.remote(num_cpus=1,num_gpus=0)(probe_worker)
        @ray.remote(num_cpus=0,num_gpus=0)
        def outer():return ray.get(inner.remote(manifest,config,src))
        result=ray.get(outer.remote(),timeout=180)
        print(json.dumps({'nested_ray_probe_passed':True,**result},ensure_ascii=False))
    finally:ray.shutdown()


if __name__=='__main__':main()
