"""Driver-only run control, independent of B's frozen loss implementation."""
import json
import hashlib
import math
import os
from pathlib import Path
import time

REQUIRED=('actor/model_world_size_1_rank_0.pt','actor/optim_world_size_1_rank_0.pt',
          'actor/extra_state_world_size_1_rank_0.pt','actor/fsdp_config.json','data.pt')
class PlannedBoundary(Exception):pass
class BudgetStop(Exception):pass
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2)+'\n');temp.replace(p)


def checkpoint(root,step):
    folder=Path(root)/f'global_step_{step}'
    for n in REQUIRED:
        p=folder/n
        if not p.is_file()or p.stat().st_size==0:raise RuntimeError('Incomplete checkpoint: '+str(p))
    if (Path(root)/'latest_checkpointed_iteration.txt').read_text().strip()!=str(step):raise RuntimeError('Checkpoint tracker mismatch')
    return str(folder)


def reserve(path,n,validation,deadline):
    if n<=0:raise ValueError('Empty generation batch')
    if time.time()>=deadline-120:raise BudgetStop('Wall budget reached; preserve latest safe checkpoint')
    state=read(path)if Path(path).exists()else {'train':0,'validation_physical':0}
    key='validation_physical'if validation else 'train';limit=2400 if validation else 24000
    if state[key]+n>limit:raise BudgetStop('Generation budget exhausted: '+key)
    state[key]+=n;write(path,state)


def install(trainer_class,diagnostics_module,control):
    if getattr(trainer_class.fit,'_b500_controlled',False):return
    original_fit=trainer_class.fit
    def fit(self):
        out=Path(control['output']);budget=out/'generation_budget.json'
        original_generate=self.async_rollout_manager.generate_sequences
        original_append=diagnostics_module.append_training_diagnostic
        def generate(batch):
            validation=batch.meta_info.get('validate')is True
            reserve(budget,len(batch),validation,control['deadline'])
            result=original_generate(batch)
            tokens=result.batch.get('response_mask')
            with (out/'generation_tokens.jsonl').open('a')as f:
                f.write(json.dumps({'validation':validation,'reserved_trajectories':len(batch),
                    'returned_Actor_tokens':int(tokens.sum().item())if tokens is not None else None,
                    'cumulative_multiturn_prompt_tokens':None,'at':time.time()})+'\n')
            return result
        def append(path,event,step,**kwargs):
            original_append(path,event,step,**kwargs)
            if event!='optimizer_step':return
            m=kwargs['metrics']
            for key in ('actor/lr','actor/grad_norm','actor/loss'):
                if key not in m or not math.isfinite(float(m[key])):raise RuntimeError('Nonfinite/missing '+key)
            if m['training/optimizer_updated']!=1 or float(m['actor/grad_norm'])<=0:raise RuntimeError('No finite nonzero Actor update')
            if float(m.get('b500/actor_tokens',0))<=0:raise RuntimeError('B weighting hook did not report actual Actor tokens')
            if not math.isclose(float(m['actor/lr']),1e-6*min(step/15,1),rel_tol=1e-5):raise RuntimeError('Unexpected LR/warmup after resume')
            if step in (1,350,500):
                saved=checkpoint(control['train_dir'],step)
                write(out/f'accepted_step_{step}.json',{'step':step,'checkpoint':saved,'lr':float(m['actor/lr']),
                    'grad_norm':float(m['actor/grad_norm']),'diagnostics_written':True})
            if step==control['stop_at']and step==350:
                write(out/'phase350_complete.json',{'step':350,'checkpoint':checkpoint(control['train_dir'],350)})
                raise PlannedBoundary()
        self.async_rollout_manager.generate_sequences=generate
        diagnostics_module.append_training_diagnostic=append
        try:return original_fit(self)
        except PlannedBoundary:
            self._shutdown_dump_executor();return None
        except BudgetStop as e:
            write(out/'budget_truncated.json',{'reason':str(e),'at':time.time()})
            self._shutdown_dump_executor();return None
        finally:
            self.async_rollout_manager.generate_sequences=original_generate
            diagnostics_module.append_training_diagnostic=original_append
    fit._b500_controlled=True;trainer_class.fit=fit


def setup_worker():
    own=Path(__file__);manifest=json.loads((own.parent.parent/'launch_manifest.json').read_text())
    key='experiments/group_weight_b500_v1/overlay/b500_run_control.py'
    if hashlib.sha256(own.read_bytes()).hexdigest()!=manifest['files_sha256'][key]:raise ValueError('Run control source drift')
    from b500_hook import setup_worker as weighting_setup
    weighting_setup()
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from shopping_grpo.training.grpo import dynamic_sampling
    path=os.environ.get('B500_CONTROL_FILE')
    if not path:raise ValueError('Missing run control; training disabled')
    install(RayPPOTrainer,dynamic_sampling,read(path))
